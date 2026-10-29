#!/usr/bin/env python3
"""Train HDiT to correct DPOT [u,v,p] from sparse pressure-tap innovations.

The channel-specific modes retain the complete DPOT field as a condition while
optimizing only the requested residual channels.  In pressure-only mode the
unoptimized velocity target is replaced by a constant, preventing the noisy
diffusion target from leaking truth velocity residuals into pressure training.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from etch.model import InvFusionResidualEDM


PRESSURE_PARAMETERS_STAGE = (
    "dpot_uvp_pressure_tap_pressure_residual_parameters_stage1_v1"
)


def physical_parameters(scenario: str, sim_id: str) -> list[float]:
    """Return deployment-known physical parameters in a stable order."""
    parts = Path(sim_id).stem.split("_")
    if scenario == "cylinder":
        return [float(parts[0])]
    if scenario == "foil":
        return [float(parts[0]), float(parts[1])]
    raise ValueError(f"Unsupported scenario: {scenario}")


def parameter_statistics(
    manifest: dict,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Compute trajectory-level Re/AoA normalization shared by all stages."""
    scenario = manifest["scenario"]
    names = ["Re"] + (["AoA_deg"] if scenario == "foil" else [])
    raw = torch.tensor(
        [physical_parameters(scenario, shard["sim_id"]) for shard in manifest["shards"]],
        dtype=torch.float32,
    )
    mean = raw.mean(dim=0)
    std = raw.std(dim=0) if len(raw) > 1 else torch.ones_like(mean)
    return names, mean, std.clamp_min(1e-6)


def validate_parameter_metadata(checkpoint: dict, dataset: Dataset) -> None:
    """Reject silently incompatible parameter encodings between stages."""
    if list(checkpoint.get("parameter_names", [])) != list(dataset.parameter_names):
        raise ValueError("Checkpoint/data parameter names differ")
    checkpoint_mean = torch.tensor(checkpoint["parameter_mean"])
    checkpoint_std = torch.tensor(checkpoint["parameter_std"])
    if not torch.allclose(checkpoint_mean, dataset.parameter_mean):
        raise ValueError("Checkpoint/data parameter means differ")
    if not torch.allclose(checkpoint_std, dataset.parameter_std):
        raise ValueError("Checkpoint/data parameter standard deviations differ")


class PhysicalParameterAdapter(nn.Module):
    """Inject normalized Re/AoA as a low-resolution spatial condition."""

    def __init__(
        self,
        parameter_dim: int,
        field_channels: int = 3,
        hidden_dim: int = 128,
        coarse_height: int = 16,
        coarse_width: int = 32,
    ) -> None:
        super().__init__()
        self.field_channels = field_channels
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.net = nn.Sequential(
            nn.Linear(parameter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(
                hidden_dim, field_channels * coarse_height * coarse_width
            ),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, condition: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        update = self.net(parameters).view(
            len(parameters),
            self.field_channels,
            self.coarse_height,
            self.coarse_width,
        )
        update = F.interpolate(
            update,
            size=condition.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (condition + 0.25 * torch.tanh(update)).clamp(0, 1)


class DPOTPressureResidualFrames(Dataset):
    def __init__(
        self,
        root: Path,
        layout_path: Path,
        frame_stride: int,
        parameter_conditioning: bool = False,
        max_shards: int | None = None,
    ):
        self.manifest = json.loads((root / "manifest.json").read_text())
        layout = json.loads(layout_path.read_text())
        self.parameter_conditioning = parameter_conditioning
        if parameter_conditioning:
            (
                self.parameter_names,
                self.parameter_mean,
                self.parameter_std,
            ) = parameter_statistics(self.manifest)
        forecast_mean = torch.tensor(self.manifest["forecast_stats"]["mean"])
        forecast_std = torch.tensor(self.manifest["forecast_stats"]["std"])
        residual_mean = torch.tensor(self.manifest["residual_stats"]["mean"])
        residual_std = torch.tensor(self.manifest["residual_stats"]["std"])
        self.forecast_mean = forecast_mean.view(3, 1, 1)
        self.forecast_std = forecast_std.clamp_min(1e-6).view(3, 1, 1)
        self.residual_mean = residual_mean.view(3, 1, 1)
        self.residual_std = residual_std.clamp_min(1e-6).view(3, 1, 1)
        forecasts, residuals, mask_ids, parameters = [], [], [], []
        mask_lookup: dict[tuple[int, ...], int] = {}
        masks = []
        expected_taps: int | None = None
        selected_shards = list(self.manifest["shards"])
        if max_shards is not None:
            if max_shards < 1:
                raise ValueError("max_shards must be positive")
            selected_shards = selected_shards[:max_shards]
        for shard in selected_shards:
            payload = torch.load(root / shard["path"], map_location="cpu", weights_only=True)
            forecast = payload["forecast_field"].reshape(-1, *payload["forecast_field"].shape[-3:])[::frame_stride]
            residual = payload["residual_field"].reshape(-1, *payload["residual_field"].shape[-3:])[::frame_stride]
            height, width = forecast.shape[-3:-1]
            sim_id = payload["sim_id"]
            if self.manifest["scenario"] == "foil":
                suffix = sim_id.rsplit("_", 1)[1]
                layout_key = next(key for key in layout["layouts"] if key.endswith(suffix))
            else:
                layout_key = next(iter(layout["layouts"]))
            # Layout coordinates refer to the released 128x256 CFD grid;
            # DPOT numerical fields use the official factor-2 subsampling.
            tap_pixels = tuple(
                (int(tap["row"]) * height // 128, int(tap["col"]) * width // 256)
                for tap in layout["layouts"][layout_key]
            )
            if expected_taps is None:
                expected_taps = len(tap_pixels)
            elif len(tap_pixels) != expected_taps:
                raise ValueError("All layouts in one scenario must use equal tap counts")
            indices = tuple(row * width + col for row, col in tap_pixels)
            if indices not in mask_lookup:
                mask = torch.zeros(3, height, width)
                for row, col in tap_pixels:
                    mask[2, row, col] = 1
                mask_lookup[indices] = len(masks)
                masks.append(mask)
            forecasts.append(forecast.permute(0, 3, 1, 2).contiguous())
            residuals.append(residual.permute(0, 3, 1, 2).contiguous())
            mask_ids.extend([mask_lookup[indices]] * len(forecast))
            if parameter_conditioning:
                raw_parameters = torch.tensor(
                    physical_parameters(self.manifest["scenario"], sim_id),
                    dtype=torch.float32,
                )
                normalized_parameters = (
                    raw_parameters - self.parameter_mean
                ) / self.parameter_std
                parameters.extend([normalized_parameters] * len(forecast))
        if expected_taps is None:
            raise ValueError("Training manifest contains no shards")
        self.forecasts = torch.cat(forecasts)
        self.residuals = torch.cat(residuals)
        self.mask_ids = torch.tensor(mask_ids)
        self.masks = torch.stack(masks)
        self.n_taps = expected_taps
        if parameter_conditioning:
            self.parameters = torch.stack(parameters)
        self.height = int(self.forecasts.shape[-2])
        self.width = int(self.forecasts.shape[-1])

    def __len__(self) -> int:
        return len(self.forecasts)

    @staticmethod
    def encode(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (((value.float() - mean) / (4 * std)).clamp(-1, 1) + 1) * .5

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        forecast = self.encode(self.forecasts[index], self.forecast_mean, self.forecast_std)
        residual = self.encode(self.residuals[index], self.residual_mean, self.residual_std)
        result = (forecast, residual, self.masks[self.mask_ids[index]])
        if self.parameter_conditioning:
            return (*result, self.parameters[index])
        return result


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensor-layout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--parameter-lr", type=float, default=1e-4)
    parser.add_argument("--parameter-conditioning", action="store_true")
    parser.add_argument("--velocity-only-loss", action="store_true")
    parser.add_argument("--pressure-only-loss", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--save-every", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.velocity_only_loss and args.pressure_only_loss:
        raise ValueError("Choose at most one channel-specific loss mode")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = DPOTPressureResidualFrames(
        args.data_dir,
        args.sensor_layout,
        args.frame_stride,
        parameter_conditioning=args.parameter_conditioning,
        max_shards=args.max_shards,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    batches = iter(loader)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    model = InvFusionResidualEDM(
        args.upstream_root, height=dataset.height, width=dataset.width, sigma_data=.5,
        field_channels=3, model_size="base", use_fp16=True,
    ).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    parameter_adapter = None
    parameter_adapter_ema = None
    parameter_groups = [{"params": model.parameters(), "lr": args.lr}]
    if args.parameter_conditioning:
        parameter_adapter = PhysicalParameterAdapter(
            len(dataset.parameter_names)
        ).to(device)
        parameter_adapter_ema = copy.deepcopy(parameter_adapter).eval().requires_grad_(False)
        parameter_groups.append(
            {"params": parameter_adapter.parameters(), "lr": args.parameter_lr}
        )
    optimizer = torch.optim.Adam(parameter_groups, betas=(.9, .99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pressure_only_loss:
        output_stage = (
            PRESSURE_PARAMETERS_STAGE
            if args.parameter_conditioning
            else "dpot_uvp_pressure_tap_pressure_residual_stage1_v1"
        )
        channel_weights = torch.tensor([0.0, 0.0, 1.0], device=device)
        loss_channels = ["p"]
    elif args.velocity_only_loss:
        output_stage = (
            "dpot_uvp_pressure_tap_velocity_parameters_stage1"
            if args.parameter_conditioning
            else "dpot_uvp_pressure_tap_residual_stage1"
        )
        channel_weights = torch.tensor([1.0, 1.0, 0.0], device=device)
        loss_channels = ["u", "v"]
    else:
        output_stage = "dpot_uvp_pressure_tap_residual_stage1"
        channel_weights = None
        loss_channels = ["u", "v", "p"]
    recent = []
    recent_sensor = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        if args.parameter_conditioning:
            forecast, residual, mask, parameters = batch
            parameters = parameters.to(device, non_blocking=True)
        else:
            forecast, residual, mask = batch
        forecast = forecast.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        training_target = residual
        if args.pressure_only_loss:
            # The HDiT target and condition both have three channels.  Do not
            # let truth velocity residuals enter the noisy target when only p
            # is observable and optimized; model-space zero is encoded as .5.
            training_target = residual.clone()
            training_target[:, :2] = 0.5
        condition = forecast
        if parameter_adapter is not None:
            condition = parameter_adapter(condition, parameters)
        optimizer.zero_grad(set_to_none=True)
        terms = model.loss_terms(
            training_target,
            condition,
            mask,
            consistency_weight=.1,
            channel_weights=channel_weights,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if parameter_adapter is not None:
            torch.nn.utils.clip_grad_norm_(parameter_adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema, model, .9999)
        if parameter_adapter is not None:
            update_ema(parameter_adapter_ema, parameter_adapter, .9999)
        recent.append(float(terms["loss"].detach()))
        recent_sensor.append(float(terms["sensor_mse"]))
        if step % 100 == 0:
            print(
                f"scenario={dataset.manifest['scenario']} step={step} "
                f"loss={np.mean(recent[-100:]):.6f} "
                f"tap_mse={np.mean(recent_sensor[-100:]):.6f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "stage": output_stage, "step": step,
                "model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scenario": dataset.manifest["scenario"], "field_channels": 3,
                "forecast_stats": dataset.manifest["forecast_stats"],
                "residual_stats": dataset.manifest["residual_stats"],
                "sensor_layout": str(args.sensor_layout),
                "n_taps": dataset.n_taps,
                "loss_channels": loss_channels,
                "target": (
                    "CFD_truth_pressure_minus_DPOT_pressure"
                    if args.pressure_only_loss
                    else "CFD_truth_minus_DPOT"
                ),
                "prediction_composition": (
                    "pressure_hat=DPOT_pressure+predicted_pressure_residual"
                    if args.pressure_only_loss
                    else "field_hat=DPOT_field+predicted_residual"
                ),
                "unoptimized_target_channels": (
                    "constant_model_space_zero"
                    if args.pressure_only_loss
                    else "unchanged"
                ),
                "normalization_type": "train_forecast_and_residual_channel_zscore",
                "normalization_sigmas": 4.0,
                "gauge_transform": "none",
                "recent_loss": float(np.mean(recent[-100:])),
                "recent_tap_mse": float(np.mean(recent_sensor[-100:])),
            }
            if parameter_adapter is not None:
                payload.update({
                    "parameter_adapter_state_dict": parameter_adapter.state_dict(),
                    "parameter_adapter_ema_state_dict": parameter_adapter_ema.state_dict(),
                    "parameter_names": dataset.parameter_names,
                    "parameter_mean": dataset.parameter_mean.tolist(),
                    "parameter_std": dataset.parameter_std.tolist(),
                })
            torch.save(payload, args.output_dir / f"checkpoint_{step:06d}.pt")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "stage": output_stage, "scenario": dataset.manifest["scenario"],
        "n_train_frames": len(dataset), "frame_stride": args.frame_stride,
        "steps": args.steps, "sensor_layout": str(args.sensor_layout),
        "n_taps": dataset.n_taps,
        "parameter_conditioning": args.parameter_conditioning,
        "loss_channels": loss_channels,
        "target": (
            "CFD truth pressure minus DPOT pressure"
            if args.pressure_only_loss
            else "CFD truth minus DPOT"
        ),
        "prediction_composition": (
            "pressure_hat = DPOT pressure + predicted pressure residual"
            if args.pressure_only_loss
            else "field_hat = DPOT field + predicted residual"
        ),
        "unoptimized_target_channels": (
            "constant_model_space_zero"
            if args.pressure_only_loss
            else "unchanged"
        ),
        "normalization_type": "train_forecast_and_residual_channel_zscore",
        "normalization_sigmas": 4.0,
        "gauge_transform": "none",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
