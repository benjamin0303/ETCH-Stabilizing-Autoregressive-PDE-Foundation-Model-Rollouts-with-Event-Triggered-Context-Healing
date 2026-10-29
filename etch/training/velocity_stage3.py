#!/usr/bin/env python3
"""Fine-tune Stage-2 HDiT with causal pressure-tap history."""

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

from etch.training.stage1 import (
    PhysicalParameterAdapter,
    parameter_statistics,
    physical_parameters,
    update_ema,
    validate_parameter_metadata,
)
from etch.training.velocity_stage2 import encode
from etch.model import InvFusionResidualEDM


class TemporalPressureResidualFrames(Dataset):
    """DPOT targets paired with causal, trajectory-local pressure histories."""

    def __init__(
        self,
        root: Path,
        layout_path: Path,
        history_length: int,
        target_stride: int,
        parameter_conditioning: bool = False,
        history_clamp: bool = True,
        max_shards: int | None = None,
    ) -> None:
        if history_length < 2:
            raise ValueError("history_length must be at least two")
        if target_stride < 1:
            raise ValueError("target_stride must be positive")
        self.manifest = json.loads((root / "manifest.json").read_text())
        layout = json.loads(layout_path.read_text())
        self.history_length = history_length
        self.parameter_conditioning = parameter_conditioning
        self.history_clamp = history_clamp
        if parameter_conditioning:
            (
                self.parameter_names,
                self.parameter_mean,
                self.parameter_std,
            ) = parameter_statistics(self.manifest)
        self.forecast_mean = torch.tensor(
            self.manifest["forecast_stats"]["mean"]
        ).view(3, 1, 1)
        self.forecast_std = torch.tensor(
            self.manifest["forecast_stats"]["std"]
        ).clamp_min(1e-6).view(3, 1, 1)
        self.residual_mean = torch.tensor(
            self.manifest["residual_stats"]["mean"]
        ).view(3, 1, 1)
        self.residual_std = torch.tensor(
            self.manifest["residual_stats"]["std"]
        ).clamp_min(1e-6).view(3, 1, 1)
        self.pressure_reference = float(
            self.forecast_mean[2] + self.residual_mean[2]
        )

        forecasts: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        histories: list[torch.Tensor] = []
        parameters: list[torch.Tensor] = []
        mask_ids: list[int] = []
        masks: list[torch.Tensor] = []
        mask_lookup: dict[tuple[int, ...], int] = {}
        expected_taps: int | None = None

        selected_shards = list(self.manifest["shards"])
        if max_shards is not None:
            if max_shards < 1:
                raise ValueError("max_shards must be positive")
            selected_shards = selected_shards[:max_shards]
        for shard in selected_shards:
            payload = torch.load(
                root / shard["path"], map_location="cpu", weights_only=True
            )
            forecast = payload["forecast_field"].reshape(
                -1, *payload["forecast_field"].shape[-3:]
            )
            residual = payload["residual_field"].reshape(
                -1, *payload["residual_field"].shape[-3:]
            )
            height, width = forecast.shape[-3:-1]
            sim_id = payload["sim_id"]
            if self.manifest["scenario"] == "foil":
                suffix = sim_id.rsplit("_", 1)[1]
                layout_key = next(
                    key for key in layout["layouts"] if key.endswith(suffix)
                )
            else:
                layout_key = next(iter(layout["layouts"]))
            tap_pixels = tuple(
                (
                    int(tap["row"]) * height // 128,
                    int(tap["col"]) * width // 256,
                )
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
            mask_id = mask_lookup[indices]

            truth_pressure = forecast[..., 2].float() + residual[..., 2].float()
            tap_series = torch.stack(
                [truth_pressure[:, row, col] for row, col in tap_pixels], dim=1
            )
            history_windows = tap_series.unfold(
                0, history_length, 1
            ).permute(0, 2, 1).contiguous()
            target_indices = torch.arange(
                history_length - 1, len(forecast), target_stride
            )
            history_indices = target_indices - (history_length - 1)
            forecasts.append(
                forecast[target_indices].permute(0, 3, 1, 2).contiguous()
            )
            residuals.append(
                residual[target_indices].permute(0, 3, 1, 2).contiguous()
            )
            histories.append(history_windows[history_indices].half())
            mask_ids.extend([mask_id] * len(target_indices))
            if parameter_conditioning:
                raw_parameters = torch.tensor(
                    physical_parameters(self.manifest["scenario"], sim_id),
                    dtype=torch.float32,
                )
                normalized_parameters = (
                    raw_parameters - self.parameter_mean
                ) / self.parameter_std
                parameters.extend([normalized_parameters] * len(target_indices))

        if expected_taps is None:
            raise ValueError("Training manifest contains no shards")
        self.forecasts = torch.cat(forecasts)
        self.residuals = torch.cat(residuals)
        self.histories = torch.cat(histories)
        self.mask_ids = torch.tensor(mask_ids)
        self.masks = torch.stack(masks)
        self.n_taps = expected_taps
        if parameter_conditioning:
            self.parameters = torch.stack(parameters)
        self.height = int(self.forecasts.shape[-2])
        self.width = int(self.forecasts.shape[-1])

    def __len__(self) -> int:
        return len(self.forecasts)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history = self.histories[index].float() - self.pressure_reference
        history = (
            history - self.residual_mean[2]
        ) / (4 * self.residual_std[2])
        if self.history_clamp:
            history = history.clamp(-1, 1)
        result = (
            self.forecasts[index].float(),
            self.residuals[index].float(),
            self.masks[self.mask_ids[index]],
            history,
        )
        if self.parameter_conditioning:
            return (*result, self.parameters[index])
        return result


class TemporalPressureAdapter(nn.Module):
    """Map a tap-time matrix to a small dense update of the HDiT prior."""

    def __init__(
        self,
        history_length: int,
        n_taps: int,
        field_channels: int = 3,
        hidden_dim: int = 512,
        coarse_height: int = 16,
        coarse_width: int = 32,
    ) -> None:
        super().__init__()
        self.field_channels = field_channels
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        input_dim = history_length * n_taps
        output_dim = field_channels * coarse_height * coarse_width
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, forecast_condition: torch.Tensor, pressure_history: torch.Tensor
    ) -> torch.Tensor:
        update = self.net(pressure_history.flatten(1)).view(
            len(pressure_history),
            self.field_channels,
            self.coarse_height,
            self.coarse_width,
        )
        update = F.interpolate(
            update,
            size=forecast_condition.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (forecast_condition + 0.25 * torch.tanh(update)).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensor-layout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--target-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--velocity-only-loss", action="store_true")
    parser.add_argument("--parameter-conditioning", action="store_true")
    parser.add_argument("--parameter-lr", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = TemporalPressureResidualFrames(
        args.data_dir,
        args.sensor_layout,
        args.history_length,
        args.target_stride,
        parameter_conditioning=args.parameter_conditioning,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )
    batches = iter(loader)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    expected_stage = (
        "dpot_uv_pressure_tap_velocity_parameters_stage2"
        if args.parameter_conditioning
        else "dpot_uv_pressure_tap_stage2"
    )
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(
            f"Expected {expected_stage}, got {checkpoint.get('stage')}"
        )
    if checkpoint.get("scenario") != dataset.manifest["scenario"]:
        raise ValueError(
            "Checkpoint/data scenario mismatch: "
            f"{checkpoint.get('scenario')} vs {dataset.manifest['scenario']}"
        )
    if args.parameter_conditioning:
        validate_parameter_metadata(checkpoint, dataset)

    backbone = InvFusionResidualEDM(
        args.upstream_root,
        dataset.height,
        dataset.width,
        sigma_data=0.5,
        field_channels=3,
        model_size="base",
        use_fp16=True,
    ).to(device)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone_ema = copy.deepcopy(backbone).eval().requires_grad_(False)
    backbone_ema.load_state_dict(checkpoint["ema_state_dict"])
    adapter = TemporalPressureAdapter(
        args.history_length, dataset.n_taps
    ).to(device)
    adapter_ema = copy.deepcopy(adapter).eval().requires_grad_(False)
    parameter_adapter = None
    parameter_adapter_ema = None
    if args.parameter_conditioning:
        parameter_adapter = PhysicalParameterAdapter(
            len(dataset.parameter_names)
        ).to(device)
        parameter_adapter.load_state_dict(
            checkpoint["parameter_adapter_state_dict"]
        )
        parameter_adapter_ema = copy.deepcopy(parameter_adapter).eval().requires_grad_(False)
        parameter_adapter_ema.load_state_dict(
            checkpoint["parameter_adapter_ema_state_dict"]
        )

    parameter_groups = [
            {"params": backbone.parameters(), "lr": args.backbone_lr},
            {"params": adapter.parameters(), "lr": args.adapter_lr},
        ]
    if parameter_adapter is not None:
        parameter_groups.append(
            {"params": parameter_adapter.parameters(), "lr": args.parameter_lr}
        )
    optimizer = torch.optim.Adam(parameter_groups, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    forecast_mean = dataset.forecast_mean.to(device)[None]
    forecast_std = dataset.forecast_std.to(device)[None]
    target_mean = dataset.residual_mean.to(device)[None]
    target_std = dataset.residual_std.to(device)[None]
    channel_weights = (
        torch.tensor([1.0, 1.0, 0.0], device=device)
        if args.velocity_only_loss
        else None
    )
    output_stage = (
        "dpot_uv_pressure_history_parameters_stage3"
        if args.parameter_conditioning and args.velocity_only_loss
        else "dpot_uv_pressure_history_stage3"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_loss: list[float] = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        if args.parameter_conditioning:
            forecast, residual, mask, history, parameters = batch
            parameters = parameters.to(device, non_blocking=True)
        else:
            forecast, residual, mask, history = batch
        forecast = forecast.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)

        reconstruction_base = forecast.clone()
        reconstruction_base[:, 2] = dataset.pressure_reference
        correction_target = residual.clone()
        correction_target[:, 2] = (
            forecast[:, 2] + residual[:, 2] - dataset.pressure_reference
        )
        forecast_encoded = encode(
            reconstruction_base, forecast_mean, forecast_std
        )
        target_encoded = encode(correction_target, target_mean, target_std)
        adapted_condition = adapter(forecast_encoded, history)
        if parameter_adapter is not None:
            adapted_condition = parameter_adapter(
                adapted_condition, parameters
            )

        optimizer.zero_grad(set_to_none=True)
        terms = backbone.loss_terms(
            target_encoded,
            adapted_condition,
            mask,
            consistency_weight=0.1,
            channel_weights=channel_weights,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if parameter_adapter is not None:
            torch.nn.utils.clip_grad_norm_(parameter_adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(backbone_ema, backbone, 0.9999)
        update_ema(adapter_ema, adapter, 0.9999)
        if parameter_adapter is not None:
            update_ema(parameter_adapter_ema, parameter_adapter, 0.9999)

        recent_loss.append(float(terms["loss"].detach()))
        if step % 100 == 0:
            print(
                f"scenario={dataset.manifest['scenario']} step={step} "
                f"loss={np.mean(recent_loss[-100:]):.6f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                    "stage": output_stage,
                    "step": step,
                    "model_state_dict": backbone.state_dict(),
                    "ema_state_dict": backbone_ema.state_dict(),
                    "adapter_state_dict": adapter.state_dict(),
                    "adapter_ema_state_dict": adapter_ema.state_dict(),
                    "scenario": dataset.manifest["scenario"],
                    "field_channels": 3,
                    "forecast_stats": dataset.manifest["forecast_stats"],
                    "residual_stats": dataset.manifest["residual_stats"],
                    "pressure_reference": dataset.pressure_reference,
                    "history_length": args.history_length,
                    "n_taps": dataset.n_taps,
                    "target_stride": args.target_stride,
                    "condition_channels": ["u", "v", "pressure_history"],
                    "loss_channels": ["u", "v"] if args.velocity_only_loss else ["u", "v", "p"],
                    "init_checkpoint": str(args.init_checkpoint),
                    "sensor_layout": str(args.sensor_layout),
                    "recent_loss": float(np.mean(recent_loss[-100:])),
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

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": output_stage,
                "scenario": dataset.manifest["scenario"],
                "n_train_frames": len(dataset),
                "history_length": args.history_length,
                "target_stride": args.target_stride,
                "condition_channels": ["u", "v", "pressure_history"],
                "loss_channels": ["u", "v"] if args.velocity_only_loss else ["u", "v", "p"],
                "steps": args.steps,
                "n_taps": dataset.n_taps,
                "init_checkpoint": str(args.init_checkpoint),
                "sensor_layout": str(args.sensor_layout),
                "parameter_conditioning": args.parameter_conditioning,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
