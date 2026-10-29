#!/usr/bin/env python3
"""Fine-tune Stage-1 HDiT without a sample-specific DPOT pressure prior."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from etch.training.stage1 import (
    DPOTPressureResidualFrames,
    PhysicalParameterAdapter,
    update_ema,
    validate_parameter_metadata,
)
from etch.model import InvFusionResidualEDM


class DPOTUVPressureTapFrames(DPOTPressureResidualFrames):
    """Return raw DPOT forecast, truth-minus-DPOT residual, and tap mask."""

    def __init__(
        self,
        root: Path,
        layout_path: Path,
        frame_stride: int,
        parameter_conditioning: bool = False,
    ):
        super().__init__(
            root,
            layout_path,
            frame_stride,
            parameter_conditioning=parameter_conditioning,
        )
        truth_pressure = self.forecasts[:, 2] + self.residuals[:, 2]
        self.pressure_reference = float(truth_pressure.mean())

    def __getitem__(self, index: int):
        result = (
            self.forecasts[index].float(),
            self.residuals[index].float(),
            self.masks[self.mask_ids[index]],
        )
        if self.parameter_conditioning:
            return (*result, self.parameters[index])
        return result


def encode(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (((value - mean) / (4 * std)).clamp(-1, 1) + 1) * 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensor-layout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--pressure-drop-ramp-steps", type=int, default=10_000)
    parser.add_argument("--velocity-only-loss", action="store_true")
    parser.add_argument("--parameter-conditioning", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--parameter-lr", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    if args.pressure_drop_ramp_steps < 1:
        raise ValueError("pressure-drop-ramp-steps must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = DPOTUVPressureTapFrames(
        args.data_dir,
        args.sensor_layout,
        args.frame_stride,
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
    if checkpoint.get("scenario") != dataset.manifest["scenario"]:
        raise ValueError(
            "Checkpoint/data scenario mismatch: "
            f"{checkpoint.get('scenario')} vs {dataset.manifest['scenario']}"
        )
    if args.parameter_conditioning:
        if checkpoint.get("stage") != "dpot_uvp_pressure_tap_velocity_parameters_stage1":
            raise ValueError(
                "Parameter-conditioned Stage-2 requires the matching velocity Stage-1"
            )
        validate_parameter_metadata(checkpoint, dataset)
    model = InvFusionResidualEDM(
        args.upstream_root,
        height=dataset.height,
        width=dataset.width,
        sigma_data=0.5,
        field_channels=3,
        model_size="base",
        use_fp16=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema.load_state_dict(checkpoint["ema_state_dict"])

    parameter_adapter = None
    parameter_adapter_ema = None
    parameter_groups = [{"params": model.parameters(), "lr": args.lr}]
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
    pressure_reference = torch.tensor(
        dataset.pressure_reference, device=device
    )
    output_stage = (
        "dpot_uv_pressure_tap_velocity_parameters_stage2"
        if args.parameter_conditioning and args.velocity_only_loss
        else "dpot_uv_pressure_tap_stage2"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_loss = []
    recent_clip = []
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

        drop_probability = min(
            1.0, (step - 1) / args.pressure_drop_ramp_steps
        )
        drop_pressure = (
            torch.rand(len(forecast), device=device) < drop_probability
        )
        pressure_base = torch.where(
            drop_pressure[:, None, None],
            pressure_reference,
            forecast[:, 2],
        )
        forecast_condition = forecast.clone()
        forecast_condition[:, 2] = pressure_base
        correction_target = residual.clone()
        correction_target[:, 2] = (
            forecast[:, 2] + residual[:, 2] - pressure_base
        )

        normalized_pressure = (
            correction_target[:, 2] - target_mean[:, 2]
        ) / (4 * target_std[:, 2])
        pressure_clip_fraction = float(
            normalized_pressure.abs().gt(1).float().mean()
        )
        forecast_encoded = encode(
            forecast_condition, forecast_mean, forecast_std
        )
        if parameter_adapter is not None:
            forecast_encoded = parameter_adapter(
                forecast_encoded, parameters
            )
        target_encoded = encode(
            correction_target, target_mean, target_std
        )

        optimizer.zero_grad(set_to_none=True)
        terms = model.loss_terms(
            target_encoded,
            forecast_encoded,
            mask,
            consistency_weight=0.1,
            channel_weights=channel_weights,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if parameter_adapter is not None:
            torch.nn.utils.clip_grad_norm_(parameter_adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema, model, 0.9999)
        if parameter_adapter is not None:
            update_ema(parameter_adapter_ema, parameter_adapter, 0.9999)

        recent_loss.append(float(terms["loss"].detach()))
        recent_clip.append(pressure_clip_fraction)
        if step % 100 == 0:
            print(
                f"step={step} loss={np.mean(recent_loss[-100:]):.6f} "
                f"pressure_drop={drop_probability:.4f} "
                f"pressure_clip={np.mean(recent_clip[-100:]):.6f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                    "stage": output_stage,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scenario": dataset.manifest["scenario"],
                    "field_channels": 3,
                    "forecast_stats": dataset.manifest["forecast_stats"],
                    "residual_stats": dataset.manifest["residual_stats"],
                    "pressure_reference": dataset.pressure_reference,
                    "pressure_drop_ramp_steps": args.pressure_drop_ramp_steps,
                    "condition_channels": ["u", "v", "pressure_taps"],
                    "loss_channels": ["u", "v"] if args.velocity_only_loss else ["u", "v", "p"],
                    "init_checkpoint": str(args.init_checkpoint),
                    "sensor_layout": str(args.sensor_layout),
                    "recent_loss": float(np.mean(recent_loss[-100:])),
                    "recent_pressure_clip_fraction": float(
                        np.mean(recent_clip[-100:])
                    ),
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
                "frame_stride": args.frame_stride,
                "steps": args.steps,
                "pressure_drop_ramp_steps": args.pressure_drop_ramp_steps,
                "condition_channels": ["u", "v", "pressure_taps"],
                "loss_channels": ["u", "v"] if args.velocity_only_loss else ["u", "v", "p"],
                "pressure_reference": dataset.pressure_reference,
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
