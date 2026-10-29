#!/usr/bin/env python3
"""Stage 2: remove sample-specific DPOT pressure from pressure HDiT."""

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


R1_MMSE_STAGE = (
    "dpot_uvp_pressure_tap_pressure_residual_parameters_one_step_mmse_v2"
)
STAGE_NAME = "dpot_uv_pressure_tap_pressure_parameters_one_step_mmse_stage2_v2"


class RawPressureFrames(DPOTPressureResidualFrames):
    """Expose physical DPOT fields and residuals instead of encoded tensors."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.forecasts[index].float(),
            self.residuals[index].float(),
            self.masks[self.mask_ids[index]],
            self.parameters[index],
        )


def encode(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    normalized = (value - mean) / (4 * std)
    if clamp:
        normalized = normalized.clamp(-1, 1)
    return (normalized + 1) * 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensor-layout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=("cylinder", "foil"), default="cylinder"
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--pressure-drop-ramp-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--parameter-lr", type=float, default=2e-5)
    parser.add_argument("--sigma", type=float, default=80.0)
    parser.add_argument("--tap-consistency-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    if args.steps < 1 or args.pressure_drop_ramp_steps < 1 or args.sigma <= 0:
        raise ValueError("steps, pressure-drop-ramp-steps, and sigma must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    dataset = RawPressureFrames(
        args.data_dir,
        args.sensor_layout,
        args.frame_stride,
        parameter_conditioning=True,
        max_shards=args.max_shards,
    )
    scenario = dataset.manifest["scenario"]
    if scenario != args.scenario:
        raise ValueError(
            f"Requested {args.scenario}, but training data are {scenario}"
        )
    checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("stage") != R1_MMSE_STAGE:
        raise ValueError(
            f"Expected {R1_MMSE_STAGE}, got {checkpoint.get('stage')}"
        )
    if checkpoint.get("loss_channels") != ["p"]:
        raise ValueError("Initialization checkpoint must be pressure-only")
    if checkpoint.get("scenario") != scenario:
        raise ValueError("Initialization checkpoint/data scenario mismatch")
    validate_parameter_metadata(checkpoint, dataset)

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
    model = InvFusionResidualEDM(
        args.upstream_root,
        height=dataset.height,
        width=dataset.width,
        sigma_data=0.5,
        field_channels=3,
        model_size="base",
        use_fp16=True,
    ).to(device)
    # Stage 2 starts from the exact EMA weights evaluated at 7.969%.
    model.load_state_dict(checkpoint["ema_state_dict"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    parameter_adapter = PhysicalParameterAdapter(
        len(dataset.parameter_names)
    ).to(device)
    parameter_adapter.load_state_dict(
        checkpoint["parameter_adapter_ema_state_dict"]
    )
    parameter_adapter_ema = (
        copy.deepcopy(parameter_adapter).eval().requires_grad_(False)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.backbone_lr},
            {"params": parameter_adapter.parameters(), "lr": args.parameter_lr},
        ],
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    pressure_weight = torch.tensor([0.0, 0.0, 1.0], device=device)
    forecast_mean = dataset.forecast_mean.to(device)[None]
    forecast_std = dataset.forecast_std.to(device)[None]
    residual_mean = dataset.residual_mean.to(device)[None]
    residual_std = dataset.residual_std.to(device)[None]
    # This is computed only from strict-training statistics and is therefore
    # available at deployment without a sample-specific pressure field.
    pressure_reference = float(
        dataset.manifest["forecast_stats"]["mean"][2]
        + dataset.manifest["residual_stats"]["mean"][2]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_loss: list[float] = []
    recent_denoising: list[float] = []
    recent_sensor: list[float] = []
    recent_drop: list[float] = []
    recent_clip: list[float] = []
    for step in range(1, args.steps + 1):
        try:
            forecast, residual, mask, parameters = next(batches)
        except StopIteration:
            batches = iter(loader)
            forecast, residual, mask, parameters = next(batches)
        forecast = forecast.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        parameters = parameters.to(device, non_blocking=True)

        # Scheduled conditioning dropout makes the transition explicit:
        # early samples retain DPOT p; after the ramp all samples use only the
        # fixed training reference, while DPOT u/v remain available.
        drop_probability = min(
            1.0, (step - 1) / args.pressure_drop_ramp_steps
        )
        drop_pressure = (
            torch.rand(len(forecast), device=device) < drop_probability
        )
        pressure_base = torch.where(
            drop_pressure[:, None, None],
            torch.as_tensor(
                pressure_reference, device=device, dtype=forecast.dtype
            ),
            forecast[:, 2],
        )
        truth_pressure = forecast[:, 2] + residual[:, 2]

        condition_physical = forecast.clone()
        condition_physical[:, 2] = pressure_base
        target_physical = residual_mean.expand_as(residual).clone()
        target_physical[:, 2] = truth_pressure - pressure_base
        normalized_pressure = (
            target_physical[:, 2] - residual_mean[:, 2]
        ) / (4 * residual_std[:, 2])
        pressure_clip_fraction = float(
            normalized_pressure.abs().gt(1).float().mean()
        )

        condition = encode(
            condition_physical, forecast_mean, forecast_std
        )
        condition = parameter_adapter(condition, parameters)
        # Preserve the R1 four-sigma scale but remove destructive clipping.
        # Rare tails can lie outside nominal [0,1]; the linear transform and
        # its decoder remain exactly invertible, including at pressure taps.
        target = encode(
            target_physical, residual_mean, residual_std, clamp=False
        )
        target[:, :2] = 0.5

        optimizer.zero_grad(set_to_none=True)
        terms = model.loss_terms_one_step_mmse(
            target,
            condition,
            mask,
            sigma_value=args.sigma,
            consistency_weight=args.tap_consistency_weight,
            channel_weights=pressure_weight,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(parameter_adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema, model, 0.9999)
        update_ema(parameter_adapter_ema, parameter_adapter, 0.9999)

        recent_loss.append(float(terms["loss"].detach()))
        recent_denoising.append(float(terms["denoising_loss"]))
        recent_sensor.append(float(terms["sensor_mse"]))
        recent_drop.append(float(drop_pressure.float().mean()))
        recent_clip.append(pressure_clip_fraction)
        if step % 100 == 0:
            print(
                f"scenario={scenario} step={step} "
                f"loss={np.mean(recent_loss[-100:]):.6f} "
                f"denoise={np.mean(recent_denoising[-100:]):.6f} "
                f"tap_mse={np.mean(recent_sensor[-100:]):.6f} "
                f"drop_probability={drop_probability:.4f} "
                f"actual_drop={np.mean(recent_drop[-100:]):.4f} "
                f"pressure_clip={np.mean(recent_clip[-100:]):.6f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "stage": STAGE_NAME,
                "step": step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "parameter_adapter_state_dict": parameter_adapter.state_dict(),
                "parameter_adapter_ema_state_dict": (
                    parameter_adapter_ema.state_dict()
                ),
                "scenario": scenario,
                "field_channels": 3,
                "forecast_stats": dataset.manifest["forecast_stats"],
                "residual_stats": dataset.manifest["residual_stats"],
                "pressure_reference": pressure_reference,
                "pressure_drop_ramp_steps": args.pressure_drop_ramp_steps,
                "final_sample_specific_dpot_pressure": False,
                "condition_channels": [
                    "DPOT_u",
                    "DPOT_v",
                    "pressure_taps",
                    *dataset.parameter_names,
                ],
                "parameter_names": dataset.parameter_names,
                "parameter_mean": dataset.parameter_mean.tolist(),
                "parameter_std": dataset.parameter_std.tolist(),
                "loss_channels": ["p"],
                "target": "CFD_truth_pressure_minus_pressure_base",
                "prediction_composition": (
                    "pressure_hat=training_pressure_reference+predicted_pressure_correction"
                ),
                "normalization_type": (
                    "train_forecast_zscore_clamped_condition_"
                    "unclipped_linear_residual_target"
                ),
                "normalization_sigmas": 4.0,
                "gauge_transform": "none",
                "training_objective": "fixed_high_sigma_direct_one_step_MSE_MMSE",
                "training_sigma": args.sigma,
                "inference_sigma": args.sigma,
                "initialized_from_ema": True,
                "init_checkpoint": str(args.init_checkpoint),
                "sensor_layout": str(args.sensor_layout),
                "n_taps": dataset.n_taps,
                "tap_consistency_weight": args.tap_consistency_weight,
                "recent_loss": float(np.mean(recent_loss[-100:])),
                "recent_tap_mse": float(np.mean(recent_sensor[-100:])),
                "recent_pressure_clip_fraction": float(
                    np.mean(recent_clip[-100:])
                ),
            }
            torch.save(
                payload, args.output_dir / f"checkpoint_{step:06d}.pt"
            )

    manifest = {
        "stage": STAGE_NAME,
        "scenario": scenario,
        "n_train_frames": len(dataset),
        "frame_stride": args.frame_stride,
        "steps": args.steps,
        "pressure_drop_ramp_steps": args.pressure_drop_ramp_steps,
        "pressure_reference": pressure_reference,
        "final_sample_specific_dpot_pressure": False,
        "condition_channels": [
            "DPOT_u",
            "DPOT_v",
            "pressure_taps",
            *dataset.parameter_names,
        ],
        "loss_channels": ["p"],
        "training_objective": "fixed_high_sigma_direct_one_step_MSE_MMSE",
        "training_sigma": args.sigma,
        "inference_sigma": args.sigma,
        "init_checkpoint": str(args.init_checkpoint),
        "sensor_layout": str(args.sensor_layout),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
