#!/usr/bin/env python3
"""Stage 4: remove DPOT velocity for taps-only pressure inference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from etch.training.pressure_stage2 import encode
from etch.training.pressure_stage3 import (
    STAGE_NAME as STAGE3_NAME,
)
from etch.training.stage1 import (
    PhysicalParameterAdapter,
    validate_parameter_metadata,
)
from etch.training.velocity_stage3 import (
    TemporalPressureAdapter,
    TemporalPressureResidualFrames,
)
from etch.model import InvFusionResidualEDM


STAGE_NAME = "pressure_history_parameters_one_step_mmse_stage4_v1"


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
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--target-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--velocity-drop-ramp-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--adapter-lr", type=float, default=5e-5)
    parser.add_argument("--parameter-lr", type=float, default=1e-5)
    parser.add_argument("--sigma", type=float, default=80.0)
    parser.add_argument("--tap-consistency-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    if args.steps < 1 or args.velocity_drop_ramp_steps < 1 or args.sigma <= 0:
        raise ValueError("steps, velocity-drop-ramp-steps, and sigma must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    dataset = TemporalPressureResidualFrames(
        args.data_dir,
        args.sensor_layout,
        args.history_length,
        args.target_stride,
        parameter_conditioning=True,
        history_clamp=False,
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
    if checkpoint.get("stage") != STAGE3_NAME:
        raise ValueError(f"Expected {STAGE3_NAME}, got {checkpoint.get('stage')}")
    if checkpoint.get("loss_channels") != ["p"]:
        raise ValueError("Initialization checkpoint must be pressure-only")
    if checkpoint.get("scenario") != scenario:
        raise ValueError("Initialization checkpoint/data scenario mismatch")
    if checkpoint.get("sample_specific_dpot_pressure") is not False:
        raise ValueError("Stage 3 must not use sample-specific DPOT pressure")
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
    backbone = InvFusionResidualEDM(
        args.upstream_root,
        height=dataset.height,
        width=dataset.width,
        sigma_data=0.5,
        field_channels=3,
        model_size="base",
        use_fp16=True,
    ).to(device)
    # Stage 3 demonstrated that its raw co-adapted endpoint is coherent while
    # the slow EMA is not; Stage 4 therefore starts from the raw triplet.
    backbone.load_state_dict(checkpoint["model_state_dict"])
    temporal_adapter = TemporalPressureAdapter(
        args.history_length, dataset.n_taps
    ).to(device)
    temporal_adapter.load_state_dict(checkpoint["adapter_state_dict"])
    parameter_adapter = PhysicalParameterAdapter(
        len(dataset.parameter_names)
    ).to(device)
    parameter_adapter.load_state_dict(
        checkpoint["parameter_adapter_state_dict"]
    )

    optimizer = torch.optim.Adam(
        [
            {"params": backbone.parameters(), "lr": args.backbone_lr},
            {"params": temporal_adapter.parameters(), "lr": args.adapter_lr},
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
    pressure_reference = dataset.pressure_reference
    truth_field_reference = forecast_mean + residual_mean

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_loss: list[float] = []
    recent_denoising: list[float] = []
    recent_sensor: list[float] = []
    recent_drop: list[float] = []
    for step in range(1, args.steps + 1):
        try:
            forecast, residual, mask, history, parameters = next(batches)
        except StopIteration:
            batches = iter(loader)
            forecast, residual, mask, history, parameters = next(batches)
        forecast = forecast.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        history = history.to(device, non_blocking=True)
        parameters = parameters.to(device, non_blocking=True)

        drop_probability = min(
            1.0, (step - 1) / args.velocity_drop_ramp_steps
        )
        drop_velocity = (
            torch.rand(len(forecast), device=device) < drop_probability
        )
        condition_physical = forecast.clone()
        condition_physical[:, :2] = torch.where(
            drop_velocity[:, None, None, None],
            truth_field_reference[:, :2],
            forecast[:, :2],
        )
        condition_physical[:, 2] = pressure_reference
        truth_pressure = forecast[:, 2] + residual[:, 2]
        target_physical = residual_mean.expand_as(residual).clone()
        target_physical[:, 2] = truth_pressure - pressure_reference

        condition = encode(
            condition_physical, forecast_mean, forecast_std
        )
        condition = temporal_adapter(condition, history)
        condition = parameter_adapter(condition, parameters)
        target = encode(
            target_physical, residual_mean, residual_std, clamp=False
        )
        target[:, :2] = 0.5

        optimizer.zero_grad(set_to_none=True)
        terms = backbone.loss_terms_one_step_mmse(
            target,
            condition,
            mask,
            sigma_value=args.sigma,
            consistency_weight=args.tap_consistency_weight,
            channel_weights=pressure_weight,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(temporal_adapter.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(parameter_adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        recent_loss.append(float(terms["loss"].detach()))
        recent_denoising.append(float(terms["denoising_loss"]))
        recent_sensor.append(float(terms["sensor_mse"]))
        recent_drop.append(float(drop_velocity.float().mean()))
        if step % 100 == 0:
            print(
                f"scenario={scenario} step={step} "
                f"loss={np.mean(recent_loss[-100:]):.6f} "
                f"denoise={np.mean(recent_denoising[-100:]):.6f} "
                f"tap_mse={np.mean(recent_sensor[-100:]):.6f} "
                f"drop_probability={drop_probability:.4f} "
                f"actual_drop={np.mean(recent_drop[-100:]):.4f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "stage": STAGE_NAME,
                "step": step,
                "model_state_dict": backbone.state_dict(),
                "adapter_state_dict": temporal_adapter.state_dict(),
                "parameter_adapter_state_dict": parameter_adapter.state_dict(),
                "scenario": scenario,
                "field_channels": 3,
                "forecast_stats": dataset.manifest["forecast_stats"],
                "residual_stats": dataset.manifest["residual_stats"],
                "truth_field_reference": truth_field_reference.flatten().tolist(),
                "pressure_reference": pressure_reference,
                "history_length": args.history_length,
                "target_stride": args.target_stride,
                "n_taps": dataset.n_taps,
                "velocity_drop_ramp_steps": args.velocity_drop_ramp_steps,
                "sample_specific_dpot_velocity": False,
                "sample_specific_dpot_pressure": False,
                "condition_channels": [
                    "causal_pressure_tap_history",
                    *dataset.parameter_names,
                ],
                "parameter_names": dataset.parameter_names,
                "parameter_mean": dataset.parameter_mean.tolist(),
                "parameter_std": dataset.parameter_std.tolist(),
                "loss_channels": ["p"],
                "target": "CFD_truth_pressure_minus_training_pressure_reference",
                "normalization_type": (
                    "fixed_train_statistics_unclipped_linear_pressure_"
                    "target_and_history"
                ),
                "normalization_sigmas": 4.0,
                "gauge_transform": "none",
                "training_objective": "fixed_high_sigma_direct_one_step_MSE_MMSE",
                "training_sigma": args.sigma,
                "inference_sigma": args.sigma,
                "checkpoint_selection": "raw_endpoint_predeclared_no_ema",
                "init_checkpoint": str(args.init_checkpoint),
                "sensor_layout": str(args.sensor_layout),
                "tap_consistency_weight": args.tap_consistency_weight,
                "recent_loss": float(np.mean(recent_loss[-100:])),
                "recent_tap_mse": float(np.mean(recent_sensor[-100:])),
            }
            torch.save(
                payload, args.output_dir / f"checkpoint_{step:06d}.pt"
            )

    manifest = {
        "stage": STAGE_NAME,
        "scenario": scenario,
        "n_train_frames": len(dataset),
        "history_length": args.history_length,
        "target_stride": args.target_stride,
        "steps": args.steps,
        "velocity_drop_ramp_steps": args.velocity_drop_ramp_steps,
        "pressure_reference": pressure_reference,
        "sample_specific_dpot_velocity": False,
        "sample_specific_dpot_pressure": False,
        "condition_channels": [
            "causal_pressure_tap_history",
            *dataset.parameter_names,
        ],
        "loss_channels": ["p"],
        "training_objective": "fixed_high_sigma_direct_one_step_MSE_MMSE",
        "checkpoint_selection": "raw_endpoint_predeclared_no_ema",
        "init_checkpoint": str(args.init_checkpoint),
        "sensor_layout": str(args.sensor_layout),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
