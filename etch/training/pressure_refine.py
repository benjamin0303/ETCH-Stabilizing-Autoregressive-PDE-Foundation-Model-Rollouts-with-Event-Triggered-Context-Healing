#!/usr/bin/env python3
"""Fine-tune pressure-only DPOT residual HDiT for its exact one-step query."""

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
    PRESSURE_PARAMETERS_STAGE,
    PhysicalParameterAdapter,
    update_ema,
    validate_parameter_metadata,
)
from etch.model import InvFusionResidualEDM


STAGE_NAME = "dpot_uvp_pressure_tap_pressure_residual_parameters_one_step_mmse_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensor-layout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--parameter-lr", type=float, default=2e-5)
    parser.add_argument("--sigma", type=float, default=80.0)
    parser.add_argument("--tap-consistency-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    if args.steps < 1 or args.sigma <= 0:
        raise ValueError("steps and sigma must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    dataset = DPOTPressureResidualFrames(
        args.data_dir,
        args.sensor_layout,
        args.frame_stride,
        parameter_conditioning=True,
        max_shards=args.max_shards,
    )
    checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("stage") != PRESSURE_PARAMETERS_STAGE:
        raise ValueError(
            f"Expected {PRESSURE_PARAMETERS_STAGE}, got {checkpoint.get('stage')}"
        )
    if checkpoint.get("scenario") != dataset.manifest["scenario"]:
        raise ValueError("Checkpoint/data scenario mismatch")
    if checkpoint.get("loss_channels") != ["p"]:
        raise ValueError("Initialization checkpoint must be pressure-only")
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
    # The deployed R1 result uses EMA weights, so fine-tune from that exact
    # evaluated model rather than reverting to the noisier raw final weights.
    model.load_state_dict(checkpoint["ema_state_dict"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    parameter_adapter = PhysicalParameterAdapter(
        len(dataset.parameter_names)
    ).to(device)
    parameter_adapter.load_state_dict(
        checkpoint["parameter_adapter_ema_state_dict"]
    )
    parameter_adapter_ema = copy.deepcopy(parameter_adapter).eval().requires_grad_(False)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_loss: list[float] = []
    recent_denoising: list[float] = []
    recent_sensor: list[float] = []
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
        target = residual.clone()
        target[:, :2] = 0.5
        condition = parameter_adapter(forecast, parameters)

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
        if step % 100 == 0:
            print(
                f"scenario={dataset.manifest['scenario']} step={step} "
                f"loss={np.mean(recent_loss[-100:]):.6f} "
                f"denoise={np.mean(recent_denoising[-100:]):.6f} "
                f"tap_mse={np.mean(recent_sensor[-100:]):.6f} "
                f"sigma={args.sigma:g}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "stage": STAGE_NAME,
                "step": step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "parameter_adapter_state_dict": parameter_adapter.state_dict(),
                "parameter_adapter_ema_state_dict": parameter_adapter_ema.state_dict(),
                "scenario": dataset.manifest["scenario"],
                "field_channels": 3,
                "forecast_stats": dataset.manifest["forecast_stats"],
                "residual_stats": dataset.manifest["residual_stats"],
                "sensor_layout": str(args.sensor_layout),
                "n_taps": dataset.n_taps,
                "parameter_names": dataset.parameter_names,
                "parameter_mean": dataset.parameter_mean.tolist(),
                "parameter_std": dataset.parameter_std.tolist(),
                "loss_channels": ["p"],
                "target": "CFD_truth_pressure_minus_DPOT_pressure",
                "prediction_composition": (
                    "pressure_hat=DPOT_pressure+predicted_pressure_residual"
                ),
                "unoptimized_target_channels": "constant_model_space_zero",
                "normalization_type": "train_forecast_and_residual_channel_zscore",
                "normalization_sigmas": 4.0,
                "gauge_transform": "none",
                "training_objective": "fixed_high_sigma_direct_one_step_MSE_MMSE",
                "training_sigma": args.sigma,
                "inference_sigma": args.sigma,
                "initialized_from_ema": True,
                "init_checkpoint": str(args.init_checkpoint),
                "tap_consistency_weight": args.tap_consistency_weight,
                "recent_loss": float(np.mean(recent_loss[-100:])),
                "recent_tap_mse": float(np.mean(recent_sensor[-100:])),
            }
            torch.save(payload, args.output_dir / f"checkpoint_{step:06d}.pt")

    manifest = {
        "stage": STAGE_NAME,
        "scenario": dataset.manifest["scenario"],
        "n_train_frames": len(dataset),
        "frame_stride": args.frame_stride,
        "steps": args.steps,
        "n_taps": dataset.n_taps,
        "parameter_names": dataset.parameter_names,
        "training_objective": "fixed_high_sigma_direct_one_step_MSE_MMSE",
        "training_sigma": args.sigma,
        "inference_sigma": args.sigma,
        "loss_channels": ["p"],
        "init_checkpoint": str(args.init_checkpoint),
        "sensor_layout": str(args.sensor_layout),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
