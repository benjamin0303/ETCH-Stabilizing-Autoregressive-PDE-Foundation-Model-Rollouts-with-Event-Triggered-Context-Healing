#!/usr/bin/env python3
"""Fine-tune temporal Stage-3 into a pressure-history + parameter Stage-4."""

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
    PhysicalParameterAdapter,
    parameter_statistics,
    physical_parameters,
    update_ema,
    validate_parameter_metadata,
)
from etch.training.velocity_stage2 import encode
from etch.training.blocks import (
    PressureTapBlocks,
    TemporalBlockAdapter,
)
from etch.model import InvFusionResidualEDM

class ParameterPressureTapBlocks(PressureTapBlocks):
    """Stage-4 blocks labeled with trajectory-level Re and AoA."""

    def __init__(self, root: Path, layout_path: Path, block_length: int) -> None:
        super().__init__(root, layout_path, block_length)
        (
            parameter_names,
            parameter_mean,
            parameter_std,
        ) = parameter_statistics(self.manifest)

        values: list[list[float]] = []
        for shard in self.manifest["shards"]:
            parameters = physical_parameters(
                self.manifest["scenario"], shard["sim_id"]
            )
            values.extend([parameters] * int(shard["n_blocks"]))
        if len(values) != len(self.forecasts):
            raise ValueError(
                f"Parameter/block mismatch: {len(values)} vs {len(self.forecasts)}"
            )
        raw = torch.tensor(values, dtype=torch.float32)
        self.parameter_names = parameter_names
        self.parameter_mean = parameter_mean
        self.parameter_std = parameter_std
        self.parameters = (raw - self.parameter_mean) / self.parameter_std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (*super().__getitem__(index), self.parameters[index])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensor-layout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--curriculum-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--history-adapter-lr", type=float, default=2e-5)
    parser.add_argument("--new-adapter-lr", type=float, default=1e-4)
    parser.add_argument("--velocity-only-loss", action="store_true")
    parser.add_argument("--pressure-only-loss", action="store_true")
    parser.add_argument("--save-every", type=int, default=5_000)
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
    checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    inherited_parameter_adapter = "parameter_adapter_state_dict" in checkpoint
    if args.pressure_only_loss:
        expected_stage = "dpot_p_pressure_history_pressure_stage3"
    elif inherited_parameter_adapter:
        expected_stage = "dpot_uv_pressure_history_parameters_stage3"
    else:
        expected_stage = "dpot_uv_pressure_history_stage3"
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(
            f"Expected {expected_stage}, got {checkpoint.get('stage')}"
        )

    dataset = ParameterPressureTapBlocks(
        args.data_dir, args.sensor_layout, args.block_length
    )
    if checkpoint.get("scenario") != dataset.manifest["scenario"]:
        raise ValueError("Checkpoint/data scenario mismatch")
    if int(checkpoint["history_length"]) != args.block_length:
        raise ValueError("Stage-3 history and Stage-4 block lengths differ")
    if int(checkpoint["n_taps"]) != dataset.n_taps:
        raise ValueError("Stage-3 and Stage-4 tap counts differ")
    if inherited_parameter_adapter:
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
    torch.set_float32_matmul_precision("high")

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

    adapter = TemporalBlockAdapter(args.block_length, dataset.n_taps).to(device)
    adapter.history_adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter_ema = copy.deepcopy(adapter).eval().requires_grad_(False)
    adapter_ema.history_adapter.load_state_dict(
        checkpoint["adapter_ema_state_dict"]
    )
    parameter_adapter = PhysicalParameterAdapter(
        len(dataset.parameter_names)
    ).to(device)
    if inherited_parameter_adapter:
        parameter_adapter.load_state_dict(
            checkpoint["parameter_adapter_state_dict"]
        )
    parameter_adapter_ema = copy.deepcopy(parameter_adapter).eval().requires_grad_(False)
    if inherited_parameter_adapter:
        parameter_adapter_ema.load_state_dict(
            checkpoint["parameter_adapter_ema_state_dict"]
        )

    optimizer = torch.optim.Adam(
        [
            {"params": backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": adapter.history_adapter.parameters(),
                "lr": args.history_adapter_lr,
            },
            {
                "params": adapter.target_embedding.parameters(),
                "lr": args.new_adapter_lr,
            },
            {
                "params": parameter_adapter.parameters(),
                "lr": (
                    args.history_adapter_lr
                    if inherited_parameter_adapter
                    else args.new_adapter_lr
                ),
            },
        ],
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    forecast_mean = dataset.forecast_mean.to(device)[None]
    forecast_std = dataset.forecast_std.to(device)[None]
    target_mean = dataset.residual_mean.to(device)[None]
    target_std = dataset.residual_std.to(device)[None]
    field_reference = dataset.field_reference.to(device)
    if args.pressure_only_loss:
        channel_weights = torch.tensor([0.0, 0.0, 1.0], device=device)
        output_stage = "pressure_history_parameters_cold_start_pressure_stage4"
    elif args.velocity_only_loss:
        channel_weights = torch.tensor([1.0, 1.0, 0.0], device=device)
        output_stage = "pressure_history_parameters_cold_start_velocity_stage4"
    else:
        channel_weights = None
        output_stage = "pressure_history_parameters_cold_start_stage4"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recent_loss: list[float] = []
    recent_clip: list[np.ndarray] = []
    for step in range(1, args.steps + 1):
        try:
            forecast_block, residual_block, mask, pressure_block, parameters = next(batches)
        except StopIteration:
            batches = iter(loader)
            forecast_block, residual_block, mask, pressure_block, parameters = next(batches)
        forecast_block = forecast_block.to(device, non_blocking=True)
        residual_block = residual_block.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pressure_block = pressure_block.to(device, non_blocking=True)
        parameters = parameters.to(device, non_blocking=True)

        curriculum = min(1.0, (step - 1) / args.curriculum_steps)
        random_offsets = torch.randint(
            args.block_length, (len(forecast_block),), device=device
        )
        use_random_offset = torch.rand(len(forecast_block), device=device) < curriculum
        target_offset = torch.where(
            use_random_offset,
            random_offsets,
            torch.full_like(random_offsets, args.block_length - 1),
        )
        batch_indices = torch.arange(len(forecast_block), device=device)
        forecast = forecast_block[batch_indices, target_offset]
        residual = residual_block[batch_indices, target_offset]

        reconstruction_base = forecast.clone()
        if args.pressure_only_loss:
            reconstruction_base[:, :2] = field_reference[None, :2]
            drop_pressure = torch.rand(len(forecast), device=device) < curriculum
            reconstruction_base[:, 2] = torch.where(
                drop_pressure[:, None, None],
                field_reference[2],
                forecast[:, 2],
            )
        else:
            reconstruction_base[:, 2] = field_reference[2]
            drop_velocity = torch.rand(len(forecast), device=device) < curriculum
            reconstruction_base[:, :2] = torch.where(
                drop_velocity[:, None, None, None],
                field_reference[None, :2],
                forecast[:, :2],
            )
        correction_target = forecast + residual - reconstruction_base
        normalized_target = (correction_target - target_mean) / (4 * target_std)
        clip_fraction = normalized_target.abs().gt(1).float().mean(dim=(0, 2, 3))
        forecast_encoded = encode(reconstruction_base, forecast_mean, forecast_std)
        target_encoded = encode(correction_target, target_mean, target_std)
        condition = adapter(forecast_encoded, pressure_block, target_offset)
        condition = parameter_adapter(condition, parameters)

        optimizer.zero_grad(set_to_none=True)
        terms = backbone.loss_terms(
            target_encoded,
            condition,
            mask,
            consistency_weight=0.1,
            channel_weights=channel_weights,
        )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(parameter_adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(backbone_ema, backbone, 0.9999)
        update_ema(adapter_ema, adapter, 0.9999)
        update_ema(parameter_adapter_ema, parameter_adapter, 0.9999)

        recent_loss.append(float(terms["loss"].detach()))
        recent_clip.append(clip_fraction.detach().cpu().numpy())
        if step % 100 == 0:
            mean_clip = np.stack(recent_clip[-100:]).mean(axis=0)
            print(
                f"scenario={dataset.manifest['scenario']} step={step} "
                f"loss={np.mean(recent_loss[-100:]):.6f} "
                f"curriculum={curriculum:.4f} clip_uvp={mean_clip.tolist()}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            mean_clip = np.stack(recent_clip[-100:]).mean(axis=0)
            torch.save(
                {
                    "stage": output_stage,
                    "step": step,
                    "model_state_dict": backbone.state_dict(),
                    "ema_state_dict": backbone_ema.state_dict(),
                    "adapter_state_dict": adapter.state_dict(),
                    "adapter_ema_state_dict": adapter_ema.state_dict(),
                    "parameter_adapter_state_dict": parameter_adapter.state_dict(),
                    "parameter_adapter_ema_state_dict": parameter_adapter_ema.state_dict(),
                    "scenario": dataset.manifest["scenario"],
                    "field_channels": 3,
                    "forecast_stats": dataset.manifest["forecast_stats"],
                    "residual_stats": dataset.manifest["residual_stats"],
                    "field_reference": field_reference.detach().cpu().flatten().tolist(),
                    "pressure_reference": dataset.pressure_reference,
                    "block_length": args.block_length,
                    "n_taps": dataset.n_taps,
                    "parameter_names": dataset.parameter_names,
                    "parameter_mean": dataset.parameter_mean.tolist(),
                    "parameter_std": dataset.parameter_std.tolist(),
                    "parameter_conditioning_from_stage1": inherited_parameter_adapter,
                    "curriculum_steps": args.curriculum_steps,
                    "init_checkpoint": str(args.init_checkpoint),
                    "sensor_layout": str(args.sensor_layout),
                    "loss_channels": (
                        ["p"] if args.pressure_only_loss else
                        ["u", "v"] if args.velocity_only_loss else
                        ["u", "v", "p"]
                    ),
                    "recent_loss": float(np.mean(recent_loss[-100:])),
                    "recent_clip_fraction_uvp": mean_clip.tolist(),
                },
                args.output_dir / f"checkpoint_{step:06d}.pt",
            )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": output_stage,
                "scenario": dataset.manifest["scenario"],
                "n_train_blocks": len(dataset),
                "block_length": args.block_length,
                "steps": args.steps,
                "curriculum_steps": args.curriculum_steps,
                "n_taps": dataset.n_taps,
                "parameter_names": dataset.parameter_names,
                "parameter_mean": dataset.parameter_mean.tolist(),
                "parameter_std": dataset.parameter_std.tolist(),
                "parameter_conditioning_from_stage1": inherited_parameter_adapter,
                "init_checkpoint": str(args.init_checkpoint),
                "sensor_layout": str(args.sensor_layout),
                "loss_channels": (
                    ["p"] if args.pressure_only_loss else
                    ["u", "v"] if args.velocity_only_loss else
                    ["u", "v", "p"]
                ),
                "final_inputs": "20-frame pressure taps, target offset, Re/AoA; no DPOT field",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
