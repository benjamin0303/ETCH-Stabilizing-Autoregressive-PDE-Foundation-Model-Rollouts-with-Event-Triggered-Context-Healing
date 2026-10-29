#!/usr/bin/env python3
"""Fine-tune temporal Stage-3 into a pressure-block-only cold-start model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from etch.training.velocity_stage3 import TemporalPressureAdapter


class PressureTapBlocks(Dataset):
    """Non-overlapping 20-frame DPOT/pressure blocks within each trajectory."""

    def __init__(self, root: Path, layout_path: Path, block_length: int) -> None:
        self.manifest = json.loads((root / "manifest.json").read_text())
        if int(self.manifest["block_frames"]) != block_length:
            raise ValueError(
                f"Expected {block_length}-frame blocks, got "
                f"{self.manifest['block_frames']}"
            )
        layout = json.loads(layout_path.read_text())
        self.block_length = block_length
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
        self.field_reference = self.forecast_mean + self.residual_mean
        self.pressure_reference = float(self.field_reference[2])

        forecasts: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        pressure_blocks: list[torch.Tensor] = []
        mask_ids: list[int] = []
        masks: list[torch.Tensor] = []
        mask_lookup: dict[tuple[int, ...], int] = {}
        expected_taps: int | None = None

        for shard in self.manifest["shards"]:
            payload = torch.load(
                root / shard["path"], map_location="cpu", weights_only=True
            )
            forecast = payload["forecast_field"]
            residual = payload["residual_field"]
            if forecast.shape[1] != block_length:
                raise ValueError(f"Unexpected block length in {shard['path']}")
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
            pressure_blocks.append(
                torch.stack(
                    [truth_pressure[..., row, col] for row, col in tap_pixels],
                    dim=2,
                ).half()
            )
            forecasts.append(forecast.permute(0, 1, 4, 2, 3).contiguous())
            residuals.append(residual.permute(0, 1, 4, 2, 3).contiguous())
            mask_ids.extend([mask_id] * len(forecast))

        if expected_taps is None:
            raise ValueError("Training manifest contains no shards")
        self.forecasts = torch.cat(forecasts)
        self.residuals = torch.cat(residuals)
        self.pressure_blocks = torch.cat(pressure_blocks)
        self.mask_ids = torch.tensor(mask_ids)
        self.masks = torch.stack(masks)
        self.n_taps = expected_taps
        self.height = int(self.forecasts.shape[-2])
        self.width = int(self.forecasts.shape[-1])

    def __len__(self) -> int:
        return len(self.forecasts)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pressure_block = self.pressure_blocks[index].float()
        pressure_block = (
            (
                pressure_block
                - self.pressure_reference
                - self.residual_mean[2]
            )
            / (4 * self.residual_std[2])
        ).clamp(-1, 1)
        return (
            self.forecasts[index].float(),
            self.residuals[index].float(),
            self.masks[self.mask_ids[index]],
            pressure_block,
        )


class TemporalBlockAdapter(nn.Module):
    """Add a target-frame embedding to the Stage-3 temporal adapter."""

    def __init__(
        self,
        block_length: int,
        n_taps: int,
        field_channels: int = 3,
        coarse_height: int = 16,
        coarse_width: int = 32,
    ) -> None:
        super().__init__()
        self.field_channels = field_channels
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.history_adapter = TemporalPressureAdapter(
            block_length,
            n_taps,
            field_channels=field_channels,
            coarse_height=coarse_height,
            coarse_width=coarse_width,
        )
        self.target_embedding = nn.Embedding(
            block_length, field_channels * coarse_height * coarse_width
        )
        nn.init.zeros_(self.target_embedding.weight)

    def forward(
        self,
        forecast_condition: torch.Tensor,
        pressure_block: torch.Tensor,
        target_offset: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.history_adapter(forecast_condition, pressure_block)
        update = self.target_embedding(target_offset).view(
            len(target_offset),
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
        return (condition + 0.25 * torch.tanh(update)).clamp(0, 1)
