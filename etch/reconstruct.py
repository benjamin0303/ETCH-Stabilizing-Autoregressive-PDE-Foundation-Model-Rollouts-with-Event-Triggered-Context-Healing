"""Frozen pressure-conditioned reconstructors and DPOT inference helpers."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from etch.training.pressure_stage2 import encode as encode_unclipped_capable
from etch.training.stage1 import PhysicalParameterAdapter
from etch.training.velocity_stage2 import encode as encode_legacy
from etch.training.velocity_stage3 import TemporalPressureAdapter
from etch.training.blocks import TemporalBlockAdapter
from etch.model import InvFusionResidualEDM

VELOCITY_STAGE = "pressure_history_parameters_cold_start_velocity_stage4"
PRESSURE_STAGE = "pressure_history_parameters_one_step_mmse_stage4_v1"


class ModelShapeDataset:
    """One shape-only sample used by RealPDEBench's model factory."""

    def __init__(self, input_field: torch.Tensor, target_field: torch.Tensor):
        self.sample = (input_field, target_field)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return self.sample

    def __len__(self) -> int:
        return 1



def tap_pixels_from_layout(
    layout_path: Path, height: int, width: int
) -> tuple[tuple[int, int], ...]:
    layout = json.loads(layout_path.read_text())
    taps = next(iter(layout["layouts"].values()))
    pixels = tuple(
        (
            int(tap["row"]) * height // 128,
            int(tap["col"]) * width // 256,
        )
        for tap in taps
    )
    if len(pixels) != len(set(pixels)):
        raise ValueError("Downsampled pressure-tap pixels are not unique")
    return pixels



def extract_taps(
    truth: torch.Tensor, tap_pixels: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    """Return absolute pressure taps with shape [time, n_taps]."""
    return torch.stack(
        [truth[:, row, col, 2] for row, col in tap_pixels], dim=1
    )



def frame_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        (prediction - target).reshape(len(target), -1), dim=1
    ) / torch.linalg.vector_norm(
        target.reshape(len(target), -1), dim=1
    ).clamp_min(1e-12)



def tap_relative_l2(
    prediction: torch.Tensor,
    measured_taps: torch.Tensor,
    tap_pixels: tuple[tuple[int, int], ...],
) -> float:
    predicted_taps = torch.stack(
        [prediction[:, row, col, 2] for row, col in tap_pixels], dim=1
    )
    return float(
        torch.linalg.vector_norm(predicted_taps - measured_taps)
        / torch.linalg.vector_norm(measured_taps).clamp_min(1e-12)
    )



def make_mask(
    count: int,
    height: int,
    width: int,
    tap_pixels: tuple[tuple[int, int], ...],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(count, 3, height, width, device=device)
    for row, col in tap_pixels:
        mask[:, 2, row, col] = 1
    return mask



class VelocityBlockReconstructor:
    """Latest low-error v6 velocity Stage 4: tap block + Re -> [u,v]."""

    def __init__(
        self,
        checkpoint_path: Path,
        upstream_root: Path,
        height: int,
        width: int,
        device: torch.device,
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("stage") != VELOCITY_STAGE:
            raise ValueError(f"Unexpected velocity stage: {checkpoint.get('stage')}")
        if checkpoint.get("scenario") != "cylinder":
            raise ValueError("Velocity checkpoint must be Cylinder")
        if checkpoint.get("loss_channels") != ["u", "v"]:
            raise ValueError("Velocity checkpoint must use u,v-only loss")
        self.device = device
        self.height = height
        self.width = width
        self.block_length = int(checkpoint["block_length"])
        self.n_taps = int(checkpoint["n_taps"])
        self.forecast_mean = torch.tensor(
            checkpoint["forecast_stats"]["mean"], device=device
        ).view(1, 3, 1, 1)
        self.forecast_std = torch.tensor(
            checkpoint["forecast_stats"]["std"], device=device
        ).clamp_min(1e-6).view(1, 3, 1, 1)
        self.residual_mean = torch.tensor(
            checkpoint["residual_stats"]["mean"], device=device
        ).view(1, 3, 1, 1)
        self.residual_std = torch.tensor(
            checkpoint["residual_stats"]["std"], device=device
        ).clamp_min(1e-6).view(1, 3, 1, 1)
        self.field_reference = torch.tensor(
            checkpoint["field_reference"], device=device
        ).view(1, 3, 1, 1)
        self.pressure_reference = float(checkpoint["pressure_reference"])
        self.parameter_mean = torch.tensor(
            checkpoint["parameter_mean"], device=device
        )
        self.parameter_std = torch.tensor(
            checkpoint["parameter_std"], device=device
        ).clamp_min(1e-6)

        self.model = InvFusionResidualEDM(
            upstream_root,
            height,
            width,
            sigma_data=0.5,
            field_channels=3,
            model_size="base",
            use_fp16=True,
        ).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(checkpoint["ema_state_dict"])
        self.adapter = TemporalBlockAdapter(
            self.block_length, self.n_taps
        ).to(device).eval().requires_grad_(False)
        self.adapter.load_state_dict(checkpoint["adapter_ema_state_dict"])
        self.parameter_adapter = PhysicalParameterAdapter(
            len(checkpoint["parameter_names"])
        ).to(device).eval().requires_grad_(False)
        self.parameter_adapter.load_state_dict(
            checkpoint["parameter_adapter_ema_state_dict"]
        )

    @torch.inference_mode()
    def reconstruct(
        self,
        tap_block: torch.Tensor,
        reynolds: float,
        tap_pixels: tuple[tuple[int, int], ...],
        seed: int,
    ) -> tuple[torch.Tensor, float]:
        if tap_block.shape != (self.block_length, self.n_taps):
            raise ValueError(f"Unexpected velocity tap block: {tap_block.shape}")
        count = self.block_length
        tap_block_device = tap_block.to(self.device)
        base = self.field_reference.expand(
            count, -1, self.height, self.width
        )
        condition = encode_legacy(
            base, self.forecast_mean, self.forecast_std
        )
        normalized_taps = (
            tap_block_device
            - self.pressure_reference
            - float(self.residual_mean[0, 2, 0, 0])
        ) / (4 * float(self.residual_std[0, 2, 0, 0]))
        normalized_taps = normalized_taps.clamp(-1, 1)[None].expand(
            count, -1, -1
        )
        offsets = torch.arange(count, device=self.device)
        condition = self.adapter(condition, normalized_taps, offsets)
        raw_parameter = torch.tensor([reynolds], device=self.device)
        parameters = (
            (raw_parameter - self.parameter_mean) / self.parameter_std
        )[None].expand(count, -1)
        condition = self.parameter_adapter(condition, parameters)

        observed_physical = self.residual_mean.expand_as(base).clone()
        for tap_index, (row, col) in enumerate(tap_pixels):
            observed_physical[:, 2, row, col] = (
                tap_block_device[:, tap_index] - self.field_reference[0, 2, 0, 0]
            )
        observed = encode_legacy(
            observed_physical, self.residual_mean, self.residual_std
        )
        mask = make_mask(
            count, self.height, self.width, tap_pixels, self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        predicted_encoded = self.model.sample_one_step(
            condition,
            observed,
            mask,
            seed=seed,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        predicted_correction = (
            (predicted_encoded * 2 - 1) * (4 * self.residual_std)
            + self.residual_mean
        )
        velocity = (base[:, :2] + predicted_correction[:, :2]).permute(
            0, 2, 3, 1
        )
        return velocity.float().cpu(), elapsed



class PressureCausalReconstructor:
    """Latest deployment Stage 4: causal 20-tap history + Re -> pressure."""

    def __init__(
        self,
        checkpoint_path: Path,
        upstream_root: Path,
        height: int,
        width: int,
        device: torch.device,
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("stage") != PRESSURE_STAGE:
            raise ValueError(f"Unexpected pressure stage: {checkpoint.get('stage')}")
        if checkpoint.get("scenario") != "cylinder":
            raise ValueError("Pressure checkpoint must be Cylinder")
        if checkpoint.get("loss_channels") != ["p"]:
            raise ValueError("Pressure checkpoint must use pressure-only loss")
        if checkpoint.get("sample_specific_dpot_velocity") is not False:
            raise ValueError("Pressure Stage 4 unexpectedly uses DPOT velocity")
        if checkpoint.get("sample_specific_dpot_pressure") is not False:
            raise ValueError("Pressure Stage 4 unexpectedly uses DPOT pressure")
        if checkpoint.get("checkpoint_selection") != "raw_endpoint_predeclared_no_ema":
            raise ValueError("Pressure checkpoint must use its predeclared raw endpoint")
        self.device = device
        self.height = height
        self.width = width
        self.history_length = int(checkpoint["history_length"])
        self.n_taps = int(checkpoint["n_taps"])
        self.forecast_mean = torch.tensor(
            checkpoint["forecast_stats"]["mean"], device=device
        ).view(1, 3, 1, 1)
        self.forecast_std = torch.tensor(
            checkpoint["forecast_stats"]["std"], device=device
        ).clamp_min(1e-6).view(1, 3, 1, 1)
        self.residual_mean = torch.tensor(
            checkpoint["residual_stats"]["mean"], device=device
        ).view(1, 3, 1, 1)
        self.residual_std = torch.tensor(
            checkpoint["residual_stats"]["std"], device=device
        ).clamp_min(1e-6).view(1, 3, 1, 1)
        self.field_reference = torch.tensor(
            checkpoint["truth_field_reference"], device=device
        ).view(1, 3, 1, 1)
        self.pressure_reference = float(checkpoint["pressure_reference"])
        self.parameter_mean = torch.tensor(
            checkpoint["parameter_mean"], device=device
        )
        self.parameter_std = torch.tensor(
            checkpoint["parameter_std"], device=device
        ).clamp_min(1e-6)
        self.inference_sigma = float(checkpoint["inference_sigma"])

        self.model = InvFusionResidualEDM(
            upstream_root,
            height,
            width,
            sigma_data=0.5,
            field_channels=3,
            model_size="base",
            use_fp16=True,
        ).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.adapter = TemporalPressureAdapter(
            self.history_length, self.n_taps
        ).to(device).eval().requires_grad_(False)
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        self.parameter_adapter = PhysicalParameterAdapter(
            len(checkpoint["parameter_names"])
        ).to(device).eval().requires_grad_(False)
        self.parameter_adapter.load_state_dict(
            checkpoint["parameter_adapter_state_dict"]
        )

    @torch.inference_mode()
    def reconstruct(
        self,
        all_taps: torch.Tensor,
        target_indices: torch.Tensor,
        reynolds: float,
        tap_pixels: tuple[tuple[int, int], ...],
        seed: int,
    ) -> tuple[torch.Tensor, float, float]:
        count = len(target_indices)
        if count < 1:
            raise ValueError("Pressure reconstruction needs at least one target")
        if int(target_indices.min()) < self.history_length - 1:
            raise ValueError("A pressure target lacks its complete causal history")
        histories = torch.stack(
            [
                all_taps[
                    int(target) - self.history_length + 1 : int(target) + 1
                ]
                for target in target_indices
            ]
        ).to(self.device)
        target_taps = all_taps[target_indices].to(self.device)
        histories = (
            histories
            - self.pressure_reference
            - float(self.residual_mean[0, 2, 0, 0])
        ) / (4 * float(self.residual_std[0, 2, 0, 0]))

        base = self.field_reference.expand(
            count, -1, self.height, self.width
        ).clone()
        base[:, 2] = self.pressure_reference
        condition = encode_unclipped_capable(
            base, self.forecast_mean, self.forecast_std
        )
        condition = self.adapter(condition, histories)
        raw_parameter = torch.tensor([reynolds], device=self.device)
        parameters = (
            (raw_parameter - self.parameter_mean) / self.parameter_std
        )[None].expand(count, -1)
        condition = self.parameter_adapter(condition, parameters)

        observed_physical = self.residual_mean.expand_as(base).clone()
        for tap_index, (row, col) in enumerate(tap_pixels):
            observed_physical[:, 2, row, col] = (
                target_taps[:, tap_index] - self.pressure_reference
            )
        observed = encode_unclipped_capable(
            observed_physical,
            self.residual_mean,
            self.residual_std,
            clamp=False,
        )
        observed[:, :2] = 0.5
        mask = make_mask(
            count, self.height, self.width, tap_pixels, self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        predicted_encoded = self.model.sample_one_step(
            condition,
            observed,
            mask,
            seed=seed,
            sigma_max=self.inference_sigma,
            clamp_output=False,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        predicted_correction = (
            (predicted_encoded[:, 2:3] * 2 - 1)
            * (4 * self.residual_std[:, 2:3])
            + self.residual_mean[:, 2:3]
        )
        pressure = (
            self.pressure_reference + predicted_correction
        ).permute(0, 2, 3, 1).float()
        reconstructed_taps = torch.stack(
            [pressure[:, row, col, 0] for row, col in tap_pixels], dim=1
        )
        tap_rmse = float(
            torch.sqrt(torch.mean(torch.square(reconstructed_taps - target_taps)))
        )
        return pressure.cpu(), elapsed, tap_rmse



@torch.inference_mode()
def reconstruct_full_context(
    velocity: VelocityBlockReconstructor,
    pressure: PressureCausalReconstructor,
    all_taps: torch.Tensor,
    target_start: int,
    reynolds: float,
    tap_pixels: tuple[tuple[int, int], ...],
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    count = velocity.block_length
    targets = torch.arange(target_start, target_start + count)
    velocity_field, velocity_seconds = velocity.reconstruct(
        all_taps[target_start : target_start + count],
        reynolds,
        tap_pixels,
        seed,
    )
    pressure_field, pressure_seconds, tap_rmse = pressure.reconstruct(
        all_taps,
        targets,
        reynolds,
        tap_pixels,
        seed + 1,
    )
    return (
        torch.cat((velocity_field, pressure_field), dim=-1),
        {
            "velocity_seconds": velocity_seconds,
            "pressure_seconds": pressure_seconds,
            "combined_seconds": velocity_seconds + pressure_seconds,
            "pressure_hard_tap_rmse": tap_rmse,
        },
    )



@torch.inference_mode()
def dpot_blocks(
    contexts: torch.Tensor,
    model: torch.nn.Module,
    normalizer: GaussianNormalizer,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    channels = contexts.shape[-1]
    current = (
        contexts.to(device) - normalizer.mean_inputs[..., :channels]
    ) / normalizer.std_inputs[..., :channels]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    prediction_normalized = model(current)
    output_channels = prediction_normalized.shape[-1]
    prediction = (
        prediction_normalized * normalizer.std_targets[..., :output_channels]
        + normalizer.mean_targets[..., :output_channels]
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    prediction = prediction[..., :3]
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Non-finite DPOT rollout")
    return prediction.float().cpu(), elapsed



def append_field_metrics(
    destination: dict[str, list[np.ndarray]],
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> None:
    destination["full"].append(frame_relative_l2(prediction, truth).numpy())
    destination["uv"].append(
        frame_relative_l2(prediction[..., :2], truth[..., :2]).numpy()
    )
    destination["p"].append(
        frame_relative_l2(prediction[..., 2:3], truth[..., 2:3]).numpy()
    )
