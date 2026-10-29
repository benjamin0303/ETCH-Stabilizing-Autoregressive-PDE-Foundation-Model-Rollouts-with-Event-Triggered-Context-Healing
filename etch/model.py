"""InvFusion-HDiT for single-frame DPOT residual reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


UPSTREAM_COMMIT = "0c4b91569cc9befbba296dc05beea8575440b18b"
UPSTREAM_REPO = "https://github.com/noamelata/InvFusion"


def load_upstream(upstream_root: Path):
    """Import the pinned InvFusion model and preconditioner."""
    required = (
        upstream_root / "models" / "hdit.py",
        upstream_root / "training" / "precond.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"InvFusion files not found: {missing}")
    root_text = str(upstream_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # Import this before HDiT changes the upstream module search path.
    from training.precond import Precond
    from models.hdit import HDiT
    from models.hyperparams import dataset_hyperparams

    return HDiT, dataset_hyperparams, Precond


class FixedQDEIMMask:
    """Fixed point-sampling operator in the interface expected by InvFusion.

    Measurements retain the full flattened grid with zeros outside sensors,
    matching InvFusion's inpainting degradation. For a binary mask, H† = H.
    """

    def __init__(
        self,
        imshape: tuple[int, int, int, int],
        mask: torch.Tensor,
    ) -> None:
        if len(imshape) != 4:
            raise ValueError("imshape must be BCHW")
        batch, channels, height, width = imshape
        if channels < 1:
            raise ValueError("imshape must contain at least one channel")
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        if (
            mask.ndim != 4
            or mask.shape[-2:] != (height, width)
            or mask.shape[1] not in (1, channels)
        ):
            raise ValueError(
                "mask must have shape HxW, BxHxW, Bx1xHxW, or BxCxHxW"
            )
        if mask.shape[0] not in (1, batch):
            raise ValueError("mask batch dimension is not broadcastable to imshape")

        self.imshape = tuple(imshape)
        self.mask = mask.ne(0).to(device=mask.device, dtype=torch.float32)[:, None]
        self.ndim = 5

    def H(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = self.imshape
        if value.shape[0] != batch or value.shape[-2:] != (height, width):
            raise ValueError("value and degradation use different grids")
        if value.ndim == 4:
            if value.shape[1] % channels:
                raise ValueError("4-D channel count must be divisible by image channels")
            groups = value.shape[1] // channels
            value = value.reshape(batch, groups, channels, height, width)
            self.ndim = 4
        elif value.ndim == 5:
            if value.shape[2] != channels:
                raise ValueError("5-D values use an unexpected image channel count")
            self.ndim = 5
        else:
            raise ValueError("value must have shape BCHW or BxKxCxHxW")
        mask = self.mask.to(device=value.device, dtype=value.dtype)
        return (value * mask).flatten(start_dim=2)

    def H_pinv(
        self,
        measurement: torch.Tensor,
        noise_level: float = 0,
    ) -> torch.Tensor:
        batch, channels, height, width = self.imshape
        if (
            measurement.ndim != 3
            or measurement.shape[0] != batch
            or measurement.shape[-1] != channels * height * width
        ):
            raise ValueError("measurement has an incompatible shape")
        value = measurement.reshape(
            batch, measurement.shape[1], channels, height, width
        )
        value = value / (1 + noise_level**2)
        value = value * self.mask.to(device=value.device, dtype=value.dtype)
        if self.ndim == 4:
            return value.reshape(
                batch, measurement.shape[1] * channels, height, width
            )
        return value


class ForecastConditionedHDiT(nn.Module):
    """Add the DPOT frame beside noisy residual and H†y model inputs."""

    def __init__(self, hdit: nn.Module) -> None:
        super().__init__()
        self.hdit = hdit

    def forward(
        self,
        value: torch.Tensor,
        sigma: torch.Tensor,
        degradation: FixedQDEIMMask,
        measurement: torch.Tensor,
        *,
        forecast_cond: torch.Tensor | None = None,
        aug_cond: torch.Tensor | None = None,
        class_cond: torch.Tensor | None = None,
        mapping_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if forecast_cond is None:
            raise ValueError("forecast_cond is required")
        if forecast_cond.shape != value.shape:
            raise ValueError("forecast_cond and residual must have the same shape")
        degradation.ndim = value.ndim
        sparse_residual = degradation.H_pinv(measurement)
        model_input = torch.cat(
            (
                value,
                sparse_residual,
                forecast_cond.to(dtype=value.dtype),
            ),
            dim=1,
        )
        return self.hdit(
            model_input,
            sigma,
            degradation,
            measurement,
            aug_cond=aug_cond,
            class_cond=class_cond,
            mapping_cond=mapping_cond,
        )


class InvFusionResidualEDM(nn.Module):
    """Official InvFusion components adapted to a RealPDE field grid."""

    _CONFIGS = {
        "smoke": {
            "patch_size": [4, 4],
            "depths": [1, 1, 1],
            "widths": [64, 128, 512],
        },
        "base": {
            "patch_size": [4, 4],
            "depths": [2, 2, 6],
            "widths": [128, 256, 512],
        },
    }

    def __init__(
        self,
        upstream_root: Path,
        height: int,
        width: int,
        sigma_data: float,
        field_channels: int = 2,
        model_size: str = "base",
        use_fp16: bool = True,
        p_mean: float = -0.8,
        p_std: float = 1.6,
    ) -> None:
        super().__init__()
        if model_size not in self._CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}")
        if field_channels < 1:
            raise ValueError("field_channels must be positive")
        config = self._CONFIGS[model_size]
        if field_channels == 3:
            # InvFusion's feature degradation reshapes the 4x4, 8x8, and
            # 16x16 level patches into (patch_h, patch_w, groups, channels).
            # These widths are divisible by the respective patch area times 3.
            config = {
                **config,
                "widths": [192, 384, 768],
            }
            attention_head_width = 48
        else:
            attention_head_width = 64
        coarsest_patch = config["patch_size"][0] * 2 ** (
            len(config["depths"]) - 1
        )
        if height % coarsest_patch or width % coarsest_patch:
            raise ValueError(
                f"Grid {(height, width)} is not divisible by {coarsest_patch}"
            )

        HDiT, dataset_hyperparams, Precond = load_upstream(
            upstream_root
        )
        config_name = f"RealPDE_{model_size}"
        dataset_hyperparams[config_name] = {
            **config,
            "joint": [True] * len(config["depths"]),
            "self_attns": [
                {"type": "global", "d_head": attention_head_width}
                for _ in config["depths"]
            ],
        }
        hdit = HDiT(
            img_channels=field_channels,
            in_mult=3,
            out_mult=1,
            img_resolution=height,
            label_dim=0,
            joint=True,
            data=config_name,
        )
        self.net = Precond(
            ForecastConditionedHDiT(hdit),
            img_resolution=height,
            img_channels=field_channels,
            label_dim=0,
            use_fp16=use_fp16,
            sigma_data=sigma_data,
        )
        self.height = height
        self.width = width
        self.field_channels = field_channels
        self.sigma_data = sigma_data
        self.model_size = model_size
        self.p_mean = p_mean
        self.p_std = p_std

    @staticmethod
    def _to_model_space(encoded: torch.Tensor) -> torch.Tensor:
        return encoded * 2 - 1

    @staticmethod
    def _to_encoded_space(
        value: torch.Tensor, *, clamp: bool = True
    ) -> torch.Tensor:
        if clamp:
            value = value.clamp(-1, 1)
        return (value + 1) * 0.5

    def loss_terms(
        self,
        normalized_residual: torch.Tensor,
        forecast_condition: torch.Tensor,
        mask: torch.Tensor,
        consistency_weight: float = 0,
        channel_weights: torch.Tensor | None = None,
        observed_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target = self._to_model_space(normalized_residual)
        forecast = self._to_model_space(forecast_condition)
        observed = (
            target
            if observed_condition is None
            else self._to_model_space(observed_condition)
        )
        if observed.shape != target.shape:
            raise ValueError("observed_condition and residual must have the same shape")
        degradation = FixedQDEIMMask(tuple(target.shape), mask)
        measurement = degradation.H(observed)

        rnd_normal = torch.randn(
            [target.shape[0], 1, 1, 1], device=target.device
        )
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma.square() + self.sigma_data**2) / (
            sigma * self.sigma_data
        ).square()
        noisy = target + torch.randn_like(target) * sigma
        denoised, logvar = self.net(
            noisy,
            sigma,
            degradation,
            measurement,
            forecast_cond=forecast,
            return_logvar=True,
        )
        denoising_error = (
            weight / logvar.exp() * (denoised - target).square() + logvar
        )
        if channel_weights is None:
            denoising_loss = denoising_error.mean()
        else:
            channel_weights = channel_weights.to(
                device=target.device, dtype=target.dtype
            ).view(1, -1, 1, 1)
            if channel_weights.shape[1] != target.shape[1]:
                raise ValueError("channel_weights must match the field channels")
            denominator = (
                channel_weights.sum()
                * target.shape[0]
                * target.shape[2]
                * target.shape[3]
            ).clamp_min(1)
            denoising_loss = (denoising_error * channel_weights).sum() / denominator
        sensor_mask = mask.to(device=target.device, dtype=target.dtype)
        if sensor_mask.ndim == 3:
            sensor_mask = sensor_mask[:, None]
        sensor_squared_error = ((denoised - target).square() * sensor_mask).sum()
        observed_values = sensor_mask.sum().clamp_min(1)
        sensor_mse = sensor_squared_error / observed_values
        total = denoising_loss + consistency_weight * sensor_mse
        return {
            "loss": total,
            "denoising_loss": denoising_loss.detach(),
            "sensor_mse": sensor_mse.detach(),
        }

    def loss_terms_one_step_mmse(
        self,
        normalized_residual: torch.Tensor,
        forecast_condition: torch.Tensor,
        mask: torch.Tensor,
        *,
        sigma_value: float = 80.0,
        consistency_weight: float = 0,
        channel_weights: torch.Tensor | None = None,
        observed_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train the exact high-noise denoising query used by one-step inference.

        This follows InvFusion's direct MSE/MMSE objective: the network receives
        pure Gaussian noise at one fixed, high sigma instead of an EDM mixture
        dominated by much smaller sigmas.  It deliberately ignores learned
        log-variance because deployment makes one deterministic denoiser call.
        """
        if sigma_value <= 0:
            raise ValueError("sigma_value must be positive")
        target = self._to_model_space(normalized_residual)
        forecast = self._to_model_space(forecast_condition)
        observed = (
            target
            if observed_condition is None
            else self._to_model_space(observed_condition)
        )
        if observed.shape != target.shape:
            raise ValueError("observed_condition and residual must have the same shape")
        degradation = FixedQDEIMMask(tuple(target.shape), mask)
        measurement = degradation.H(observed)
        sigma = torch.full(
            (len(target),),
            float(sigma_value),
            device=target.device,
            dtype=target.dtype,
        )
        noisy = torch.randn_like(target) * sigma_value
        denoised, logvar = self.net(
            noisy,
            sigma,
            degradation,
            measurement,
            forecast_cond=forecast,
            return_logvar=True,
        )
        # Preserve a zero-gradient connection to logvar so DDP/optimizer setups
        # that expect every model output remain valid, as in upstream InvFusion.
        squared_error = (denoised - target).square() + 0 * logvar
        if channel_weights is None:
            denoising_loss = squared_error.mean()
        else:
            channel_weights = channel_weights.to(
                device=target.device, dtype=target.dtype
            ).view(1, -1, 1, 1)
            if channel_weights.shape[1] != target.shape[1]:
                raise ValueError("channel_weights must match the field channels")
            denominator = (
                channel_weights.sum()
                * target.shape[0]
                * target.shape[2]
                * target.shape[3]
            ).clamp_min(1)
            denoising_loss = (squared_error * channel_weights).sum() / denominator
        sensor_mask = mask.to(device=target.device, dtype=target.dtype)
        if sensor_mask.ndim == 3:
            sensor_mask = sensor_mask[:, None]
        sensor_squared_error = ((denoised - target).square() * sensor_mask).sum()
        observed_values = sensor_mask.sum().clamp_min(1)
        sensor_mse = sensor_squared_error / observed_values
        total = denoising_loss + consistency_weight * sensor_mse
        return {
            "loss": total,
            "denoising_loss": denoising_loss.detach(),
            "sensor_mse": sensor_mse.detach(),
            "sigma": sigma.detach().mean(),
        }

    def forward(
        self,
        normalized_residual: torch.Tensor,
        forecast_condition: torch.Tensor,
        mask: torch.Tensor,
        consistency_weight: float = 0,
    ) -> torch.Tensor:
        return self.loss_terms(
            normalized_residual,
            forecast_condition,
            mask,
            consistency_weight=consistency_weight,
        )["loss"]

    @torch.no_grad()
    def sample_one_step(
        self,
        forecast_condition: torch.Tensor,
        observed_residual: torch.Tensor,
        mask: torch.Tensor,
        seed: int = 0,
        sigma_max: float = 80.0,
        hard_condition: bool = True,
        clamp_output: bool = True,
    ) -> torch.Tensor:
        """Run one direct EDM denoiser call at the sampler's maximum noise."""
        forecast = self._to_model_space(forecast_condition)
        observed = self._to_model_space(observed_residual)
        degradation = FixedQDEIMMask(tuple(observed.shape), mask)
        measurement = degradation.H(observed)
        devices = (
            [forecast.device.index]
            if forecast.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            noisy = torch.randn_like(forecast) * sigma_max
            sigma = torch.full(
                (len(noisy),),
                sigma_max,
                device=noisy.device,
            )
            reconstructed = self.net(
                noisy,
                sigma,
                degradation,
                measurement,
                forecast_cond=forecast,
            )
        if hard_condition:
            reconstructed = torch.where(mask.bool(), observed, reconstructed)
        return self._to_encoded_space(reconstructed, clamp=clamp_output)
