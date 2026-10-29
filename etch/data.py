"""Cylinder data and frozen DPOT loading."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    split = json.loads(Path(config["split"]).read_text())
    if set(split["train"]) & set(split["test"]):
        raise ValueError("Training and test trajectories overlap")
    if config["dpot"]["in_timesteps"] != 20 or config["dpot"]["out_timesteps"] != 20:
        raise ValueError("This release uses 20 input frames and 20 output frames")
    return config, split


def cylinder_dataset(config, mode="test"):
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    return CylinderHFDataset(
        dataset_name="cylinder", dataset_root=config["data_root"],
        dataset_type="numerical", mode=mode, test_mode="all",
        in_step=40, out_step=3950, N_autoregressive=1,
        mask_prob=0., noise_scale=0., n_sim_frame=3990,
    )


def read_trajectory(dataset, sim_id):
    dataset._indices = [{"sim_id": sim_id, "time_id": 0}]
    first, rest = dataset[0]
    field = torch.cat((first[..., :3], rest[..., :3]))
    if field.shape != (3990, 64, 128, 3):
        raise ValueError(f"Expected the released Cylinder trajectory, got {field.shape}")
    return field


def load_normalizer(path, device):
    # Never estimate statistics from a test trajectory.
    values = torch.load(path, map_location=device, weights_only=True)
    names = ("mean_inputs", "mean_targets", "std_inputs", "std_targets")
    stats = dict(zip(names, values, strict=True))
    for name in ("std_inputs", "std_targets"):
        stats[name] = torch.where(stats[name] == 0, torch.ones_like(stats[name]), stats[name])
    return SimpleNamespace(**stats)


def load_dpot(config, device):
    from realpdebench.model.load_model import load_model
    from etch.reconstruct import ModelShapeDataset
    shape = torch.empty(20, 64, 128, 3)
    model = load_model(
        ModelShapeDataset(shape, shape), device=device,
        checkpoint_path=None, **config["dpot"],
    )
    checkpoint = torch.load(config["dpot_checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    state = {key.removeprefix("module.").removeprefix("dpot_model."): value
             for key, value in state.items()}
    # A complete fine-tuned checkpoint makes the pretraining file unnecessary.
    # Reject partial loading instead of silently leaving random parameters.
    model.dpot_model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
