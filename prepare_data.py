"""Prepare the disjoint training trajectories for the HDiT curriculum."""
import argparse
import json
from pathlib import Path

import torch

from etch.data import cylinder_dataset, load_config, load_dpot, load_normalizer


class ChannelMoments:
    def __init__(self):
        self.count = 0
        self.total = torch.zeros(3, dtype=torch.float64)
        self.total_sq = torch.zeros(3, dtype=torch.float64)

    def update(self, values):
        flat = values.reshape(-1, 3).double()
        self.count += len(flat)
        self.total += flat.sum(dim=0)
        self.total_sq += flat.square().sum(dim=0)

    def result(self):
        mean = self.total / self.count
        variance = self.total_sq / self.count - mean.square()
        return {"mean": mean.tolist(), "std": variance.clamp_min(0).sqrt().tolist()}


@torch.inference_mode()
def training_forecasts(context, model, normalizer, device):
    current = (context[None].to(device) - normalizer.mean_inputs[..., :3]) / normalizer.std_inputs[..., :3]
    outputs = []
    for _ in range(198):
        prediction = model(current) * normalizer.std_targets[..., :3] + normalizer.mean_targets[..., :3]
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite DPOT training rollout")
        outputs.append(prediction[0].cpu())
        current = (prediction - normalizer.mean_inputs[..., :3]) / normalizer.std_inputs[..., :3]
    return torch.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cylinder.yaml")
    parser.add_argument("--output", type=Path, default=Path("data/training"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config, split = load_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dataset = cylinder_dataset(config, "train")
    dataset.in_step = 20
    dataset.out_step = 3960
    dataset.horizon = 3980
    normalizer = load_normalizer(config["normalizer"], device)
    model = load_dpot(config, device)
    forecast_moments, residual_moments = ChannelMoments(), ChannelMoments()
    shards = []
    for index, sim_id in enumerate(split["train"]):
        dataset._indices = [{"sim_id": sim_id, "time_id": 0}]
        context, truth = dataset[0]
        forecast = training_forecasts(context[..., :3], model, normalizer, device)
        target = truth[..., :3].reshape_as(forecast)
        residual = target - forecast
        forecast_moments.update(forecast)
        residual_moments.update(residual)
        name = f"{index:03d}_{Path(sim_id).stem}_t0.pt"
        torch.save(dict(forecast_field=forecast.half(), residual_field=residual.half(),
                        sim_id=sim_id, time_id=0), args.output / name)
        shards.append(dict(path=name, sim_id=sim_id, time_id=0, n_blocks=198))
        print(f"[{index+1}/{len(split['train'])}] {sim_id}", flush=True)
    manifest = dict(scenario="cylinder", split="train", block_frames=20,
                    field_names=["u", "v", "p"], height=64, width=128,
                    forecast_stats=forecast_moments.result(), residual_stats=residual_moments.result(),
                    n_trajectories=len(shards), shards=shards)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
