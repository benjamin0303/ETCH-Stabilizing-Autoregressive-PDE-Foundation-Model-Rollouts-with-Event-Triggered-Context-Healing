"""Evaluate DPOT + ETCH, DPOT-only, or HDiT-only through the final frame."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from etch.data import cylinder_dataset, load_config, load_dpot, load_normalizer, read_trajectory
from etch.reconstruct import (
    PressureCausalReconstructor, VelocityBlockReconstructor, dpot_blocks,
    extract_taps, reconstruct_full_context, tap_pixels_from_layout,
)
from etch.rollout import TIMING_COLUMNS, run_dpot, run_etch, run_hdit, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cylinder.yaml")
    parser.add_argument("--mode", choices=("etch", "dpot", "hdit"), default="etch")
    parser.add_argument("--initialization", choices=("hdit", "truth"), default="hdit",
                        help="DPOT-only initialization; ETCH always starts from pressure taps")
    parser.add_argument("--trajectory", help="One test ID, such as 2531; default: all 43")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-fields", action="store_true")
    args = parser.parse_args()
    config, split = load_config(args.config)
    if args.initialization == "truth" and args.mode != "dpot":
        parser.error("--initialization truth applies only to --mode dpot")
    cases = split["test"]
    if args.trajectory:
        requested = args.trajectory.removesuffix(".h5") + ".h5"
        if requested not in cases:
            parser.error(f"{requested} is not in the test split")
        cases = [requested]
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    dataset = cylinder_dataset(config)
    pixels = tap_pixels_from_layout(Path(config["tap_layout"]), 64, 128)
    reports, all_timings = [], []
    for sim_id in cases:
        seed = split["test"].index(sim_id) * 1_000_000
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        truth = read_trajectory(dataset, sim_id)
        taps = extract_taps(truth, pixels)
        reynolds = float(Path(sim_id).stem)
        normalizer = load_normalizer(config["normalizer"], device)
        dpot = load_dpot(config, device) if args.mode != "hdit" else None
        upstream = Path(config["invfusion_root"])
        velocity = VelocityBlockReconstructor(Path(config["velocity_checkpoint"]), upstream, 64, 128, device)
        pressure = PressureCausalReconstructor(Path(config["pressure_checkpoint"]), upstream, 64, 128, device)
        for _ in range(5):
            context, _ = reconstruct_full_context(velocity, pressure, taps[:40], 20, reynolds, pixels, seed)
            if dpot is not None:
                dpot_blocks(context[None], dpot, normalizer, device)
        if args.mode == "etch":
            arrays = run_etch(velocity, pressure, dpot, normalizer, truth, taps, pixels, reynolds,
                              seed, config["threshold"], config["frame_interval_ms"], args.save_fields)
        elif args.mode == "hdit":
            arrays = run_hdit(velocity, pressure, truth, taps, pixels, reynolds, seed, args.save_fields)
        else:
            initial = truth[20:40] if args.initialization == "truth" else context
            arrays = run_dpot(initial, dpot, normalizer, truth, device, args.save_fields)
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError(f"Non-finite result for {sim_id}")
        report = summarize(arrays, config["frame_interval_ms"])
        report.update(trajectory=sim_id, seed=seed)
        reports.append(report)
        if "timings_ms" in arrays:
            timings = arrays["timings_ms"]
            total = timings[:, -1] if timings.ndim == 2 else timings
            all_timings.append(total[arrays["complete"]])
        stem = args.output / Path(sim_id).stem
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        del velocity, pressure, dpot, truth, taps, context, arrays
    summary = dict(mode=args.mode, initialization=args.initialization, config=config,
                   n_trajectories=len(reports), stride=20, reports=reports,
                   device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   torch_version=str(torch.__version__),
                   cuda_version=torch.version.cuda, timing_columns=TIMING_COLUMNS if args.mode == "etch" else ["total"])
    summary["errors"] = {
        key: {"mean": float(np.mean([r["mean_errors"][key] for r in reports])),
              "std": float(np.std([r["mean_errors"][key] for r in reports]))}
        for key in reports[0]["mean_errors"]
    }
    if all_timings:
        total = np.concatenate(all_timings)
        summary.update(n_complete_windows=len(total), mean_cycle_ms=float(total.mean()),
                       median_cycle_ms=float(np.median(total)),
                       mean_csr=float(total.mean() / (20 * config["frame_interval_ms"])))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
