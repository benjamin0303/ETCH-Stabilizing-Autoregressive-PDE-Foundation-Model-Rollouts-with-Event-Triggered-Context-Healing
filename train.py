"""Run one stage of the velocity or pressure HDiT curriculum."""
import argparse
import json
from pathlib import Path
import runpy
import sys

STAGES = {
    "v1": ("stage1", None, 30000, ["--lr", "5e-5", "--parameter-lr", "1e-4", "--parameter-conditioning", "--velocity-only-loss"]),
    "v2": ("velocity_stage2", "v1", 20000, ["--lr", "2e-5", "--parameter-lr", "5e-5", "--parameter-conditioning", "--velocity-only-loss", "--pressure-drop-ramp-steps", "10000"]),
    "v3": ("velocity_stage3", "v2", 20000, ["--backbone-lr", "2e-5", "--adapter-lr", "1e-4", "--parameter-lr", "5e-5", "--parameter-conditioning", "--velocity-only-loss"]),
    "v4": ("velocity_stage4", "v3", 30000, ["--backbone-lr", "1e-5", "--history-adapter-lr", "2e-5", "--new-adapter-lr", "1e-4", "--curriculum-steps", "10000", "--velocity-only-loss"]),
    "p1": ("stage1", None, 30000, ["--lr", "5e-5", "--parameter-lr", "1e-4", "--parameter-conditioning", "--pressure-only-loss"]),
    "p1_refine": ("pressure_refine", "p1", 5000, ["--backbone-lr", "1e-5", "--parameter-lr", "2e-5"]),
    "p2": ("pressure_stage2", "p1_refine", 10000, ["--backbone-lr", "1e-5", "--parameter-lr", "2e-5", "--pressure-drop-ramp-steps", "5000"]),
    "p3": ("pressure_stage3", "p2", 10000, ["--backbone-lr", "5e-6", "--adapter-lr", "1e-4", "--parameter-lr", "1e-5", "--adapter-warmup-steps", "1000"]),
    "p4": ("pressure_stage4", "p3", 10000, ["--backbone-lr", "5e-6", "--adapter-lr", "5e-5", "--parameter-lr", "1e-5", "--velocity-drop-ramp-steps", "5000"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--sensor-layout", default="configs/taps.json")
    parser.add_argument("--upstream-root", default="third_party/InvFusion")
    args, overrides = parser.parse_known_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    split = json.loads((Path(__file__).parent / "configs/split.json").read_text())
    actual = {shard["sim_id"] for shard in manifest["shards"]}
    if manifest["scenario"] != "cylinder" or actual != set(split["train"]):
        parser.error("Training data must contain exactly the 49 released training trajectories")
    module, previous, steps, options = STAGES[args.stage]
    argv = ["--data-dir", str(args.data_dir), "--sensor-layout", args.sensor_layout,
            "--upstream-root", args.upstream_root, "--output-dir", str(args.output_dir / args.stage),
            "--steps", str(steps), "--batch-size", "4", "--seed", "0", *options]
    if previous:
        endpoint = args.output_dir / previous / f"checkpoint_{STAGES[previous][2]:06d}.pt"
        argv += ["--init-checkpoint", str(endpoint)]
    if args.stage in ("v1", "v2", "p1", "p1_refine", "p2"):
        argv += ["--frame-stride", "5"]
    elif args.stage == "v4":
        argv += ["--block-length", "20"]
    else:
        argv += ["--history-length", "20", "--target-stride", "5"]
    if args.stage in ("p1_refine", "p2", "p3", "p4"):
        argv += ["--sigma", "80", "--tap-consistency-weight", "0.1"]
    sys.argv = [f"etch.training.{module}", *argv, *overrides]
    runpy.run_module(f"etch.training.{module}", run_name="__main__")


if __name__ == "__main__":
    main()
