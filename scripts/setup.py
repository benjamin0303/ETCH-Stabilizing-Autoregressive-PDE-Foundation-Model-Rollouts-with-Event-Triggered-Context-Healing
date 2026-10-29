"""Install the two pinned upstream code dependencies."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    "RealPDEBench": (
        "https://github.com/AI4Science-WestlakeU/RealPDEBench.git",
        "62f4c80ab17f78933d046f2b038531dbc6a478a0",
    ),
    "InvFusion": (
        "https://github.com/noamelata/InvFusion.git",
        "0c4b91569cc9befbba296dc05beea8575440b18b",
    ),
}


def main():
    for name, (url, revision) in REPOSITORIES.items():
        destination = ROOT / "third_party" / name
        if not destination.exists():
            subprocess.run(["git", "clone", url, str(destination)], check=True)
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--detach", revision],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise RuntimeError(f"{name}: expected {revision}, found {actual}")

    # Read only the ID column. Reading each full Arrow row loads all flow fields.
    loader = ROOT / "third_party/RealPDEBench/realpdebench/data/fluid_hf_dataset.py"
    original = (
        '        for i in range(len(self.trajectories)):\n'
        '            sim_id = self.trajectories[i]["sim_id"]\n'
    )
    replacement = '        for i, sim_id in enumerate(self.trajectories["sim_id"]):\n'
    source = loader.read_text()
    if original in source:
        loader.write_text(source.replace(original, replacement, 1))
    elif replacement not in source:
        raise RuntimeError("Unexpected RealPDEBench loader; cannot apply the ID-column fix")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e",
         str(ROOT / "third_party/RealPDEBench")], check=True,
    )


if __name__ == "__main__":
    main()
