# ETCH

Training and inference code for **Stabilizing Autoregressive PDE Foundation Model
Rollouts with Event-Triggered Context Healing** ([paper](https://arxiv.org/abs/2609.32321)).

This release uses DPOT-S on the RealPDEBench Cylinder dataset. ETCH compares
forecast pressure with 12 surface pressure taps. When the discrepancy exceeds
the threshold, two HDiT branches reconstruct velocity and pressure. The
reconstructed block becomes the context for the next DPOT forecast.

## Install

Use Python 3.10 or 3.11 and a CUDA GPU for training and inference.
Run the following commands from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup.py
```

The setup script checks out pinned versions of
[RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench) and
[InvFusion](https://github.com/noamelata/InvFusion) under `third_party/`.
These dependencies retain their upstream terms and attribution.

## Data and checkpoints

Obtain the numerical Cylinder trajectories using the
[RealPDEBench data instructions](https://realpdebench.github.io/getting-started/).
The expected local layout is:

```text
data/cylinder/hf_dataset/
  numerical/                       # Arrow trajectories
  train_index_numerical.json
  test_index_numerical.json
checkpoints/
  dpot_cylinder.pth                 # Complete Cylinder-finetuned DPOT-S
  velocity.pt                      # Final velocity Stage 4
  pressure.pt                      # Final pressure Stage 4
  mean_std.pt                      # Fixed DPOT normalization statistics
```

Set these local paths in `configs/cylinder.yaml`. Model weights and data are
not included in Git. The complete fine-tuned DPOT checkpoint is loaded strictly;
inference does not need the separate DPOT pretraining checkpoint.

`configs/split.json` specifies 49 training trajectories and 43 test trajectories.
The groups are disjoint. `configs/taps.json` contains the fixed tap locations.
The normalization file must be the one used to train the supplied DPOT model.
Inference never estimates normalization statistics from test data.

## Inference

Run DPOT + ETCH on all 43 test trajectories through their final stored frames:

```bash
python inference.py --mode etch --output outputs/etch
```

Run the matched baselines:

```bash
python inference.py --mode dpot --output outputs/dpot
python inference.py --mode hdit --output outputs/hdit
```

The DPOT-only baseline uses the same HDiT initial context as ETCH. Add
`--initialization truth` to use the ground-truth initial context instead.
For one trajectory, add `--trajectory 2531`.
Add `--save-fields` to save the physical `[u,v,p]` fields as well as the errors.
Use an empty output directory for each run.

The default protocol uses 20 input frames, 20 output frames, stride 20, and
threshold `0.05`. Frame spacing is 5 ms. Frames 0–19 supply pressure history;
frames 20–39 form the initial reconstructed context. Evaluation covers frames
40–3989, including the final partial block. Each HDiT branch makes one direct
denoiser call per reconstruction, matching the released checkpoints.

Each trajectory produces an `.npz` file and a JSON report. `summary.json`
aggregates the selected trajectories. Errors are per-frame relative L2 in
physical units; velocity uses the joint vector `[u,v]`. ETCH saves both
`issued_*` forecast errors and `post_replacement_*` corrected-state errors.
Corrections occur after observations arrive and affect subsequent forecasts.

Timing includes discrepancy calculation, trigger evaluation, reconstruction
when triggered, context update, and the next DPOT call including transfers.
Loading, warm-up, sensor acquisition, and offline error calculation are excluded.
Only complete 20-frame cycles contribute to timing summaries. The report stores
**mean** and **median** separately, pooling the individual complete cycles.
`mean_csr` is mean cycle time divided by 100 ms. GPU and software differences
can change timing.

## Training

DPOT stays frozen during the HDiT curriculum. This release uses the
RealPDEBench Cylinder-finetuned DPOT model. To train DPOT itself, use the
[upstream training interface](https://realpdebench.github.io/getting-started/).

First prepare DPOT forecast/residual pairs from the 49 training trajectories:

```bash
python prepare_data.py --output data/training
```

Then run the stages in order:

```bash
for stage in v1 v2 v3 v4; do
  python train.py --stage "$stage"
done

for stage in p1 p1_refine p2 p3 p4; do
  python train.py --stage "$stage"
done
```

| Stage | Conditioning change | Updates |
|---|---|---:|
| `v1` / `p1` | DPOT field, current pressure taps, and Reynolds number | 30,000 each |
| `v2` / `p2` | Remove DPOT pressure conditioning | 20,000 / 10,000 |
| `v3` / `p3` | Add a causal 20-frame tap history | 20,000 / 10,000 |
| `v4` / `p4` | Remove DPOT velocity conditioning | 30,000 / 10,000 |

Velocity uses the mixed-noise EDM objective. Pressure starts with EDM in `p1`,
then uses a 5,000-update direct-denoising refinement in `p1_refine` and the
same fixed-noise objective in `p2`–`p4`. The training code preserves the
checkpoint selection and normalization used by these models: velocity
inference uses EMA weights; pressure inference uses the raw Stage-4 endpoint.

Training saves checkpoints under `runs/<stage>/`. For inference, set
`velocity_checkpoint` to `runs/v4/checkpoint_030000.pt` and
`pressure_checkpoint` to `runs/p4/checkpoint_010000.pt`.
Use `CUDA_VISIBLE_DEVICES` to select the GPU. Stage options can be supplied
after the common arguments, for example `--batch-size 4`.
