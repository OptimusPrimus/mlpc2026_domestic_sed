# MLPC 2026 Deep Learning Tutorial

This repository contains a PyTorch Lightning training pipeline for domestic sound event detection.

## Conda setup

Create and activate a conda environment with a supported Python version, then install the package in editable mode with the development dependencies:

```bash
conda create -n mlpc2026-dl python=3.12 -y
conda activate mlpc2026-dl
pip install -e ".[dev]"
```

If you want to log runs to Weights & Biases, authenticate once in the environment:

```bash
wandb login
```

## Run one experiment from the command line

To run one experiment with `learning-rate=1e-4`, use:

```bash
python -m domestic_sed.training \
  --data-root /path/to/MLPC2026_challenge_dataset_raw \
  --wandb-save-dir ./wandb \
  --learning-rate 1e-4 \
  --architecture-depth 8 \
  --architecture-base-multiplier 2 \
  --max-epochs 100
```

## Run a W&B sweep

The sweep config in [`sweeps/sweep_1_architecture_depth_8_x2.yaml`](/home/paul/repos/mlpc2026_dl_tutorial/sweeps/sweep_1_architecture_depth_8_x2.yaml) contains hardcoded paths in the `command` section. Update those paths first so they match your machine:

- `--data-root`
- `--wandb-save-dir`

Then create the sweep:

```bash
wandb sweep sweeps/sweep_1_architecture_depth_8_x2.yaml
```

W&B will print a command containing the generated sweep ID. Start an agent with that ID:

```bash
wandb agent domestic_sed/<SWEEP_ID>
```

If you use a different W&B entity or project, replace `domestic_sed/<SWEEP_ID>` with the value printed by `wandb sweep`.
