# MTRL

## Overview
MTRL is a JAX/Flax research codebase for multi-task reinforcement learning on the Meta-World benchmark.
It packages environment wrappers, neural network architectures, optimisation utilities, and experiment
scripts used to reproduce a range of baseline comparisons and ablations for large-scale multi-task
continuous-control training.

## Getting started

### 1. Prerequisites
- **Python**: The repository targets Python 3.12; create your environment with that interpreter.
- **Accelerator**: Training requires access to either a GPU or TPU device. The main experiment runner
  aborts early if no accelerator is detected.
- **Dependency resolver**: The project requires [uv](https://github.com/astral-sh/uv) for managing
  environments and installing dependencies.

If uv is not available locally, install it first (pipx is recommended):
```bash
pipx install uv
# or, if pipx is unavailable
python -m pip install --user uv
```

### 2. Create a virtual environment
```bash
uv venv
source .venv/bin/activate
```

### 3. Install the package
Install in editable mode so that local changes to the library are picked up by experiment scripts. Use
the extra that matches your hardware:
```bash
# CPU-only
uv pip install -e ".[cpu]"

# CUDA 12 GPUs
uv pip install -e ".[cuda12]"

# Apple Metal
uv pip install -e ".[metal]"
```
The project depends on a custom Meta-World fork that is pulled automatically during installation.

## Running experiments
Each experiment is a small Tyro-powered CLI that instantiates an `Experiment` dataclass with the desired
environment, algorithm, and training configuration before calling `experiment.run()`.
The runner handles workspace creation, checkpointing, and hardware checks for you.

### Quick smoke test
```bash
uv run python experiments/example.py --experiment-name debug_run --seed 1
```
This launches multi-task Soft Actor-Critic on the MT10 benchmark with multi-head actor and critic
networks, saving results under `experiment_results/debug_run/` by default.

### Baseline scripts
Canonical baselines and ablations live under `experiments/`. A few useful entry points:
- `experiments/baselines/mt10_sac_v2.py` – single-task network SAC baseline for MT10.
- `experiments/mt10_mtmhsac.py` and `experiments/mt50_mtmhsac_v2.py` – multi-task multi-head SAC
  configurations.
- `experiments/width_scaling/` – architectural scaling sweeps for different model widths.

To log metrics to Weights & Biases, supply the `--track`, `--wandb-project`, and `--wandb-entity`
flags that each CLI exposes. The experiment manager will reuse run IDs when resuming from checkpoints.

### Outputs, checkpoints, and evaluation
Results are written to `experiment_results/<name>/<name>_<seed>/` along with a `checkpoints/` folder
containing Orbax-managed snapshots of the agent, replay buffer (for off-policy algorithms), RNG states,
and evaluation metadata. When `resume=True`, the latest checkpoint is restored automatically, including
replay buffers for off-policy runs and environment states. Evaluation uses Meta-World’s vector
environments and task-specific reward implementations, with support for MT1, MT10, MT25, and MT50 suites
as well as per-task runs.

## Repository structure
```
.
├── mtrl/                  # Core library code
│   ├── config/            # Dataclass configuration objects for networks, optimizers, and RL loops
│   ├── envs/              # Environment interfaces and Meta-World wrappers
│   ├── nn/                # Flax neural network modules (Multi-Head, Soft Modules, PaCo, CARE, FiLM, MOORE, ...)
│   ├── optim/             # Optimisation utilities such as PCGrad and GradNorm
│   ├── rl/                # Algorithm implementations (MTSAC, MTPPO, SAC) and shared components
│   └── monitoring/        # Metrics and video utilities for evaluation logging
├── experiments/           # Tyro CLIs for baselines, ablations, and parameter sweeps
├── plots/                 # Figure generation scripts plus `export_all.sh` convenience wrapper
├── figures/               # Pre-rendered SVG/PNG assets for publication-quality plots
├── plots/get_data.py      # Helpers for collecting experiment metrics into plotting tables
├── pyproject.toml         # Project metadata, dependency list, and optional extras
└── uv.lock                # Locked dependency versions for reproducibility
```
Supporting references: configs define reusable neural network, optimizer, and RL settings used by
algorithms. Environment wrappers are provided by `MetaworldConfig` and friends for MT suites. Neural
network modules expose architecture variants selected through config dataclasses. Optimisation utilities
include GradNorm and PCGrad for balancing gradients across tasks. Algorithm implementations wire these
pieces together and choose SAC, MT-SAC, or MT-PPO based on the supplied configuration. Plot scripts can
be batch-executed via `plots/export_all.sh` to regenerate the paper figures stored under `figures/`.

## Development tips
- **Testing**: Install the `testing` extra and run `pytest` to execute the available test suite.
- **Formatting**: The project ships with Ruff configuration for linting (see `pyproject.toml`).
- **Type checking**: The codebase uses type hints extensively (including jaxtyping) and is compatible
  with Pyright.

## Citation
If you use this codebase in academic work, please cite the associated paper (add BibTeX entry here).
