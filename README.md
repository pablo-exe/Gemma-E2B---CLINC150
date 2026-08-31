# Gemma E2B × CLINC150

[![CI](https://github.com/pablo-exe/Gemma-E2B---CLINC150/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-exe/Gemma-E2B---CLINC150/actions/workflows/ci.yml)

A reproducible knowledge-distillation study: can a small, locally runnable Gemma model become a better intent classifier through supervised and teacher-generated data?


The project uses **Gemma 4 E2B IT** as the student and **CLINC150** as the benchmark. CLINC150 contains 150 in-scope intents across 10 domains plus out-of-scope (OOS) queries. Phase 1 establishes the untouched zero-shot baseline against which every later experiment will be compared.

## Research plan

| Phase | Experiment | Status |
|---|---|---|
| 1 | Reproducible zero-shot baseline | Implemented |
| 2 | QLoRA supervised fine-tuning on official training data | Planned |
| 3 | Response-based distillation from a frontier teacher | Planned |
| 4 | Error-focused distillation on confusing intent pairs | Planned |
| 5 | Final ablations and analysis | Planned |

The primary metric is macro-F1 over all 151 labels. In-scope accuracy, OOS precision/recall/F1, invalid-output rate and the most frequent confusions are reported separately.

## Repository layout

```text
.
├── configs/                 # Versioned experiment definitions
├── data/                    # Downloaded datasets (ignored by Git)
├── artifacts/               # Predictions and metrics (ignored by Git)
├── docs/                    # Protocol, architecture and contribution guide
├── src/gemma_clinc/         # Reusable Python package
├── tests/                   # Fast unit tests; no model download required
└── .github/                 # CI and pull-request templates
```

## Quick start

Requirements: Python 3.11, an NVIDIA CUDA GPU, and a Hugging Face account with access to the gated Gemma checkpoint.

```powershell
uv sync --extra gpu
uv run hf auth login
uv run gemma-clinc baseline --config configs/phase1_zero_shot.yaml --limit 25
```

The committed configuration uses a measured low-VRAM text-only path: Gemma 4's
large per-layer embedding table stays in system RAM and the unused audio and
vision towers are released. On the development RTX 4070 Laptop GPU, a real QLoRA
optimizer step over the longest compact CLINC150 training example peaks at
2,337 MiB according to `nvidia-smi`.
See [the hardware memory profile](docs/hardware_memory_profile.md) for the method,
raw measurements and the reproducible profiling command.

The Windows lockfile selects PyTorch's official CUDA 12.8 wheels, so a separate CUDA Toolkit installation is not required. A current NVIDIA driver is still required. The first GPU installation downloads a large CUDA-enabled PyTorch wheel and can take several minutes.

### Why the CUDA index is configured

The `gpu` extra declares the PyTorch version, but the PyTorch version alone does not guarantee a CUDA-enabled wheel. During setup, resolving `torch==2.11.0` by default selected the CPU build, so PyTorch reported `cuda=False` despite the machine having an NVIDIA RTX 4070 Laptop GPU. The project therefore points Windows to PyTorch's official `cu128` index, which resolves the CUDA 12.8 build (`torch==2.11.0+cu128`).

This decision is intentionally scoped to Windows because this project is developed and evaluated on the author's Windows laptop. The marker does not mean CUDA is Windows-only: Linux supports CUDA too, but its platform-specific dependency resolution is not pinned here. On non-Windows systems, `uv` resolves the regular PyPI package instead. Supporting additional CUDA platforms will be a separate portability change rather than an implicit behavior.

Remove `--limit 25` for the full official test split. Results are written to a timestamped directory under `artifacts/phase1/`; raw model outputs are retained for auditing.

For a CPU-only development environment, install only the test dependencies:

```powershell
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Reproducibility rules

- The official CLINC150 train, validation and test partitions are never reshuffled.
- Test examples are never sent to a teacher model or used for prompt development.
- All experiment behavior comes from a committed YAML configuration.
- Raw predictions and run metadata are saved before aggregate metrics are calculated.
- Dependencies are locked with `uv.lock`.
- On Windows, the GPU extra resolves PyTorch from the official CUDA 12.8 index because the default resolution selected a CPU build.

See [the experiment protocol](docs/experiment_protocol.md) for the evaluation contract and [the contribution guide](CONTRIBUTING.md) for the branch and pull-request workflow.

## Data and model access

The dataset is downloaded from the [official CLINC repository](https://github.com/clinc/oos-eval). Gemma weights are downloaded directly from [Google's Hugging Face repository](https://huggingface.co/google/gemma-4-E2B-it) and are never stored in this project.

Dataset and model licenses remain with their respective owners. This repository contains only the experiment code.
