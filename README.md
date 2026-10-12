# Gemma E2B × CLINC150

[![CI](https://github.com/pablo-exe/Gemma-E2B---CLINC150/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-exe/Gemma-E2B---CLINC150/actions/workflows/ci.yml)

A reproducible study of how to improve **Gemma 4 E2B IT** on **CLINC150**: supervised fine-tuning when labeled data is available, and teacher–student logit distillation as an alternative research direction. Frontier-model evaluations at different reasoning levels provide a separate reference for classification performance.

CLINC150 contains 150 in-scope intents across 10 domains plus out-of-scope (OOS) queries. The primary metric is macro-F1 across all 151 labels; accuracy, in-scope accuracy, OOS precision/recall/F1 and invalid-output rate expose different failure modes.

## Results and research evidence

| Setup | Macro-F1 ↑ | Accuracy ↑ | In-scope accuracy ↑ | OOS F1 ↑ | Invalid outputs ↓ | Evidence |
|---|---:|---:|---:|---:|---:|---|
| Zero-shot | 72.05 | 72.42 | 74.33 | 23.08 | 1.52 | [Metrics][zs-val] |
| QLoRA | **96.96** | 95.13 | 96.63 | 62.89 | 2.52 | [Metrics][sft-e3] |
| Teacher–student KD | 93.93 | 92.77 | 93.43 | 68.22 | 1.90 | [Results](docs/results/distillation_20261012.md) |
| Luna low | 52.45 | 56.70 | 48.64 | 48.60 | **0.00** | [Metrics][luna-low] |
| Luna max | 89.78 | 93.20 | 93.32 | 91.05 | **0.00** | [Metrics][luna-max] |
| Sol 6.1 low | 96.11 | **97.10** | 97.65 | 95.04 | **0.00** | [Metrics][sol-low] |


### Teacher–student distillation

The Modal L4 run completed all three epochs over 15,100 training examples. Each
epoch's 3,100 validation predictions was checked against the official split and its
metrics independently recomputed. The selected epoch-3 adapter contains 1,339,392
finite FP32 parameters; logged losses and gradient norms are finite. Downloaded
artifacts and the verification report are under
`artifacts/distillation/modal_completed_full_gpu_bf16/`.

| Epoch | Validation macro-F1 | Accuracy | OOS F1 | Invalid outputs |
|---|---:|---:|---:|---:|
| 1 | 92.41 | 90.10 | 59.89 | 4.45 |
| 2 | 92.09 | 90.06 | 50.51 | 2.00 |
| **3 (selected)** | **93.93** | **92.77** | **68.22** | **1.90** |

This run's macro-F1 is below the recorded SFT result (96.96%), while OOS F1 and
invalid-output rate improve. These are single validation runs, with BF16 computation
for KD versus FP16 for SFT; they do not isolate the effect of distillation. The held-out
test has not been evaluated for the KD adapter.

Gemma scores above use all 3,100 validation examples; frontier scores use a
1,000-example test subset. They are separate evaluations rather than a shared leaderboard.

[zs-val]: artifacts/phase1/phase1_zero_shot_20261011T130041Z/metrics.json
[zs-val-pred]: artifacts/phase1/phase1_zero_shot_20261011T130041Z/predictions.jsonl
[zs-test]: artifacts/phase1/phase1_zero_shot_20260831T182656Z%20%28Full%29/metrics.json
[sft-e1]: artifacts/phase2/phase2_qlora_sft_20260901T161247Z/validation_metrics_epoch_001.json
[sft-e2]: artifacts/phase2/phase2_qlora_sft_20260901T161247Z/validation_metrics_epoch_002.json
[sft-e3]: artifacts/phase2/phase2_qlora_sft_20260901T161247Z/validation_metrics_epoch_003.json
[sft-summary]: artifacts/phase2/phase2_qlora_sft_20260901T161247Z/training_summary.json
[sft-meta]: artifacts/phase2/phase2_qlora_sft_20260901T161247Z/run_metadata.json
[sft-pred]: artifacts/phase2/phase2_qlora_sft_20260901T161247Z/validation_epoch_003.jsonl
[luna-low]: artifacts/Luna_low/clinc150_agent_blind_1000_metrics.json
[luna-low-pred]: artifacts/Luna_low/clinc150_agent_blind_1000_evaluated.csv
[luna-max]: artifacts/Luna_max/clinc150_agent_blind_1000_metrics.json
[luna-max-pred]: artifacts/Luna_max/clinc150_agent_blind_1000_evaluated.csv
[sol-low]: artifacts/Sol%206.1_low/clinc150_agent_blind_1000_metrics.json
[sol-low-pred]: artifacts/Sol%206.1_low/clinc150_agent_blind_1000_evaluated.csv
[luna-100]: artifacts/Luna_low/clinc150_chatgpt_blind_100_metrics.json

## Repository layout

### Offline teacher logits on RunPod

The [RunPod runbook](docs/runpod_teacher_logits.md) covers setup, upload, a tokenizer
preflight, a stratified pilot, resumable INT8 Gemma 4 31B extraction and local
download verification. Start with `scripts/generate_teacher_logits.py --prepare-only
--limit 151` using the project environment. This exports sparse teacher distributions
on gold training responses. The [local student training runbook](docs/distillation_training.md)
covers the response-only CE + grouped-KL loss, resumable checkpoints and detached launch.

The [Modal workflow](docs/modal_teacher_logits.md) uses an A100 80 GB with the
unquantized BF16 teacher, persistent model/result volumes and automatic download
verification. Its separate configuration is `configs/teacher_logits_modal.yaml`.
The full run completed on 11 October 2026: 15,100 gold-response examples, top-128
log probabilities plus residual mass at temperature 2, and a peak of 58.69 GiB
reserved GPU memory. The [local manifest](artifacts/teacher_logits/train_bf16_t2_k128/manifest.json)
records provenance; the [archive](artifacts/teacher_logits/train_bf16_t2_k128.tar.gz)
is 80.9 MiB. These ignored artifacts are present in the local checkout, not published
with the repository. This is evidence of target extraction, not of a student KD gain.
The full KD run completed on Modal L4, with checkpoints retained in the
`finetuning-student-training` volume. Local launch/status receipts are under
`artifacts/distillation/full_gpu_bf16/`; see the [detached student runbook](docs/modal_student_training.md).
Selected adapter, validation predictions and verification results were downloaded to
`artifacts/distillation/modal_completed_full_gpu_bf16/`. No Modal apps remain running.


```text
.
├── configs/                 # Versioned experiment definitions
├── data/                    # Downloaded datasets (ignored by Git)
├── artifacts/               # Predictions and metrics (ignored by Git)
├── docs/                    # Protocol, architecture and contribution guide
├── notebooks/               # Educational baseline and QLoRA walkthroughs
├── src/gemma_clinc/         # Reusable Python package
├── tests/                   # Fast unit tests; no model download required
└── .github/                 # CI and pull-request templates
```

The [educational notebooks](notebooks/README.md) walk through zero-shot evaluation and supervised fine-tuning
cell by cell and write artifacts in the same formats as the CLI.

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

## Supervised fine-tuning with QLoRA

Supervised fine-tuning trains a response-only LoRA adapter on the official CLINC150 training
split and selects checkpoints using macro-F1 on validation. Start with the
balanced pilot before launching the complete run:

```powershell
uv sync --extra gpu --extra train
uv run gemma-clinc train-sft `
  --config configs/phase2_qlora_sft.yaml `
  --train-limit 1510 `
  --eval-limit 302 `
  --max-steps 95
```

Remove all three limit flags for the committed three-epoch experiment. See the
[supervised fine-tuning runbook](docs/phase2_sft.md) for smoke tests, expected runtime, artifacts
and final adapter evaluation.

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
