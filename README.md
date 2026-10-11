# Gemma E2B × CLINC150

[![CI](https://github.com/pablo-exe/Gemma-E2B---CLINC150/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-exe/Gemma-E2B---CLINC150/actions/workflows/ci.yml)

A reproducible study of how to improve **Gemma 4 E2B IT** on **CLINC150**: supervised fine-tuning when labeled data is available, and teacher–student logit distillation as an alternative research direction. Frontier-model evaluations at different reasoning levels provide a separate reference for classification performance.

CLINC150 contains 150 in-scope intents across 10 domains plus out-of-scope (OOS) queries. The primary metric is macro-F1 across all 151 labels; accuracy, in-scope accuracy, OOS precision/recall/F1 and invalid-output rate expose different failure modes.

## Results and research evidence

Snapshot from the local artifacts on **11 October 2026**. Scores below are percentages, rounded to two decimals. Research tracks are independent approaches, with no required phase order.

| Research track | Current progress | Evidence / remaining work |
|---|---|---|
| Zero-shot Gemma reference | ✅ Full validation and test evaluations recorded | Fixed prompt, raw predictions and aggregate metrics |
| Supervised fine-tuning with labeled data | ✅ Full QLoRA run completed on 15,100 training examples | Three checkpoints evaluated on validation; epoch 3 selected; full test evaluation still pending |
| Teacher–student logits | ⬜ Not implemented yet | No teacher logits, distillation run or student results recorded |
| Frontier models × reasoning levels | ✅ Blind 1,000-example evaluations recorded | Luna low/max and Sol 6.1 low; reference evaluations, not evidence of student distillation |

### Gemma: baseline and supervised fine-tuning

The validation rows use the complete official validation split (3,000 in-scope + 100 OOS examples). QLoRA uses the official labeled train split and a compact training/inference prompt; the zero-shot reference uses the full label vocabulary. The measured gain therefore describes the complete SFT setup, including its prompt, rather than an isolated adapter ablation.

| Setup | Split | N | Macro-F1 ↑ | Accuracy ↑ | In-scope accuracy ↑ | OOS F1 ↑ | Invalid outputs ↓ | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | Validation | 3,100 | 72.05 | 72.42 | 74.33 | 23.08 | 1.52 | [Metrics][zs-val] |
| QLoRA · epoch 1 | Validation | 3,100 | 94.72 | 91.00 | 93.70 | 18.02 | 6.26 | [Metrics][sft-e1] |
| QLoRA · epoch 2 | Validation | 3,100 | 96.04 | 93.71 | 95.33 | 58.82 | 3.58 | [Metrics][sft-e2] |
| **QLoRA · epoch 3 (selected)** | **Validation** | **3,100** | **96.96** | **95.13** | **96.63** | **62.89** | **2.52** | [Metrics][sft-e3] |
| Zero-shot | Test | 5,500 | 66.95 | 61.49 | 72.24 | 23.04 | 3.84 | [Metrics][zs-test] |
| QLoRA · selected adapter | Test | — | — | — | — | — | — | No full test result recorded |

**Validation progress:** macro-F1 **72.05 → 94.72 → 96.04 → 96.96**. The selected SFT setup gains **24.91 percentage points** over zero-shot on validation. OOS remains the main weakness: epoch 3 has **84.75% OOS precision**, **50.00% recall**, and 37 OOS examples parsed as invalid. Its overall invalid-output rate also remains above zero-shot.

The [training summary][sft-summary] records 2,832 optimizer steps, 1,339,392 trainable parameters and about 7.12 hours elapsed. The [run metadata][sft-meta] records the configuration, dataset hash, model revision and Git commit. Per-example evidence is retained in [zero-shot validation predictions][zs-val-pred] and [selected-adapter validation predictions][sft-pred]. Validation was used for checkpoint selection; these scores are not final held-out test results.

### Frontier models: reasoning-level comparison

The three agent evaluations contain the **same 1,000 test sample IDs, utterances and gold labels**, in the same order: 808 in-scope and 192 OOS examples, covering all 151 labels. Model names and reasoning levels follow the artifact directory names; exact provider model revisions and generation settings are not recorded in these metrics files.

| Model | Reasoning | N | Macro-F1 ↑ | Accuracy ↑ | In-scope accuracy ↑ | OOS F1 ↑ | Invalid outputs ↓ | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Luna | low | 1,000 | 52.45 | 56.70 | 48.64 | 48.60 | 0.00 | [Metrics][luna-low] · [Predictions][luna-low-pred] |
| Luna | max | 1,000 | 89.78 | 93.20 | 93.32 | 91.05 | 0.00 | [Metrics][luna-max] · [Predictions][luna-max-pred] |
| **Sol 6.1** | **low** | **1,000** | **96.11** | **97.10** | **97.65** | **95.04** | **0.00** | [Metrics][sol-low] · [Predictions][sol-low-pred] |

Luna's recorded max-reasoning run improves macro-F1 by **37.33 percentage points** over low. Luna low often maps valid intents to OOS (OOS precision **33.21%**, recall **90.63%**). Sol 6.1 low has the highest scores among these recorded frontier runs. These are single-run observations; frontier results on a test subset and Gemma results on full validation/test splits do not establish a shared leaderboard or a distillation gain.

<details>
<summary>Additional exploratory evidence and artifact availability</summary>

The separate [Luna low ChatGPT 100-example evaluation][luna-100] reports 60.00% accuracy, 72.51% macro-F1, 0.00% OOS F1 and 29.00% invalid outputs. This smaller evaluation is kept separate because its sample size and output behavior differ. Tiny QLoRA smoke runs (1–4 validation examples) verify the execution path and are excluded from the performance table.

Artifact links resolve in a local checkout containing the runs. `artifacts/` is ignored by Git, so these raw files are not published with the repository; this README preserves the result snapshot. Historical `phase1`/`phase2` paths are retained as run identifiers, not as a research sequence.

</details>

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
