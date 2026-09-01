# Phase 2 supervised QLoRA runbook

Phase 2 establishes the conventional supervised baseline that later distillation
runs must beat. It trains Gemma 4 E2B only on the official CLINC150 train split,
selects an adapter on validation and leaves test untouched until the experiment
is frozen.

## What is implemented

- NF4 QLoRA over text-language `q_proj` and `v_proj`, rank 8.
- Gemma 4 PLE lookup on CPU and text-only model execution.
- Loss only over assistant label tokens, not the instruction or utterance.
- Micro-batch 1 with gradient accumulation 16.
- Linear learning-rate schedule with 3% warmup.
- Adapter checkpoint and validation evaluation after every epoch.
- Best-checkpoint selection by validation macro-F1.
- Raw predictions, metrics, loss history, configuration, Git commit and dataset
  SHA-256 stored with every run.
- Deterministic stratified subsets for smoke tests and pilots.
- Standalone adapter evaluation on validation or held-out test.

The base checkpoint is pinned to revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`.

## Hardware smoke result

The complete path was executed on the RTX 4070 Laptop GPU with two training
examples, one optimizer update and adapter validation:

| Measurement | Result |
|---|---:|
| Trainable parameters | 1,339,392 |
| Peak CUDA allocated | 2,300 MiB |
| Peak CUDA reserved | 2,336 MiB |
| Time for two micro-batches and one update | 1.71 s |
| Initial mean loss | 6.6720, finite |
| Adapter save and fresh-process reload | Passed |

Gradient checkpointing was also tested. It reduced allocated memory by about
263 MiB but doubled the optimizer-step time, so it is disabled for the compact
61-token CLINC prompts. Set `gradient_checkpointing: true` if future distilled
examples become substantially longer.

The tiny smoke is not an accuracy result. After one update, validation outputs
were plausible synonyms such as `translation`, but not exact CLINC labels; the
strict parser correctly counted them as invalid.

## Recommended sequence

Install the training dependencies:

```powershell
uv sync --extra gpu --extra train
```

Run a quick end-to-end check:

```powershell
uv run gemma-clinc train-sft `
  --config configs/phase2_qlora_sft.yaml `
  --train-limit 4 `
  --eval-limit 4 `
  --max-steps 1
```

Run a balanced pilot with ten training examples and two validation examples per
class. Ninety-five optimizer steps equal one pass over the 1,510-example subset:

```powershell
uv run gemma-clinc train-sft `
  --config configs/phase2_qlora_sft.yaml `
  --train-limit 1510 `
  --eval-limit 302 `
  --max-steps 95
```

If loss, invalid-output rate and macro-F1 move in the expected direction, launch
the committed full experiment:

```powershell
uv run gemma-clinc train-sft --config configs/phase2_qlora_sft.yaml
```

The two-example benchmark projects roughly 11-13 hours for three complete epochs
plus validation on this laptop. Thermal throttling and background GPU use can
change that estimate materially.

## Artifacts

Each run under `artifacts/phase2/` contains:

- `run_metadata.json`: exact configuration, Git commit, limits and dataset hash.
- `label_vocabulary.json`: the expected 151 labels.
- `training_history.jsonl`: optimizer-step losses, LR and CUDA peaks.
- `adapter_epoch_*`: independently reloadable PEFT checkpoints.
- `validation_epoch_*.jsonl`: auditable raw generations.
- `validation_metrics_epoch_*.json`: checkpoint metrics.
- `training_summary.json`: best adapter and final resource summary.

Re-evaluate a saved adapter on validation:

```powershell
uv run gemma-clinc evaluate-adapter `
  --config configs/phase2_qlora_sft.yaml `
  --adapter artifacts/phase2/<run>/adapter_epoch_003_step_<step> `
  --split val
```

Use `--split test` exactly once after selecting and freezing the best Phase 2
adapter. Test results must not influence hyperparameters or checkpoint choice.
