# Gemma 4 E2B offline distillation — 12 October 2026

The complete run finished at 2026-10-12 02:55 UTC (04:55 Europe/Madrid). It
processed 15,100 official training examples for three epochs: 45,300 sample
presentations and 2,832 optimizer steps. The selected adapter is epoch 3. The
held-out test split was not used for training, teacher extraction or selection,
and has not yet been evaluated with this adapter.

## Method and environment

- Teacher: `google/gemma-4-31B-it`, BF16 without quantization, Modal A100 80 GB.
- Student: `google/gemma-4-E2B-it`, revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, fresh QLoRA adapter.
- Teacher targets: response-only top-128 log probabilities and residual mass at
  temperature 2, aligned token IDs, verified against all official training rows.
- Objective: `0.5 * gold CE + 0.5 * T² * grouped forward KL`, T=2.
- Student runtime: Modal L4, PyTorch 2.11.0+cu128, PEFT 0.20.0, NF4 base weights,
  BF16 computation and FP32 adapters, GPU-resident embeddings.
- Training: rank 8 / alpha 16 q/v LoRA, accumulation 16, learning rate 2e-4,
  seed 42; configuration in `configs/distillation_qlora_gpu.yaml`.

The job resumed the verified Windows checkpoint at step 50, completed a Modal
probe to step 58, then ran detached to completion. Recorded cumulative training
time was 3.06 hours and validation time 0.89 hours; these counters exclude setup
and some checkpoint overhead. Peak reserved GPU memory was 6,690 MiB.

## Validation results

All rows use the full official validation split, 3,000 in-scope and 100 OOS
examples. Values are percentages; macro-F1 covers all 151 labels.

| Epoch | Macro-F1 | Accuracy | In-scope accuracy | OOS F1 | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| 1 | 92.41 | 90.10 | 91.23 | 59.89 | 4.45 |
| 2 | 92.09 | 90.06 | 90.60 | 50.51 | 2.00 |
| **3 (selected)** | **93.93** | **92.77** | **93.43** | **68.22** | **1.90** |

Epoch 3 OOS precision is 64.04% and recall is 73.00%. The recorded SFT adapter
has higher macro-F1 (96.96%) and accuracy (95.13%); this KD run has higher OOS
F1 and fewer invalid outputs. These are single runs, with BF16 computation for
KD versus FP16 for SFT, so the comparison does not isolate the loss change.

## Provenance and verification

Downloaded artifacts are retained locally under
`artifacts/distillation/modal_completed_full_gpu_bf16/`, and remote checkpoints
remain in Modal volume `finetuning-student-training`, directory `/full_gpu_bf16`.
Raw logits, predictions, checkpoints and model weights are excluded from Git.

The verification checked all 3,100 validation IDs, texts and labels per epoch
against the official split and independently recomputed every recorded metric.
All logged numeric values are finite. The selected adapter contains 100 finite
FP32 tensors with 1,339,392 parameters. Trainer and teacher-manifest hashes match
the immutable run metadata. `verification_report.json` records these checks.

| Artifact | SHA-256 |
|---|---|
| Teacher archive | `f57dd8557d16f2425448f6b9b3a9a4028d79788bea08abfa2dd691ed8fb30dd2` |
| Teacher manifest | `049a98a996e9d7e8a7286e5d301ce1784f687ff9b55d33decff4113d09ff473b` |
| Trainer source | `551cf4b10efbb0ef7cdf6bbb4d16d77d702d0da68e847711231468a0cfbccfbd` |
| Selected adapter weights | `d6e71f888aa04503190b92cf82e499bcfa367613cf0e4b6cddcb5a4a28ab0900` |

Source and workflow details: [local training](../distillation_training.md),
[Modal student](../modal_student_training.md), and
[Modal teacher extraction](../modal_teacher_logits.md).
