# Hardware memory profile

This project uses measured memory figures from the development machine instead of
estimating feasibility from parameter counts or checkpoint marketing figures.

## Test system

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB VRAM
- CPU: AMD Ryzen 9 8945HS
- System RAM: 32 GiB
- NVIDIA driver: 610.88
- Student: `google/gemma-4-E2B-it`
- Runtime: PyTorch 2.11.0 + CUDA 12.8, Transformers 5.15.0,
  bitsandbytes 0.50.0 and PEFT 0.20.0

## Measurements

All values below are deltas measured in a fresh process on this machine. The
selected configuration was rerun on 2026-08-25. `nvidia-smi` includes the CUDA context and allocator reservation;
PyTorch's allocated peak describes tensors actively held by the workload.

| Configuration | Workload | PyTorch peak | `nvidia-smi` delta | Result |
|---|---|---:|---:|---|
| NF4, complete model on GPU | one inference | 6,335 MiB | 6,412 MiB | valid label |
| NF4, PLE moved to CPU after GPU load | one inference | 1,925 MiB | 6,412 MiB | valid label; CUDA cache remains large |
| NF4, PLE loaded directly on CPU, text-only | inference with all 151 labels | 1,837 MiB | 2,191 MiB | valid label |
| NF4, PLE on CPU, text-only, full label vocabulary | response-only QLoRA step, 834 tokens | 5,338 MiB | 5,618 MiB | finite loss and gradients |
| NF4, PLE on CPU, text-only, compact SFT prompt | QLoRA step on longest train example, 61 tokens | 2,038 MiB | 2,337 MiB | finite loss and gradients |

The selected compact SFT configuration leaves about 5,851 MiB of physical VRAM
free during the measured optimizer step. The loss was 7.4926 and all 100 LoRA
gradient tensors were finite. The high initial loss is expected before training;
the relevant feasibility checks are that it and every gradient are finite.

The PLE is confirmed on CPU with shape `262144 x 8960` and size 4,480 MiB. Model
loading increased process RSS by 4,975 MiB; 11,709 MiB of system RAM remained
available during the measurement on the 32 GiB machine.

## Why this works

Gemma 4 E2B has 5.1 billion total parameters despite its 2.3 billion effective
parameter count. Inspection of the downloaded checkpoint shows that
`embed_tokens_per_layer.weight` has shape `262144 x 8960` and occupies about
4.38 GiB in BF16. It is an embedding lookup, not a dense operation over the full
table. Loading it directly into system RAM and transferring only selected rows
keeps the GPU peak low.

This matches Google's explanation that Gemma 4's large embedding tables account
for the gap between effective parameter count and static weight memory in the
[official model overview](https://ai.google.dev/gemma/docs/core).

Accelerate's ordinary CPU offload hook is not sufficient here: it stages the
entire table on CUDA before each call. The project therefore replaces that one
frozen lookup with a CPU lookup after retrieving its CPU-offloaded weight. The
audio tower, vision tower and their projectors are then removed because CLINC150
is strictly text-to-text.

For supervised fine-tuning and response distillation, the loss is computed only
over the assistant's label tokens. The compact training prompt does not repeat all
151 labels for every example: they are the mapping the adapter is meant to learn.
The profiler scans the complete official training split and measures its longest
tokenized example. Gradient accumulation should be used to increase the effective
batch size without increasing the measured micro-batch of one.

## Reproduce the training measurement

```powershell
uv sync --extra gpu --extra train
uv run python scripts/profile_qlora_memory.py
```

Close other GPU applications before comparing the `nvidia-smi` delta. This is a
one-step feasibility check, not a throughput benchmark or a substitute for the
full Phase 2 validation run.
