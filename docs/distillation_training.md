# Local student distillation

**Current execution:** the local technical tests passed, but sustained GPU-frequency
reductions made the laptop ETA unreliable. The full run completed after moving its
checkpoint to [Modal L4](modal_student_training.md). Both local training processes
and the transient display guard have stopped. The local instructions below remain
available as a fallback; do not run them concurrently with the cloud experiment.

The student is the pinned Gemma 4 E2B IT base with a fresh QLoRA adapter, not the
previous SFT adapter. NF4 weights, BF16 computation and removal of unused
multimodal towers follow the repository's low-VRAM implementation. The active
`distillation_qlora_gpu.yaml` configuration keeps the PLE embeddings on GPU:
compact response-only training fits on the RTX 4070 8 GB and avoids the observed
CPU-offload slowdown. The original CPU-PLE configuration remains available.

`configs/distillation_qlora.yaml` selects three epochs, micro-batch one, accumulation
16, AdamW learning rate 2e-4, rank 8 on q/v projections, gradient clipping at 1,
and validation-based selection. No test queries are used.

The loss is `0.5 * CE(gold) + 0.5 * 2^2 * KL(grouped teacher || student)`.
Teacher top-128 values and tail are already normalized at T=2. Student log
probabilities are normalized over the full vocabulary; the tail is evaluated
directly with masked logsumexp. The grouped KL is an approximation to full KL.
Only saved assistant-response positions contribute; the causal prediction index
is the saved target position minus one. Full input IDs are taken from the export,
avoiding response retokenization or changes to the template.

The trainer checks every teacher file's hash, shape, target alignment and mass;
matches source labels against official train, checks tokenizer/model revisions and
vocabulary, and checks prompt and temperature. NaN/Inf losses or gradient norms
abort and write `failure.json`/a failed `status.json`.

## Pilot and checkpoint recovery

```powershell
.venv/Scripts/python.exe -u scripts/train_distillation.py --run-dir artifacts/distillation/pilot_bf16_verified --train-limit 151 --eval-limit 32 --max-steps 12
.venv/Scripts/python.exe -u scripts/train_distillation.py --run-dir artifacts/distillation/pilot_bf16_verified --train-limit 151 --eval-limit 32 --max-steps 14 --resume
```

Pilots select a stratified subset and also include the longest student sequence.
They exercise training, validation, optimizer updates and saving. `max_steps` is
only an early stop; the scheduler horizon remains the planned full pilot run, so
continuation does not silently change the learning rate. These pilot validation
scores are not model-quality conclusions.

Checkpoints are written in temporary directories and atomically renamed; only
complete checkpoints update `latest_checkpoint.json`. They contain adapter weights,
optimizer/scheduler, sample cursor and CPU/CUDA/Python random states. Resume requires
unchanged training source/config/data. Validation predictions use a temporary file
to avoid duplicated predictions after a restart during validation.

## Detached full run on Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_distillation.ps1
```

This starts a hidden independent Python process and records its PID and logs in
`artifacts/distillation/full_gpu_bf16/launch.json`. It runs all 15,100 examples for
three epochs (2,832 optimizer steps), validates on all 3,100 validation examples
after each epoch, and records the best adapter. Checkpoints are saved every 50
optimizer steps and after each epoch. Keep the computer powered and plugged in;
the trainer prevents idle sleep while alive, but cannot prevent manually suspending
the computer or closing a lid configured to suspend it. Closing this chat does not
stop the detached process.

Inspect progress:

```powershell
Get-Content artifacts/distillation/full_gpu_bf16/status.json
Get-Content artifacts/distillation/full_gpu_bf16/train.stdout.log -Tail 5
Get-Content artifacts/distillation/full_gpu_bf16/train.stderr.log -Tail 5
```

`status.json` updates after each optimizer step. The training ETA excludes upcoming
validation; add validation time when estimating total completion. During validation,
status indicates that phase. `training_summary.json` and completed status mark a
successful finish; `best_adapter.json` identifies the selected adapter.

After interruption, first ensure the old PID is no longer running, then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_distillation.ps1 -Resume
```

Do not edit source, change the YAML or upgrade packages during a running experiment.

## Launch evidence: 12 October 2026

The technical pilot exercised all 151 labels plus the longest saved student
sequence. It completed 12 optimizer steps, a 32-example validation and saved an
adapter with changed LoRA weights. Losses/gradients remained finite; peak reserved
memory was 2,238 MiB. Pilot classification scores are not final evaluation results.
The final trainer source was also tested by stopping a fresh run at step 2 and
resuming it to step 4, restoring adapter, optimizer, scheduler and random states.
All 37 unit tests and Ruff checks passed before the full launch.

The preserved local GPU-resident run uses `artifacts/distillation/full_gpu_bf16/`. The earlier
CPU-PLE run was stopped after a severe sustained throughput drop; its step-50
checkpoint is preserved in `artifacts/distillation/full_bf16/`. The GPU-resident
pilot passed with 6,688 MiB reserved memory and the same initial CE/KD values.
On Windows the
virtual-environment launcher may have a different PID from the actual Python
worker: `status.json` identifies the worker, while `launch.json` records both.
The GPU-resident throughput is about 2.4 training examples/s, projecting roughly
5.3 hours of training plus validation/checkpoint overhead (about 7–9 hours total).
The launcher opts this process out of Windows' inferred energy throttling and
uses above-normal priority. This applies only to the worker and does not alter
the system power plan; details are recorded in `performance.json`. See
[Microsoft's process power-throttling API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation).
