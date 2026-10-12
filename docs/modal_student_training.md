# Detached E2B distillation on Modal

The completed student run used a **Modal NVIDIA L4 (24 GB)**, NF4 base weights,
BF16 computation and FP32 LoRA adapters. It keeps all embeddings on GPU. The
Windows laptop showed a sustained GPU-frequency drop after a few minutes, so
the validated checkpoint was moved to Modal instead of relying on laptop power
behavior for an overnight run. The local training and display guard were stopped.

## Verified completion — 12 October 2026

The run completed at 02:55 UTC with 2,832 optimizer steps and 45,300 sample
presentations (three full epochs). Recorded training time is 3.06 hours and validation
time 0.89 hours; these counters exclude setup and some checkpoint overhead. No Modal
apps remain running. Persistent volumes are retained.

Results are downloaded under `artifacts/distillation/modal_completed_full_gpu_bf16/`.
`verification_report.json` records independent metric recomputation from all 3,100
official validation examples in each epoch, finite logged losses/gradients and all
100 adapter tensors (1,339,392 FP32 parameters), and matching trainer/teacher hashes.
Epoch 3 is selected: macro-F1 93.93%, accuracy 92.77%, OOS F1 68.22%, invalid outputs
1.90%. It has lower macro-F1 than the recorded SFT run (96.96%); the held-out test
remains unevaluated. The adapter is in `adapter_epoch_003/` within the downloaded
results directory. Remote best-adapter paths refer to Modal's internal mount and
must be replaced by this local path for inference.

The objective and training settings are unchanged: 0.5 gold CE plus 0.5 T²
grouped KL, T=2, three epochs, 15,100 training examples, accumulation 16, rank 8
q/v LoRA, and selection on all 3,100 validation examples. No test examples are used.

The Linux image pins PyTorch 2.11.0+cu128, torchvision 0.26.0, PEFT 0.20.0 and
the repository runtime dependencies. Linux PyTorch 2.11 requires setuptools<82;
`requirements-modal-training.lock.txt` resolves that separately from the RunPod
PyTorch 2.8 lock. GPU checkpoints restore adapter weights, AdamW, scheduler,
sample cursor and CPU/CUDA/Python random states.

Teacher samples are extracted from the verified archive to the container's local
disk before loading; repeated small-file reads from a remote volume are avoided.
Checkpoints and validation results are saved in the persistent Modal volume
`finetuning-student-training`, directory `/full_gpu_bf16`. The mounted model cache
is `finetuning-hf-cache` and teacher data is `finetuning-teacher-data`.

## Validation and launch

The technical local pilot, longest-sequence check and checkpoint recovery passed;
37 tests and Ruff checks passed. A Modal probe restored local step 50 and completed
eight updates to step 58, with finite losses/gradients, approximately 6.5 GiB GPU
reservation and about 2.5 examples/s. The remaining training is roughly five hours;
with validation/checkpoint overhead the initial estimate is **6–8 hours**.

Commands from the repository root (Modal's globally installed Python/CLI):

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
modal run scripts/modal_student_training.py --stage probe
modal run --detach scripts/modal_student_training.py --stage start
```

**Do not repeat `probe` on an active run:** it uploads the local checkpoint and
would overwrite remote progress. `start` is for a stopped/interrupted run only;
never create simultaneous writers to the same training directory.

The active detached app/call IDs are recorded in
`artifacts/distillation/full_gpu_bf16/modal_launch.json`. `train.spawn()` starts
the remote call and returns immediately. Detached mode keeps it running after
the CLI exits or the laptop disconnects. Successful completion writes
`training_summary.json`, completed `status.json` and `best_adapter.json`.
GPU containers scale down after the call; persistent volumes have separate storage
billing. This is a single job, not a deployment, scheduler or recurring monitor.

## Check progress without allocating a GPU

```powershell
python scripts/modal_training_status.py
```

This also mirrors the latest status to local `modal_status.json`. The training ETA
in status excludes upcoming validation. Use the Modal app link in the launch
receipt for logs. The laptop can be closed or shut down while the cloud run executes.

## Retrieve results after completion

```powershell
modal volume get finetuning-student-training /full_gpu_bf16/best_adapter.json artifacts/distillation/full_gpu_bf16/best_adapter_modal.json
modal volume get finetuning-student-training /full_gpu_bf16/training_summary.json artifacts/distillation/full_gpu_bf16/training_summary_modal.json
```

Download the adapter epoch indicated by `best_adapter.json`, plus validation
metrics/predictions and run metadata. Its remote absolute path must be translated
to the downloaded local directory before adapter evaluation. Checkpoints remain
available for recovery. Do not delete the volume before obtaining verified results.
