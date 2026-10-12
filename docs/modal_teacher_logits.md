# Teacher logits on Modal

This workflow uses **A100 80 GB, unquantized BF16**, as requested. It does not
fine-tune the teacher or train the student. It extracts teacher-forced sparse
distributions over gold training responses, downloads the complete artifact and
verifies its checksums locally.

The Modal client runs in the user's globally installed Python; the local project
environment is used only for final verification. Authenticate with `modal setup`
before running. A cached Hugging Face token (or `HF_TOKEN`) is passed through an
ephemeral Modal Secret, never embedded in the image or printed.

Run in local PowerShell from the repository root:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
modal run --detach scripts/modal_teacher_logits.py
```

The UTF-8 settings are necessary for Modal's console output on Windows. The task:

1. Builds a Linux Python 3.11 image with PyTorch 2.8/CUDA 12.8 and the locked
   dependencies. No local Windows environment is uploaded.
2. On CPU, prepares all 15,100 train examples and downloads the teacher weights
   into the `finetuning-hf-cache` persistent volume.
3. On an A100 80 GB, checks BF16 with a tiny real Gemma model and runs a stratified
   151-example teacher pilot. The teacher is not quantized.
4. Runs the full train split and saves each example atomically to the
   `finetuning-teacher-data` volume, including provenance and sparse distributions.
5. Checks the complete index and packages the result in a CPU task, downloads the archive to
   `artifacts/teacher_logits/`, extracts it, and verifies all examples locally.

Every saved sample is validated and hashed while building the final index. The
Modal workflow avoids an additional serial verification on the GPU: small remote
file reads were slow in the measured run. Packaging uses 24 concurrent readers;
the final downloaded files receive a complete checksum, alignment and probability
verification locally.

Configuration: `configs/teacher_logits_modal.yaml`. Run name:
`train_bf16_t2_k128`. The separate RunPod INT8 configuration remains available.
The stored top-128 log probabilities and residual mass are FP32 at temperature 2;
these are sparse targets, not complete raw vocabulary logits.

For a pilot only:

```powershell
modal run --detach scripts/modal_teacher_logits.py --pilot-only
```

To retrieve a completed remote run without extracting again:

```powershell
modal run scripts/modal_teacher_logits.py --download-only
```

Repeating the original command skips validated samples. Keep the exact source,
dependency versions and configuration for a resume. Modal commits persistent
volumes in the background and the workflow commits explicitly before returning.
Do not start concurrent writers to the same run directory.

Detached mode keeps the currently executing remote function alive after a local
disconnect; the local orchestration/download does not continue if the local
process is killed. Resume from the same command once the remote function ends.
Functions scale down after work; the data and model-cache volumes persist and
may incur storage charges. These volumes are retained to allow recovery/resume.

See [Modal GPU configuration](https://modal.com/docs/guide/gpu) and
[persistent volumes](https://modal.com/docs/guide/volumes).

## Completed run: 11 October 2026

- Teacher: `google/gemma-4-31B-it`, unquantized BF16 on NVIDIA A100-SXM4-80GB.
- All 15,100 train examples generated; 58.46 GiB peak allocated and 58.69 GiB
  peak reserved GPU memory. Extraction plus loading and index checks: 5,253.7 s.
- The redundant serial remote verification was stopped after the complete index
  had already been validated and written. The GPU app was shut down; a CPU task
  packaged the files with parallel reads and the downloaded data passed complete
  local verification (SHA-256, target alignment and probability normalization).
- Local archive: `artifacts/teacher_logits/train_bf16_t2_k128.tar.gz`, 84,868,014 bytes.
- SHA-256: `f57dd8557d16f2425448f6b9b3a9a4028d79788bea08abfa2dd691ed8fb30dd2`.
- An exact copy of the extractor used is retained as
  `artifacts/modal_distillation_extractor_used.py`; its hash matches the manifest.
- The GPU app and packaging app are stopped. Both persistent volumes are retained
  for model-cache reuse and output recovery; volume storage has separate billing.
