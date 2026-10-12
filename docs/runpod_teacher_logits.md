# Offline teacher logits on RunPod

This extractor produces train-only, gold-response teacher-forcing targets, not
teacher-generated labels. The student training loop does not yet consume these
files. No paid pod or 31B weights are needed for the local tokenizer preflight.

## Before renting

Run from the repository root in PowerShell, with the project's GPU dependencies
installed (the preflight uses Transformers but does not load model weights):

```powershell
.venv/Scripts/python.exe scripts/generate_teacher_logits.py --prepare-only --limit 151
```

This downloads CLINC150 and both tokenizers, resolves model revisions to immutable
commits, checks identical token-to-ID mappings and special tokens, and checks gold
response alignment including turn terminators. It records maximum sequence length.
There is no truncation. A mismatch aborts before loading teacher weights; do not
disable the check to force incompatible distributions through.

Prepare an SSH key locally and register the public key before deployment:

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\runpod_ed25519"
Get-Content "$env:USERPROFILE\.ssh\runpod_ed25519.pub"
```

Choose A40 48 GB, CUDA 12.8 and a PyTorch template with SSH enabled, TCP port 22
exposed and preferably 150 GB **Volume Disk**, mounted at `/workspace`. Temporary
Container Disk is not persistent. The original checkpoint occupies approximately
62 GB on disk even when loaded in INT8. Use SSH over exposed TCP, not Basic SSH,
for SCP transfers. See [RunPod's SSH guide](https://www.runpod.io/blog/how-to-connect-to-your-pod-with-ssh-and-move-files-with-scp-and-filezilla).
The intended image is
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.
The setup verifies actual Python/PyTorch/CUDA versions rather than trusting the
template display name.

## Upload current local source

Do this in local PowerShell from the repository root. Replace IP and PUERTO with
the Connect panel values. The archive includes uncommitted script changes and
excludes local environments, cached weights, dataset and previous artifacts.

```powershell
tar --exclude=__pycache__ -czf "$env:TEMP\finetuning-pod.tar.gz" pyproject.toml uv.lock README.md src scripts configs
scp -P PUERTO -i "$env:USERPROFILE\.ssh\runpod_ed25519" "$env:TEMP\finetuning-pod.tar.gz" root@IP:/workspace/
ssh -p PUERTO -i "$env:USERPROFILE\.ssh\runpod_ed25519" root@IP
```

## Prepare the pod

The following commands run on Linux, inside the SSH session:

```bash
mkdir -p /workspace/finetuning
cd /workspace/finetuning
tar -xzf /workspace/finetuning-pod.tar.gz
bash scripts/setup_runpod.sh
source .venv-pod/bin/activate
export HF_HOME=/workspace/hf-cache
hf auth login
nvidia-smi
df -h /workspace
```

The installer reuses the template's **PyTorch 2.8.0 + CUDA 12.8** with a
`--system-site-packages` environment. It never installs the repository's local
`gpu` extra, PyTorch 2.11 or a new CUDA stack. It checks the actual template
versions before installing and constrains torch, torchvision, torchaudio and
Triton so dependency resolution cannot silently replace them.

`requirements-pod.lock.txt` pins Linux CPython 3.11 dependencies separately from
the Windows/local `uv.lock`. The installer runs `pip check` and a real tiny Gemma 4
INT8/BF16 CUDA forward pass using SDPA and `logits_to_keep`, then verifies sparse
probabilities and an NPZ round trip. This needs no HF account or downloaded teacher
weights. The report is `artifacts/pod_preflight.json`. It requires at least 44 GiB
free GPU memory on the pod. It does not demonstrate that the full 31B model fits;
the teacher pilot remains necessary. Template optional packages are inherited;
broken imports or dependency conflicts abort setup before the teacher download.

The installation itself uses time, network and disk. Never put authentication
tokens in source/config. Use a fresh `.venv-pod` if you previously ran the old
installer that created an isolated PyTorch 2.11 environment.

## Preflight, pilot and full extraction

```bash
python scripts/generate_teacher_logits.py --prepare-only --limit 151
tmux new -s teacher
python scripts/generate_teacher_logits.py --limit 151
```

If `tmux` is absent, install it with `apt-get update` then `apt-get install -y tmux`.
Detach using Ctrl+B then D; reconnect with `tmux attach -t teacher`. Closing SSH
does not stop the pod or billing. The pilot is stratified (one example per label).
Read its `summary.json` for runtime and GPU allocated/reserved memory; loading and
verification are included in session time, so it is not a pure throughput benchmark.
Cross-check total GPU memory using `nvidia-smi`. No A40 measurement is claimed yet.

After a successful pilot, run all 15,100 train examples:

```bash
python scripts/generate_teacher_logits.py
```

Repeat the same command after interruption to resume. Completed NPZ files are
validated and skipped. Pilots get separate `_pilot151` directories. Changes to
configuration, dataset, tokenizer identity or recorded runtime require a new
output directory. Extraction stops on errors, including CUDA OOM; it never silently
changes precision or truncates inputs. Concurrent writers to one output directory
are not supported. Keep all files under `/workspace`.

## Artifact contract

`artifacts/teacher_logits/train_int8_t2_k128/` contains:

- `manifest.json`: configuration, resolved revisions, vocab/dataset hashes, token
  length, runtime versions and extractor hash. `main` is resolved once per run and
  its saved commit reused on restart.
- `teacher_tokenizer/`, `student_tokenizer/`: saved tokenizers and chat templates.
- `samples/train-NNNNN.npz`: one atomically written file per example, with metadata,
  teacher/student full `input_ids`, target positions, common gold `target_ids`,
  `topk_ids` (INT32), `topk_log_probs` and `tail_log_prob` (FP32).
- `examples.jsonl`: ordered example text, gold label, split and per-NPZ SHA-256.
- `summary.json`: complete flag, counts and session memory/time, written after success.

Positions index the **target token** in the saved sequence; the corresponding
causal prediction is at `position - 1`. Teacher and student prompts differ, but
their entire response suffixes must match. Only response-position logits are
computed using `logits_to_keep`; no full-prompt vocabulary tensor is created.
No sampling, repetition penalty or generation-time probability filtering is used.
The gold sequence preserves the exact no-thinking generation prefix, including
any empty thought-channel marker, and appends the label with the chat template's
turn terminator. This handles templates that omit that marker when rendering a
completed assistant message.

Top-k values are log probabilities normalized over the **whole vocabulary** at
temperature 2. The complement is stored as a log probability, computed directly
to avoid cancellation. Future training can compute a KL over k tokens plus a rest
bucket. This is an approximation to full-vocabulary KL, not full raw logits, and
does not allow exact changes of temperature. FP32 avoids FP16 rounding changing
the probability mass. Plain text metadata uses JSON; NPZ requires no pickle.

The teacher is frozen INT8 on one GPU, with SDPA, batch size 1, no gradients or KV
cache. Multimodal weights are loaded as part of the checkpoint but no images/audio
are supplied. Test and validation cannot be exported as training targets here.

## Download and verify on Windows

In a local PowerShell window from the repository root:

```powershell
.\scripts\download_teacher_logits.ps1 -PodIp IP -SshPort PUERTO
```

This downloads and verifies automatically. Use `-RunName train_int8_t2_k128_pilot151`
for the pilot, or `-KeyPath` for another SSH key. The equivalent manual commands are:

```powershell
New-Item -ItemType Directory -Force .\artifacts\teacher_logits
scp -r -P PUERTO -i "$env:USERPROFILE\.ssh\runpod_ed25519" root@IP:/workspace/finetuning/artifacts/teacher_logits/train_int8_t2_k128 .\artifacts\teacher_logits\
.venv/Scripts/python.exe scripts/generate_teacher_logits.py --verify artifacts/teacher_logits/train_int8_t2_k128
```

Verification needs neither CUDA nor Transformers/network. It checks counts,
checksums, response alignment and probability normalization. Download the entire
directory, not only `samples`. Verify locally before terminating the pod. Stopping
preserves a Volume Disk but continues storage billing; terminating removes it.
Network volumes have a separate lifecycle. See [RunPod storage](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage).

## Refreshing the Linux dependency lock

For maintainers only; never resolve latest dependencies during a paid extraction:
resolve `scripts/requirements-pod.txt` with `scripts/constraints-pod.txt`, targeting
Linux x86_64 and Python 3.11, and omit torch, torchvision, torchaudio, Triton and
NVIDIA runtime packages from the output lock (they belong to the template).
Retest the tiny CUDA model after changing the lock. Windows training dependencies
are independent and remain unchanged.

## Validation recorded on 11 October 2026

- Windows: 33 tests, Ruff lint/format, Bash syntax check.
- Linux Ubuntu 24.04 under WSL: isolated Python 3.11, PyTorch 2.8.0+cu128,
  torchvision 0.23.0+cu128 and the Linux dependency lock installed successfully;
  `pip check` and 33 tests passed. A real tiny Gemma 4 save/load, SDPA forward,
  selected-vs-full logits comparison and sparse NPZ round trip passed on CPU.
  Report: `artifacts/pod_audit/linux_torch28_cpu_smoke.json`.
- The tokenizer preflight passed for all 15,100 train examples with a maximum
  teacher length of 856 tokens.
- WSL had no working NVIDIA driver access, so a Linux CUDA/INT8 pass could not
  be executed locally. Setup runs that check on the real pod before downloading
  teacher weights. The complete RunPod image and A40 31B memory fit have not yet
  been tested; successful CPU checks do not establish those hardware results.
