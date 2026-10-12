"""Resume the student on a Modal L4; probe, then detached full training."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("finetuning-student-kd")
teacher_volume = modal.Volume.from_name("finetuning-teacher-data")
cache_volume = modal.Volume.from_name("finetuning-hf-cache")
training_volume = modal.Volume.from_name("finetuning-student-training", create_if_missing=True)
secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    token = os.environ.get("HF_TOKEN")
    token_path = Path.home() / ".cache/huggingface/token"
    if not token and token_path.exists():
        token = token_path.read_text().strip()
    if token:
        secret = modal.Secret.from_dict({"HF_TOKEN": token})
    image = (
        image.add_local_file(
            ROOT / "scripts/constraints-modal-training.txt",
            "/opt/training-constraints.txt",
            copy=True,
        )
        .pip_install(
            "torch==2.11.0",
            "torchvision==0.26.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install_from_requirements(
            str(ROOT / "scripts/requirements-modal-training.lock.txt"),
            extra_options="--constraint /opt/training-constraints.txt",
        )
        .pip_install("peft==0.20.0", extra_options="--constraint /opt/training-constraints.txt")
        .run_commands("python -m pip check")
        .env(
            {
                "HF_HOME": "/hf-cache",
                "PYTHONPATH": "/opt/finetuning/src",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .workdir("/opt")
        .add_local_dir(ROOT / "src", "/opt/finetuning/src", ignore=["**/__pycache__/**"])
        .add_local_file(ROOT / "configs/distillation_qlora_gpu.yaml", "/opt/student_config.yaml")
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={"/hf-cache": cache_volume},
    cpu=4,
    memory=16384,
    timeout=3600,
    scaledown_window=2,
)
def prepare_model():
    import yaml
    from huggingface_hub import snapshot_download

    config = yaml.safe_load(Path("/opt/student_config.yaml").read_text())
    snapshot_download(
        config["model"]["id"],
        revision=config["model"]["revision"],
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model"],
    )
    cache_volume.commit()
    return {"student_weights_cached": True}


@app.function(
    image=image,
    secrets=[secret],
    gpu="L4",
    cpu=4,
    memory=16384,
    volumes={
        "/hf-cache": cache_volume,
        "/teacher": teacher_volume,
        "/student_runs": training_volume,
    },
    timeout=43200,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def train(probe: bool = False):
    import importlib.metadata
    import time

    import torch

    from gemma_clinc.distillation import write_json
    from gemma_clinc.kd_training import run_training

    work = Path("/tmp/student_job")
    work.mkdir(exist_ok=True)
    targets = work / "artifacts/teacher_logits"
    if not (targets / "train_bf16_t2_k128/manifest.json").exists():
        targets.mkdir(parents=True, exist_ok=True)
        with tarfile.open("/teacher/train_bf16_t2_k128.tar.gz") as handle:
            handle.extractall(targets, filter="data")
    os.chdir(work)
    run_dir = Path("/student_runs/full_gpu_bf16")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
        raise RuntimeError("Student GPU must have at least 20 GiB")
    write_json(
        run_dir / "modal_environment.json",
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "peft": importlib.metadata.version("peft"),
            "cuda": torch.version.cuda,
            "precision": "NF4 base / BF16 compute / FP32 adapter",
            "probe": probe,
        },
    )
    metadata = json.loads((run_dir / "latest_checkpoint.json").read_text())
    start_step = metadata["global_step"]
    started = time.monotonic()
    try:
        run_training(
            Path("/opt/student_config.yaml"),
            run_dir,
            resume=True,
            max_steps=start_step + 8 if probe else None,
        )
    except Exception as exc:
        write_json(run_dir / "failure.json", {"state": "failed", "error": str(exc)})
        raise
    finally:
        training_volume.commit()
    status = json.loads((run_dir / "status.json").read_text())
    status["modal_session_seconds"] = time.monotonic() - started
    return status


@app.local_entrypoint()
def main(stage: str = "probe"):
    local = ROOT / "artifacts/distillation/full_gpu_bf16"
    if stage == "probe":
        latest = json.loads((local / "latest_checkpoint.json").read_text())
        with training_volume.batch_upload(force=True) as batch:
            batch.put_file(local / "run_metadata.json", "/full_gpu_bf16/run_metadata.json")
            batch.put_file(
                local / "latest_checkpoint.json", "/full_gpu_bf16/latest_checkpoint.json"
            )
            batch.put_directory(local / latest["path"], f"/full_gpu_bf16/{latest['path']}")
        print(prepare_model.remote(), flush=True)
        result = train.remote(probe=True)
        print(json.dumps(result), flush=True)
        (local / "modal_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    elif stage == "start":
        call = train.spawn()
        receipt = {
            "call_id": call.object_id,
            "app_id": app.app_id,
            "volume": "finetuning-student-training",
            "run": "/full_gpu_bf16",
        }
        (local / "modal_launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    else:
        raise ValueError("stage must be probe or start")
