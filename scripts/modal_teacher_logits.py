"""Run teacher extraction on Modal and download the verified archive locally.

Run with the authenticated Modal CLI: modal run --detach scripts/modal_teacher_logits.py
Use --pilot-only for a pilot, or --download-only to retrieve an existing full run.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/finetuning")
WORKSPACE = Path("/workspace")
RUN_NAME = "train_bf16_t2_k128"
DATA_VOLUME = "finetuning-teacher-data"
CACHE_VOLUME = "finetuning-hf-cache"

app = modal.App("finetuning-teacher-logits")
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
if modal.is_local():
    token_path = Path(
        os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token"))
    )
    token = os.environ.get("HF_TOKEN")
    if not token and token_path.is_file():
        token = token_path.read_text().strip()
    if token:
        hf_secret = modal.Secret.from_dict({"HF_TOKEN": token})
    # Constraints must exist during image build, before pip resolves packages.
    image = (
        image.add_local_file(
            ROOT / "scripts/constraints-pod.txt", "/opt/pytorch-constraints.txt", copy=True
        )
        .pip_install(
            "torch==2.8.0",
            "torchvision==0.23.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install_from_requirements(
            str(ROOT / "scripts/requirements-pod.lock.txt"),
            extra_options="--constraint /opt/pytorch-constraints.txt",
        )
        .run_commands("python -m pip check")
        .env(
            {
                "HF_HOME": "/hf-cache",
                "PYTHONPATH": (REMOTE_ROOT / "src").as_posix(),
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .workdir("/opt")
        .add_local_dir(ROOT / "src", (REMOTE_ROOT / "src").as_posix(), ignore=["**/__pycache__/**"])
        .add_local_file(
            ROOT / "configs/teacher_logits_modal.yaml",
            (REMOTE_ROOT / "configs/teacher_logits_modal.yaml").as_posix(),
        )
    )

COMMON = {
    "image": image,
    "volumes": {WORKSPACE.as_posix(): data_volume, "/hf-cache": cache_volume},
    "secrets": [hf_secret],
    "max_containers": 1,
    "retries": 0,
}


@app.function(**COMMON, cpu=4, memory=98304, timeout=3600)
def prepare() -> dict:
    import yaml
    from huggingface_hub import snapshot_download

    from gemma_clinc.distillation import run

    os.chdir(WORKSPACE)
    config_path = REMOTE_ROOT / "configs/teacher_logits_modal.yaml"
    run(config_path, limit=None, prepare_only=True)
    config = yaml.safe_load(config_path.read_text())
    snapshot_download(
        config["teacher"]["id"],
        revision=config["teacher"]["revision"],
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model"],
    )
    cache_volume.commit()
    data_volume.commit()
    return {"prepared": True}


@app.function(**COMMON, gpu="A100-80GB", cpu=8, memory=98304, timeout=43200, scaledown_window=2)
def extract(limit: int | None = None) -> dict:
    import gc

    import torch

    from gemma_clinc.distillation import run, write_json
    from gemma_clinc.pod_check import check_environment

    os.chdir("/opt")
    data_volume.reload()
    cache_volume.reload()
    os.chdir(WORKSPACE)
    gc.collect()
    torch.cuda.empty_cache()
    report = check_environment(minimum_vram_gib=72, quantization="none")
    write_json(WORKSPACE / "pod_preflight.json", report)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        run(
            REMOTE_ROOT / "configs/teacher_logits_modal.yaml",
            limit=limit,
            prepare_only=False,
            verify_after_run=False,
        )
    finally:
        data_volume.commit()
        gc.collect()
        torch.cuda.empty_cache()
    directory = (
        WORKSPACE
        / "artifacts/teacher_logits"
        / (RUN_NAME if limit is None else f"{RUN_NAME}_pilot{limit}")
    )
    return json.loads((directory / "summary.json").read_text())


@app.function(**COMMON, cpu=2, memory=8192, timeout=1800)
def pack(limit: int | None = None) -> dict:
    import io
    from concurrent.futures import ThreadPoolExecutor

    from gemma_clinc.distillation import digest

    os.chdir("/opt")
    data_volume.reload()
    name = RUN_NAME if limit is None else f"{RUN_NAME}_pilot{limit}"
    directory = WORKSPACE / "artifacts/teacher_logits" / name
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    records = [json.loads(line) for line in (directory / "examples.jsonl").read_text().splitlines()]
    if not summary["complete"] or summary["completed"] != manifest["expected_examples"]:
        raise RuntimeError("Cannot download an incomplete extraction")
    if len(records) != summary["completed"] or len({r["id"] for r in records}) != len(records):
        raise RuntimeError("Invalid extraction index")
    archive = WORKSPACE / f"{name}.tar.gz"
    # Parallel reads avoid serial latency on the remote filesystem. The full
    # checksum/alignment/probability verification runs on the downloaded files.
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    files.append(WORKSPACE / "pod_preflight.json")
    print(f"Packing {len(files)} files using 24 concurrent readers...", flush=True)
    with ThreadPoolExecutor(max_workers=24) as pool:
        with tarfile.open(archive, "w:gz", compresslevel=3) as handle:
            for index, (path, payload) in enumerate(
                zip(files, pool.map(Path.read_bytes, files), strict=True)
            ):
                relative = (
                    path.relative_to(directory)
                    if path.is_relative_to(directory)
                    else Path(path.name)
                )
                member = tarfile.TarInfo(f"{name}/{relative.as_posix()}")
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))
                if index % 1000 == 0:
                    print(f"Packed {index + 1}/{len(files)} files", flush=True)
    data_volume.commit()
    return {"archive": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size}


def download_archive(info: dict, name: str) -> Path:
    import hashlib
    import subprocess

    destination = ROOT / "artifacts/teacher_logits"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / info["archive"]
    temporary = archive.with_suffix(".download")
    with temporary.open("wb") as handle:
        for chunk in data_volume.read_file(info["archive"]):
            handle.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != info["sha256"]:
        raise RuntimeError("Archive download checksum mismatch")
    temporary.replace(archive)
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    local_python = ROOT / ".venv/Scripts/python.exe"
    if not local_python.exists():
        raise RuntimeError("Local project .venv required for final artifact verification")
    subprocess.run(
        [
            str(local_python),
            str(ROOT / "scripts/generate_teacher_logits.py"),
            "--verify",
            str(destination / name),
        ],
        check=True,
    )
    return destination / name


@app.local_entrypoint()
def main(pilot_only: bool = False, download_only: bool = False, resume_full: bool = False) -> None:
    name = f"{RUN_NAME}_pilot151" if pilot_only else RUN_NAME
    if not download_only:
        if not resume_full:
            print("Preparing dataset/tokenizers and teacher weights on CPU...")
            print(prepare.remote())
            print("Testing unquantized BF16 on A100 80GB and extracting 151 stratified examples...")
            print(extract.remote(151))
        if not pilot_only:
            print("Extracting all 15,100 train examples...")
            print(extract.remote())
    info = pack.remote(151 if pilot_only else None)
    print("Downloading archive:", info)
    path = download_archive(info, name)
    print("Downloaded and verified:", path)
