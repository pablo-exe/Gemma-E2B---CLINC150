"""Offline sparse KD + gold CE, with resumable response-only QLoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from gemma_clinc.config import load_config
from gemma_clinc.data import Example, download_dataset, labels_from_examples, load_examples
from gemma_clinc.distillation import digest, json_hash, verify_example, write_json
from gemma_clinc.model import GemmaClassifier
from gemma_clinc.training import (
    _append_jsonl,
    _evaluate,
    _prepare_qlora_model,
    optimizer_step_count,
    stratified_limit,
)


def grouped_kd_loss(
    student_logits: Any,
    topk_ids: Any,
    teacher_log_probs: Any,
    teacher_tail_log_prob: Any,
    temperature: float,
) -> Any:
    """Forward KL on teacher-selected tokens + their complement; token mean.

    The student tail is computed directly in log space, not as 1 minus a rounded
    probability sum. T^2 scaling is applied by the caller, not here.
    """
    import torch
    import torch.nn.functional as functional

    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if student_logits.shape[:-1] != topk_ids.shape[:-1]:
        raise ValueError("Student/teacher response positions differ")
    if topk_ids.shape != teacher_log_probs.shape:
        raise ValueError("Teacher top-k shapes differ")
    if teacher_tail_log_prob.shape != student_logits.shape[:-1]:
        raise ValueError("Teacher tail shape differs")
    if topk_ids.shape[-1] >= student_logits.shape[-1]:
        raise ValueError("Teacher top-k must leave a nonempty tail")
    log_probs = functional.log_softmax(student_logits.float() / temperature, dim=-1)
    student_top = log_probs.gather(-1, topk_ids)
    student_tail = torch.logsumexp(log_probs.scatter(-1, topk_ids, -torch.inf), dim=-1)
    top = (teacher_log_probs.exp() * (teacher_log_probs - student_top)).sum(-1)
    tail = teacher_tail_log_prob.exp() * (teacher_tail_log_prob - student_tail)
    return (top + tail).mean()


def load_targets(directory: Path, limit: int | None) -> tuple[list, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] or summary["completed"] != manifest["expected_examples"]:
        raise ValueError("Teacher export is incomplete")
    records = [
        json.loads(line)
        for line in (directory / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != manifest["expected_examples"] or any(r["split"] != "train" for r in records):
        raise ValueError("Invalid teacher index; only train targets are permitted")
    examples = [Example(**{k: r[k] for k in ("id", "text", "label", "split")}) for r in records]
    selected = {e.id for e in stratified_limit(examples, limit)}
    samples = []
    longest = None
    for record in tqdm(records, desc="Load teacher targets", unit="example", mininterval=2):
        path = directory / "samples" / f"{record['id']}.npz"
        if digest(path) != record["sha256"]:
            raise ValueError(f"Teacher checksum mismatch: {path}")
        verify_example(path)
        with np.load(path, allow_pickle=False) as data:
            sample = {
                key: data[key].copy()
                for key in (
                    "student_input_ids",
                    "student_target_positions",
                    "target_ids",
                    "topk_ids",
                    "topk_log_probs",
                    "tail_log_prob",
                )
            }
        sample["example"] = Example(**{k: record[k] for k in ("id", "text", "label", "split")})
        if longest is None or len(sample["student_input_ids"]) > len(longest["student_input_ids"]):
            longest = sample
        if record["id"] in selected:
            samples.append(sample)
    # Every pilot includes the actual longest student sequence to exercise peak memory.
    if limit is not None and longest["example"].id not in selected:
        samples.append(longest)
    return samples, manifest


def save_checkpoint(
    run_dir: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    state: dict,
) -> Path:
    import torch

    step = state["global_step"]
    name = f"checkpoint_step_{step:06d}_epoch_{state['epoch']:03d}"
    target = run_dir / name
    temporary = run_dir / (name + ".tmp")
    root = run_dir.resolve()
    if not target.resolve().is_relative_to(root) or not temporary.resolve().is_relative_to(root):
        raise ValueError("Checkpoint path escapes the requested run directory")
    if temporary.exists():
        shutil.rmtree(temporary)
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter")
    torch.save(
        {
            "state": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        },
        temporary / "training_state.pt",
    )
    write_json(temporary / "cursor.json", state)
    temporary.rename(target)
    write_json(run_dir / "latest_checkpoint.json", {"path": name, "global_step": step})
    return target


def restore_checkpoint(run_dir: Path, model: Any, optimizer: Any, scheduler: Any) -> dict:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    latest = json.loads((run_dir / "latest_checkpoint.json").read_text())
    directory = run_dir / latest["path"]
    weights = load_file(str(directory / "adapter/adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
    missing_adapter = [key for key in result.missing_keys if "lora_" in key]
    if missing_adapter:
        raise ValueError(f"Missing adapter weights: {missing_adapter}")
    # Trusted checkpoint created by this script; never load arbitrary pickle files.
    saved = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    return saved["state"]


def run_training(
    config_path: Path,
    run_dir: Path,
    *,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    max_steps: int | None = None,
    resume: bool = False,
) -> None:
    import torch
    import torch.nn.functional as functional
    from transformers import get_linear_schedule_with_warmup

    config = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kd = raw["distillation"]
    training = config.training
    if training is None or config.dataset.split != "val":
        raise ValueError("Training configuration must select on validation")
    if not 0 <= kd["alpha"] <= 1 or kd["temperature"] <= 0:
        raise ValueError("Invalid distillation alpha or temperature")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.output.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    samples, manifest = load_targets(Path(kd["teacher_data_dir"]), train_limit)
    if manifest["config"]["temperature"] != kd["temperature"]:
        raise ValueError("Temperature differs from the fixed teacher export")
    if manifest["config"]["student"]["id"] != config.model.id:
        raise ValueError("Student model differs from exported targets")
    if manifest["revisions"]["student"] != config.model.revision:
        raise ValueError("Student revision differs from exported targets")
    if manifest["config"]["student_system"] != config.prompt.system:
        raise ValueError("Student prompt differs from exported targets")
    dataset_path = download_dataset(config.dataset.source_url, config.dataset.cache_path)
    if digest(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("CLINC dataset differs from teacher export")
    all_train = load_examples(dataset_path, "train", include_oos=True)
    by_id = {e.id: e for e in all_train}
    if any(by_id.get(s["example"].id) != s["example"] for s in samples):
        raise ValueError("Teacher sample text/gold label differs from official train")
    validation = stratified_limit(load_examples(dataset_path, "val", include_oos=True), eval_limit)
    labels = labels_from_examples(all_train)
    planned_steps = optimizer_step_count(
        len(samples), training.epochs, training.gradient_accumulation_steps
    )
    identity = {
        "config": raw,
        "teacher_manifest_sha256": digest(Path(kd["teacher_data_dir"]) / "manifest.json"),
        "sample_ids_sha256": json_hash([s["example"].id for s in samples]),
        "training_samples": len(samples),
        "validation_samples": len(validation),
        "planned_optimizer_steps": planned_steps,
        "trainer_sha256": digest(Path(__file__)),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    if resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Configuration/data/source changed; cannot resume this run")
    elif metadata_path.exists():
        raise FileExistsError("Run already exists; use --resume or a new --run-dir")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    classifier = GemmaClassifier(config.model, config.generation)
    classifier.load()
    if json_hash(classifier.processor.tokenizer.get_vocab()) != manifest["vocab_sha256"]:
        raise ValueError("Loaded student vocabulary differs from teacher targets")
    model = _prepare_qlora_model(classifier, training)
    classifier.model = model
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(planned_steps * training.warmup_ratio), planned_steps
    )
    if not resume:
        write_json(
            metadata_path,
            {
                "identity": identity,
                "started_at": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "max_student_tokens": max(len(s["student_input_ids"]) for s in samples),
            },
        )
    state = {
        "epoch": 1,
        "position": 0,
        "global_step": 0,
        "examples_seen": 0,
        "train_seconds": 0.0,
        "validation_seconds": 0.0,
        "best_macro_f1": -1.0,
        "best_adapter": None,
        "epoch_ce_sum": 0.0,
        "epoch_kd_sum": 0.0,
        "epoch_loss_sum": 0.0,
        "epoch_samples": 0,
    }
    if resume:
        state = restore_checkpoint(run_dir, model, optimizer, scheduler)
        if max_steps is not None and max_steps <= state["global_step"]:
            raise ValueError("max_steps is already reached by the restored checkpoint")
        print(
            f"Resumed optimizer step {state['global_step']}, "
            f"epoch {state['epoch']}, position {state['position']}",
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    history = run_dir / "training_history.jsonl"
    alpha, temperature = kd["alpha"], kd["temperature"]
    previous_checkpoint = None
    while state["epoch"] <= training.epochs:
        epoch = state["epoch"]
        order = list(range(len(samples)))
        random.Random(config.seed + epoch).shuffle(order)
        # Exercise the longest sequence in the first accumulation group of pilots.
        if train_limit is not None:
            longest = max(order, key=lambda i: len(samples[i]["student_input_ids"]))
            order.remove(longest)
            order.insert(0, longest)
        model.train()
        progress = tqdm(
            range(state["position"], len(order)),
            desc=f"KD epoch {epoch}",
            unit="example",
            mininterval=5,
        )
        group_metrics = []
        group_started = time.perf_counter()
        for position in progress:
            sample = samples[order[position]]
            device = model.device
            ids = torch.tensor(sample["student_input_ids"].astype(np.int64), device=device)[None]
            positions = torch.tensor(
                sample["student_target_positions"].astype(np.int64) - 1, device=device
            )
            targets = torch.tensor(sample["target_ids"].astype(np.int64), device=device)
            top_ids = torch.tensor(sample["topk_ids"].astype(np.int64), device=device)
            top_logp = torch.tensor(sample["topk_log_probs"], device=device)
            tail_logp = torch.tensor(sample["tail_log_prob"], device=device)
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                logits_to_keep=positions,
                use_cache=False,
            )
            logits = output.logits[0]
            ce = functional.cross_entropy(logits.float(), targets)
            kl = grouped_kd_loss(logits, top_ids, top_logp, tail_logp, temperature)
            loss = (1 - alpha) * ce + alpha * temperature**2 * kl
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss on {sample['example'].id}")
            group_start = (
                position // training.gradient_accumulation_steps
            ) * training.gradient_accumulation_steps
            group_size = min(training.gradient_accumulation_steps, len(order) - group_start)
            (loss / group_size).backward()
            numbers = (float(ce.detach()), float(kl.detach()), float(loss.detach()))
            group_metrics.append(numbers)
            state["epoch_ce_sum"] += numbers[0]
            state["epoch_kd_sum"] += numbers[1]
            state["epoch_loss_sum"] += numbers[2]
            state["epoch_samples"] += 1
            state["examples_seen"] += 1
            state["position"] = position + 1
            del output, logits, ce, kl, loss, ids, positions, targets, top_ids, top_logp, tail_logp
            if state["position"] % training.gradient_accumulation_steps and state[
                "position"
            ] != len(order):
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, training.max_grad_norm, error_if_nonfinite=True
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["train_seconds"] += time.perf_counter() - group_started
            group_started = time.perf_counter()
            speed = state["examples_seen"] / max(state["train_seconds"], 1e-6)
            remaining = len(samples) * training.epochs - state["examples_seen"]
            metrics = np.mean(group_metrics, axis=0).tolist()
            record = {
                "event": "optimizer_step",
                "epoch": epoch,
                "global_step": state["global_step"],
                "position": state["position"],
                "examples_seen": state["examples_seen"],
                "ce": metrics[0],
                "kd": metrics[1],
                "loss": metrics[2],
                "grad_norm": float(grad_norm),
                "learning_rate": learning_rate,
                "train_seconds": state["train_seconds"],
                "examples_per_second": speed,
                "estimated_remaining_training_hours": remaining / speed / 3600,
                "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                "updated_at": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "state": "training",
            }
            write_json(run_dir / "status.json", record)
            if state["global_step"] % training.log_every == 0 or state["global_step"] <= 3:
                _append_jsonl(history, record)
                print(json.dumps(record), flush=True)
            group_metrics.clear()
            progress.set_postfix(loss=f"{metrics[2]:.4f}")
            if state["global_step"] % config.output.checkpoint_every == 0:
                previous_checkpoint = save_checkpoint(
                    run_dir, model, optimizer, scheduler, dict(state)
                )
            if max_steps is not None and state["global_step"] >= max_steps:
                if previous_checkpoint is None or not previous_checkpoint.name.startswith(
                    f"checkpoint_step_{state['global_step']:06d}_"
                ):
                    save_checkpoint(run_dir, model, optimizer, scheduler, dict(state))
                write_json(
                    run_dir / "pilot_summary.json",
                    {
                        **record,
                        "state": "stopped_at_max_steps",
                        "adapter_weights_changed": any(
                            torch.count_nonzero(p.detach()).item() > 0
                            for name, p in model.named_parameters()
                            if p.requires_grad and "lora_B" in name
                        ),
                    },
                )
                return
        adapter = run_dir / f"adapter_epoch_{epoch:03d}"
        model.save_pretrained(adapter)
        validation_started = time.perf_counter()
        write_json(
            run_dir / "status.json",
            {
                "state": "validation",
                "epoch": epoch,
                "global_step": state["global_step"],
                "pid": os.getpid(),
            },
        )
        prediction_path = run_dir / f"validation_epoch_{epoch:03d}.jsonl"
        temporary_predictions = prediction_path.with_suffix(".jsonl.tmp")
        temporary_predictions.unlink(missing_ok=True)
        metrics = _evaluate(
            classifier,
            validation,
            labels,
            config.prompt.system,
            temporary_predictions,
        )
        temporary_predictions.replace(prediction_path)
        state["validation_seconds"] += time.perf_counter() - validation_started
        write_json(run_dir / f"validation_metrics_epoch_{epoch:03d}.json", metrics)
        if metrics["macro_f1"] > state["best_macro_f1"]:
            state["best_macro_f1"] = metrics["macro_f1"]
            state["best_adapter"] = str(adapter.resolve())
            write_json(
                run_dir / "best_adapter.json",
                {"path": str(adapter.resolve()), "macro_f1": metrics["macro_f1"], "epoch": epoch},
            )
        _append_jsonl(history, {"event": "epoch_end", "epoch": epoch, "validation": metrics})
        print(f"Epoch {epoch} validation: {json.dumps(metrics)}", flush=True)
        state.update(
            {
                "epoch": epoch + 1,
                "position": 0,
                "epoch_samples": 0,
                "epoch_ce_sum": 0.0,
                "epoch_kd_sum": 0.0,
                "epoch_loss_sum": 0.0,
            }
        )
        previous_checkpoint = save_checkpoint(run_dir, model, optimizer, scheduler, dict(state))
    write_json(
        run_dir / "training_summary.json",
        {
            **state,
            "state": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        },
    )
    write_json(
        run_dir / "status.json",
        {
            "state": "completed",
            "global_step": state["global_step"],
            "best_adapter": state["best_adapter"],
            "pid": os.getpid(),
        },
    )
    print(f"Training complete. Best adapter: {state['best_adapter']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/distillation_qlora.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Keep the Windows machine awake while the detached training process lives.
    # This is thread-scoped and does not change persistent power settings.
    if os.name == "nt":
        import ctypes

        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise ctypes.WinError()
    try:
        run_training(
            args.config,
            args.run_dir,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            max_steps=args.max_steps,
            resume=args.resume,
        )
    except Exception as exc:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "state": "failed",
            "error": str(exc),
            "type": type(exc).__name__,
            "pid": os.getpid(),
            "failed_at": datetime.now(UTC).isoformat(),
        }
        write_json(args.run_dir / "failure.json", failure)
        write_json(args.run_dir / "status.json", failure)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
