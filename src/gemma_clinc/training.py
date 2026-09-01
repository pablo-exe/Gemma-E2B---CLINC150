"""Phase 2 QLoRA supervised fine-tuning orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from gemma_clinc.config import ExperimentConfig, TrainingConfig
from gemma_clinc.data import Example, download_dataset, labels_from_examples, load_examples
from gemma_clinc.metrics import compute_metrics
from gemma_clinc.model import GemmaClassifier
from gemma_clinc.prompting import build_sft_messages, parse_prediction, response_token_span


def optimizer_step_count(n_examples: int, epochs: int, accumulation_steps: int) -> int:
    """Return the number of optimizer updates for micro-batch size one."""
    if n_examples < 1 or epochs < 1 or accumulation_steps < 1:
        raise ValueError("examples, epochs and accumulation steps must be positive")
    return math.ceil(n_examples / accumulation_steps) * epochs


def select_lora_target_names(
    named_modules: Iterable[tuple[str, Any]],
    linear_type: type,
    requested_suffixes: tuple[str, ...],
) -> list[str]:
    """Select quantized language-layer projections without touching multimodal towers."""
    names = [
        name
        for name, module in named_modules
        if isinstance(module, linear_type)
        and ".language_model.layers." in name
        and name.endswith(requested_suffixes)
    ]
    if not names:
        raise ValueError(f"no quantized LoRA targets matched {requested_suffixes!r}")
    return names


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _limited(examples: list[Example], limit: int | None, name: str) -> list[Example]:
    if limit is None:
        return examples
    if limit < 1:
        raise ValueError(f"{name} must be a positive integer")
    return examples[:limit]


def _prepare_qlora_model(classifier: GemmaClassifier, config: TrainingConfig) -> Any:
    from bitsandbytes.nn import Linear4bit
    from peft import LoraConfig, get_peft_model

    model = classifier.model
    targets = select_lora_target_names(
        model.named_modules(), Linear4bit, config.target_modules
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
            revision=classifier.model_config.revision,
        ),
    )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA injection produced no trainable parameters")
    return model


def _encode_training_example(
    processor: Any, example: Example, system_prompt: str, device: Any
) -> tuple[dict[str, Any], Any, Any]:
    import torch

    prompt_messages = build_sft_messages(example, system_prompt, include_answer=False)
    full_messages = build_sft_messages(example, system_prompt, include_answer=True)
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    batch = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = prompt["input_ids"][0].tolist()
    full_ids = batch["input_ids"][0].tolist()
    target_start, target_end = response_token_span(prompt_ids, full_ids)
    batch = batch.to(device)
    targets = batch["input_ids"][:, target_start:target_end]
    logit_indices = torch.arange(target_start - 1, target_end - 1, device=device)
    return batch, targets, logit_indices


def _evaluate(
    classifier: GemmaClassifier,
    examples: list[Example],
    labels: list[str],
    system_prompt: str,
    predictions_path: Path,
) -> dict[str, Any]:
    model = classifier.model
    model.eval()
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    y_true: list[str] = []
    y_pred: list[str | None] = []
    try:
        for example in tqdm(examples, desc="Validation", unit="example"):
            messages = build_sft_messages(example, system_prompt, include_answer=False)
            raw_output = classifier.predict(messages)
            prediction = parse_prediction(raw_output, labels)
            _append_jsonl(
                predictions_path,
                {
                    "id": example.id,
                    "text": example.text,
                    "expected": example.label,
                    "predicted": prediction,
                    "raw_output": raw_output,
                    "correct": prediction == example.label,
                },
            )
            y_true.append(example.label)
            y_pred.append(prediction)
    finally:
        model.config.use_cache = previous_cache
        model.train()
    return compute_metrics(y_true, y_pred)


def run_sft(
    config: ExperimentConfig,
    *,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    max_steps: int | None = None,
) -> Path:
    """Train the Phase 2 adapter and return its artifact directory."""
    import torch
    import torch.nn.functional as functional
    from transformers import get_linear_schedule_with_warmup

    if config.training is None:
        raise ValueError("Phase 2 requires a training section in the configuration")
    if config.dataset.split != "val":
        raise ValueError("Phase 2 model selection must use the validation split")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    training = config.training
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    dataset_path = download_dataset(config.dataset.source_url, config.dataset.cache_path)
    all_train_examples = load_examples(
        dataset_path, "train", include_oos=config.dataset.include_oos
    )
    train_examples = _limited(all_train_examples, train_limit, "train_limit")
    validation_examples = _limited(
        load_examples(dataset_path, "val", include_oos=config.dataset.include_oos),
        eval_limit,
        "eval_limit",
    )
    labels = labels_from_examples(all_train_examples)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output.root_dir / f"{config.name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "training_history.jsonl"
    _write_json(run_dir / "label_vocabulary.json", labels)
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "train_limit": train_limit,
            "eval_limit": eval_limit,
            "max_steps": max_steps,
            "dataset_sha256": _sha256(dataset_path),
            "config": asdict(config),
        },
    )

    classifier = GemmaClassifier(config.model, config.generation)
    classifier.load()
    model = _prepare_qlora_model(classifier, training)
    classifier.model = model
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    planned_steps = optimizer_step_count(
        len(train_examples), training.epochs, training.gradient_accumulation_steps
    )
    total_steps = min(planned_steps, max_steps) if max_steps is not None else planned_steps
    warmup_steps = round(total_steps * training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    global_step = 0
    best_macro_f1 = -1.0
    best_adapter: str | None = None
    stop_training = False
    started = time.perf_counter()

    for epoch in range(1, training.epochs + 1):
        order = list(range(len(train_examples)))
        random.Random(config.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        epoch_examples = 0
        model.train()
        progress = tqdm(order, desc=f"Train epoch {epoch}", unit="example")
        for position, index in enumerate(progress, start=1):
            batch, targets, logit_indices = _encode_training_example(
                classifier.processor,
                train_examples[index],
                config.prompt.system,
                model.device,
            )
            output = model(**batch, logits_to_keep=logit_indices, use_cache=False)
            loss = functional.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
            group_start = ((position - 1) // training.gradient_accumulation_steps) * (
                training.gradient_accumulation_steps
            )
            group_size = min(
                training.gradient_accumulation_steps,
                len(order) - group_start,
            )
            (loss / group_size).backward()
            epoch_loss += float(loss.detach())
            epoch_examples += 1

            should_update = (
                position % training.gradient_accumulation_steps == 0
                or position == len(order)
            )
            if should_update:
                learning_rate_used = optimizer.param_groups[0]["lr"]
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % training.log_every == 0 or global_step == 1:
                    record = {
                        "event": "optimizer_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "mean_epoch_loss": epoch_loss / epoch_examples,
                        "learning_rate_used": learning_rate_used,
                        "next_learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": time.perf_counter() - started,
                        "cuda_peak_allocated_mib": round(
                            torch.cuda.max_memory_allocated() / 2**20, 1
                        ),
                        "cuda_peak_reserved_mib": round(
                            torch.cuda.max_memory_reserved() / 2**20, 1
                        ),
                    }
                    _append_jsonl(history_path, record)
                    progress.set_postfix(loss=f"{record['mean_epoch_loss']:.4f}")
                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break

        adapter_dir = run_dir / f"adapter_epoch_{epoch:03d}_step_{global_step:06d}"
        model.save_pretrained(adapter_dir)
        epoch_record: dict[str, Any] = {
            "event": "epoch_end",
            "epoch": epoch,
            "global_step": global_step,
            "mean_loss": epoch_loss / max(epoch_examples, 1),
            "adapter": str(adapter_dir),
        }
        should_evaluate = epoch % training.eval_every_epochs == 0 or stop_training
        if should_evaluate:
            predictions_path = run_dir / f"validation_epoch_{epoch:03d}.jsonl"
            validation_metrics = _evaluate(
                classifier,
                validation_examples,
                labels,
                config.prompt.system,
                predictions_path,
            )
            _write_json(run_dir / f"validation_metrics_epoch_{epoch:03d}.json", validation_metrics)
            epoch_record["validation"] = validation_metrics
            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation_metrics["macro_f1"]
                best_adapter = str(adapter_dir)
        _append_jsonl(history_path, epoch_record)
        if stop_training:
            break

    summary = {
        "completed_at": datetime.now(UTC).isoformat(),
        "optimizer_steps": global_step,
        "trainable_parameters": trainable_parameters,
        "best_macro_f1": best_macro_f1,
        "best_adapter": best_adapter,
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(run_dir / "training_summary.json", summary)
    return run_dir
