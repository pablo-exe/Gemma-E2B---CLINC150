"""Resumable sparse teacher distributions, aligned with gold student responses.

No teacher generation or test data is used. Each NPZ is an atomic recovery unit.
Sparse probabilities are normalized over the full vocabulary at a fixed T.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from gemma_clinc.data import download_dataset, labels_from_examples, load_examples
from gemma_clinc.prompting import build_messages, build_sft_messages, response_token_span


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_config(config: dict) -> None:
    if config["dataset"]["split"] != "train":
        raise ValueError("Only the official train split may produce distillation targets")
    if config["teacher"]["quantization"] not in {"8bit", "4bit", "none"}:
        raise ValueError("Unsupported quantization")
    if config["teacher"]["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported dtype")
    if not math.isfinite(float(config["temperature"])) or config["temperature"] <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(config["top_k"], int) or config["top_k"] < 1:
        raise ValueError("top_k must be a positive integer")
    if config["max_length"] < 1:
        raise ValueError("max_length must be positive")


def assert_tokenizers_match(teacher: Any, student: Any) -> str:
    if teacher.get_vocab() != student.get_vocab():
        raise ValueError("Teacher/student token-to-ID vocabularies differ")
    if teacher.all_special_tokens != student.all_special_tokens:
        raise ValueError("Teacher/student special tokens differ")
    return json_hash(teacher.get_vocab())


def encode_gold(tokenizer: Any, messages: list[dict], gold: str) -> tuple[list[int], int]:
    kwargs = {"tokenize": False, "enable_thinking": False}
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    rendered = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": gold}],
        add_generation_prompt=False,
        **kwargs,
    )
    # Gemma's generation prefix may include an empty thought channel, while its
    # rendered gold conversation omits that channel. Preserve the actual inference
    # prefix and append the gold answer plus the template's turn terminator.
    before, answer, terminator = rendered.rpartition(gold)
    if not answer or not before or not terminator:
        raise ValueError("Could not locate gold response and turn terminator in chat template")
    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt_text + answer + terminator, add_special_tokens=False)["input_ids"]
    start, _ = response_token_span(prompt, full)
    return full, start


def sparse_distribution(logits: Any, top_k: int, temperature: float) -> dict[str, np.ndarray]:
    import torch

    scaled = logits.float() / temperature
    if not torch.isfinite(scaled).all():
        raise ValueError("Non-finite teacher logits")
    if top_k >= scaled.shape[-1]:
        raise ValueError("top_k must be smaller than the model vocabulary")
    log_probs = torch.log_softmax(scaled, dim=-1)
    values, indices = torch.topk(log_probs, top_k, dim=-1)
    # Compute the tail directly rather than subtracting nearly one in float32.
    tail = log_probs.clone()
    tail.scatter_(-1, indices, -torch.inf)
    tail_log_probs = torch.logsumexp(tail, dim=-1)
    return {
        "topk_ids": indices.cpu().numpy().astype(np.int32),
        "topk_log_probs": values.cpu().numpy().astype(np.float32),
        "tail_log_prob": tail_log_probs.cpu().numpy().astype(np.float32),
    }


def save_example(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays, metadata=np.array(json.dumps(metadata)))
    temporary.replace(path)


def verify_example(path: Path, expected: dict | None = None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if expected is not None and metadata != expected:
            raise ValueError(f"Example metadata mismatch: {path}")
        targets = data["target_ids"]
        ids, logp, tail = data["topk_ids"], data["topk_log_probs"], data["tail_log_prob"]
        if ids.shape != logp.shape or ids.shape[0] != len(targets) or tail.shape != targets.shape:
            raise ValueError(f"Invalid target shapes: {path}")
        if not np.isfinite(logp).all() or not np.isfinite(tail).all():
            raise ValueError(f"Invalid probabilities: {path}")
        mass = np.exp(logp.astype(np.float64)).sum(-1) + np.exp(tail.astype(np.float64))
        if not np.allclose(mass, 1, atol=2e-5):
            raise ValueError(f"Probability mass mismatch: {path}")
        for prefix in ("teacher", "student"):
            tokens, positions = data[f"{prefix}_input_ids"], data[f"{prefix}_target_positions"]
            if not np.array_equal(tokens[positions], targets):
                raise ValueError(f"Target alignment mismatch: {path}")
    return metadata


def verify_run(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] or summary["completed"] != manifest["expected_examples"]:
        raise ValueError("Run is incomplete")
    records = [
        json.loads(line)
        for line in (directory / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != manifest["expected_examples"] or len({r["id"] for r in records}) != len(
        records
    ):
        raise ValueError("Invalid example count or duplicate IDs")
    for record in tqdm(records, desc="Verify"):
        path = directory / "samples" / f"{record['id']}.npz"
        if digest(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        metadata = verify_example(path)
        if metadata["example"]["id"] != record["id"]:
            raise ValueError("Index/metadata mismatch")
    print(f"Verified {len(records)} examples in {directory}")


def run(
    config_path: Path, limit: int | None, prepare_only: bool, *, verify_after_run: bool = True
) -> None:
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    directory = Path(config["output_dir"])
    if limit is not None:
        directory = directory.with_name(directory.name + f"_pilot{limit}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old and old["config"] != config:
        raise ValueError("Configuration changed; use a new output directory")
    revisions = (
        old["revisions"]
        if old
        else {
            key: HfApi().model_info(config[key]["id"], revision=config[key]["revision"]).sha
            for key in ("teacher", "student")
        }
    )
    dataset_path = download_dataset(
        config["dataset"]["source_url"], Path(config["dataset"]["cache_path"])
    )
    all_examples = load_examples(dataset_path, "train", include_oos=True)
    labels = labels_from_examples(all_examples)
    if limit is None:
        examples = all_examples
    else:
        from gemma_clinc.training import stratified_limit

        examples = stratified_limit(all_examples, limit)
    tokenizers = {
        key: AutoTokenizer.from_pretrained(config[key]["id"], revision=revisions[key])
        for key in ("teacher", "student")
    }
    vocab_hash = assert_tokenizers_match(tokenizers["teacher"], tokenizers["student"])
    prepared = []
    for example in examples:
        teacher_messages = build_messages(example.text, labels, config["teacher_system"])
        student_messages = build_sft_messages(
            example, config["student_system"], include_answer=False
        )
        teacher_ids, teacher_start = encode_gold(
            tokenizers["teacher"], teacher_messages, example.label
        )
        student_ids, student_start = encode_gold(
            tokenizers["student"], student_messages, example.label
        )
        if teacher_ids[teacher_start:] != student_ids[student_start:]:
            raise ValueError(f"Response tokenization differs: {example.id}")
        if max(len(teacher_ids), len(student_ids)) > config["max_length"]:
            raise ValueError(f"Example too long (no silent truncation): {example.id}")
        arrays = {
            "teacher_input_ids": np.array(teacher_ids, dtype=np.int32),
            "student_input_ids": np.array(student_ids, dtype=np.int32),
            "teacher_target_positions": np.arange(teacher_start, len(teacher_ids), dtype=np.int32),
            "student_target_positions": np.arange(student_start, len(student_ids), dtype=np.int32),
            "target_ids": np.array(teacher_ids[teacher_start:], dtype=np.int32),
        }
        prepared.append((example, arrays))
    identity = {
        "format_version": 1,
        "config": config,
        "revisions": revisions,
        "dataset_sha256": digest(dataset_path),
        "vocab_sha256": vocab_hash,
        "expected_examples": len(examples),
        "example_ids_sha256": json_hash([e.id for e in examples]),
        "max_teacher_tokens": max(len(a["teacher_input_ids"]) for _, a in prepared),
        "target_source": "gold",
        "probability_dtype": "float32",
        "enable_thinking": False,
    }
    if old and any(old.get(key) != value for key, value in identity.items()):
        raise ValueError("Dataset, tokenizer or run identity changed; cannot resume")
    for key, tokenizer in tokenizers.items():
        tokenizer.save_pretrained(directory / f"{key}_tokenizer")
    if not old:
        write_json(manifest_path, identity)
    print(f"Prepared {len(examples)} examples; max teacher length {identity['max_teacher_tokens']}")
    if prepare_only:
        return

    import torch
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use --prepare-only for CPU preflight")
    samples = directory / "samples"
    samples.mkdir(exist_ok=True)
    pending = []
    run_hash = json_hash(identity)
    for example, arrays in prepared:
        metadata = {"example": asdict(example), "run_hash": run_hash}
        path = samples / f"{example.id}.npz"
        if path.exists():
            verify_example(path, metadata)
        else:
            pending.append((example, arrays, metadata))
    model = None
    started = time.monotonic()
    if pending:
        write_json(
            directory / "summary.json",
            {
                "complete": False,
                "completed": len(prepared) - len(pending),
            },
        )
        quantization = config["teacher"]["quantization"]
        kwargs = {}
        if quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, config["teacher"]["dtype"]),
            )
        model = AutoModelForMultimodalLM.from_pretrained(
            config["teacher"]["id"],
            revision=revisions["teacher"],
            device_map={"": 0},
            dtype=getattr(torch, config["teacher"]["dtype"]),
            attn_implementation="sdpa",
            **kwargs,
        ).eval()
        model.requires_grad_(False)
        runtime = {
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "accelerate",
                    "bitsandbytes",
                    "numpy",
                    "huggingface-hub",
                )
            },
        }
        try:
            runtime["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            runtime["git_commit"] = None
        runtime["extractor_sha256"] = digest(Path(__file__))
        identity["runtime"] = runtime
        if old and old.get("runtime") is not None and old["runtime"] != runtime:
            raise ValueError("Runtime changed; resume with original environment or a new directory")
        write_json(manifest_path, identity)
        torch.cuda.reset_peak_memory_stats()
    for example, arrays, metadata in tqdm(pending, desc="Teacher", unit="example"):
        input_ids = torch.tensor(arrays["teacher_input_ids"].astype(np.int64), device="cuda")[None]
        positions = torch.tensor(
            arrays["teacher_target_positions"].astype(np.int64) - 1, device="cuda"
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=positions,
            )
            logits = output.logits[0]
            if logits.shape[0] != len(arrays["target_ids"]):
                raise RuntimeError("Model did not honor logits_to_keep")
            sparse = sparse_distribution(logits, config["top_k"], config["temperature"])
        save_example(samples / f"{example.id}.npz", {**arrays, **sparse}, metadata)
        del output, logits, input_ids, positions
    index = []
    for example, _ in prepared:
        path = samples / f"{example.id}.npz"
        verify_example(path, {"example": asdict(example), "run_hash": run_hash})
        index.append({**asdict(example), "sha256": digest(path)})
    temporary = directory / "examples.jsonl.tmp"
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index), encoding="utf-8"
    )
    temporary.replace(directory / "examples.jsonl")
    write_json(
        directory / "summary.json",
        {
            "complete": True,
            "completed": len(index),
            "generated_this_session": len(pending),
            "session_seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if model else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if model else None,
        },
    )
    if verify_after_run:
        verify_run(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/teacher_logits.yaml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Download data/tokenizers, no weights"
    )
    parser.add_argument("--verify", type=Path, help="Verify downloaded run; no GPU/network needed")
    args = parser.parse_args()
    if args.verify:
        verify_run(args.verify)
    else:
        run(args.config, args.limit, args.prepare_only)
