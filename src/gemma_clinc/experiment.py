"""Phase 1 zero-shot experiment orchestration."""

from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from gemma_clinc.config import ExperimentConfig
from gemma_clinc.data import download_dataset, labels_from_examples, load_examples
from gemma_clinc.metrics import compute_metrics
from gemma_clinc.model import GemmaClassifier
from gemma_clinc.prompting import build_messages, parse_prediction


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_zero_shot(config: ExperimentConfig, *, limit: int | None = None) -> Path:
    """Run the zero-shot baseline and return its artifact directory."""
    random.seed(config.seed)
    dataset_path = download_dataset(config.dataset.source_url, config.dataset.cache_path)
    examples = load_examples(
        dataset_path, config.dataset.split, include_oos=config.dataset.include_oos
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        examples = examples[:limit]

    labels = labels_from_examples(
        load_examples(dataset_path, "train", include_oos=config.dataset.include_oos)
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output.root_dir / f"{config.name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"

    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "limit": limit,
            "config": asdict(config),
        },
    )

    classifier = GemmaClassifier(config.model, config.generation)
    classifier.load()
    y_true: list[str] = []
    y_pred: list[str | None] = []

    for example in tqdm(examples, desc="Zero-shot evaluation", unit="example"):
        messages = build_messages(example.text, labels, config.prompt.system)
        raw_output = classifier.predict(messages)
        prediction = parse_prediction(raw_output, labels)
        record = {
            "id": example.id,
            "text": example.text,
            "expected": example.label,
            "predicted": prediction,
            "raw_output": raw_output,
            "correct": prediction == example.label,
        }
        _append_jsonl(predictions_path, record)
        y_true.append(example.label)
        y_pred.append(prediction)

        if len(y_true) % config.output.checkpoint_every == 0:
            _write_json(run_dir / "metrics.partial.json", compute_metrics(y_true, y_pred))

    metrics = compute_metrics(y_true, y_pred)
    _write_json(run_dir / "metrics.json", metrics)
    partial_metrics = run_dir / "metrics.partial.json"
    if partial_metrics.exists():
        partial_metrics.unlink()
    return run_dir
