"""Download and load the official CLINC150 splits."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OOS_LABEL = "out_of_scope"
SOURCE_OOS_LABEL = "oos"
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    split: str


def download_dataset(source_url: str, cache_path: Path) -> Path:
    """Download CLINC150 once and return its local path."""
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source_url, headers={"User-Agent": "gemma-clinc/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
    cache_path.write_bytes(payload)
    return cache_path


def load_examples(path: Path, split: str, *, include_oos: bool = True) -> list[Example]:
    """Load an official split without altering its order."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    rows = list(raw[split])
    if include_oos:
        rows.extend(raw[f"oos_{split}"])

    return [
        Example(
            id=f"{split}-{index:05d}",
            text=text,
            label=OOS_LABEL if label == SOURCE_OOS_LABEL else label,
            split=split,
        )
        for index, (text, label) in enumerate(rows)
    ]


def labels_from_examples(examples: list[Example]) -> list[str]:
    """Return stable alphabetically ordered labels, with OOS last."""
    labels = sorted({example.label for example in examples if example.label != OOS_LABEL})
    if any(example.label == OOS_LABEL for example in examples):
        labels.append(OOS_LABEL)
    return labels
