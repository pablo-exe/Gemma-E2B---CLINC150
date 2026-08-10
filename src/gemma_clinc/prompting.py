"""Prompt construction and strict prediction parsing."""

from __future__ import annotations

import re


def build_messages(text: str, labels: list[str], system_prompt: str) -> list[dict[str, str]]:
    """Build a deterministic zero-shot classification conversation."""
    label_block = "\n".join(f"- {label}" for label in labels)
    user_prompt = f"Allowed labels:\n{label_block}\n\nUser query:\n{text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_prediction(raw_output: str, labels: list[str]) -> str | None:
    """Extract one known label while rejecting ambiguous or malformed output."""
    normalized = raw_output.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.strip("`'\".,:;!()[]{}")
    label_set = set(labels)
    if normalized in label_set:
        return normalized

    labelled = re.search(r"(?:intent|label)_*[:=]_*([a-z0-9_]+)", normalized)
    if labelled and labelled.group(1) in label_set:
        return labelled.group(1)

    found = {
        label
        for label in labels
        if re.search(rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])", normalized)
    }
    return next(iter(found)) if len(found) == 1 else None
