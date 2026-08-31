"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    source_url: str
    cache_path: Path
    split: str = "test"
    include_oos: bool = True


@dataclass(frozen=True)
class ModelConfig:
    id: str
    quantization: str = "4bit"
    dtype: str = "float16"
    device_map: str = "auto"


@dataclass(frozen=True)
class PromptConfig:
    system: str
    include_label_descriptions: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 16
    do_sample: bool = False


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    checkpoint_every: int = 25


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    prompt: PromptConfig
    generation: GenerationConfig
    output: OutputConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    experiment = raw["experiment"]
    return ExperimentConfig(
        name=experiment["name"],
        seed=int(experiment["seed"]),
        dataset=DatasetConfig(
            source_url=raw["dataset"]["source_url"],
            cache_path=Path(raw["dataset"]["cache_path"]),
            split=raw["dataset"].get("split", "test"),
            include_oos=bool(raw["dataset"].get("include_oos", True)),
        ),
        model=ModelConfig(**raw["model"]),
        prompt=PromptConfig(**raw["prompt"]),
        generation=GenerationConfig(**raw["generation"]),
        output=OutputConfig(
            root_dir=Path(raw["output"]["root_dir"]),
            checkpoint_every=int(raw["output"].get("checkpoint_every", 25)),
        ),
    )
