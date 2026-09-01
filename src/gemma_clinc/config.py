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
    revision: str | None = None
    quantization: str = "4bit"
    dtype: str = "float16"
    device_map: str | dict[str, str | int] = "auto"
    offload_per_layer_embeddings: bool = False
    text_only: bool = False


@dataclass(frozen=True)
class PromptConfig:
    system: str
    include_label_descriptions: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 16
    do_sample: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    log_every: int = 25
    eval_every_epochs: int = 1

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps must be at least 1")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not self.target_modules:
            raise ValueError("training.target_modules cannot be empty")
        if self.eval_every_epochs < 1:
            raise ValueError("training.eval_every_epochs must be at least 1")


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
    training: TrainingConfig | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    experiment = raw["experiment"]
    training_raw = raw.get("training")
    training = None
    if training_raw is not None:
        training_values = dict(training_raw)
        if "target_modules" in training_values:
            training_values["target_modules"] = tuple(training_values["target_modules"])
        training = TrainingConfig(**training_values)

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
        training=training,
    )
