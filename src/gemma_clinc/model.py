"""Lazy-loading Gemma inference backend."""

from __future__ import annotations

from typing import Any

from gemma_clinc.config import GenerationConfig, ModelConfig


class GemmaClassifier:
    """Text-only classifier backed by the instruction-tuned Gemma model."""

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig) -> None:
        self.model_config = model_config
        self.generation_config = generation_config
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load processor and model only when an experiment is executed."""
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

        dtype = getattr(torch, self.model_config.dtype)
        model_kwargs: dict[str, Any] = {
            "device_map": self.model_config.device_map,
            "dtype": dtype,
        }
        if self.model_config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.model_config.quantization != "none":
            raise ValueError("quantization must be '4bit' or 'none'")

        self.processor = AutoProcessor.from_pretrained(self.model_config.id)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.model_config.id, **model_kwargs)
        self.model.eval()

    def predict(self, messages: list[dict[str, str]]) -> str:
        """Generate a label for one conversation."""
        if self.model is None or self.processor is None:
            raise RuntimeError("load() must be called before predict()")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_config.max_new_tokens,
            do_sample=self.generation_config.do_sample,
        )
        return self.processor.decode(outputs[0][input_length:], skip_special_tokens=True)
