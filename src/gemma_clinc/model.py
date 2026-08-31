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
        device_map = self._device_map()
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if self.model_config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=(
                    self.model_config.offload_per_layer_embeddings
                ),
            )
        elif self.model_config.quantization != "none":
            raise ValueError("quantization must be '4bit' or 'none'")

        self.processor = AutoProcessor.from_pretrained(self.model_config.id)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.model_config.id, **model_kwargs)
        if self.model_config.offload_per_layer_embeddings:
            self._replace_offloaded_embeddings()
        if self.model_config.text_only:
            self._remove_multimodal_towers()
        self.model.eval()

    def _device_map(self) -> str | dict[str, str | int]:
        """Place Gemma 4's large per-layer embedding table directly on CPU."""
        if not self.model_config.offload_per_layer_embeddings:
            return self.model_config.device_map
        if self.model_config.quantization != "4bit":
            raise ValueError("per-layer embedding offload currently requires 4-bit loading")

        if isinstance(self.model_config.device_map, dict):
            device_map = dict(self.model_config.device_map)
        elif self.model_config.device_map == "auto":
            device_map = {"": "cuda:0"}
        else:
            raise ValueError("embedding offload requires device_map='auto' or a device-map dict")
        device_map["model.language_model.embed_tokens_per_layer"] = "cpu"
        return device_map

    def _replace_offloaded_embeddings(self) -> None:
        """Keep PLE lookup on CPU instead of Accelerate moving the full table per call.

        Accelerate represents an offloaded parameter as a meta tensor and stores its
        actual CPU value in the module hook. Calling the stock module would stage the
        entire 4.4 GiB table on CUDA. This replacement performs only the requested
        rows on CPU and transfers the small lookup result to the model device.
        """
        import torch.nn as nn
        import torch.nn.functional as functional

        embedding = self.model.model.language_model.embed_tokens_per_layer
        hook = getattr(embedding, "_hf_hook", None)
        hooks = getattr(hook, "hooks", (hook,))
        weights_map = next(
            (
                item.weights_map
                for item in hooks
                if getattr(item, "offload", False)
                and getattr(item, "weights_map", None) is not None
            ),
            None,
        )
        if weights_map is None:
            raise RuntimeError("could not locate Accelerate's CPU-offloaded PLE weights")

        cpu_weight = weights_map["weight"].detach()
        if cpu_weight.device.type != "cpu":
            raise RuntimeError("per-layer embedding weights were not loaded on CPU")
        padding_idx = embedding.padding_idx
        scalar_embed_scale = embedding.scalar_embed_scale

        class CpuPerLayerEmbedding(nn.Module):
            def __init__(self, weight: Any, padding: int | None, scale: float) -> None:
                super().__init__()
                # A plain tensor is intentional: the frozen base weight must not be
                # moved by model.to() or included in the trainable adapter state.
                self.weight = weight
                self.padding_idx = padding
                self.scalar_embed_scale = scale

            def forward(self, input_ids: Any) -> Any:
                rows = functional.embedding(input_ids.cpu(), self.weight, self.padding_idx)
                return (rows * self.scalar_embed_scale).to(input_ids.device)

        self.model.model.language_model.embed_tokens_per_layer = CpuPerLayerEmbedding(
            cpu_weight, padding_idx, scalar_embed_scale
        )

    def _remove_multimodal_towers(self) -> None:
        """Release components that cannot be reached by the text-only experiment."""
        self.model.model.vision_tower = None
        self.model.model.audio_tower = None
        self.model.model.embed_vision = None
        self.model.model.embed_audio = None

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
