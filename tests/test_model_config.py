import pytest

from gemma_clinc.config import GenerationConfig, ModelConfig
from gemma_clinc.model import GemmaClassifier


def test_embedding_offload_adds_cpu_override_without_mutating_config():
    configured_map = {"": 0}
    classifier = GemmaClassifier(
        ModelConfig(
            id="test/model",
            device_map=configured_map,
            offload_per_layer_embeddings=True,
        ),
        GenerationConfig(),
    )

    assert classifier._device_map() == {
        "": 0,
        "model.language_model.embed_tokens_per_layer": "cpu",
    }
    assert configured_map == {"": 0}


def test_embedding_offload_rejects_unquantized_loading():
    classifier = GemmaClassifier(
        ModelConfig(
            id="test/model",
            quantization="none",
            offload_per_layer_embeddings=True,
        ),
        GenerationConfig(),
    )

    with pytest.raises(ValueError, match="requires 4-bit"):
        classifier._device_map()
