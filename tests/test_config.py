from gemma_clinc.config import load_config


def test_phase1_config_loads():
    config = load_config("configs/phase1_zero_shot.yaml")

    assert config.name == "phase1_zero_shot"
    assert config.model.id == "google/gemma-4-E2B-it"
    assert config.model.quantization == "4bit"
    assert config.model.device_map == {"": 0}
    assert config.model.offload_per_layer_embeddings is True
    assert config.model.text_only is True
    assert config.dataset.split == "test"
    assert config.dataset.include_oos is True
    assert config.generation.do_sample is False
