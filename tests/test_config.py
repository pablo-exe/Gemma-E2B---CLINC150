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
    assert config.training is None


def test_phase2_training_config_loads():
    config = load_config("configs/phase2_qlora_sft.yaml")

    assert config.name == "phase2_qlora_sft"
    assert config.dataset.split == "val"
    assert config.training is not None
    assert config.training.epochs == 3
    assert config.training.gradient_accumulation_steps == 16
    assert config.training.target_modules == ("q_proj", "v_proj")
