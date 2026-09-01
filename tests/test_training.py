import pytest
import torch

from gemma_clinc.data import Example
from gemma_clinc.training import (
    _encode_training_example,
    optimizer_step_count,
    select_lora_target_names,
)


class QuantizedLinear:
    pass


class OtherModule:
    pass


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        if add_generation_prompt:
            return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3, 7, 8]]))


def test_optimizer_step_count_rounds_each_epoch_up():
    assert optimizer_step_count(101, epochs=3, accumulation_steps=16) == 21


@pytest.mark.parametrize("value", [0, -1])
def test_optimizer_step_count_rejects_invalid_examples(value):
    with pytest.raises(ValueError):
        optimizer_step_count(value, epochs=1, accumulation_steps=1)


def test_select_lora_targets_only_uses_requested_language_projections():
    modules = [
        ("model.language_model.layers.0.self_attn.q_proj", QuantizedLinear()),
        ("model.language_model.layers.0.self_attn.k_proj", QuantizedLinear()),
        ("model.vision_tower.layers.0.self_attn.q_proj", QuantizedLinear()),
        ("model.language_model.layers.0.self_attn.v_proj", OtherModule()),
    ]

    assert select_lora_target_names(modules, QuantizedLinear, ("q_proj", "v_proj")) == [
        "model.language_model.layers.0.self_attn.q_proj"
    ]


def test_select_lora_targets_requires_at_least_one_match():
    with pytest.raises(ValueError, match="no quantized LoRA targets"):
        select_lora_target_names([], QuantizedLinear, ("q_proj",))


def test_encode_training_example_aligns_logits_with_only_assistant_tokens():
    example = Example(id="train-1", text="hello", label="greeting", split="train")

    batch, targets, logit_indices = _encode_training_example(
        FakeProcessor(), example, "Classify", torch.device("cpu")
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 7, 8]]
    assert targets.tolist() == [[7, 8]]
    assert logit_indices.tolist() == [2, 3]
