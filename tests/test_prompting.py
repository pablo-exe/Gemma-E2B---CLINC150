import pytest

from gemma_clinc.data import Example
from gemma_clinc.prompting import (
    build_messages,
    build_sft_messages,
    parse_prediction,
    response_token_span,
)

LABELS = ["card_arrival", "cash_withdrawal", "out_of_scope"]


def test_build_messages_contains_query_and_every_label():
    messages = build_messages("Where is my card?", LABELS, "Choose one label.")

    assert messages[0] == {"role": "system", "content": "Choose one label."}
    assert "Where is my card?" in messages[1]["content"]
    assert all(label in messages[1]["content"] for label in LABELS)


def test_parse_prediction_accepts_exact_label():
    assert parse_prediction("card_arrival", LABELS) == "card_arrival"


def test_parse_prediction_accepts_common_wrapper():
    assert parse_prediction("Intent: cash_withdrawal.", LABELS) == "cash_withdrawal"


def test_parse_prediction_rejects_unknown_or_ambiguous_output():
    assert parse_prediction("something_else", LABELS) is None
    assert parse_prediction("card_arrival or cash_withdrawal", LABELS) is None


def test_build_sft_messages_only_adds_answer_for_training():
    example = Example(id="train-1", text="where is my card", label="card_arrival", split="train")

    prompt = build_sft_messages(example, "Classify", include_answer=False)
    training = build_sft_messages(example, "Classify", include_answer=True)

    assert [message["role"] for message in prompt] == ["system", "user"]
    assert training[-1] == {"role": "assistant", "content": "card_arrival"}


def test_response_token_span_accepts_exact_prompt_prefix():
    assert response_token_span([1, 2, 3], [1, 2, 3, 7, 8]) == (3, 5)


@pytest.mark.parametrize("full_ids", [[1, 9, 3, 7], [1, 2, 3]])
def test_response_token_span_rejects_invalid_training_sequence(full_ids):
    with pytest.raises(ValueError):
        response_token_span([1, 2, 3], full_ids)
