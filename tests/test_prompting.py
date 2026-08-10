from gemma_clinc.prompting import build_messages, parse_prediction

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
