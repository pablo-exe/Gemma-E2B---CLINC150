import json

import pytest

from gemma_clinc.data import OOS_LABEL, labels_from_examples, load_examples


def test_load_examples_preserves_order_and_appends_oos(tmp_path):
    dataset_path = tmp_path / "clinc.json"
    dataset_path.write_text(
        json.dumps(
            {
                "train": [["hello", "greeting"]],
                "val": [],
                "test": [["freeze my card", "card_arrival"]],
                "oos_train": [["unknown train", "oos"]],
                "oos_val": [],
                "oos_test": [["unknown test", "oos"]],
            }
        ),
        encoding="utf-8",
    )

    examples = load_examples(dataset_path, "test", include_oos=True)

    assert [example.text for example in examples] == ["freeze my card", "unknown test"]
    assert [example.label for example in examples] == ["card_arrival", OOS_LABEL]
    assert [example.id for example in examples] == ["test-00000", "test-00001"]


def test_labels_are_sorted_with_oos_last(tmp_path):
    dataset_path = tmp_path / "clinc.json"
    dataset_path.write_text(
        json.dumps(
            {
                "train": [["b", "zeta"], ["a", "alpha"]],
                "val": [],
                "test": [],
                "oos_train": [["x", "oos"]],
                "oos_val": [],
                "oos_test": [],
            }
        ),
        encoding="utf-8",
    )

    labels = labels_from_examples(load_examples(dataset_path, "train", include_oos=True))

    assert labels == ["alpha", "zeta", OOS_LABEL]


def test_invalid_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="split must be"):
        load_examples(tmp_path / "unused.json", "development")
