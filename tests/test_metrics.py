import pytest

from gemma_clinc.metrics import compute_metrics


def test_compute_metrics_tracks_invalid_and_oos():
    metrics = compute_metrics(
        ["intent_a", "intent_b", "out_of_scope", "out_of_scope"],
        ["intent_a", None, "out_of_scope", "intent_a"],
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["in_scope_accuracy"] == pytest.approx(0.5)
    assert metrics["oos_precision"] == pytest.approx(1.0)
    assert metrics["oos_recall"] == pytest.approx(0.5)
    assert metrics["invalid_output_rate"] == pytest.approx(0.25)
    assert metrics["top_confusions"][0]["count"] == 1


def test_compute_metrics_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="equal length"):
        compute_metrics(["intent_a"], [])
