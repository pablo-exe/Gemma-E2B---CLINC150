import numpy as np
import pytest
import torch

from gemma_clinc.kd_training import grouped_kd_loss


def test_grouped_kl_matches_analytic_distribution_and_has_gradients():
    student = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    logits = student.log().requires_grad_()
    ids = torch.tensor([[0, 2]])
    teacher_top = torch.tensor([[0.6, 0.1]])
    teacher_tail = torch.tensor([0.3])
    loss = grouped_kd_loss(logits, ids, teacher_top.log(), teacher_tail.log(), 1)
    expected = 0.6 * np.log(0.6 / 0.4) + 0.1 * np.log(0.1 / 0.2) + 0.3 * np.log(0.3 / 0.4)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_matching_distributions_have_zero_kl_at_temperature_two():
    logits = torch.tensor([[6.0, 3.0, 1.0, -2.0]], requires_grad=True)
    probabilities = (logits.detach() / 2).softmax(-1)
    ids = torch.tensor([[0, 1]])
    top = probabilities[:, :2]
    tail = probabilities[:, 2:].sum(-1)
    loss = grouped_kd_loss(logits, ids, top.log(), tail.log(), 2)
    assert loss.item() == pytest.approx(0, abs=1e-6)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_tail_remains_finite_when_top_probability_rounds_to_one():
    logits = torch.tensor([[50.0, 0.0, -1.0]], requires_grad=True)
    loss = grouped_kd_loss(
        logits, torch.tensor([[0]]), torch.tensor([[0.8]]).log(), torch.tensor([0.2]).log(), 1
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_rejects_temperature_or_alignment_mismatch():
    with pytest.raises(ValueError, match="temperature"):
        grouped_kd_loss(
            torch.ones(1, 4),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1),
            torch.zeros(1),
            0,
        )
    with pytest.raises(ValueError, match="positions differ"):
        grouped_kd_loss(
            torch.ones(2, 4),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1),
            torch.zeros(1),
            2,
        )
