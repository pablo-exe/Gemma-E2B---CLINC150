import pytest


def test_real_tiny_gemma_sdpa_and_artifact_roundtrip():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_clinc.pod_check import check_environment

    report = check_environment(cpu_only=True)
    assert "sdpa_forward" in report["checks_passed"]
    assert "logits_to_keep" in report["checks_passed"]
    assert "npz_roundtrip" in report["checks_passed"]
    assert "int8_cuda_bf16" not in report["checks_passed"]
