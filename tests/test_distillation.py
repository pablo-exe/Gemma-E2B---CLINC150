import json

import numpy as np
import pytest

from gemma_clinc.distillation import (
    assert_tokenizers_match,
    digest,
    encode_gold,
    save_example,
    sparse_distribution,
    validate_config,
    verify_example,
    verify_run,
    write_json,
)


def test_gold_encoding_preserves_actual_generation_prefix():
    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            if add_generation_prompt:
                return "user<prompt><empty_thought>"
            return "user<prompt>" + messages[-1]["content"] + "<end>"

        def __call__(self, text, **kwargs):
            return {"input_ids": list(text.encode())}

    ids, start = encode_gold(Tokenizer(), [{"role": "user", "content": "user"}], "gold")
    assert bytes(ids[:start]).decode() == "user<prompt><empty_thought>"
    assert bytes(ids[start:]).decode() == "gold<end>"


def test_sparse_distribution_preserves_tail_and_temperature():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[4.0, 2.0, 1.0, -1.0], [50.0, 1.0, 0.0, -20.0]])
    result = sparse_distribution(logits, 2, 2.0)
    full = (logits.double() / 2).softmax(-1).numpy()
    ids = result["topk_ids"]
    top = np.take_along_axis(full, ids, axis=-1)
    np.testing.assert_allclose(np.exp(result["topk_log_probs"]), top, rtol=2e-6)
    mask = np.ones_like(full, dtype=bool)
    np.put_along_axis(mask, ids, False, axis=-1)
    np.testing.assert_allclose(np.exp(result["tail_log_prob"]), (full * mask).sum(-1), rtol=3e-6)


def test_test_split_is_rejected():
    with pytest.raises(ValueError, match="train split"):
        validate_config({"dataset": {"split": "test"}})


def test_token_id_mismatch_is_rejected():
    class Tokenizer:
        all_special_tokens = ["eos"]

        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    with pytest.raises(ValueError, match="vocabularies differ"):
        assert_tokenizers_match(Tokenizer({"a": 1}), Tokenizer({"a": 2}))


def test_atomic_artifact_verification_and_corruption(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    path = samples / "train-00000.npz"
    metadata = {"example": {"id": "train-00000"}, "run_hash": "abc"}
    arrays = {
        "teacher_input_ids": np.array([1, 2, 3]),
        "student_input_ids": np.array([4, 3]),
        "teacher_target_positions": np.array([2]),
        "student_target_positions": np.array([1]),
        "target_ids": np.array([3]),
        "topk_ids": np.array([[3, 4]]),
        "topk_log_probs": np.log(np.array([[0.5, 0.3]], dtype=np.float32)),
        "tail_log_prob": np.log(np.array([0.2], dtype=np.float32)),
    }
    save_example(path, arrays, metadata)
    assert verify_example(path, metadata) == metadata
    assert not path.with_suffix(".npz.tmp").exists()
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_example(path, {"other": 1})
    write_json(tmp_path / "manifest.json", {"expected_examples": 1})
    write_json(tmp_path / "summary.json", {"complete": True, "completed": 1})
    (tmp_path / "examples.jsonl").write_text(
        json.dumps({"id": "train-00000", "sha256": digest(path)}) + "\n"
    )
    verify_run(tmp_path)
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_run(tmp_path)


def test_wrong_causal_alignment_is_rejected(tmp_path):
    arrays = {
        "teacher_input_ids": np.array([1, 2]),
        "student_input_ids": np.array([1, 2]),
        "teacher_target_positions": np.array([0]),
        "student_target_positions": np.array([1]),
        "target_ids": np.array([2]),
        "topk_ids": np.array([[2]]),
        "topk_log_probs": np.log(np.array([[0.8]])),
        "tail_log_prob": np.log(np.array([0.2])),
    }
    path = tmp_path / "bad.npz"
    save_example(path, arrays, {})
    with pytest.raises(ValueError, match="alignment mismatch"):
        verify_example(path)
