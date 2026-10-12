"""Small real Gemma 4/SDPA/INT8 forward pass before loading the 31B teacher."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from gemma_clinc.distillation import save_example, sparse_distribution, verify_example, write_json


def check_environment(
    cpu_only: bool = False, minimum_vram_gib: float = 0, quantization: str = "8bit"
) -> dict:
    import torch
    from transformers import (
        AutoModelForMultimodalLM,
        BitsAndBytesConfig,
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4TextConfig,
    )

    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "numpy",
                "huggingface-hub",
            )
        },
        "torch_cuda": torch.version.cuda,
        "cpu_only": cpu_only,
        "quantization": quantization if not cpu_only else "none",
    }
    if not cpu_only:
        if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.8":
            raise RuntimeError("Expected RunPod PyTorch 2.8.0 + CUDA 12.8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported")
        free, total = torch.cuda.mem_get_info()
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_gib": total / 2**30,
                "gpu_free_gib": free / 2**30,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
        if free / 2**30 < minimum_vram_gib:
            raise RuntimeError(f"Need {minimum_vram_gib} GiB free VRAM; found {free / 2**30:.1f}")
    torch.manual_seed(42)
    text = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=128,
        attention_k_eq_v=True,
    )
    config = Gemma4Config(text_config=text, vision_config=None, audio_config=None)
    config._attn_implementation = "sdpa"
    model = Gemma4ForConditionalGeneration(config).eval()
    with tempfile.TemporaryDirectory(prefix="gemma-pod-check-") as temporary:
        directory = Path(temporary)
        model.save_pretrained(directory / "tiny_model")
        del model
        kwargs = {"dtype": torch.float32, "attn_implementation": "sdpa"}
        if not cpu_only:
            kwargs.update(
                {
                    "dtype": torch.bfloat16,
                    "device_map": {"": 0},
                }
            )
            if quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForMultimodalLM.from_pretrained(directory / "tiny_model", **kwargs).eval()
        if not cpu_only and quantization == "8bit":
            from bitsandbytes.nn import Linear8bitLt

            count = sum(isinstance(module, Linear8bitLt) for module in model.modules())
            if count == 0:
                raise RuntimeError("No INT8 modules were loaded")
            report["int8_modules"] = count
        device = "cpu" if cpu_only else "cuda"
        inputs = torch.tensor([[2, 3, 4, 5]], device=device)
        positions = torch.tensor([1, 2], device=device)
        with torch.inference_mode():
            all_logits = model(input_ids=inputs, use_cache=False).logits
            selected = model(input_ids=inputs, use_cache=False, logits_to_keep=positions).logits
        if selected.shape != (1, 2, 128):
            raise RuntimeError("Gemma 4 logits_to_keep shape mismatch")
        torch.testing.assert_close(selected, all_logits[:, positions], rtol=0.01, atol=0.005)
        sparse = sparse_distribution(selected[0], 16, 2.0)
        arrays = {
            "teacher_input_ids": np.array([2, 3, 4, 5], dtype=np.int32),
            "student_input_ids": np.array([2, 3, 4, 5], dtype=np.int32),
            "teacher_target_positions": np.array([2, 3], dtype=np.int32),
            "student_target_positions": np.array([2, 3], dtype=np.int32),
            "target_ids": np.array([4, 5], dtype=np.int32),
            **sparse,
        }
        save_example(directory / "sample.npz", arrays, {"smoke_test": True})
        verify_example(directory / "sample.npz", {"smoke_test": True})
    report["checks_passed"] = [
        "gemma4_import",
        "save_load",
        "sdpa_forward",
        "logits_to_keep",
        "sparse_probabilities",
        "npz_roundtrip",
    ]
    if not cpu_only:
        report["checks_passed"].append("int8_cuda_bf16" if quantization == "8bit" else "cuda_bf16")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--minimum-vram-gib", type=float, default=0)
    parser.add_argument("--report", type=Path, default=Path("artifacts/pod_preflight.json"))
    args = parser.parse_args()
    report = check_environment(args.cpu_only, args.minimum_vram_gib)
    disk = shutil.disk_usage(Path.cwd())
    report["working_disk_free_gib"] = disk.free / 2**30
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(f"Environment checks passed. Report: {args.report}")


if __name__ == "__main__":
    main()
