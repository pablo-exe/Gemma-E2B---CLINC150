"""Run one measured QLoRA optimizer step on the configured student model."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess

import psutil
import torch
import torch.nn.functional as functional
from bitsandbytes.nn import Linear4bit
from peft import LoraConfig, get_peft_model

from gemma_clinc.config import load_config
from gemma_clinc.data import download_dataset, labels_from_examples, load_examples
from gemma_clinc.model import GemmaClassifier


def _gpu_used_mib() -> int:
    output = subprocess.check_output(  # noqa: S603
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _mib(value: int) -> float:
    return round(value / 2**20, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_zero_shot.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    baseline_mib = _gpu_used_mib()
    classifier = GemmaClassifier(config.model, config.generation)
    classifier.load()
    model = classifier.model
    ple = model.model.language_model.embed_tokens_per_layer
    if ple.weight.device.type != "cpu":
        raise RuntimeError("PLE offload is not active: embedding weight is not on CPU")
    load_rss = process.memory_info().rss
    load_gpu_mib = _gpu_used_mib()
    system_available_after_load = psutil.virtual_memory().available

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, Linear4bit)
        and ".language_model.layers." in name
        and name.endswith(("q_proj", "v_proj"))
    ]
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    dataset_path = download_dataset(config.dataset.source_url, config.dataset.cache_path)
    train_examples = load_examples(dataset_path, "train", include_oos=config.dataset.include_oos)
    known_labels = labels_from_examples(train_examples)
    system_prompt = (
        "Classify the user request into one CLINC150 intent. Return only the "
        "lowercase label with underscores. Valid labels were demonstrated during training."
    )
    tokenized_examples = [
        (
            len(
                classifier.processor.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item.text},
                        {"role": "assistant", "content": item.label},
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            ),
            item,
        )
        for item in train_examples
    ]
    _, example = max(tokenized_examples, key=lambda pair: pair[0])
    if example.label not in known_labels:
        raise RuntimeError("training example has an unknown label")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.text},
    ]
    prompt_batch = classifier.processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to("cuda")
    messages.append({"role": "assistant", "content": example.label})
    batch = classifier.processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
        enable_thinking=False,
    ).to("cuda")
    prompt_tokens = prompt_batch.input_ids.shape[-1]
    if not torch.equal(batch.input_ids[:, :prompt_tokens], prompt_batch.input_ids):
        raise RuntimeError("assistant response does not follow the tokenized prompt prefix")
    targets = batch.input_ids[:, prompt_tokens:]
    logit_indices = torch.arange(
        prompt_tokens - 1,
        batch.input_ids.shape[-1] - 1,
        device=batch.input_ids.device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-4,
    )

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = model(**batch, logits_to_keep=logit_indices)
    loss = functional.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]), targets.reshape(-1)
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print(
        {
            "sequence_tokens": batch.input_ids.shape[-1],
            "supervised_tokens": targets.shape[-1],
            "example_id": example.id,
            "ple_device": ple.weight.device.type,
            "ple_mib": _mib(ple.weight.numel() * ple.weight.element_size()),
            "model_load_rss_delta_mib": _mib(load_rss - baseline_rss),
            "model_load_gpu_delta_mib": load_gpu_mib - baseline_mib,
            "system_available_mib_after_load": _mib(system_available_after_load),
            "loss": float(loss.detach()),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": finite_gradients,
            "gradient_tensors": len(gradients),
            "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
            "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved()),
            "nvidia_smi_delta_mib": _gpu_used_mib() - baseline_mib,
        }
    )


if __name__ == "__main__":
    main()
