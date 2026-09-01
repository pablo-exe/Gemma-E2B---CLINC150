"""Command-line interface for reproducible experiments."""

from __future__ import annotations

import argparse

from gemma_clinc.config import load_config
from gemma_clinc.experiment import run_zero_shot
from gemma_clinc.training import run_adapter_evaluation, run_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-clinc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run the Phase 1 zero-shot baseline")
    baseline.add_argument("--config", required=True, help="Path to experiment YAML")
    baseline.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")

    train = subparsers.add_parser("train-sft", help="Run Phase 2 supervised QLoRA")
    train.add_argument("--config", required=True, help="Path to experiment YAML")
    train.add_argument("--train-limit", type=int, default=None)
    train.add_argument("--eval-limit", type=int, default=None)
    train.add_argument("--max-steps", type=int, default=None)

    evaluate = subparsers.add_parser(
        "evaluate-adapter", help="Evaluate a saved Phase 2 adapter"
    )
    evaluate.add_argument("--config", required=True, help="Path to experiment YAML")
    evaluate.add_argument("--adapter", required=True, help="Saved PEFT adapter directory")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        run_dir = run_zero_shot(load_config(args.config), limit=args.limit)
        print(f"Artifacts written to {run_dir}")
    elif args.command == "train-sft":
        run_dir = run_sft(
            load_config(args.config),
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            max_steps=args.max_steps,
        )
        print(f"Artifacts written to {run_dir}")
    elif args.command == "evaluate-adapter":
        run_dir = run_adapter_evaluation(
            load_config(args.config),
            args.adapter,
            split=args.split,
            limit=args.limit,
        )
        print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
