"""Command-line interface for reproducible experiments."""

from __future__ import annotations

import argparse

from gemma_clinc.config import load_config
from gemma_clinc.experiment import run_zero_shot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-clinc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run the Phase 1 zero-shot baseline")
    baseline.add_argument("--config", required=True, help="Path to experiment YAML")
    baseline.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        run_dir = run_zero_shot(load_config(args.config), limit=args.limit)
        print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
