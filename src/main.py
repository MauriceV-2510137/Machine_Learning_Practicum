"""Entry point; `--steps` selects which model steps to run (default: all, see --help)."""

import argparse

from src.pipeline import MODEL_STEPS, run_pipeline
from src.timer import timer


def parse_args():
    parser = argparse.ArgumentParser(description="Student dropout prediction pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(MODEL_STEPS.keys()),
        default=None,
        help="Which model steps to run (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with timer("student-dropout-prediction"):
        run_pipeline(selected_steps=args.steps)
