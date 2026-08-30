"""Entry point; bare invocation runs everything (all steps + full-year + summary); flags narrow that down."""

import argparse

from src.pipeline import MODEL_STEPS, run_pipeline
from src.timer import timer


def parse_args():
    parser = argparse.ArgumentParser(description="Student dropout prediction pipeline")
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=list(MODEL_STEPS.keys()),
        default=None,
        help="Which model steps to run (default: all; pass with no values to run none).",
    )
    parser.add_argument(
        "--full-year",
        action="store_true",
        default=None,
        help="Also run the early-warning vs full-year comparison (default: on only for a bare, no-flags invocation).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=None,
        help="Also run the final CV + held-out test summary table/plot (default: on only for a bare, no-flags invocation).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bare = args.steps is None
    include_full_year = args.full_year if args.full_year is not None else bare
    include_summary = args.summary if args.summary is not None else bare
    with timer("student-dropout-prediction"):
        run_pipeline(
            selected_steps=args.steps,
            include_full_year=include_full_year,
            include_summary=include_summary,
        )
