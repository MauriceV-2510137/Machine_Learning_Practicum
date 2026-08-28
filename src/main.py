"""
Entry point.
"""

from src.pipeline import run_pipeline
from src.timer import timer

if __name__ == "__main__":
    with timer("student-dropout-prediction"):
        run_pipeline()