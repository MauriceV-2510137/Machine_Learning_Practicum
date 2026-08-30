"""Timing helper."""

import time
from contextlib import contextmanager


@contextmanager
def timer(label: str = "Run", results: dict | None = None):
    """Context manager that prints how long the wrapped block took, and optionally records it in results[label]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"\n{label} finished in {elapsed:.2f}s")
        if results is not None:
            results[label] = elapsed
