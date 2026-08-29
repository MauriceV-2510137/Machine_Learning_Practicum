"""Timing helper."""

import time
from contextlib import contextmanager


@contextmanager
def timer(label: str = "Run"):
    """Context manager that prints how long the wrapped block took."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"\n{label} finished in {elapsed:.2f}s")
