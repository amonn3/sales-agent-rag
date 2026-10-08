"""Small cross-cutting helpers (a typed decorator lives here)."""

from collections.abc import Callable
import functools
import logging
import time
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed(logger: logging.Logger) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log how long the wrapped call took, even if it raises."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("%s took %.0f ms", func.__qualname__, elapsed_ms)

        return wrapper

    return decorator
