"""Optional Langfuse tracing. A no-op unless langfuse is installed and configured."""

from collections.abc import Callable
import os
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def traced(name: str) -> Callable[[F], F]:
    """Wrap a function with Langfuse's ``observe`` when enabled.

    Enabled only if LANGFUSE_PUBLIC_KEY is set *before import* and the
    ``langfuse`` package is installed. Otherwise returns the function untouched.
    """

    def decorator(func: F) -> F:
        if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            return func
        try:
            from langfuse import observe
        except ImportError:
            return func
        return cast(F, observe(name=name)(func))

    return decorator
