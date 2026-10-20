
import time
import math
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_transient(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.25,
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
) -> T:
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be at least 1")
    if not math.isfinite(base_delay) or base_delay < 0:
        raise ValueError("base_delay must be finite and nonnegative")

    for attempt in range(attempts):
        try:
            return operation()
        except retry_on:
            if attempt == attempts - 1:
                raise

            time.sleep(base_delay * (2 ** attempt))

    raise RuntimeError("Unreachable")
