"""Exponential backoff retry for NVIDIA/OpenAI-compatible API calls."""
import time
import logging
import functools
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 529, 503, 502, 500}


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Call fn(), retrying on transient API and connection errors."""
    from openai import APIConnectionError, APIStatusError

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except APIStatusError as e:
            if e.status_code not in _RETRYABLE_STATUS or attempt == max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(f"LLM API {e.status_code} (attempt {attempt}/{max_attempts}). Retrying in {delay:.0f}s...")
            time.sleep(delay)
        except APIConnectionError:
            if attempt == max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(f"Connection error (attempt {attempt}/{max_attempts}). Retrying in {delay:.0f}s...")
            time.sleep(delay)
