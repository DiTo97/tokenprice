"""Async caching utilities using async-lru.

This module provides small helpers to support TTL-like behavior by bucketing
time into intervals. Prefer decorating private helpers and exposing public
functions that compute and pass the TTL bucket argument.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def ttl_bucket(ttl_seconds: int) -> int:
    """Return an integer bucket for the given TTL window.

    Example usage:
        @alru_cache(maxsize=1)
        async def _compute(_bucket: int) -> T:
            ...

        async def public() -> T:
            return await _compute(ttl_bucket(3600))
    """
    if ttl_seconds <= 0:
        return int(time.time())
    return int(time.time() // ttl_seconds)
