# Caching and Async I/O

This project requires strict caching rules and async HTTP for all network operations.

Status summary
- Pricing cache: implemented (6h TTL, TTL-bucket strategy via async-lru).
- FX cache: planned (24h per currency pair using forex-python); not implemented yet.

## Pricing Data Cache (implemented)

- Source: LLMTracker prices JSON
- TTL: 6 hours (aligned with upstream update frequency)
- Behavior:
  - Cache successful fetches for 6 hours using a time-bucket key.
  - If a fetch fails, bubble up the error; do not fabricate data.
  - Validate payload shape before caching (Pydantic models).

## Exchange Rate Cache (planned)

- Source: forex-python
- TTL: 24 hours (forex-python updates daily)
- Per-pair cache: key by `BASE->QUOTE`
- Acceptance criteria:
  - Public conversion API with type hints and clear exceptions.
  - Per-pair 24h TTL cache; misses call through to forex-python.
  - Failures propagate; no silent fallback to stale unknown data.
  - Tests for cache hit/miss and TTL expiry.

## Async HTTP Requirements

- Use `httpx` (AsyncClient) or `aiohttp`; never blocking `requests`.
- Set explicit timeouts and handle non-200 responses.
- Optional: limited retries with backoff for transient errors (no infinite retries).
- All I/O functions are `async` and documented as such.

## Concurrency and Safety

- Guard cache invalidation to avoid race conditions.
- Prefer per-process in-memory caches with TTL; document invalidation behavior.
- Avoid leaking global mutable state; encapsulate caches in modules.

## Testing Guidance

- Mock HTTP clients and forex-python; no real network calls in tests.
- Validate TTL behavior by controlling the time source (injectable clock or monkeypatch).
- Cover failure modes: timeouts, malformed payloads, provider errors.