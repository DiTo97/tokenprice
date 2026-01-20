# Testing Guidelines (TDD)

All work follows TDD: write failing tests before implementation. Use pytest for test discovery and execution.

## Structure

- tests/test_*.py files mirror public APIs
- Prefer small, focused tests over broad integration when possible
- Use type hints in tests where helpful to clarify intent

## External API Mocking

- LLMTracker pricing fetches: mock network responses
- forex-python exchange: planned — mock once implemented
- Use `unittest.mock` (patch) or pytest fixtures to inject doubles

### HTTP Mocking (httpx/aiohttp)

- Prefer `httpx.AsyncClient` or `aiohttp` with async tests
- Patch the client method used by the code (e.g., `httpx.AsyncClient.get`)
- Return representative JSON payloads and status codes
- Cover error paths: timeouts, non-200, malformed payload

## Caching Tests

- Pricing cache TTL: 6 hours (implemented)
- FX cache TTL: 24 hours (planned; per currency pair)
- Strategy:
  - First call: expect underlying provider to be invoked.
  - Subsequent calls within TTL: ensure provider is NOT invoked; return cached result.
  - After TTL: ensure provider is invoked again.
- For time control, prefer designing cache utilities to accept an injectable "now" or clock function; otherwise monkeypatch time source where feasible.

## Failure-first Tests

- Confirm tests fail for the intended reason (e.g., missing key, incorrect value, uncalled mock)
- Keep the error messages clear; assert on specific behaviors, not broad exceptions

## Running Tests and Quality Checks

```bash
# Run tests
uv run pytest -q

# Re-run a specific test node
uv run pytest tests/test_pricing.py::test_fetch_prices_async -q

# Format and lint
uv run ruff format src
uv run ruff check --select I --fix
uv run ruff check --fix
```

## Coverage

- Aim for meaningful coverage on public APIs
- Validate edge cases: missing currencies, stale cache, API failures, invalid inputs

## Test Data

- Keep fixtures small and realistic
- Store small JSON payloads inline in tests when clearer; otherwise use dedicated fixture helpers