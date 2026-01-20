# tokenprice — AGENTS.md

This file orients AI coding agents working on tokenprice. Keep it concise, follow progressive disclosure, and always link to detailed docs in AGENTS-docs.

## Why
- Provide up-to-date LLM token pricing across providers using LLMTracker (updates ~6h).
- Focus on pricing data only — no token counting. For counting, see tokencost.
- Credit LLMTracker in code and docs: repo and website.

## What
- Stack: Python 3.12, `httpx` (async), `async-lru` (6h TTL via buckets), `pydantic` models, `pytest` (+pytest-asyncio), `ruff`, package manager `uv`.
- Key modules:
  - src/tokenprice/pricing.py — async fetch + 6h TTL cache of LLMTracker JSON.
  - src/tokenprice/modeling.py — Pydantic models for dataset, search helpers.
  - src/tokenprice/cache.py — small TTL bucket helper.
  - src/tokenprice/core.py — public facade exposing `get_pricing` and `compute_cost`.
  - src/tokenprice/__init__.py — exports only `get_pricing` and `compute_cost`.
- Current state (truth):
  - Pricing fetch + cache implemented.
  - No CLI implemented yet.
  - No currency conversion/forex wrapper yet.
- Project map: see repository tree; tests live in tests/test_*.py.

## How
Follow the mandatory workflow in AGENTS-docs/workflow.md. In short: plan → review → TDD (failing) → implement (minimal) → verify → report.

Common commands
```bash
# Setup
uv sync

# Test
uv run pytest -q

# Format & lint
uv run ruff format src
uv run ruff check --select I --fix
uv run ruff check --fix
```

Policies
- Async I/O: All network calls are async via `httpx` (or `aiohttp`). No `requests`.
- Caching: Pricing data cached for 6h; bucketed TTL strategy. Do not extend TTL beyond 6h.
- Documentation: Keep README, AGENTS.md (this file) and AGENTS-docs in sync with code behavior.
- Scope: Pricing data library only. Do not add token counting features.

Public API policy
- Expose only `get_pricing(model_id)` and `compute_cost(model_id, input_tokens, output_tokens)`.
- Caching must remain transparent to callers (no cache controls in public API).

Progressive disclosure
- Workflow and reporting: AGENTS-docs/workflow.md
- Caching + async rules (current + planned FX): AGENTS-docs/caching-and-async.md
- CLI spec (planned): AGENTS-docs/CLI-design.md
- Testing guidance (TDD, mocking, TTL): AGENTS-docs/testing-guidelines.md
- Contribution checklist (quality gates): AGENTS-docs/contribution-checklist.md

Design decisions
- No Token Counting: different tokenizers and use-cases make estimates unreliable; https://github.com/AgentOps-AI/tokencost already covers this very well.
- Data Source: LLMTracker JSON at https://raw.githubusercontent.com/MrUnreal/LLMTracker/main/data/current/prices.json
- Multi-currency: planned via forex-python (https://github.com/MicroPyramid/forex-python) with 24h per-pair cache; not implemented yet.
- CLI: planned (Typer recommended); not implemented yet.

What not to do
- Don't fetch more often than needed; rely on 6h cache.
- Don't add dependencies without justification.
- Don't use package managers other than `uv`.
- Don't write hypey or overly verbose docs.

Credits
- Pricing data: LLMTracker (repo and website: https://raw.githubusercontent.com/MrUnreal/LLMTracker and https://mrunreal.github.io/LLMTracker/).
