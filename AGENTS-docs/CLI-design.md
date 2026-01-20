# CLI Design (planned)

Status: No CLI is currently shipped. This document specifies the future CLI to query current pricing (with optional currency conversion).

## Requirements

- Console script entry in pyproject when implemented.
- Query by model id; optional `--provider` to disambiguate names.
- Output input/output prices; default USD.
- Optional `--currency EUR` using cached forex rates (when FX exists).
- Human-friendly default output; optional `--json` for scripting.

## Framework

- Prefer `typer` for fast, typed CLIs; `click` acceptable.
- Commands discoverable via `--help` with clear error messages.

## Example UX

```bash
# Price for a model in USD (default)
tokenprice price gpt-4o

# Price converted to EUR (when FX is implemented)
tokenprice price gpt-4o --currency EUR

# Disambiguate by provider
tokenprice price "command-r+" --provider anthropic

# JSON output for scripting
tokenprice price gpt-4o --json
```

## Behavior

- All network ops are async; wrap async in the CLI entry point.
- Respect caches (6h pricing, 24h FX when available).
- Clear errors for unknown model/provider/currency.

## Acceptance Criteria

- End-to-end tests for parsing and output (mocking network/FX).
- Graceful errors and helpful usage text.
- Minimal dependencies; follows project policies in AGENTS.md.