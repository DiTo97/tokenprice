# tokenprice

[![PyPI version](https://img.shields.io/pypi/v/tokenprice)](https://pypi.org/project/tokenprice/)

API pricing estimates for 1k+ LLMs from [LLMTracker](https://mrunreal.github.io/LLMTracker) with multi-currency support.

## Why tokenprice?

Token pricing for LLMs changes frequently across different providers. This library provides up-to-date pricing information by leveraging [LLMTracker](https://github.com/MrUnreal/LLMTracker), which updates pricing data every six hours from various sources.

**Important:** This library does **not** estimate token counts from strings or messages. Any estimation would be too approximate for anything beyond plain text, and the [tokencost](https://github.com/AgentOps-AI/tokencost) package already handles that use case well. tokenprice focuses solely on providing accurate, current pricing data.

## Features

- Up-to-date LLM pricing from [LLMTracker](https://mrunreal.github.io/LLMTracker/)
- Async caching via async-lru with a 6-hour TTL for pricing data
- Multi-currency conversion via JSDelivr currency API with a 24-hour cached USD rates map
- Clean, typed data models (Pydantic)

## Installation

```bash
uv add tokenprice
```

Or with pip:

```bash
pip install tokenprice
```

## Usage

The public API exposes only two async functions:
- `get_pricing(model_id, currency="USD")`
- `compute_cost(model_id, input_tokens, output_tokens, currency="USD")`

```python
import asyncio

from tokenprice import get_pricing, compute_cost


async def main():
    model_id = "openai/gpt-5.2"

    # Get pricing (cached transparently for ~6 hours)
    pricing = await get_pricing(model_id, currency="EUR")
    print(f"Pricing for {model_id} ({pricing.currency}):")
    print(f"  Input per 1M tokens: €{pricing.input_per_million:.2f}")
    print(f"  Output per 1M tokens: €{pricing.output_per_million:.2f}")

    # Compute total cost for a usage
    total = await compute_cost(model_id, input_tokens=1000, output_tokens=500, currency="EUR")
    print(f"Total cost (EUR): €{total:.6f}")


asyncio.run(main())
```

## Data Source

Pricing data is sourced from [LLMTracker](https://github.com/MrUnreal/LLMTracker), which aggregates and updates pricing information from various LLM providers every six hours. The raw data is available at:
```
https://raw.githubusercontent.com/MrUnreal/LLMTracker/main/data/current/prices.json
```

Caching uses async-lru with a 6-hour TTL aligned to LLMTracker's refresh cadence. Caching is fully transparent to callers of the public API.

Note: Pricing data from LLMTracker is denominated in USD; currency conversion uses daily USD base rates from the JSDelivr currency API with a 24h cache (keys uppercased).

## Development

This project uses `uv` as the package manager.

### Setup

```bash
uv sync
```

### Running Tests

```bash
uv run pytest
```

### Quality (pre-commit)

Use pre-commit to run formatting and linting consistently:

```bash
# One-time setup
uv run pre-commit install

# Run all hooks on the codebase
uv run pre-commit run -a
```

This runs `ruff-format` and `ruff` with `--fix`, along with basic repo hygiene checks.

### CI

Pre-commit hooks run in CI via GitHub Actions using `uv` (see [.github/workflows/pre-commit.yml](.github/workflows/pre-commit.yml)). Pushes and pull requests to `main`/`master` execute the same checks as local runs.

## API Surface

Only `get_pricing` and `compute_cost` are part of the public API. Internal modules and models are not considered public and may change.

## Credits

- Pricing data: [LLMTracker](https://github.com/MrUnreal/LLMTracker) by MrUnreal

## CLI

Install via UV or pip, then use the `tokenprice` command.

```bash
# Show price per 1M tokens (USD default)
tokenprice pricing openai/gpt-5.2

# Convert to another currency (uses cached FX rates)
tokenprice pricing openai/gpt-5.2 --currency EUR

# JSON output for scripting
tokenprice pricing openai/gpt-5.2 --json

# Compute total cost for a usage
tokenprice cost openai/gpt-5.2 --in 1000 --out 500 --currency EUR
```

## License

See [LICENSE](LICENSE) file for details.
