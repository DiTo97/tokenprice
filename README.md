# tokenprice

[![PyPI version](https://img.shields.io/pypi/v/tokenprice)](https://pypi.org/project/tokenprice/)

A Python library for fetching LLM token pricing across providers with multi-currency support.

## Why tokenprice?

Token pricing for LLMs changes frequently across different providers. This library provides up-to-date pricing information by leveraging [LLMTracker](https://github.com/MrUnreal/LLMTracker), which updates pricing data every six hours from various sources.

**Important:** This library does **not** estimate token counts from strings or messages. Any estimation would be too approximate for anything beyond plain text, and the [tokencost](https://github.com/AgentOps-AI/tokencost) package already handles that use case well. tokenprice focuses solely on providing accurate, current pricing data.

## Features

- Up-to-date LLM pricing from [LLMTracker](https://mrunreal.github.io/LLMTracker/)
- Multi-currency support via forex-python
- Built-in caching (6 hours for pricing data, 24 hours for exchange rates)
- Simple, clean API

## Installation

```bash
uv add tokenprice
```

Or with pip:

```bash
pip install tokenprice
```

## Usage

```python
from tokenprice import get_model_price

# Get price in USD (default)
price = get_model_price("gpt-4")
print(f"Input: ${price['input']}/1M tokens")
print(f"Output: ${price['output']}/1M tokens")

# Get price in EUR
price_eur = get_model_price("gpt-4", currency="EUR")
print(f"Input: €{price_eur['input']}/1M tokens")

# Get price in GBP
price_gbp = get_model_price("claude-3-opus", currency="GBP")
```

## Data Source

Pricing data is sourced from [LLMTracker](https://github.com/MrUnreal/LLMTracker), which aggregates and updates pricing information from various LLM providers every six hours. The raw data is available at:
```
https://raw.githubusercontent.com/MrUnreal/LLMTracker/main/data/current/prices.json
```

Exchange rates are provided by forex-python and cached for 24 hours.

## Development

This project uses `uv` as the package manager.

### Setup

```bash
uv sync
```

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check
```

### Formatting

```bash
ruff format
```

## Credits

- Pricing data: [LLMTracker](https://github.com/MrUnreal/LLMTracker) by MrUnreal
- Token counting: For estimating token counts, see [tokencost](https://github.com/AgentOps-AI/tokencost)

## License

See [LICENSE](LICENSE) file for details.
