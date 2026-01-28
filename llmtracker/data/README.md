# LLMTracker data directory

This directory contains scraped and normalized LLM pricing data.

## Structure

- `current/` - Latest scraped data
  - `openrouter.json` - Raw OpenRouter API response
  - `litellm.json` - Raw LiteLLM pricing data
  - `prices.json` - Normalized unified schema with cache pricing

## Schema

The `prices.json` file includes:

```json
{
  "models": {
    "model_id": {
      "pricing": {
        "input_per_million": 2.5,
        "output_per_million": 10.0,
        "cache_read_per_million": 1.25,
        "cache_creation_per_million": null,
        "currency": "USD"
      }
    }
  }
}
```

## Credits

Data sourced from:
- [OpenRouter](https://openrouter.ai)
- [LiteLLM](https://github.com/BerriAI/litellm)
