# Development Guidelines for AI Agents

This document contains instructions and context for AI agents working on the tokenprice project.

## Project Overview

tokenprice is a Python library that provides up-to-date LLM token pricing across different providers with multi-currency support.

### Key Design Decisions

1. **No Token Counting**: This library intentionally does NOT implement token counting from strings/messages. Reasoning:
   - Any estimation would be too approximate for anything beyond plain text
   - Different tokenizers (GPT, Claude, etc.) produce different results
   - The [tokencost](https://github.com/AgentOps-AI/tokencost) package already handles this well
   - We focus on what we do best: providing accurate pricing data

2. **Data Source**: All pricing data comes from [LLMTracker](https://github.com/MrUnreal/LLMTracker)
   - Updates every 6 hours
   - Data available at: `https://raw.githubusercontent.com/MrUnreal/LLMTracker/main/data/current/prices.json`
   - Always credit LLMTracker in documentation
   - Reference either their [repo](https://github.com/MrUnreal/LLMTracker) or [website](https://mrunreal.github.io/LLMTracker/)

3. **Multi-Currency Support**: Using forex-python for currency conversion
   - forex-python updates exchange rates daily
   - forex-python does NOT have built-in caching
   - We must wrap their main currency exchange function with caching

4. **Async I/O**: All network requests MUST use async technology
   - Use aiohttp or httpx for HTTP requests (not synchronous requests library)
   - Ensure proper async/await patterns throughout the codebase
   - This applies to fetching pricing data from LLMTracker and any other external APIs

5. **CLI Requirement**: The package MUST provide a command-line interface
   - Users should easily query current pricing for any model via CLI
   - CLI should be intuitive and well-documented
   - Consider using Click or Typer for CLI implementation

## Caching Strategy

### Pricing Data Cache
- Cache duration: 6 hours (aligns with LLMTracker update frequency)
- Invalidate cache if data fetch fails

### Exchange Rate Cache  
- Cache duration: 24 hours (forex-python updates daily)
- Wrap forex-python's currency exchange function with caching
- Cache per currency pair to optimize performance

## Dependencies

Required dependencies:
- `httpx` or `aiohttp`: for async HTTP requests (NOT requests)
- `forex-python`: for currency conversion (required)
- CLI framework: `click` or `typer` (choose one)

Package manager: **uv** (not pip, not poetry)

## Code Style

### General Guidelines
- Avoid excessive emojis in code or documentation
- Write clean, human-readable documentation
- No "AI slop" formatting (overly enthusiastic tone, excessive bullet points, etc.)
- Keep README concise and practical
- Follow Python best practices and PEP 8
- **Always use type hinting** (Python 3.12+ syntax)
- **Use modeling classes** (dataclasses, Pydantic models, etc.) for structured data
- **Respect DRY (Don't Repeat Yourself)**: Eliminate code duplication
- **Respect KISS (Keep It Simple, Stupid)**: Favor simple, clear solutions
- **Respect SOLID principles**: Design maintainable, extensible code

### Documentation Style
- Be direct and informative
- Use code examples where helpful
- Avoid marketing-speak or hype
- Don't over-explain obvious things
- Technical accuracy over friendliness
- **Keep documentation in sync with code**: Any significant codebase change that affects information in the README MUST be reflected in the README immediately

## Testing

- **TDD Approach**: Write tests BEFORE implementation
- Use pytest for all tests
- Test caching mechanisms thoroughly
- Mock external API calls (LLMTracker, forex-python)
- Test edge cases: missing currencies, API failures, cache expiration
- Ensure test coverage for all public APIs

## Project Structure

```
tokenprice/
├── src/
│   └── tokenprice/         # Main package (rename from 'jokes')
│       ├── __init__.py
│       ├── pricing.py      # Core pricing logic
│       ├── cache.py        # Caching implementation
│       └── currency.py     # Currency conversion wrapper
├── tests/
│   └── test_*.py
├── examples/
│   └── main.py
├── pyproject.toml
├── README.md
├── AGENTS.md               # This file
└── LICENSE
```

## Common Tasks

### Adding a New Feature
1. Check if it aligns with project scope (pricing data only)
2. Write tests first (TDD approach)
3. Implement feature with proper type hints
4. Ensure async/await for any network operations
5. Update README if user-facing (keep docs in sync)
6. Run code quality checks (see below)
7. Verify all tests pass

## Code Quality Workflow

### Formatting and Linting
Always run these commands before committing code:

```bash
# Format code
uv run ruff format src

# Fix import sorting
uv run ruff check --select I --fix

# Fix auto-fixable linting issues
uv run ruff check --fix
```

### Code Quality Checks
Before pushing, verify code quality:

```bash
# Check for linting issues
uv run ruff check src

# Check all rules
uv run ruff check -e

# Check import sorting
uv run ruff check --select I -e
```

All code MUST pass these checks before being committed.

### Updating Dependencies
```bash
uv add <package-name>
```

### Publishing
- Handled by GitHub Actions on push to main
- Ensure version is bumped in pyproject.toml
- Trusted publisher configured on PyPI

## What NOT to Do

- Don't implement token counting features
- Don't add dependencies without good reason
- Don't cache pricing data for more than 6 hours
- Don't use package managers other than uv
- Don't write overly casual or emoji-heavy documentation
- Don't ignore the LLMTracker credit requirement
- Don't fetch pricing data more frequently than needed (respect caching)

## Credits

Always acknowledge:
- LLMTracker for pricing data
- tokencost for token counting (when discussing why we don't do it)
- forex-python for exchange rates
