# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langsmith",
#     "openai",
#     "tokenprice",
# ]
# ///
"""LangSmith cost tracking with tokenprice.

This example demonstrates how to integrate tokenprice with LangSmith for
automatic cost tracking. Instead of hardcoding pricing details, tokenprice
provides up-to-date pricing that can be used to compute costs dynamically.

Based on: https://docs.langchain.com/langsmith/cost-tracking#llm-calls:-sending-costs-directly

Run with: uv run examples/langsmith_cost_tracking.py

Prerequisites:
- Set LANGSMITH_API_KEY environment variable
- Set OPENAI_API_KEY environment variable (optional, uses mock otherwise)
"""

from __future__ import annotations

import os

from langsmith import traceable
from langsmith.run_trees import get_current_run_tree

from tokenprice import get_pricing_sync


def get_cost_per_token(model_id: str, currency: str = "USD") -> tuple[float, float]:
    """Get per-token costs using tokenprice.

    Args:
        model_id: The model identifier (e.g., "openai/gpt-4o-mini")
        currency: Currency code for pricing

    Returns:
        Tuple of (input_cost_per_token, output_cost_per_token)
    """
    pricing = get_pricing_sync(model_id, currency=currency)
    # Convert from per-million to per-token
    input_cost = pricing.input_per_million / 1_000_000
    output_cost = pricing.output_per_million / 1_000_000
    return input_cost, output_cost


@traceable(
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"},
)
def chat_with_cost_tracking(messages: list[dict]) -> dict:
    """Chat completion with automatic cost tracking via tokenprice.

    This function demonstrates the LangSmith cost tracking pattern,
    but uses tokenprice to get up-to-date pricing instead of hardcoding.
    """
    # In a real scenario, this would be an actual API call
    # For demonstration, we simulate a response with token counts
    model = "gpt-4o-mini"
    model_id = f"openai/{model}"

    # Simulated response (replace with actual OpenAI call in production)
    prompt_tokens = 25
    completion_tokens = 42

    response_content = "Hello! I'm here to help. How can I assist you today?"

    # Get up-to-date pricing from tokenprice
    input_cost_per_token, output_cost_per_token = get_cost_per_token(model_id)

    # Calculate costs
    input_cost = prompt_tokens * input_cost_per_token
    output_cost = completion_tokens * output_cost_per_token

    # Set usage metadata for LangSmith cost tracking
    # This follows the LangSmith pattern but with dynamic pricing
    usage_metadata = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        # LangSmith expects costs in dollars
        "input_cost": input_cost,
        "output_cost": output_cost,
    }

    # Report to LangSmith
    run = get_current_run_tree()
    if run:
        run.set(usage_metadata=usage_metadata)

    return {
        "role": "assistant",
        "content": response_content,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost": {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        },
    }


@traceable(
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"},
)
def chat_with_openai(messages: list[dict]) -> dict:
    """Real OpenAI chat completion with cost tracking.

    This version makes an actual API call when OPENAI_API_KEY is set.
    """
    from openai import OpenAI

    model = "gpt-4o-mini"
    model_id = f"openai/{model}"

    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)

    if response.usage is None:
        raise ValueError("No usage data in response")

    # Get up-to-date pricing from tokenprice
    input_cost_per_token, output_cost_per_token = get_cost_per_token(model_id)

    # Calculate costs
    input_cost = response.usage.prompt_tokens * input_cost_per_token
    output_cost = response.usage.completion_tokens * output_cost_per_token

    # Set usage metadata for LangSmith cost tracking
    usage_metadata = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
    }

    run = get_current_run_tree()
    if run:
        run.set(usage_metadata=usage_metadata)

    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "cost": {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        },
    }


def main() -> None:
    """Demonstrate LangSmith cost tracking with tokenprice."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Show the pricing being used
    model_id = "openai/gpt-4o-mini"
    pricing = get_pricing_sync(model_id)
    print(f"Current pricing for {model_id}:")
    print(f"  Input: ${pricing.input_per_million:.4f} per 1M tokens")
    print(f"  Output: ${pricing.output_per_million:.4f} per 1M tokens")
    print()

    # Choose real or mock based on API key availability
    if os.getenv("OPENAI_API_KEY"):
        print("Using real OpenAI API call...")
        result = chat_with_openai(messages)
    else:
        print("Note: OPENAI_API_KEY not set, using mock response")
        print("(Set LANGSMITH_API_KEY to see traces in LangSmith)")
        print()
        result = chat_with_cost_tracking(messages)

    print(f"\nResponse: {result['content']}")
    print(f"\nUsage:")
    print(f"  Prompt tokens: {result['usage']['prompt_tokens']}")
    print(f"  Completion tokens: {result['usage']['completion_tokens']}")
    print(f"\nCost (computed via tokenprice):")
    print(f"  Input cost: ${result['cost']['input_cost']:.8f}")
    print(f"  Output cost: ${result['cost']['output_cost']:.8f}")
    print(f"  Total cost: ${result['cost']['total_cost']:.8f}")

    if os.getenv("LANGSMITH_API_KEY"):
        print("\n✓ Cost data sent to LangSmith!")
    else:
        print("\nTip: Set LANGSMITH_API_KEY to track costs in LangSmith UI")


if __name__ == "__main__":
    main()
