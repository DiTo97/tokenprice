# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "tokenprice",
# ]
# ///
"""Compute cost from an OpenAI API chat completions response.

This example shows how to integrate tokenprice with the OpenAI Python SDK
to compute the cost of an API call from the response's usage data.

Run with: uv run examples/openai_completions_response.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openai import OpenAI

from tokenprice import compute_cost_sync

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


def compute_cost_from_response(response: ChatCompletion, currency: str = "USD") -> float:
    """Compute the cost of an OpenAI API call from its response.

    Args:
        response: The ChatCompletion response from the OpenAI API
        currency: Currency code for the cost (default: USD)

    Returns:
        Total cost of the API call in the specified currency
    """
    if response.usage is None:
        raise ValueError("Response does not contain usage information")

    # Map OpenAI model names to tokenprice format (provider/model)
    model_id = f"openai/{response.model}"

    return compute_cost_sync(
        model_id,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        currency=currency,
    )


def main() -> None:
    """Demonstrate cost computation from an OpenAI API response."""
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: OPENAI_API_KEY not set, using mock response for demonstration\n")
        # Mock response for demonstration when API key is not available
        demo_with_mock_response()
        return

    # Real API call
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )

    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message.content}")
    print()

    if response.usage:
        print("Usage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        print()

    # Compute cost using tokenprice
    cost_usd = compute_cost_from_response(response)
    cost_eur = compute_cost_from_response(response, currency="EUR")

    print("Cost:")
    print(f"  USD: ${cost_usd:.6f}")
    print(f"  EUR: €{cost_eur:.6f}")


def demo_with_mock_response() -> None:
    """Demonstrate with a mock response when no API key is available."""
    # Simulate a typical response structure
    mock_usage = {
        "prompt_tokens": 25,
        "completion_tokens": 42,
        "total_tokens": 67,
    }
    model = "gpt-4o-mini"
    model_id = f"openai/{model}"

    print(f"Model: {model}")
    print("Response: Paris is the capital of France.")
    print()
    print("Usage (mock):")
    print(f"  Prompt tokens: {mock_usage['prompt_tokens']}")
    print(f"  Completion tokens: {mock_usage['completion_tokens']}")
    print(f"  Total tokens: {mock_usage['total_tokens']}")
    print()

    # Compute cost using tokenprice
    cost_usd = compute_cost_sync(
        model_id,
        input_tokens=mock_usage["prompt_tokens"],
        output_tokens=mock_usage["completion_tokens"],
    )
    cost_eur = compute_cost_sync(
        model_id,
        input_tokens=mock_usage["prompt_tokens"],
        output_tokens=mock_usage["completion_tokens"],
        currency="EUR",
    )

    print("Cost:")
    print(f"  USD: ${cost_usd:.6f}")
    print(f"  EUR: €{cost_eur:.6f}")


if __name__ == "__main__":
    main()
