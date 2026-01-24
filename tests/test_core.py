"""Tests for synchronous public API wrappers.

Tests that sync versions work correctly and use the same cache as async versions.
"""

from unittest.mock import Mock, patch

import pytest

from tokenprice import compute_cost_sync, get_pricing_sync


@pytest.fixture
def sample_llmtracker_response() -> dict:
    """Sample response from LLMTracker API."""
    return {
        "generated_at": "2026-01-20T06:05:10.791612+00:00",
        "models": {
            "openai/gpt-4": {
                "provider": "openai",
                "model_id": "openai/gpt-4",
                "display_name": "OpenAI: GPT-4",
                "pricing": {
                    "input_per_million": 30.0,
                    "output_per_million": 60.0,
                    "currency": "USD",
                },
                "context_window": 8192,
                "max_output_tokens": 4096,
                "model_type": "chat",
                "supports_vision": False,
                "supports_function_calling": True,
                "supports_streaming": True,
                "category": "flagship",
            }
        },
        "providers": {
            "openai": {
                "name": "OpenAI",
                "website": "https://openai.com",
                "pricing_page": "https://openai.com/pricing",
                "affiliate_link": "https://platform.openai.com/signup",
            }
        },
        "metadata": {
            "total_models": 1,
            "sources": ["openrouter"],
            "last_scrape": "2026-01-20T06:05:10.791612+00:00",
            "categories": {"flagship": 1},
        },
    }


@pytest.fixture
def sample_currency_response() -> dict:
    """Sample response from currency API."""
    return {
        "date": "2026-01-23",
        "usd": {
            "eur": 0.92,
            "gbp": 0.79,
            "cny": 7.25,
        },
    }


def test_get_pricing_sync_usd(sample_llmtracker_response):
    """Test sync get_pricing returns correct pricing in USD."""
    with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_llmtracker_response
        mock_get.return_value = mock_response

        pricing = get_pricing_sync("openai/gpt-4")

        assert pricing.input_per_million == 30.0
        assert pricing.output_per_million == 60.0
        assert pricing.currency == "USD"


def test_get_pricing_sync_with_currency(
    sample_llmtracker_response, sample_currency_response, monkeypatch
):
    """Test sync get_pricing converts currency correctly."""
    import tokenprice.pricing as pricing_mod
    import tokenprice.currency as currency_mod
    from decimal import Decimal

    # Clear caches
    pricing_mod._get_pricing_data_bucketed.cache_clear()
    currency_mod._get_usd_rates_bucketed.cache_clear()

    # Mock the pricing data fetch
    with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_pricing:
        mock_pricing_response = Mock()
        mock_pricing_response.json.return_value = sample_llmtracker_response
        mock_pricing.return_value = mock_pricing_response

        # Mock the currency rates fetch
        def fake_sync_get_usd_rates():
            return {
                "EUR": Decimal("0.92"),
                "GBP": Decimal("0.79"),
                "CNY": Decimal("7.25"),
            }

        monkeypatch.setattr(
            currency_mod, "_sync_get_usd_rates", fake_sync_get_usd_rates
        )

        pricing = get_pricing_sync("openai/gpt-4", currency="EUR")

        # 30.0 * 0.92 = 27.6
        assert abs(pricing.input_per_million - 27.6) < 0.01
        assert abs(pricing.output_per_million - 55.2) < 0.01
        assert pricing.currency == "EUR"


def test_get_pricing_sync_not_found(sample_llmtracker_response):
    """Test sync get_pricing raises ValueError for unknown model."""
    with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_llmtracker_response
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Model not found"):
            get_pricing_sync("fake/model")


def test_compute_cost_sync_usd(sample_llmtracker_response):
    """Test sync compute_cost calculates correctly in USD."""
    with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_llmtracker_response
        mock_get.return_value = mock_response

        cost = compute_cost_sync("openai/gpt-4", input_tokens=1000, output_tokens=500)

        # (1000 / 1M) * 30 + (500 / 1M) * 60 = 0.03 + 0.03 = 0.06
        assert abs(cost - 0.06) < 0.0001


def test_compute_cost_sync_with_currency(
    sample_llmtracker_response, sample_currency_response, monkeypatch
):
    """Test sync compute_cost calculates correctly with currency conversion."""
    import tokenprice.pricing as pricing_mod
    import tokenprice.currency as currency_mod
    from decimal import Decimal

    # Clear caches
    pricing_mod._get_pricing_data_bucketed.cache_clear()
    currency_mod._get_usd_rates_bucketed.cache_clear()

    # Mock the pricing data fetch
    with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_pricing:
        mock_pricing_response = Mock()
        mock_pricing_response.json.return_value = sample_llmtracker_response
        mock_pricing.return_value = mock_pricing_response

        # Mock the currency rates fetch
        def fake_sync_get_usd_rates():
            return {
                "EUR": Decimal("0.92"),
                "GBP": Decimal("0.79"),
                "CNY": Decimal("7.25"),
            }

        monkeypatch.setattr(
            currency_mod, "_sync_get_usd_rates", fake_sync_get_usd_rates
        )

        cost = compute_cost_sync(
            "openai/gpt-4", input_tokens=1000, output_tokens=500, currency="EUR"
        )

        # (1000 / 1M) * 27.6 + (500 / 1M) * 55.2 = 0.0276 + 0.0276 = 0.0552
        assert abs(cost - 0.0552) < 0.0001


def test_compute_cost_sync_negative_tokens(sample_llmtracker_response):
    """Test sync compute_cost raises ValueError for negative token counts."""
    with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_llmtracker_response
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            compute_cost_sync("openai/gpt-4", input_tokens=-1, output_tokens=500)

        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            compute_cost_sync("openai/gpt-4", input_tokens=1000, output_tokens=-1)
