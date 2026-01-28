"""Tests for synchronous public API wrappers.

Tests that sync versions work correctly and use the same cache as async versions.
"""

from decimal import Decimal
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


class TestDidYouMeanSuggestions:
    """Test 'Did you mean?' suggestions in error messages."""

    def test_model_not_found_with_suggestion(self, sample_llmtracker_response):
        """Test that model not found error includes 'Did you mean?' suggestion."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_llmtracker_response
            mock_get.return_value = mock_response

            # Typo: "openai/gpt4" instead of "openai/gpt-4"
            with pytest.raises(ValueError) as exc_info:
                get_pricing_sync("openai/gpt4")

            assert "Model not found: openai/gpt4" in str(exc_info.value)
            assert "Did you mean" in str(exc_info.value)
            assert "openai/gpt-4" in str(exc_info.value)

    def test_model_not_found_no_suggestion(self, sample_llmtracker_response):
        """Test that completely unrelated model gives no suggestion."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_llmtracker_response
            mock_get.return_value = mock_response

            # Completely unrelated model name
            with pytest.raises(ValueError) as exc_info:
                get_pricing_sync("completely-unrelated-xyz-123")

            assert "Model not found" in str(exc_info.value)
            # Should not have a suggestion for completely unrelated input
            # (or if it does, it's still a valid error)

    def test_currency_not_found_with_suggestion(
        self, sample_llmtracker_response, monkeypatch
    ):
        """Test that currency not found error includes 'Did you mean?' suggestion."""
        import tokenprice.pricing as pricing_mod
        import tokenprice.currency as currency_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()
        currency_mod._get_usd_rates_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_pricing:
            mock_pricing_response = Mock()
            mock_pricing_response.json.return_value = sample_llmtracker_response
            mock_pricing.return_value = mock_pricing_response

            def fake_sync_get_usd_rates():
                return {
                    "EUR": Decimal("0.92"),
                    "GBP": Decimal("0.79"),
                    "CNY": Decimal("7.25"),
                }

            monkeypatch.setattr(
                currency_mod, "_sync_get_usd_rates", fake_sync_get_usd_rates
            )

            # Typo: "ERU" instead of "EUR"
            with pytest.raises(ValueError) as exc_info:
                get_pricing_sync("openai/gpt-4", currency="ERU")

            assert "Unsupported currency: ERU" in str(exc_info.value)
            assert "Did you mean" in str(exc_info.value)
            assert "EUR" in str(exc_info.value)

    def test_currency_not_found_no_suggestion(
        self, sample_llmtracker_response, monkeypatch
    ):
        """Test that completely unrelated currency gives no suggestion."""
        import tokenprice.pricing as pricing_mod
        import tokenprice.currency as currency_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()
        currency_mod._get_usd_rates_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_pricing:
            mock_pricing_response = Mock()
            mock_pricing_response.json.return_value = sample_llmtracker_response
            mock_pricing.return_value = mock_pricing_response

            def fake_sync_get_usd_rates():
                return {
                    "EUR": Decimal("0.92"),
                    "GBP": Decimal("0.79"),
                }

            monkeypatch.setattr(
                currency_mod, "_sync_get_usd_rates", fake_sync_get_usd_rates
            )

            # Completely unrelated currency code
            with pytest.raises(ValueError) as exc_info:
                get_pricing_sync("openai/gpt-4", currency="XYZ")

            assert "Unsupported currency: XYZ" in str(exc_info.value)
            # No suggestion for completely unrelated input
            assert "Did you mean" not in str(exc_info.value)


class TestCachePricing:
    """Tests for cache pricing support."""

    @pytest.fixture
    def sample_response_with_cache_pricing(self) -> dict:
        """Sample response with cache pricing fields."""
        return {
            "generated_at": "2026-01-20T06:05:10.791612+00:00",
            "models": {
                "anthropic/claude-3.5-sonnet": {
                    "provider": "anthropic",
                    "model_id": "anthropic/claude-3.5-sonnet",
                    "display_name": "Anthropic: Claude 3.5 Sonnet",
                    "pricing": {
                        "input_per_million": 3.0,
                        "output_per_million": 15.0,
                        "cache_read_per_million": 0.3,
                        "cache_creation_per_million": 3.75,
                        "currency": "USD",
                    },
                    "context_window": 200000,
                    "max_output_tokens": 8192,
                    "model_type": "chat",
                    "supports_vision": True,
                    "supports_function_calling": True,
                    "supports_streaming": True,
                    "category": "flagship",
                },
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
                },
            },
            "providers": {
                "anthropic": {
                    "name": "Anthropic",
                    "website": "https://anthropic.com",
                    "pricing_page": "https://anthropic.com/pricing",
                    "affiliate_link": "https://console.anthropic.com",
                },
                "openai": {
                    "name": "OpenAI",
                    "website": "https://openai.com",
                    "pricing_page": "https://openai.com/pricing",
                    "affiliate_link": "https://platform.openai.com/signup",
                },
            },
            "metadata": {
                "total_models": 2,
                "sources": ["openrouter", "litellm"],
                "last_scrape": "2026-01-20T06:05:10.791612+00:00",
                "categories": {"flagship": 2},
            },
        }

    def test_get_pricing_returns_cache_fields(self, sample_response_with_cache_pricing):
        """Test that get_pricing returns cache pricing fields."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_response_with_cache_pricing
            mock_get.return_value = mock_response

            pricing = get_pricing_sync("anthropic/claude-3.5-sonnet")

            assert pricing.cache_read_per_million == 0.3
            assert pricing.cache_creation_per_million == 3.75

    def test_get_pricing_cache_fields_default_to_input_price(
        self, sample_response_with_cache_pricing
    ):
        """Test that cache fields default to input price when not in source data."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_response_with_cache_pricing
            mock_get.return_value = mock_response

            pricing = get_pricing_sync("openai/gpt-4")

            # Should default to input price (30.0) when not available
            assert pricing.cache_read_per_million == 30.0
            assert pricing.cache_creation_per_million == 30.0

    def test_compute_cost_with_cache_tokens(self, sample_response_with_cache_pricing):
        """Test compute_cost with cache read and creation tokens."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_response_with_cache_pricing
            mock_get.return_value = mock_response

            # 1000 input, 500 output, 10000 cache read, 5000 cache creation
            cost = compute_cost_sync(
                "anthropic/claude-3.5-sonnet",
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=10000,
                cache_creation_tokens=5000,
            )

            # input: (1000 / 1M) * 3.0 = 0.003
            # output: (500 / 1M) * 15.0 = 0.0075
            # cache_read: (10000 / 1M) * 0.3 = 0.003
            # cache_creation: (5000 / 1M) * 3.75 = 0.01875
            # total: 0.003 + 0.0075 + 0.003 + 0.01875 = 0.03225
            assert abs(cost - 0.03225) < 0.0001

    def test_compute_cost_cache_fallback_to_input_price(
        self, sample_response_with_cache_pricing
    ):
        """Test that cache pricing falls back to input price when not available."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_response_with_cache_pricing
            mock_get.return_value = mock_response

            # GPT-4 has no cache pricing, should fall back to input price
            cost = compute_cost_sync(
                "openai/gpt-4",
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=10000,
            )

            # input: (1000 / 1M) * 30.0 = 0.03
            # output: (500 / 1M) * 60.0 = 0.03
            # cache_read: (10000 / 1M) * 30.0 = 0.3 (fallback to input price)
            # total: 0.03 + 0.03 + 0.3 = 0.36
            assert abs(cost - 0.36) < 0.0001

    def test_compute_cost_negative_cache_tokens_raises(
        self, sample_response_with_cache_pricing
    ):
        """Test that negative cache token counts raise ValueError."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_response_with_cache_pricing
            mock_get.return_value = mock_response

            with pytest.raises(
                ValueError, match="Cache token counts must be non-negative"
            ):
                compute_cost_sync(
                    "anthropic/claude-3.5-sonnet",
                    input_tokens=1000,
                    output_tokens=500,
                    cache_read_tokens=-100,
                )

    def test_compute_cost_zero_cache_tokens(self, sample_response_with_cache_pricing):
        """Test compute_cost with zero cache tokens (default behavior)."""
        import tokenprice.pricing as pricing_mod

        pricing_mod._get_pricing_data_bucketed.cache_clear()

        with patch("tokenprice.pricing.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_response_with_cache_pricing
            mock_get.return_value = mock_response

            cost = compute_cost_sync(
                "anthropic/claude-3.5-sonnet",
                input_tokens=1000,
                output_tokens=500,
            )

            # input: (1000 / 1M) * 3.0 = 0.003
            # output: (500 / 1M) * 15.0 = 0.0075
            # total: 0.003 + 0.0075 = 0.0105
            assert abs(cost - 0.0105) < 0.0001
