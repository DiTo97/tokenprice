"""tokenprice - LLM token pricing library.

Public API:
- get_pricing(model_id)
- compute_cost(model_id, input_tokens, output_tokens)

Data source: LLMTracker (https://github.com/MrUnreal/LLMTracker)
Website: https://mrunreal.github.io/LLMTracker/
"""

from tokenprice.core import compute_cost, get_pricing

# Version will be set by package manager
__version__ = "0.1.0"

__all__ = [
    "__version__",
    "get_pricing",
    "compute_cost",
]
