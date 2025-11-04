"""
Fallback strategy implementation
"""

from typing import List, Optional


class SimpleFallbackStrategy:
    """Simple fallback strategy - use parallel_models list"""

    def __init__(self, parallel_models: List[str], verbose: bool = False):
        self.parallel_models = parallel_models or []
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[FallbackStrategy] {message}")

    def get_models(self, primary_model: Optional[str] = None) -> List[str]:
        """
        Get ordered list of models to try.

        If primary_model is provided, try it first, then parallel_models.
        Otherwise, just use parallel_models.
        """
        if primary_model:
            models = [primary_model] + [m for m in self.parallel_models if m != primary_model]
        else:
            models = self.parallel_models

        if not models:
            raise ValueError("No models configured for inference")

        self._log(f"Fallback chain: {models}")
        return models
