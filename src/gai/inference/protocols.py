"""
Protocol definitions for inference components
"""

from typing import Protocol, Dict, List, Any, Optional
from gai.inference.models import InferenceRequest, InferenceResponse


class ProviderAdapter(Protocol):
    """Interface that all provider adapters must implement"""

    def request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request to provider and return raw response.

        Should raise:
        - RateLimitError if rate limit hit
        - ProviderError for other errors
        """
        ...

    def parse_response(self, response: Dict[str, Any]) -> str:
        """Parse provider response to extract text content"""
        ...

    def get_usage(self, response: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract token usage from response"""
        ...


class RetryStrategy(Protocol):
    """Interface for retry strategies"""

    def execute(self, fn, on_error=None):
        """Execute function with retry logic"""
        ...


class FallbackStrategy(Protocol):
    """Interface for fallback strategies"""

    def get_models(self, primary_model: Optional[str]) -> List[str]:
        """Get ordered list of models to try"""
        ...


class RateLimitStrategy(Protocol):
    """Interface for rate limit strategies"""

    def check_can_use(self, model: str):
        """Check if model can be used (raise if not)"""
        ...

    def record_usage(self, model: str, usage: Dict[str, int]):
        """Record token usage for model"""
        ...

    def handle(self, error: Exception, attempt: int) -> bool:
        """Handle error, return True if should retry"""
        ...
