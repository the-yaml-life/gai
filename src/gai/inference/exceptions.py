"""
Unified exceptions for inference engine
"""


class InferenceError(Exception):
    """Base exception for all inference errors"""
    pass


class RateLimitError(InferenceError):
    """Rate limit exceeded"""
    def __init__(self, message: str, model: str = None, wait_time: float = 60.0):
        super().__init__(message)
        self.model = model
        self.wait_time = wait_time


class ProviderError(InferenceError):
    """Provider-specific error (non-rate-limit)"""
    def __init__(self, message: str, status_code: int = None, provider: str = None, skip_retry: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.skip_retry = skip_retry  # If True, don't retry this error


class BillingError(ProviderError):
    """Billing/credit error - don't retry"""
    def __init__(self, message: str, provider: str = None):
        super().__init__(message, provider=provider, skip_retry=True)


class AllModelsFailedError(InferenceError):
    """All models in fallback chain failed"""
    def __init__(self, errors: dict = None):
        self.errors = errors or {}
        models = list(errors.keys()) if errors else []

        # Create detailed error message
        if errors:
            error_summary = f"All {len(models)} models failed:\n"
            for model, error in list(errors.items())[:5]:  # Show first 5
                error_str = str(error)[:100]  # Truncate long errors
                error_summary += f"  • {model}: {error_str}\n"
            if len(errors) > 5:
                error_summary += f"  ... and {len(errors) - 5} more\n"
            super().__init__(error_summary)
        else:
            super().__init__(f"All {len(models)} models failed")
