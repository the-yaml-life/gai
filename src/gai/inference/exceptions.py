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
    def __init__(self, message: str, status_code: int = None, provider: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class AllModelsFailedError(InferenceError):
    """All models in fallback chain failed"""
    def __init__(self, errors: dict = None):
        self.errors = errors or {}
        models = list(errors.keys()) if errors else []
        super().__init__(f"All {len(models)} models failed: {models}")
