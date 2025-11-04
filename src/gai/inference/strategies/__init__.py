"""
Inference strategies
"""

from gai.inference.strategies.retry import SimpleRetryStrategy
from gai.inference.strategies.fallback import SimpleFallbackStrategy
from gai.inference.strategies.rate_limit import IntelligentRateLimitStrategy

__all__ = [
    'SimpleRetryStrategy',
    'SimpleFallbackStrategy',
    'IntelligentRateLimitStrategy',
]
