"""
Rate limit tracking for Groq API
"""

import time
from typing import Dict


class RateLimiter:
    """Track TPM usage per model"""

    # Model TPM limits (free tier)
    MODEL_LIMITS = {
        "meta-llama/llama-4-scout-17b-16e-instruct": 30_000,
        "llama-3.3-70b-versatile": 12_000,
        "llama-3.1-8b-instant": 6_000,
    }

    def __init__(self):
        self.buckets: Dict[str, Dict] = {}

    def _get_bucket(self, model: str) -> Dict:
        """Get or create bucket for model"""
        if model not in self.buckets:
            self.buckets[model] = {
                "limit": self.MODEL_LIMITS.get(model, 30_000),
                "used": 0,
                "reset_at": time.time() + 60
            }
        return self.buckets[model]

    def _reset_if_needed(self, bucket: Dict):
        """Reset bucket if minute has passed"""
        if time.time() >= bucket["reset_at"]:
            bucket["used"] = 0
            bucket["reset_at"] = time.time() + 60

    def can_use(self, model: str, estimated_tokens: int) -> bool:
        """Check if we can use this many tokens"""
        bucket = self._get_bucket(model)
        self._reset_if_needed(bucket)

        return bucket["used"] + estimated_tokens <= bucket["limit"]

    def wait_time(self, model: str) -> float:
        """Get seconds to wait until bucket resets"""
        bucket = self._get_bucket(model)
        return max(0, bucket["reset_at"] - time.time())

    def record_usage(self, model: str, tokens_used: int):
        """Record tokens used"""
        bucket = self._get_bucket(model)
        self._reset_if_needed(bucket)
        bucket["used"] += tokens_used

    def get_usage(self, model: str) -> Dict:
        """Get current usage for model"""
        bucket = self._get_bucket(model)
        self._reset_if_needed(bucket)

        return {
            "used": bucket["used"],
            "limit": bucket["limit"],
            "remaining": bucket["limit"] - bucket["used"],
            "reset_in": self.wait_time(model)
        }
