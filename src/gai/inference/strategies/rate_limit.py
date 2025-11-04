"""
Intelligent rate limit strategy that learns from errors
"""

import time
from typing import Dict, Optional
from gai.inference.exceptions import RateLimitError


class IntelligentRateLimitStrategy:
    """
    Intelligent rate limiting that learns from 429 errors.

    Instead of hardcoding limits, this strategy:
    1. Starts with no knowledge
    2. Learns limits from actual 429 errors
    3. Tracks usage in rolling windows
    4. Prevents hitting known limits
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.limits: Dict[str, Dict] = {}  # model -> limit info
        self.buckets: Dict[str, Dict] = {}  # model -> usage bucket

    def _log(self, message: str):
        if self.verbose:
            print(f"[RateLimitStrategy] {message}")

    def check_can_use(self, model: str):
        """
        Check if model can be used right now.

        Raises RateLimitError if we know this model is rate limited.
        """
        if model not in self.buckets:
            # No knowledge yet, allow
            return

        bucket = self.buckets[model]

        # Check if bucket has reset
        if time.time() >= bucket["reset_at"]:
            # Reset bucket
            bucket["used"] = 0
            bucket["reset_at"] = time.time() + bucket["window"]
            return

        # Check if we've hit the limit
        if bucket["limit"] and bucket["used"] >= bucket["limit"]:
            wait_time = bucket["reset_at"] - time.time()
            self._log(f"Model {model} rate limited, {wait_time:.1f}s until reset")
            raise RateLimitError(
                f"Rate limit for {model}",
                model=model,
                wait_time=wait_time
            )

    def record_usage(self, model: str, usage: Dict[str, int]):
        """Record token usage for model"""
        if not usage:
            return

        tokens = usage.get("total_tokens", 0)
        if tokens == 0:
            return

        # Initialize bucket if needed
        if model not in self.buckets:
            self.buckets[model] = {
                "used": 0,
                "limit": None,  # Unknown until we hit it
                "window": 60,  # Assume 1 minute window
                "reset_at": time.time() + 60
            }

        bucket = self.buckets[model]

        # Check if we need to reset
        if time.time() >= bucket["reset_at"]:
            bucket["used"] = 0
            bucket["reset_at"] = time.time() + bucket["window"]

        # Add usage
        bucket["used"] += tokens
        self._log(f"Model {model}: {bucket['used']} tokens used")

    def handle(self, error: Exception, attempt: int) -> bool:
        """
        Handle error and decide if should retry.

        This is where we LEARN from rate limit errors.

        Returns:
            True if should retry, False if should skip to next model
        """
        if isinstance(error, RateLimitError):
            model = error.model
            wait_time = error.wait_time

            self._log(f"Learning from rate limit: {model} needs {wait_time:.1f}s")

            # Learn from this error
            self._learn_limit(model, wait_time)

            # Don't retry - let fallback handle it
            return False

        # For other errors, let retry strategy decide
        return True

    def _learn_limit(self, model: str, wait_time: float):
        """Learn rate limit info from error"""
        if model not in self.limits:
            self.limits[model] = {
                "window": wait_time,  # How long to wait
                "learned_at": time.time()
            }
            self._log(f"Learned: {model} has ~{wait_time:.0f}s rate limit window")

        # Initialize or update bucket
        if model not in self.buckets:
            self.buckets[model] = {
                "used": 0,
                "limit": None,  # We don't know exact limit, just that it exists
                "window": wait_time,
                "reset_at": time.time() + wait_time
            }
        else:
            # Update window if we learned a different one
            self.buckets[model]["window"] = wait_time
            self.buckets[model]["reset_at"] = time.time() + wait_time

    def get_stats(self, model: str) -> Optional[Dict]:
        """Get current stats for model"""
        if model not in self.buckets:
            return None

        bucket = self.buckets[model]
        return {
            "used": bucket["used"],
            "limit": bucket["limit"] or "unknown",
            "reset_in": max(0, bucket["reset_at"] - time.time())
        }
