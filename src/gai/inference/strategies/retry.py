"""
Retry strategy implementation
"""

import time
from typing import Callable, Optional


class SimpleRetryStrategy:
    """Simple retry strategy with exponential backoff"""

    def __init__(self, max_retries: int = 3, backoff_base: float = 2.0, verbose: bool = False):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[RetryStrategy] {message}")

    def _calculate_wait(self, attempt: int) -> float:
        """Calculate exponential backoff wait time"""
        return self.backoff_base ** attempt

    def execute(self, fn: Callable, on_error: Optional[Callable] = None):
        """
        Execute function with retry logic.

        Args:
            fn: Function to execute
            on_error: Optional callback(error, attempt) -> should_retry

        Returns:
            Result from fn()

        Raises:
            Last exception if all retries exhausted
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return fn()

            except Exception as e:
                last_error = e
                self._log(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                # Call error handler if provided
                if on_error:
                    should_retry = on_error(e, attempt)
                    if not should_retry:
                        self._log("Error handler says: don't retry")
                        raise

                # If this was last attempt, raise
                if attempt >= self.max_retries - 1:
                    self._log("Max retries exhausted")
                    raise

                # Wait before retry
                wait_time = self._calculate_wait(attempt)
                self._log(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

        # Should never reach here, but just in case
        raise last_error or Exception("Retry failed with no error?")
