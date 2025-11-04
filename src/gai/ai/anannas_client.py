"""
Anannas.ai API client with rate limiting and fallback

Anannas provides a unified gateway to 500+ models from multiple providers
through an OpenAI-compatible API.
"""

import requests
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from gai.ai.rate_limiter import RateLimiter
from gai.core.tokens import estimate_tokens
from gai.core.stats import get_stats


class AnannasError(Exception):
    """Anannas API error"""
    pass


class RateLimitError(AnannasError):
    """Rate limit exceeded"""
    def __init__(self, message: str, wait_time: float):
        super().__init__(message)
        self.wait_time = wait_time


class AnannasClient:
    """OpenAI-compatible Anannas client with rate limiting and multi-model fallback"""

    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: Optional[List[str]] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        verbose: bool = False,
        api_url: str = "https://api.anannas.ai/v1/chat/completions"
    ):
        self.api_key = api_key
        self.model = model
        self.fallback_models = fallback_models or []
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.api_url = api_url
        self.rate_limiter = RateLimiter()

    def _log(self, message: str):
        """Log if verbose"""
        if self.verbose:
            print(f"[AnannasClient] {message}")

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 500,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make API request"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self._log(f"Request to {model} ({max_tokens} max tokens)")

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})

            # Record usage
            tokens_used = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            self.rate_limiter.record_usage(model, tokens_used)

            # Record stats
            get_stats().record_api_call(
                model=model,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )

            self._log(f"Success: {prompt_tokens} in, {completion_tokens} out")

            return data

        elif response.status_code == 429:
            # Rate limit
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")

            # Extract wait time
            wait_match = re.search(r'try again in ([\d.]+)s', error_msg)
            wait_time = float(wait_match.group(1)) + 1 if wait_match else 60

            raise RateLimitError(error_msg, wait_time)

        else:
            # Other error
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = response.text or "Unknown error"

            raise AnannasError(f"API error ({response.status_code}): {error_msg}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        retry_with_fallback: bool = True
    ) -> str:
        """
        Generate completion with automatic retry and fallback.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature override
            retry_with_fallback: Whether to try fallback models on failure

        Returns:
            Generated text

        Raises:
            AnannasError: If all attempts fail
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(prompt)
        if system_prompt:
            estimated_tokens += estimate_tokens(system_prompt)
        estimated_tokens += max_tokens  # Add completion tokens

        # Try models in order: primary -> fallbacks
        models_to_try = [self.model] + (self.fallback_models if retry_with_fallback else [])

        last_error = None

        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    # Check rate limit
                    if not self.rate_limiter.can_use(model, estimated_tokens):
                        wait_time = self.rate_limiter.wait_time(model)
                        self._log(f"Rate limit approaching for {model}, waiting {wait_time:.1f}s")
                        time.sleep(wait_time + 1)

                    # Make request
                    response = self._make_request(messages, model, max_tokens, temperature)

                    # Extract content
                    content = response["choices"][0]["message"]["content"]
                    return content.strip()

                except RateLimitError as e:
                    self._log(f"Rate limit hit: {e.wait_time:.1f}s wait")

                    if attempt < self.max_retries - 1:
                        # Retry with same model after waiting
                        self._log(f"Waiting and retrying...")
                        time.sleep(e.wait_time)
                    else:
                        # Move to next model
                        last_error = e
                        break

                except AnannasError as e:
                    self._log(f"Error: {e}")
                    last_error = e

                    # Don't retry on non-rate-limit errors
                    break

                except Exception as e:
                    self._log(f"Unexpected error: {e}")
                    last_error = AnannasError(str(e))
                    break

            # If we got here without returning, try next model
            if model != models_to_try[-1]:
                self._log(f"Falling back to next model...")

        # All attempts failed
        raise last_error or AnannasError("All generation attempts failed")

    def get_usage_stats(self) -> Dict[str, Dict]:
        """Get current rate limit usage for all models"""
        stats = {}
        for model in [self.model] + self.fallback_models:
            stats[model] = self.rate_limiter.get_usage(model)
        return stats

    def list_models(self, raw=False) -> List[Dict[str, Any]]:
        """
        List available models from the API.

        Args:
            raw: If True, return raw API response without modifications

        Returns:
            List of model information dicts (tier info only if API provides it)
        """
        # Get base URL from api_url
        base_url = self.api_url.rsplit('/', 2)[0]  # Remove /chat/completions
        models_url = f"{base_url}/models"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                # If raw mode, return unmodified response
                if raw:
                    return models

                # Sort by id
                models.sort(key=lambda m: m.get("id", ""))

                return models
            else:
                self._log(f"Failed to list models: {response.status_code}")
                return []

        except Exception as e:
            self._log(f"Error listing models: {e}")
            return []

    def generate_parallel(
        self,
        prompts: List[Tuple[str, str, int]],
        models: List[str],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate completions in parallel using different models with intelligent retry.

        Args:
            prompts: List of (prompt_text, description, max_tokens) tuples
            models: List of model names to use (pool to rotate from)
            system_prompt: Optional system prompt for all requests

        Returns:
            List of generated texts (same order as prompts)

        Raises:
            AnannasError: If all models have rate limit > 10 minutes
        """
        if len(prompts) > len(models):
            raise AnannasError(f"Not enough models ({len(models)}) for prompts ({len(prompts)})")

        self._log(f"Parallel generation: {len(prompts)} prompts with {len(models)} models available")

        # Track pending tasks: {idx: (prompt_data, models_tried)}
        pending = {idx: (prompt_data, []) for idx, prompt_data in enumerate(prompts)}
        results = {}

        MAX_WAIT_TIME = 600  # 10 minutes in seconds

        while pending:
            # Find model with shortest wait time for each pending task
            tasks_to_run = []

            for idx, (prompt_data, tried_models) in pending.items():
                # Find best available model (not tried yet, shortest wait time)
                available_models = [m for m in models if m not in tried_models]

                if not available_models:
                    # All models tried, check wait times
                    min_wait = min(self.rate_limiter.wait_time(m) for m in models)

                    if min_wait > MAX_WAIT_TIME:
                        # All models rate limited for > 10min, fail
                        raise AnannasError(
                            f"All {len(models)} models have rate limit > 10 minutes. "
                            f"Shortest wait: {min_wait/60:.1f}min"
                        )

                    # Wait and reset tried list
                    self._log(f"[{idx}] All models tried, waiting {min_wait:.1f}s...")
                    time.sleep(min_wait + 1)
                    tried_models.clear()
                    available_models = models[:]

                # Pick model with shortest wait time
                best_model = min(available_models, key=lambda m: self.rate_limiter.wait_time(m))
                wait_time = self.rate_limiter.wait_time(best_model)

                if wait_time > 0:
                    self._log(f"[{idx}] Waiting {wait_time:.1f}s for {best_model}")
                    time.sleep(wait_time + 1)

                tasks_to_run.append((idx, prompt_data, best_model))

            # Execute batch in parallel
            def generate_one(idx: int, prompt_data: Tuple[str, str, int], model: str) -> Tuple[int, str, bool]:
                """Generate one completion and return (index, result, success)"""
                prompt_text, description, max_tokens = prompt_data
                self._log(f"[{idx}] {description} using {model}")

                try:
                    # Build messages
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt_text})

                    # Make request directly (bypass generate() to avoid fallback)
                    response = self._make_request(messages, model, max_tokens, self.temperature)
                    content = response["choices"][0]["message"]["content"]

                    self._log(f"[{idx}] ✓ Complete with {model}")
                    return (idx, content.strip(), True)

                except RateLimitError as e:
                    self._log(f"[{idx}] Rate limit on {model}: {e.wait_time:.1f}s wait")
                    return (idx, "", False)

                except Exception as e:
                    self._log(f"[{idx}] Error with {model}: {e}")
                    return (idx, "", False)

            # Run tasks in parallel
            with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
                futures = []
                for idx, prompt_data, model in tasks_to_run:
                    future = executor.submit(generate_one, idx, prompt_data, model)
                    futures.append((future, idx, model))

                # Collect results
                for future, idx, model in futures:
                    result_idx, result_text, success = future.result()

                    if success:
                        # Task completed successfully
                        results[result_idx] = result_text
                        if result_idx in pending:
                            del pending[result_idx]
                    else:
                        # Task failed, mark model as tried and retry
                        if result_idx in pending:
                            pending[result_idx][1].append(model)

        # Return in original order
        return [results[i] for i in range(len(prompts))]
