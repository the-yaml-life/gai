"""
OpenRouter API client

OpenRouter provides a unified gateway to 300+ models from multiple providers
through an OpenAI-compatible API.

Note: OpenRouter manages rate limiting server-side, so this client does not
implement client-side rate limiting (similar to OllamaClient).
"""

import requests
import time
import re
from typing import Dict, List, Optional, Any
from gai.core.tokens import estimate_tokens
from gai.core.stats import get_stats


class OpenRouterError(Exception):
    """OpenRouter API error"""
    pass


class RateLimitError(OpenRouterError):
    """Rate limit exceeded"""
    def __init__(self, message: str, wait_time: float):
        super().__init__(message)
        self.wait_time = wait_time


class OpenRouterClient:
    """OpenAI-compatible OpenRouter client with multi-model fallback (no client-side rate limiting)"""

    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: Optional[List[str]] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        verbose: bool = False,
        api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ):
        self.api_key = api_key
        self.model = model
        self.fallback_models = fallback_models or []
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.api_url = api_url

    def _log(self, message: str):
        """Log if verbose"""
        if self.verbose:
            print(f"[OpenRouterClient] {message}")

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
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/gai",  # Optional: for rankings
            "X-Title": "gai - Git AI Assistant"  # Optional: for rankings
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

            # Record stats
            get_stats().record_api_call(
                model=f"openrouter/{model}",
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

            # Extract wait time from error message if available
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

            raise OpenRouterError(f"API error ({response.status_code}): {error_msg}")

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
            OpenRouterError: If all attempts fail
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Try models in order: primary -> fallbacks
        models_to_try = [self.model] + (self.fallback_models if retry_with_fallback else [])

        last_error = None

        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    # Make request (no client-side rate limiting for OpenRouter)
                    response = self._make_request(messages, model, max_tokens, temperature)

                    # Extract content
                    content = response["choices"][0]["message"]["content"]
                    return content.strip()

                except RateLimitError as e:
                    self._log(f"Rate limit hit: {e.wait_time:.1f}s wait")
                    last_error = e

                    # Don't wait - immediately try next model in fallback list
                    # This is especially important for free models with low rate limits
                    self._log(f"Skipping to next model...")
                    break

                except OpenRouterError as e:
                    self._log(f"Error: {e}")
                    last_error = e

                    # Don't retry on non-rate-limit errors
                    break

                except Exception as e:
                    self._log(f"Unexpected error: {e}")
                    last_error = OpenRouterError(str(e))
                    break

            # If we got here without returning, try next model
            if model != models_to_try[-1]:
                self._log(f"Falling back to next model...")

        # All attempts failed
        raise last_error or OpenRouterError("All generation attempts failed")

    def get_usage_stats(self) -> Dict[str, Dict]:
        """
        Get current rate limit usage for all models.

        Note: OpenRouter doesn't use client-side rate limiting,
        so this returns empty stats.
        """
        return {}

    def list_models(self, raw=False) -> List[Dict[str, Any]]:
        """
        List available models from the API.

        Args:
            raw: If True, return raw API response without modifications

        Returns:
            List of model information dicts
        """
        # OpenRouter models endpoint
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
