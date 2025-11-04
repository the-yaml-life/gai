"""
Ollama API client compatible with GroqClient interface
"""

import requests
import time
from typing import Dict, List, Optional, Any
from gai.core.tokens import estimate_tokens
from gai.core.stats import get_stats


class OllamaError(Exception):
    """Ollama API error"""
    pass


class OllamaClient:
    """Ollama API client with same interface as GroqClient"""

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        verbose: bool = False,
        base_url: str = "http://localhost:11434"
    ):
        # Strip "ollama/" prefix if present
        self.model = model.replace("ollama/", "")
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/chat"

    def _log(self, message: str):
        """Log if verbose"""
        if self.verbose:
            print(f"[OllamaClient] {message}")

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 500,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make API request to Ollama"""
        payload = {
            "model": model.replace("ollama/", ""),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens
            }
        }

        self._log(f"Request to Ollama:{model} ({max_tokens} max tokens)")

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Ollama can be slower
            )

            if response.status_code == 200:
                data = response.json()

                # Ollama response format:
                # {
                #   "model": "llama3",
                #   "message": {"role": "assistant", "content": "..."},
                #   "done": true,
                #   "total_duration": 123456,
                #   "prompt_eval_count": 10,
                #   "eval_count": 20
                # }

                content = data.get("message", {}).get("content", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                tokens_used = prompt_tokens + completion_tokens

                # Record stats
                get_stats().record_api_call(
                    model=f"ollama/{model}",
                    tokens_used=tokens_used,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )

                self._log(f"Success: {prompt_tokens} in, {completion_tokens} out")

                # Return in GroqClient-compatible format
                return {
                    "choices": [{
                        "message": {
                            "content": content
                        }
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": tokens_used
                    }
                }

            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                except:
                    error_msg = response.text or "Unknown error"

                raise OllamaError(f"Ollama error ({response.status_code}): {error_msg}")

        except requests.exceptions.ConnectionError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)"
            )
        except requests.exceptions.Timeout:
            raise OllamaError("Ollama request timed out")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        retry_with_fallback: bool = True
    ) -> str:
        """
        Generate completion with automatic retry.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature override
            retry_with_fallback: Ignored for Ollama (no fallback concept)

        Returns:
            Generated text

        Raises:
            OllamaError: If generation fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._make_request(messages, self.model, max_tokens, temperature)
                content = response["choices"][0]["message"]["content"]
                return content.strip()

            except OllamaError as e:
                self._log(f"Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                last_error = e

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self._log(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

            except Exception as e:
                self._log(f"Unexpected error: {e}")
                raise OllamaError(str(e))

        # All retries failed
        raise last_error or OllamaError("All generation attempts failed")

    def list_models(self, raw=False) -> List[Dict[str, Any]]:
        """
        List available models from Ollama.

        Args:
            raw: If True, return raw API response without modifications

        Returns:
            List of model information dicts
        """
        models_url = f"{self.base_url}/api/tags"

        try:
            response = requests.get(models_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                # If raw mode, return unmodified response
                if raw:
                    return models

                # Format to match GroqClient output
                formatted = []
                for model in models:
                    formatted.append({
                        "id": f"ollama/{model['name']}",
                        "name": model['name'],
                        "size": model.get('size', 0),
                        "modified_at": model.get('modified_at', ''),
                        "tier": "local"  # All Ollama models run locally
                    })

                return formatted
            else:
                self._log(f"Failed to list Ollama models: {response.status_code}")
                return []

        except requests.exceptions.ConnectionError:
            self._log(f"Cannot connect to Ollama at {self.base_url}")
            return []
        except Exception as e:
            self._log(f"Error listing Ollama models: {e}")
            return []
