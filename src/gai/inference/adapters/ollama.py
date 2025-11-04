"""
Ollama provider adapter
"""

import requests
from typing import Dict, List, Any, Optional
from gai.inference.exceptions import ProviderError
from gai.core.stats import get_stats


class OllamaAdapter:
    """Ollama API adapter - handles HTTP requests and response parsing"""

    def __init__(self, base_url: str = "http://localhost:11434", verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/chat"
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[OllamaAdapter] {message}")

    def request(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to Ollama API"""
        self._log(f"Request to {model} at {self.base_url}")

        # Ollama uses slightly different format
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.3),
                "num_predict": kwargs.get("max_tokens", 50000)
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                data = response.json()

                # Extract usage (Ollama format is different)
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens

                # Record stats
                get_stats().record_api_call(
                    model=f"ollama/{model}",
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )

                self._log(f"Success: {prompt_tokens} in, {completion_tokens} out")

                # Convert to standard format
                return {
                    "choices": [{
                        "message": {
                            "content": data.get("message", {}).get("content", "")
                        }
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }

            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                except:
                    error_msg = response.text or "Unknown error"

                raise ProviderError(
                    f"Ollama error: {error_msg}",
                    status_code=response.status_code,
                    provider="ollama"
                )

        except requests.exceptions.ConnectionError:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.",
                provider="ollama"
            )
        except requests.exceptions.Timeout:
            raise ProviderError("Ollama request timed out", provider="ollama")

    def parse_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Ollama response"""
        return response["choices"][0]["message"]["content"]

    def get_usage(self, response: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract usage from Ollama response"""
        return response.get("usage")
