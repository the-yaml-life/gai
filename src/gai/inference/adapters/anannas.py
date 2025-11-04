"""
Anannas provider adapter
"""

import re
import requests
from typing import Dict, List, Any, Optional
from gai.inference.exceptions import RateLimitError, ProviderError
from gai.core.stats import get_stats


class AnannasAdapter:
    """Anannas API adapter - handles HTTP requests and response parsing"""

    def __init__(self, api_key: str, api_url: str = "https://api.anannas.ai/v1/chat/completions", verbose: bool = False):
        self.api_key = api_key
        self.api_url = api_url
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[AnannasAdapter] {message}")

    def request(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to Anannas API"""
        self._log(f"Request to {model}")

        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()

            # Record stats
            usage = data.get("usage", {})
            get_stats().record_api_call(
                model=f"anannas/{model}",
                tokens_used=usage.get("total_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0)
            )

            self._log(f"Success: {usage.get('prompt_tokens', 0)} in, {usage.get('completion_tokens', 0)} out")
            return data

        elif response.status_code == 429:
            # Rate limit - extract wait time
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")

            wait_match = re.search(r'try again in ([\d.]+)s', error_msg)
            wait_time = float(wait_match.group(1)) + 1 if wait_match else 60.0

            raise RateLimitError(error_msg, model=model, wait_time=wait_time)

        else:
            # Other error
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = response.text or "Unknown error"

            raise ProviderError(
                f"Anannas API error: {error_msg}",
                status_code=response.status_code,
                provider="anannas"
            )

    def parse_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Anannas response"""
        return response["choices"][0]["message"]["content"]

    def get_usage(self, response: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract usage from Anannas response"""
        return response.get("usage")
