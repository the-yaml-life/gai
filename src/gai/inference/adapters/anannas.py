"""
Anannas provider adapter
"""

import re
import requests
from typing import Dict, List, Any, Optional
from gai.inference.exceptions import RateLimitError, ProviderError, BillingError
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

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        except requests.exceptions.Timeout as e:
            self._log(f"Timeout after 60s for {model}")
            raise ProviderError(
                f"Anannas API timeout after 60s: {str(e)}",
                status_code=None,
                provider="anannas"
            )
        except requests.exceptions.ConnectionError as e:
            self._log(f"Connection error for {model}: {e}")
            raise ProviderError(
                f"Anannas API connection error: {str(e)}",
                status_code=None,
                provider="anannas"
            )
        except requests.exceptions.RequestException as e:
            self._log(f"Request exception for {model}: {e}")
            raise ProviderError(
                f"Anannas API request error: {str(e)}",
                status_code=None,
                provider="anannas"
            )

        self._log(f"Response status: {response.status_code}")

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
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "")
            except Exception:
                error_msg = response.text[:200] if response.text else "Rate limit (no details)"

            self._log(f"Rate limit: {error_msg}")

            wait_match = re.search(r'try again in ([\d.]+)s', error_msg)
            wait_time = float(wait_match.group(1)) + 1 if wait_match else 60.0

            raise RateLimitError(error_msg, model=model, wait_time=wait_time)

        else:
            # Other error - try to extract meaningful error message
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                error_type = error_data.get("error", {}).get("type", "unknown")
                error_details = f"{error_type}: {error_msg}"
            except Exception:
                # Not JSON or malformed - use raw text
                error_details = response.text[:500] if response.text else "No error details"
                error_msg = error_details

            self._log(f"HTTP {response.status_code}: {error_details}")

            # Check if it's a billing error
            if "billing" in error_msg.lower() or "credit" in error_msg.lower() or "insufficient" in error_msg.lower():
                raise BillingError(error_details, provider="anannas")

            raise ProviderError(
                f"Anannas API HTTP {response.status_code}: {error_details}",
                status_code=response.status_code,
                provider="anannas"
            )

    def parse_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Anannas response"""
        return response["choices"][0]["message"]["content"]

    def get_usage(self, response: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract usage from Anannas response"""
        return response.get("usage")
