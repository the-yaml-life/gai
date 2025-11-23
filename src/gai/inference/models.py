"""
Data models for inference engine
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class InferenceRequest:
    """Request for inference generation"""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    max_tokens: int = 4096  # Reduced from 50000 to avoid payload errors
    temperature: float = 0.3

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> 'InferenceRequest':
        """Helper to create request from simple prompt"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return cls(messages=messages, model=model, **kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        """Get parameters for API request"""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


@dataclass
class InferenceResponse:
    """Response from inference generation"""
    text: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens from usage"""
        return self.usage.get("prompt_tokens", 0) if self.usage else 0

    @property
    def completion_tokens(self) -> int:
        """Get completion tokens from usage"""
        return self.usage.get("completion_tokens", 0) if self.usage else 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens from usage"""
        return self.usage.get("total_tokens", 0) if self.usage else 0
