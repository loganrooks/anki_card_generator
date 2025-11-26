"""Google Gemini LLM provider (FREE tier available).

Gemini offers a generous free tier:
- 15 requests per minute
- 1 million tokens per day
- 1,500 requests per day

This makes it excellent for personal/educational use.
See: https://ai.google.dev/pricing
"""

import logging
import os
import time
from typing import Optional

from .base import LLMProvider, LLMResponse
from ..core.config import ProviderConfig
from ..core.exceptions import ProviderError

logger = logging.getLogger(__name__)

# Gemini pricing (as of 2024) - free tier has no cost
GEMINI_PRICING = {
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},  # Paid
    "gemini-1.0-pro": {"input": 0.0, "output": 0.0},  # Free tier
}


class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Free tier includes:
    - gemini-1.5-flash: Fast, good for most tasks
    - gemini-1.0-pro: Stable, well-tested

    Setup:
    1. Get API key from https://makersuite.google.com/app/apikey
    2. Set GOOGLE_API_KEY environment variable
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

        # Default to free model
        if not config.model:
            config.model = "gemini-1.5-flash"

        # Set pricing
        pricing = GEMINI_PRICING.get(config.model, {"input": 0.0, "output": 0.0})
        config.cost_per_1k_input = pricing["input"]
        config.cost_per_1k_output = pricing["output"]

    def _get_client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ProviderError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai",
                    provider="gemini",
                    retryable=False,
                )

            api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ProviderError(
                    "Google API key not found. Get one free at "
                    "https://makersuite.google.com/app/apikey and set GOOGLE_API_KEY",
                    provider="gemini",
                    retryable=False,
                )

            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model)

        return self._client

    def get_name(self) -> str:
        return "gemini"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using Gemini API."""
        client = self._get_client()

        # Combine system and user prompts (Gemini doesn't have separate system prompt)
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        start_time = time.time()

        try:
            response = client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_output_tokens": self.config.max_tokens,
                },
            )
        except Exception as e:
            error_str = str(e).lower()
            retryable = any(x in error_str for x in ["rate", "quota", "resource", "503", "429"])
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                retryable=retryable,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        content = response.text if response.text else ""

        # Gemini provides token counts
        input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        estimated_cost = self._calculate_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=content.strip(),
            model=self.config.model,
            provider="gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
        )
