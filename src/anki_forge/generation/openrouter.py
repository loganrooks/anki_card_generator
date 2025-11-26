"""OpenRouter LLM provider.

OpenRouter provides a unified API for many models:
- Claude (Anthropic)
- GPT-4, GPT-3.5 (OpenAI)
- Llama, Mistral, Mixtral (Meta, Mistral AI)
- Gemini (Google)
- And many more

Benefits:
- Single API for all models
- Often cheaper than direct APIs
- Free models available (with rate limits)
- Easy model switching

See: https://openrouter.ai/docs
"""

import logging
import os
import time
from typing import Optional

from .base import LLMProvider, LLMResponse
from ..core.config import ProviderConfig
from ..core.exceptions import ProviderError

logger = logging.getLogger(__name__)

# Popular models and their pricing (per 1K tokens)
OPENROUTER_MODELS = {
    # Free models
    "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
    "mistralai/mistral-7b-instruct:free": {"input": 0.0, "output": 0.0},
    "google/gemma-2-9b-it:free": {"input": 0.0, "output": 0.0},
    # Cheap models
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.00035, "output": 0.0004},
    "mistralai/mixtral-8x7b-instruct": {"input": 0.00024, "output": 0.00024},
    # Quality models
    "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "openai/gpt-4o": {"input": 0.005, "output": 0.015},
}


class OpenRouterProvider(LLMProvider):
    """OpenRouter unified API provider.

    Provides access to many models through a single API.
    Some models are FREE (with rate limits).

    Setup:
    1. Get API key from https://openrouter.ai/keys
    2. Set OPENROUTER_API_KEY environment variable

    Free models (rate-limited but no cost):
    - meta-llama/llama-3.1-8b-instruct:free
    - mistralai/mistral-7b-instruct:free
    - google/gemma-2-9b-it:free
    """

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

        # Default to free model
        if not config.model:
            config.model = "meta-llama/llama-3.1-8b-instruct:free"

        # Set pricing
        pricing = OPENROUTER_MODELS.get(
            config.model,
            {"input": 0.001, "output": 0.002}  # Default fallback
        )
        config.cost_per_1k_input = pricing["input"]
        config.cost_per_1k_output = pricing["output"]

    def _get_client(self):
        """Lazy-load the OpenAI-compatible client for OpenRouter."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai",
                    provider="openrouter",
                    retryable=False,
                )

            api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ProviderError(
                    "OpenRouter API key not found. Get one at "
                    "https://openrouter.ai/keys and set OPENROUTER_API_KEY",
                    provider="openrouter",
                    retryable=False,
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.API_BASE,
            )

        return self._client

    def get_name(self) -> str:
        return "openrouter"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using OpenRouter API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                extra_headers={
                    "HTTP-Referer": "https://github.com/loganrooks/anki_card_generator",
                    "X-Title": "Anki Forge",
                },
            )
        except Exception as e:
            error_str = str(e).lower()
            retryable = any(x in error_str for x in ["rate", "timeout", "connection", "503", "429"])
            raise ProviderError(
                f"OpenRouter API error: {e}",
                provider="openrouter",
                retryable=retryable,
            )

        latency_ms = (time.time() - start_time) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        estimated_cost = self._calculate_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=content.strip(),
            model=self.config.model,
            provider="openrouter",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
        )

    @staticmethod
    def list_free_models() -> list[str]:
        """List available free models."""
        return [m for m in OPENROUTER_MODELS.keys() if ":free" in m]

    @staticmethod
    def list_cheap_models() -> list[str]:
        """List cheap models (< $0.001 per 1K tokens)."""
        return [
            m for m, p in OPENROUTER_MODELS.items()
            if p["input"] < 0.001 and p["output"] < 0.001
        ]
