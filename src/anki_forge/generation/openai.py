"""OpenAI LLM provider implementation."""

import logging
import os
import time
from typing import Optional

from .base import LLMProvider, LLMResponse
from ..core.config import ProviderConfig
from ..core.exceptions import ProviderError

logger = logging.getLogger(__name__)

# OpenAI pricing (as of 2024) - can be overridden in config
OPENAI_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4o-mini, GPT-4o, etc.)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

        # Set pricing if not configured
        if config.cost_per_1k_input == 0:
            pricing = OPENAI_PRICING.get(config.model, {"input": 0.001, "output": 0.002})
            config.cost_per_1k_input = pricing["input"]
            config.cost_per_1k_output = pricing["output"]

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai",
                    provider="openai",
                    retryable=False,
                )

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ProviderError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or provide api_key in config.",
                    provider="openai",
                    retryable=False,
                )

            kwargs = {"api_key": api_key}
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base

            self._client = OpenAI(**kwargs)

        return self._client

    def get_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
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
            )
        except Exception as e:
            error_str = str(e).lower()
            retryable = any(x in error_str for x in ["rate", "timeout", "connection", "503", "429"])
            raise ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                retryable=retryable,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        choice = response.choices[0]
        content = choice.message.content or ""

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        estimated_cost = self._calculate_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=content.strip(),
            model=self.config.model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )
