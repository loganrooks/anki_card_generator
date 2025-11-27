"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging
import time

from ..core.config import ProviderConfig
from ..core.exceptions import ProviderError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Cost tracking
    estimated_cost: float = 0.0

    # Timing
    latency_ms: float = 0.0

    # Metadata
    finish_reason: Optional[str] = None
    raw_response: Optional[dict] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement:
    - generate(prompt, system_prompt) -> LLMResponse
    - get_name() -> str
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt (the text chunk)
            system_prompt: System instructions

        Returns:
            LLMResponse with generated content

        Raises:
            ProviderError: If generation fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return provider name."""
        pass

    def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> LLMResponse:
        """Generate with exponential backoff retry.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            max_retries: Maximum retry attempts
            backoff_factor: Multiplier for retry delay

        Returns:
            LLMResponse

        Raises:
            ProviderError: If all retries fail
        """
        last_error = None
        delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = self.generate(prompt, system_prompt)
                self._update_stats(response)
                return response

            except ProviderError as e:
                last_error = e
                if not e.retryable or attempt == max_retries:
                    raise

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor

        raise last_error

    def _update_stats(self, response: LLMResponse) -> None:
        """Update internal statistics."""
        self._request_count += 1
        self._total_tokens += response.input_tokens + response.output_tokens
        self._total_cost += response.estimated_cost

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost

    def get_stats(self) -> dict:
        """Get provider usage statistics."""
        return {
            "provider": self.get_name(),
            "model": self.config.model,
            "requests": self._request_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
