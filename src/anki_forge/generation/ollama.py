"""Ollama LLM provider implementation (local, FREE).

Ollama allows running open-source LLMs locally without any API costs.
Recommended models:
- llama3.1:8b - Good balance of quality and speed
- mistral:7b - Fast and capable
- phi3:mini - Very fast, good for testing
"""

import logging
import time
from typing import Optional

from .base import LLMProvider, LLMResponse
from ..core.config import ProviderConfig
from ..core.exceptions import ProviderError

logger = logging.getLogger(__name__)

# Default Ollama API endpoint
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference (FREE).

    Requires Ollama to be installed and running locally.
    Install: https://ollama.ai/download

    Example usage:
        1. Install Ollama
        2. Run: ollama pull llama3.1:8b
        3. Configure provider with name="ollama", model="llama3.1:8b"
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._available = None

        # Ollama is free, so no cost
        config.cost_per_1k_input = 0.0
        config.cost_per_1k_output = 0.0

        # Default API base
        if not config.api_base:
            config.api_base = DEFAULT_OLLAMA_BASE

    def _check_availability(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available

        try:
            import requests
            response = requests.get(f"{self.config.api_base}/api/tags", timeout=5)
            self._available = response.status_code == 200
        except Exception:
            self._available = False

        if not self._available:
            logger.warning(
                f"Ollama not available at {self.config.api_base}. "
                "Make sure Ollama is installed and running."
            )

        return self._available

    def get_name(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using Ollama API."""
        try:
            import requests
        except ImportError:
            raise ProviderError(
                "requests package not installed. Install with: pip install requests",
                provider="ollama",
                retryable=False,
            )

        if not self._check_availability():
            raise ProviderError(
                f"Ollama not available at {self.config.api_base}. "
                "Make sure Ollama is installed and running: https://ollama.ai/download",
                provider="ollama",
                retryable=False,
            )

        # Build the request
        url = f"{self.config.api_base}/api/generate"

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }

        start_time = time.time()

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            raise ProviderError(
                "Ollama request timed out. The model may be loading or the request is too large.",
                provider="ollama",
                retryable=True,
            )
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                f"Could not connect to Ollama at {self.config.api_base}",
                provider="ollama",
                retryable=True,
            )
        except Exception as e:
            raise ProviderError(
                f"Ollama API error: {e}",
                provider="ollama",
                retryable=False,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        content = data.get("response", "")

        # Ollama provides token counts in some responses
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content.strip(),
            model=self.config.model,
            provider="ollama",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=0.0,  # Ollama is free!
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason"),
            raw_response=data,
        )

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            import requests
            response = requests.get(f"{self.config.api_base}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")

        return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            import requests
            response = requests.post(
                f"{self.config.api_base}/api/pull",
                json={"name": model_name},
                timeout=600,  # Model downloads can take a while
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
