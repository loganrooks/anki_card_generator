"""LLM providers for card generation."""

from .base import LLMProvider, LLMResponse
from .openai import OpenAIProvider
from .ollama import OllamaProvider
from .cache import ResponseCache, CachedProvider

from ..core.config import ProviderConfig

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "OllamaProvider",
    "ResponseCache",
    "CachedProvider",
    "get_provider",
]


def get_provider(config: ProviderConfig, cache: ResponseCache = None) -> LLMProvider:
    """Get appropriate LLM provider based on configuration.

    Args:
        config: Provider configuration
        cache: Optional response cache (wraps provider with caching if provided)

    Returns:
        LLMProvider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        # Future: "anthropic": AnthropicProvider,
        # Future: "groq": GroqProvider,
    }

    provider_class = providers.get(config.name.lower())
    if not provider_class:
        available = ", ".join(providers.keys())
        raise ValueError(
            f"Unknown provider: {config.name}. Available: {available}"
        )

    provider = provider_class(config)

    if cache:
        return CachedProvider(provider, cache)

    return provider
