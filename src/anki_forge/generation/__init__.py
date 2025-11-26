"""LLM providers for card generation."""

from .base import LLMProvider, LLMResponse
from .openai import OpenAIProvider
from .ollama import OllamaProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .cache import ResponseCache, CachedProvider

from ..core.config import ProviderConfig

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "OllamaProvider",
    "GeminiProvider",
    "OpenRouterProvider",
    "ResponseCache",
    "CachedProvider",
    "get_provider",
]


# Provider registry with info
PROVIDERS = {
    "openai": {
        "class": OpenAIProvider,
        "free": False,
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "ollama": {
        "class": OllamaProvider,
        "free": True,
        "env_var": None,  # Local, no API key needed
        "default_model": "llama3.1:8b",
    },
    "gemini": {
        "class": GeminiProvider,
        "free": True,  # Free tier available
        "env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-1.5-flash",
    },
    "openrouter": {
        "class": OpenRouterProvider,
        "free": True,  # Free models available
        "env_var": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
    },
}


def get_provider(config: ProviderConfig, cache: ResponseCache = None) -> LLMProvider:
    """Get appropriate LLM provider based on configuration.

    Args:
        config: Provider configuration
        cache: Optional response cache (wraps provider with caching if provided)

    Returns:
        LLMProvider instance
    """
    provider_info = PROVIDERS.get(config.name.lower())
    if not provider_info:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {config.name}. Available: {available}"
        )

    provider = provider_info["class"](config)

    if cache:
        return CachedProvider(provider, cache)

    return provider


def list_providers() -> dict:
    """List available providers with their info."""
    return {
        name: {
            "free": info["free"],
            "env_var": info["env_var"],
            "default_model": info["default_model"],
        }
        for name, info in PROVIDERS.items()
    }


def get_free_providers() -> list[str]:
    """Get list of providers with free tiers."""
    return [name for name, info in PROVIDERS.items() if info["free"]]
