"""Response caching for LLM providers.

Caches LLM responses based on content hash to avoid
regenerating cards for identical input chunks.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .base import LLMResponse

logger = logging.getLogger(__name__)


class ResponseCache:
    """File-based cache for LLM responses.

    Caches responses by hashing:
    - prompt content
    - system prompt
    - model name
    - temperature (affects output)

    This allows reusing responses when the same chunk is processed
    with the same settings, saving significant API costs.
    """

    def __init__(self, cache_dir: str = "./.cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
    ) -> str:
        """Generate cache key from request parameters."""
        key_data = {
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "model": model,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.json"

    def get(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
    ) -> Optional[LLMResponse]:
        """Get cached response if available."""
        key = self._make_key(prompt, system_prompt, model, temperature)
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self._misses += 1
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            self._hits += 1
            logger.debug(f"Cache hit for key {key[:8]}...")

            return LLMResponse(
                content=data["content"],
                model=data["model"],
                provider=data["provider"],
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                estimated_cost=data.get("estimated_cost", 0.0),
                latency_ms=0.0,  # Cached, so no latency
                finish_reason=data.get("finish_reason"),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache entry {key[:8]}: {e}")
            cache_path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def put(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        response: LLMResponse,
    ) -> None:
        """Store response in cache."""
        key = self._make_key(prompt, system_prompt, model, temperature)
        cache_path = self._get_cache_path(key)

        data = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "estimated_cost": response.estimated_cost,
            "finish_reason": response.finish_reason,
            "cached_at": datetime.now().isoformat(),
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cached response for key {key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cache_dir": str(self.cache_dir),
        }

    def clear(self) -> int:
        """Clear all cached responses. Returns number of entries cleared."""
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
        return count


class CachedProvider:
    """Wrapper that adds caching to any LLM provider."""

    def __init__(self, provider, cache: ResponseCache):
        self.provider = provider
        self.cache = cache

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate with caching."""
        # Check cache first
        cached = self.cache.get(
            prompt,
            system_prompt,
            self.provider.config.model,
            self.provider.config.temperature,
        )
        if cached:
            return cached

        # Generate new response
        response = self.provider.generate(prompt, system_prompt)

        # Cache it
        self.cache.put(
            prompt,
            system_prompt,
            self.provider.config.model,
            self.provider.config.temperature,
            response,
        )

        return response

    def generate_with_retry(self, *args, **kwargs) -> LLMResponse:
        """Delegate to underlying provider."""
        return self.provider.generate_with_retry(*args, **kwargs)

    def get_name(self) -> str:
        return f"{self.provider.get_name()}+cache"

    def get_stats(self) -> dict:
        stats = self.provider.get_stats()
        stats["cache"] = self.cache.get_stats()
        return stats
