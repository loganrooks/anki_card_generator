"""Batch processing for large context window providers.

This module enables batching multiple chunks into single API calls,
which is especially useful for providers like Gemini with:
- Large context windows (1M+ tokens)
- Low rate limits (15 RPM for free tier)

By batching, we can process more content within rate limits.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ..core.models import Chunk, ClozeTarget, ClozeTargetType
from ..core.parsing import parse_targets_from_list
from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# Context limits by provider/model (in tokens, conservative estimates)
CONTEXT_LIMITS = {
    # Gemini models have huge context windows
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.0-pro": 30_720,
    # OpenRouter free models
    "meta-llama/llama-3.1-8b-instruct:free": 128_000,
    "mistralai/mistral-7b-instruct:free": 32_000,
    "google/gemma-2-9b-it:free": 8_000,
    # OpenAI
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    # Ollama (varies by model)
    "llama3.1:8b": 128_000,
    "mistral": 32_000,
}

# Rate limits (requests per minute)
RATE_LIMITS = {
    "gemini": 15,  # Free tier
    "openai": 60,
    "openrouter": 60,
    "ollama": None,  # Local, no limit
}

# Approximate chars per token (conservative)
CHARS_PER_TOKEN = 3.5


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    # Maximum tokens per batch (leave room for output)
    max_tokens_per_batch: int = 100_000
    # Maximum chunks per batch
    max_chunks_per_batch: int = 20
    # Minimum chunks to bother batching
    min_chunks_for_batch: int = 3
    # Requests per minute limit
    rate_limit_rpm: Optional[int] = None
    # Delay between batches (seconds)
    batch_delay: float = 0.0


@dataclass
class BatchResult:
    """Result from processing a batch of chunks."""
    chunk_index: int
    targets: list[ClozeTarget]
    success: bool
    error: Optional[str] = None


@dataclass
class BatchStats:
    """Statistics from batch processing."""
    total_chunks: int = 0
    batches_processed: int = 0
    chunks_per_batch_avg: float = 0.0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    processing_time_s: float = 0.0


class BatchProcessor:
    """Processes multiple chunks in batched API calls.

    This is particularly useful for providers like Gemini with:
    - Huge context windows (1M tokens)
    - Low rate limits (15 RPM for free tier)

    By batching 10-20 chunks per request, we can process
    much more content within the rate limit.
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[BatchConfig] = None,
    ):
        self.provider = provider
        self.config = config or self._auto_config()
        self._last_request_time = 0.0

    def _auto_config(self) -> BatchConfig:
        """Auto-configure based on provider."""
        model = self.provider.config.model or ""
        provider_name = self.provider.get_name()

        # Get context limit
        max_context = CONTEXT_LIMITS.get(model, 32_000)
        # Use 60% of context for input to leave room for output
        max_tokens = int(max_context * 0.6)

        # Get rate limit
        rate_limit = RATE_LIMITS.get(provider_name)

        # Calculate batch delay if rate limited
        batch_delay = 0.0
        if rate_limit:
            batch_delay = 60.0 / rate_limit + 0.5  # Add buffer

        return BatchConfig(
            max_tokens_per_batch=max_tokens,
            rate_limit_rpm=rate_limit,
            batch_delay=batch_delay,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return int(len(text) / CHARS_PER_TOKEN)

    def should_batch(self, chunks: list[Chunk]) -> bool:
        """Determine if batching makes sense for these chunks."""
        if len(chunks) < self.config.min_chunks_for_batch:
            return False

        # If provider has rate limits, batching helps
        if self.config.rate_limit_rpm:
            return True

        # If total tokens fit in one batch, batch
        total_tokens = sum(self.estimate_tokens(c.text) for c in chunks)
        return total_tokens <= self.config.max_tokens_per_batch

    def create_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into optimal batches."""
        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.estimate_tokens(chunk.text)

            # Check if chunk fits in current batch
            if (current_tokens + chunk_tokens <= self.config.max_tokens_per_batch and
                len(current_batch) < self.config.max_chunks_per_batch):
                current_batch.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk_tokens

        # Don't forget last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def process_batch(
        self,
        chunks: list[Chunk],
        system_prompt: str,
        target_density_pct: int = 30,
    ) -> list[BatchResult]:
        """Process a batch of chunks in a single API call.

        Args:
            chunks: Chunks to process together
            system_prompt: System prompt for target identification
            target_density_pct: Target cloze density percentage

        Returns:
            List of BatchResult for each chunk
        """
        # Respect rate limit
        self._wait_for_rate_limit()

        # Build batch prompt
        batch_prompt = self._build_batch_prompt(chunks, target_density_pct)

        try:
            response = self.provider.generate_with_retry(batch_prompt, system_prompt)
            results = self._parse_batch_response(response.content, chunks)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return failure results for all chunks
            results = [
                BatchResult(chunk_index=c.index, targets=[], success=False, error=str(e))
                for c in chunks
            ]

        return results

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        if not self.config.batch_delay:
            return

        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.batch_delay:
            wait_time = self.config.batch_delay - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        self._last_request_time = time.time()

    def _build_batch_prompt(
        self,
        chunks: list[Chunk],
        target_density_pct: int,
    ) -> str:
        """Build a batched prompt for multiple chunks."""
        prompt_parts = [
            f"Analyze the following {len(chunks)} text sections and identify cloze targets for each.",
            f"Target approximately {target_density_pct}% of each section for cloze deletion.",
            "",
            "Output a JSON object with section numbers as keys:",
            "{",
            '  "1": [{"text": "term", "type": "KEY_TERM", "importance": 8, "reason": "why", "group": 1}, ...],',
            '  "2": [...],',
            "  ...",
            "}",
            "",
            "IMPORTANT:",
            "- Each section's array should have targets for THAT section only",
            "- Include importance score (1-10) for each target",
            "- Include brief reason for why this should be cloze deleted",
            "",
        ]

        for i, chunk in enumerate(chunks, 1):
            prompt_parts.extend([
                f"=== SECTION {i} (chunk {chunk.index}) ===",
                chunk.text,
                "",
            ])

        return "\n".join(prompt_parts)

    def _parse_batch_response(
        self,
        content: str,
        chunks: list[Chunk],
    ) -> list[BatchResult]:
        """Parse batched response back to individual chunk results."""
        results = []

        try:
            # Find JSON object in response
            match = re.search(r'\{[\s\S]*\}', content)
            if not match:
                logger.warning("No JSON object found in batch response")
                return [
                    BatchResult(chunk_index=c.index, targets=[], success=False,
                               error="No JSON in response")
                    for c in chunks
                ]

            data = json.loads(match.group())

            # Parse each section
            for i, chunk in enumerate(chunks, 1):
                section_key = str(i)
                section_data = data.get(section_key, [])

                targets = self._parse_targets(section_data)
                results.append(BatchResult(
                    chunk_index=chunk.index,
                    targets=targets,
                    success=True,
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse batch response JSON: {e}")
            return [
                BatchResult(chunk_index=c.index, targets=[], success=False,
                           error=f"JSON parse error: {e}")
                for c in chunks
            ]

        return results

    def _parse_targets(self, target_list: list) -> list[ClozeTarget]:
        """Parse a list of target dicts into ClozeTarget objects."""
        return parse_targets_from_list(target_list, max_cloze_groups=3)

    def process_chunks(
        self,
        chunks: list[Chunk],
        system_prompt: str,
        target_density_pct: int = 30,
    ) -> tuple[list[BatchResult], BatchStats]:
        """Process all chunks with optimal batching.

        Args:
            chunks: All chunks to process
            system_prompt: System prompt for target identification
            target_density_pct: Target cloze density percentage

        Returns:
            Tuple of (results list, stats)
        """
        start_time = time.time()
        all_results = []
        stats = BatchStats(total_chunks=len(chunks))

        # Create batches
        batches = self.create_batches(chunks)
        stats.batches_processed = len(batches)

        logger.info(f"Processing {len(chunks)} chunks in {len(batches)} batches")

        for batch_num, batch in enumerate(batches, 1):
            logger.debug(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} chunks)")

            results = self.process_batch(batch, system_prompt, target_density_pct)
            all_results.extend(results)

        # Calculate stats
        stats.processing_time_s = time.time() - start_time
        stats.chunks_per_batch_avg = len(chunks) / len(batches) if batches else 0

        # Get provider stats
        provider_stats = self.provider.get_stats()
        stats.total_tokens = provider_stats.get("total_tokens", 0)
        stats.estimated_cost = provider_stats.get("total_cost", 0.0)

        return all_results, stats
