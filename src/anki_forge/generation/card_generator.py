"""Card generation pipeline.

This module implements TWO approaches to card generation:

1. DIRECT MODE (current, simpler):
   - LLM generates complete cards with cloze deletions
   - Density is requested via prompt but not guaranteed
   - Good when LLM follows instructions well

2. HYBRID MODE (recommended for precise density):
   - LLM identifies WHAT to cloze (semantic task)
   - ClozeEngine applies clozes with density control (precision task)
   - Guarantees target density is met

The hybrid approach solves the "7-9 words instead of 30%" problem.

BATCHING: For providers with large context windows (Gemini, etc.),
multiple chunks can be processed in a single API call, which helps
stay within rate limits (e.g., Gemini free tier: 15 RPM).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.models import Card, Chunk, ClozeTarget, ClozeTargetType, GenerationSettings
from ..core.config import ProviderConfig
from ..core.exceptions import GenerationError
from ..core.parsing import parse_targets_from_json
from ..validation.cloze_engine import ClozeEngine
from ..validation.cloze_validator import validate_cloze_syntax, extract_cloze_stats
from .base import LLMProvider, LLMResponse
from .batch import BatchProcessor, BatchConfig, BatchStats

logger = logging.getLogger(__name__)


# Prompt for HYBRID mode - LLM identifies targets with importance scores
IDENTIFY_TARGETS_PROMPT = """You are analyzing text to identify what should be cloze-deleted for Anki flashcards.

Your task is to identify:
1. KEY_TERM: Technical/philosophical terminology
2. DEFINITION: Definitions and explanations (the defined term + key phrases)
3. FOREIGN: German, Greek, Latin, French terms and phrases
4. PHRASE: Important phrases that should be memorized together
5. CONCEPT: Abstract concepts central to the argument

For each target, provide:
- "text": The exact text to cloze (as it appears in the source)
- "type": One of KEY_TERM, DEFINITION, FOREIGN, PHRASE, CONCEPT
- "importance": Score 1-10 indicating memorization priority:
    10 = Core concept, essential for understanding the text
    7-9 = Important term/phrase, should definitely be included
    4-6 = Helpful but not essential
    1-3 = Minor detail, include only if space allows
- "reason": Brief (5-10 words) explanation of why this matters
- "group": 1, 2, or 3 (group related items together)

Example output:
[
  {{"text": "Dasein", "type": "KEY_TERM", "importance": 10, "reason": "Central Heideggerian concept", "group": 1}},
  {{"text": "être-au-monde", "type": "FOREIGN", "importance": 9, "reason": "French phenomenological term", "group": 3}},
  {{"text": "being-in-the-world", "type": "KEY_TERM", "importance": 8, "reason": "Translation of Dasein's mode", "group": 1}},
  {{"text": "existence precedes essence", "type": "PHRASE", "importance": 10, "reason": "Sartre's fundamental thesis", "group": 2}}
]

IMPORTANT:
- Identify roughly {target_pct}% of the text (by character count) for cloze deletion
- Over-identify slightly - the system will select the best targets based on importance
- Be generous with importance scores for truly critical concepts (8-10)
- Prioritize: foreign terms > key terminology > definitions > phrases
- Return ONLY valid JSON array, no other text"""


# Prompt for DIRECT mode - LLM generates complete cards
DIRECT_GENERATION_PROMPT = """You create Anki flashcards with cloze deletions from text.

Rules:
- Create cards with {{{{c1::...}}}}, {{{{c2::...}}}}, {{{{c3::...}}}} cloze deletions
- Target approximately {target_pct}% of text for cloze deletion
- Group related clozes: c1 for main concepts, c2 for related terms, c3 for foreign phrases
- Each card should have 3-8 sentences for context
- ALWAYS cloze delete foreign terms (German, Greek, Latin, French)

Output ONLY a JSON array:
[
  {{"Text": "The {{{{c1::concept}}}} is defined as...", "Citation": "[source]"}},
  ...
]"""


@dataclass
class GenerationResult:
    """Result from card generation."""
    cards: list[Card]
    mode: str  # "direct", "hybrid", or "hybrid_batched"

    # Stats
    chunks_processed: int = 0
    chunks_failed: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    # Batching stats
    batches_used: int = 0
    chunks_per_batch_avg: float = 0.0

    # Quality metrics
    avg_density: float = 0.0
    density_in_range: int = 0  # Cards within target density
    avg_importance_used: float = 0.0  # Average importance of included clozes

    errors: list[dict] = field(default_factory=list)


class CardGenerator:
    """Generates Anki cards from document chunks.

    Supports three modes:
    - direct: LLM generates complete cards (simpler, less precise)
    - hybrid: LLM identifies targets, rules apply clozes (precise density)
    - hybrid_batched: Like hybrid, but batches chunks for large context providers

    Use hybrid_batched with providers like Gemini that have:
    - Large context windows (1M+ tokens)
    - Low rate limits (15 RPM for free tier)
    """

    def __init__(
        self,
        provider: LLMProvider,
        settings: GenerationSettings,
        mode: str = "direct",  # "direct", "hybrid", or "hybrid_batched"
        batch_config: Optional[BatchConfig] = None,
    ):
        self.provider = provider
        self.settings = settings
        self.mode = mode
        self.batch_config = batch_config

        if mode in ("hybrid", "hybrid_batched"):
            self.cloze_engine = ClozeEngine(settings)
        else:
            self.cloze_engine = None

        if mode == "hybrid_batched":
            self.batch_processor = BatchProcessor(provider, batch_config)
        else:
            self.batch_processor = None

    def generate(
        self,
        chunks: list[Chunk],
        author: str = "",
    ) -> GenerationResult:
        """Generate cards from chunks.

        Args:
            chunks: Text chunks to process
            author: Author name for citations

        Returns:
            GenerationResult with cards and stats
        """
        result = GenerationResult(cards=[], mode=self.mode)

        # Use batched processing for hybrid_batched mode
        if self.mode == "hybrid_batched" and self.batch_processor:
            return self._generate_hybrid_batched(chunks, author)

        # Process chunks individually for direct/hybrid modes
        importance_scores = []  # Track for avg_importance calculation

        for chunk in chunks:
            try:
                if self.mode == "hybrid":
                    cards, avg_imp = self._generate_hybrid(chunk, author)
                    if avg_imp > 0:
                        importance_scores.append(avg_imp)
                else:
                    cards = self._generate_direct(chunk, author)

                result.cards.extend(cards)
                result.chunks_processed += 1

            except Exception as e:
                logger.warning(f"Error processing chunk {chunk.index}: {e}")
                result.chunks_failed += 1
                result.errors.append({
                    "chunk_index": chunk.index,
                    "error": str(e),
                })

        # Calculate quality metrics
        self._calculate_quality_metrics(result, importance_scores)

        return result

    def _calculate_quality_metrics(
        self,
        result: GenerationResult,
        importance_scores: Optional[list[float]] = None,
    ) -> None:
        """Calculate quality metrics for the result."""
        if result.cards:
            densities = [c.cloze_density for c in result.cards]
            result.avg_density = sum(densities) / len(densities)

            target = self.settings.target_density
            tolerance = self.settings.density_tolerance
            result.density_in_range = sum(
                1 for d in densities
                if target - tolerance <= d <= target + tolerance
            )

        if importance_scores:
            result.avg_importance_used = sum(importance_scores) / len(importance_scores)

        # Get token/cost stats from provider
        stats = self.provider.get_stats()
        result.total_tokens = stats.get("total_tokens", 0)
        result.estimated_cost = stats.get("total_cost", 0.0)

    def _generate_direct(self, chunk: Chunk, author: str) -> list[Card]:
        """Generate cards using direct mode (LLM does everything)."""
        target_pct = int(self.settings.target_density * 100)

        system_prompt = DIRECT_GENERATION_PROMPT.format(target_pct=target_pct)

        user_prompt = f"""Create cloze deletion flashcards from this text:

{chunk.text}

Citation to use: {chunk.citation}
Author: {author}

Output ONLY the JSON array."""

        response = self.provider.generate_with_retry(user_prompt, system_prompt)

        return self._parse_cards_response(response.content, chunk.citation)

    def _generate_hybrid(self, chunk: Chunk, author: str) -> tuple[list[Card], float]:
        """Generate cards using hybrid mode (LLM identifies, rules apply).

        Returns:
            Tuple of (cards list, average importance of used targets)
        """
        # Step 1: LLM identifies what to cloze
        target_pct = int(self.settings.target_density * 100)
        system_prompt = IDENTIFY_TARGETS_PROMPT.format(target_pct=target_pct)

        user_prompt = f"""Analyze this text and identify cloze targets:

{chunk.text}

Identify roughly {target_pct}% of the text for cloze deletion.
Output ONLY the JSON array of targets."""

        response = self.provider.generate_with_retry(user_prompt, system_prompt)

        # Parse targets (now includes importance and reason)
        targets = self._parse_targets_response(response.content)

        if not targets:
            logger.warning(f"No targets identified for chunk {chunk.index}, falling back to direct")
            return self._generate_direct(chunk, author), 0.0

        # Step 2: Apply clozes with density control (importance-based selection)
        application = self.cloze_engine.apply(chunk.text, targets)

        logger.debug(
            f"Chunk {chunk.index}: {len(application.targets_used)} targets applied, "
            f"{len(application.targets_skipped)} skipped, "
            f"density: {application.actual_density:.1%}, "
            f"avg importance: {application.avg_importance_used:.1f}"
        )

        # Step 3: Split into cards if text is long
        cards = self._split_into_cards(application.text, chunk.citation)

        return cards, application.avg_importance_used

    def _generate_hybrid_batched(
        self,
        chunks: list[Chunk],
        author: str,
    ) -> GenerationResult:
        """Generate cards using batched hybrid mode.

        Processes multiple chunks in single API calls for efficiency
        with rate-limited providers like Gemini.
        """
        result = GenerationResult(cards=[], mode="hybrid_batched")
        importance_scores = []

        target_pct = int(self.settings.target_density * 100)
        system_prompt = IDENTIFY_TARGETS_PROMPT.format(target_pct=target_pct)

        # Process in batches
        batch_results, batch_stats = self.batch_processor.process_chunks(
            chunks, system_prompt, target_pct
        )

        # Update batch stats
        result.batches_used = batch_stats.batches_processed
        result.chunks_per_batch_avg = batch_stats.chunks_per_batch_avg

        # Create a mapping of chunk index to chunk for easy lookup
        chunk_map = {c.index: c for c in chunks}

        # Process each batch result
        for batch_result in batch_results:
            chunk = chunk_map.get(batch_result.chunk_index)
            if not chunk:
                continue

            if not batch_result.success:
                result.chunks_failed += 1
                result.errors.append({
                    "chunk_index": batch_result.chunk_index,
                    "error": batch_result.error or "Unknown error",
                })
                continue

            targets = batch_result.targets
            if not targets:
                logger.warning(f"No targets for chunk {batch_result.chunk_index}")
                result.chunks_failed += 1
                continue

            # Apply clozes with importance-based selection
            application = self.cloze_engine.apply(chunk.text, targets)

            logger.debug(
                f"Chunk {batch_result.chunk_index}: "
                f"{len(application.targets_used)} targets applied, "
                f"density: {application.actual_density:.1%}"
            )

            # Create cards
            cards = self._split_into_cards(application.text, chunk.citation)
            result.cards.extend(cards)
            result.chunks_processed += 1

            if application.avg_importance_used > 0:
                importance_scores.append(application.avg_importance_used)

        # Calculate quality metrics
        self._calculate_quality_metrics(result, importance_scores)

        return result

    def _parse_cards_response(self, content: str, default_citation: str) -> list[Card]:
        """Parse cards from LLM JSON response."""
        cards = []

        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', content)
            if not match:
                logger.warning(f"No JSON array found in LLM response (length={len(content)})")
                logger.debug(f"Response preview: {content[:200]}...")
                return cards

            data = json.loads(match.group())
            if not isinstance(data, list):
                logger.warning(f"Expected JSON array, got {type(data).__name__}")
                return cards

            for item in data:
                if not isinstance(item, dict):
                    continue
                text = item.get("Text", item.get("text", ""))
                citation = item.get("Citation", item.get("citation", default_citation))

                if text:
                    # Validate cloze syntax
                    is_valid, errors = validate_cloze_syntax(text)
                    if is_valid:
                        cards.append(Card(text=text, citation=citation))
                    else:
                        logger.debug(f"Invalid card skipped: {errors}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response preview: {content[:200]}...")

        return cards

    def _parse_targets_response(self, content: str) -> list[ClozeTarget]:
        """Parse cloze targets from LLM response with importance scoring."""
        return parse_targets_from_json(content, self.settings.max_cloze_groups)

    def _split_into_cards(self, text: str, citation: str) -> list[Card]:
        """Split text into multiple cards if too long."""
        # For now, return as single card
        # TODO: Implement smart splitting at sentence boundaries

        if not text.strip():
            return []

        # Check if text has clozes
        stats = extract_cloze_stats(text)
        if stats.total_clozes == 0:
            logger.warning("No clozes in processed text")
            return []

        return [Card(text=text, citation=citation)]
