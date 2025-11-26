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
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.models import Card, Chunk, ClozeTarget, ClozeTargetType, GenerationSettings
from ..core.config import ProviderConfig
from ..core.exceptions import GenerationError
from ..validation.cloze_engine import ClozeEngine
from ..validation.cloze_validator import validate_cloze_syntax, extract_cloze_stats
from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# Prompt for HYBRID mode - LLM only identifies targets
IDENTIFY_TARGETS_PROMPT = """You are analyzing text to identify what should be cloze-deleted for Anki flashcards.

Your task is to identify:
1. KEY_TERM: Technical/philosophical terminology
2. DEFINITION: Definitions and explanations (the defined term + key phrases)
3. FOREIGN: German, Greek, Latin, French terms and phrases
4. PHRASE: Important phrases that should be memorized together

Output a JSON array of targets. For each target specify:
- "text": The exact text to cloze (as it appears in the source)
- "type": One of KEY_TERM, DEFINITION, FOREIGN, PHRASE
- "group": 1, 2, or 3 (group related items together)

Example output:
[
  {"text": "Dasein", "type": "KEY_TERM", "group": 1},
  {"text": "être-au-monde", "type": "FOREIGN", "group": 3},
  {"text": "being-in-the-world", "type": "KEY_TERM", "group": 1},
  {"text": "existence precedes essence", "type": "PHRASE", "group": 2}
]

IMPORTANT:
- Identify roughly {target_pct}% of the text for cloze deletion
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
    mode: str  # "direct" or "hybrid"

    # Stats
    chunks_processed: int = 0
    chunks_failed: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    # Quality metrics
    avg_density: float = 0.0
    density_in_range: int = 0  # Cards within target density

    errors: list[dict] = field(default_factory=list)


class CardGenerator:
    """Generates Anki cards from document chunks.

    Supports two modes:
    - direct: LLM generates complete cards (simpler, less precise)
    - hybrid: LLM identifies targets, rules apply clozes (precise density)
    """

    def __init__(
        self,
        provider: LLMProvider,
        settings: GenerationSettings,
        mode: str = "direct",  # "direct" or "hybrid"
    ):
        self.provider = provider
        self.settings = settings
        self.mode = mode

        if mode == "hybrid":
            self.cloze_engine = ClozeEngine(settings)
        else:
            self.cloze_engine = None

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

        for chunk in chunks:
            try:
                if self.mode == "hybrid":
                    cards = self._generate_hybrid(chunk, author)
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
        if result.cards:
            densities = [c.cloze_density for c in result.cards]
            result.avg_density = sum(densities) / len(densities)

            target = self.settings.target_density
            tolerance = self.settings.density_tolerance
            result.density_in_range = sum(
                1 for d in densities
                if target - tolerance <= d <= target + tolerance
            )

        # Get token/cost stats from provider
        stats = self.provider.get_stats()
        result.total_tokens = stats.get("total_tokens", 0)
        result.estimated_cost = stats.get("total_cost", 0.0)

        return result

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

    def _generate_hybrid(self, chunk: Chunk, author: str) -> list[Card]:
        """Generate cards using hybrid mode (LLM identifies, rules apply)."""
        # Step 1: LLM identifies what to cloze
        target_pct = int(self.settings.target_density * 100)
        system_prompt = IDENTIFY_TARGETS_PROMPT.format(target_pct=target_pct)

        user_prompt = f"""Analyze this text and identify cloze targets:

{chunk.text}

Identify roughly {target_pct}% of the text for cloze deletion.
Output ONLY the JSON array of targets."""

        response = self.provider.generate_with_retry(user_prompt, system_prompt)

        # Parse targets
        targets = self._parse_targets_response(response.content)

        if not targets:
            logger.warning(f"No targets identified for chunk {chunk.index}, falling back to direct")
            return self._generate_direct(chunk, author)

        # Step 2: Apply clozes with density control
        application = self.cloze_engine.apply(chunk.text, targets)

        logger.debug(
            f"Chunk {chunk.index}: {len(application.targets_used)} targets applied, "
            f"{len(application.targets_skipped)} skipped, "
            f"density: {application.actual_density:.1%}"
        )

        # Step 3: Split into cards if text is long
        cards = self._split_into_cards(application.text, chunk.citation)

        return cards

    def _parse_cards_response(self, content: str, default_citation: str) -> list[Card]:
        """Parse cards from LLM JSON response."""
        cards = []

        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', content)
            if match:
                data = json.loads(match.group())
                for item in data:
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

        return cards

    def _parse_targets_response(self, content: str) -> list[ClozeTarget]:
        """Parse cloze targets from LLM response."""
        targets = []

        try:
            match = re.search(r'\[[\s\S]*\]', content)
            if match:
                data = json.loads(match.group())

                for item in data:
                    text = item.get("text", "")
                    type_str = item.get("type", "KEY_TERM").upper()
                    group = item.get("group", 1)

                    # Map type string to enum
                    type_map = {
                        "KEY_TERM": ClozeTargetType.KEY_TERM,
                        "DEFINITION": ClozeTargetType.DEFINITION,
                        "FOREIGN": ClozeTargetType.FOREIGN_PHRASE,
                        "PHRASE": ClozeTargetType.FULL_PHRASE,
                        "CONCEPT": ClozeTargetType.CONCEPT,
                    }
                    target_type = type_map.get(type_str, ClozeTargetType.KEY_TERM)

                    if text:
                        targets.append(ClozeTarget(
                            text=text,
                            target_type=target_type,
                            cloze_group=min(group, self.settings.max_cloze_groups),
                        ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse targets response: {e}")

        return targets

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
