"""Hybrid cloze deletion engine with importance-based selection.

This engine combines LLM-identified targets with rule-based application
to achieve precise control over cloze deletion density.

Key features:
1. IMPORTANCE-BASED SELECTION: Targets have scores 1-10, highest priority first
2. QUALITY THRESHOLD: Can require minimum importance to ensure quality
3. DENSITY CONTROL: Guarantees target density ± tolerance
4. SMART FILLING: If under density, adds lower-importance targets

The insight: LLM identifies WHAT to cloze (semantic), rules control HOW MUCH (precision).
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..core.models import ClozeTarget, ClozeTargetType, GenerationSettings

logger = logging.getLogger(__name__)


@dataclass
class ClozeApplication:
    """Result of applying cloze deletions."""
    text: str                           # Text with clozes applied
    targets_used: list[ClozeTarget]     # Targets that were applied
    targets_skipped: list[ClozeTarget]  # Targets skipped (density/overlap)
    actual_density: float
    target_density: float

    # Quality metrics
    avg_importance_used: float = 0.0    # Average importance of used targets
    min_importance_used: int = 0        # Lowest importance included


class ClozeEngine:
    """Engine for applying cloze deletions with importance-based selection.

    Selection algorithm:
    1. Sort targets by importance (10 highest, 1 lowest)
    2. Apply targets in importance order until density target reached
    3. If under minimum density, add lower-importance targets
    4. Skip overlapping targets (keep higher importance one)
    """

    def __init__(
        self,
        settings: GenerationSettings,
        min_importance: int = 1,  # Minimum importance to consider
    ):
        self.settings = settings
        self.target_density = settings.target_density
        self.density_tolerance = settings.density_tolerance
        self.max_groups = settings.max_cloze_groups
        self.min_importance = min_importance

    def apply(
        self,
        text: str,
        targets: list[ClozeTarget],
    ) -> ClozeApplication:
        """Apply cloze deletions to text with importance-based selection.

        Args:
            text: Original text
            targets: Identified cloze targets with importance scores

        Returns:
            ClozeApplication with modified text and quality metrics
        """
        if not targets:
            return ClozeApplication(
                text=text,
                targets_used=[],
                targets_skipped=[],
                actual_density=0.0,
                target_density=self.target_density,
            )

        # Filter by minimum importance
        valid_targets = [t for t in targets if t.importance >= self.min_importance]
        if not valid_targets:
            logger.warning(f"No targets meet minimum importance {self.min_importance}")
            valid_targets = targets  # Fall back to all targets

        # Sort by importance (highest first), then by type priority for ties
        sorted_targets = self._sort_by_importance(valid_targets)

        # Find positions in text
        positioned_targets = self._find_positions(text, sorted_targets)

        # Apply with density control
        result_text, used, skipped = self._apply_with_importance_priority(
            text, positioned_targets
        )

        # Calculate metrics
        actual_density = self._calculate_density(result_text)
        avg_importance = sum(t.importance for t in used) / len(used) if used else 0
        min_importance = min((t.importance for t in used), default=0)

        return ClozeApplication(
            text=result_text,
            targets_used=used,
            targets_skipped=skipped,
            actual_density=actual_density,
            target_density=self.target_density,
            avg_importance_used=avg_importance,
            min_importance_used=min_importance,
        )

    def _sort_by_importance(self, targets: list[ClozeTarget]) -> list[ClozeTarget]:
        """Sort targets by importance, with type as tiebreaker."""
        # Type priority for tiebreaking (lower = higher priority)
        type_priority = {
            ClozeTargetType.FOREIGN_PHRASE: 0,  # Always prioritize foreign terms
            ClozeTargetType.KEY_TERM: 1,
            ClozeTargetType.DEFINITION: 2,
            ClozeTargetType.CONCEPT: 3,
            ClozeTargetType.FULL_PHRASE: 4,
        }

        return sorted(
            targets,
            key=lambda t: (
                -t.importance,  # Higher importance first
                type_priority.get(t.target_type, 99),  # Type as tiebreaker
            )
        )

    def _find_positions(
        self,
        text: str,
        targets: list[ClozeTarget],
    ) -> list[ClozeTarget]:
        """Find start/end positions of targets in text."""
        positioned = []
        used_positions = set()  # Track which character positions are used

        for target in targets:
            if target.start_pos is not None:
                positioned.append(target)
                continue

            # Find all occurrences and pick the first unused one
            pattern = re.escape(target.text)
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.start(), match.end()

                # Check if this position overlaps with already-positioned targets
                position_range = set(range(start, end))
                if not position_range & used_positions:
                    target.start_pos = start
                    target.end_pos = end
                    used_positions.update(position_range)
                    positioned.append(target)
                    break
            else:
                logger.debug(f"Target not found or all occurrences used: '{target.text[:30]}...'")

        return positioned

    def _apply_with_importance_priority(
        self,
        text: str,
        targets: list[ClozeTarget],
    ) -> tuple[str, list[ClozeTarget], list[ClozeTarget]]:
        """Apply targets in importance order, respecting density limits."""
        text_length = len(text)
        min_cloze_chars = int(text_length * (self.target_density - self.density_tolerance))
        max_cloze_chars = int(text_length * (self.target_density + self.density_tolerance))
        ideal_cloze_chars = int(text_length * self.target_density)

        # Separate targets into positioned (sorted by position) and unpositioned
        positioned = [t for t in targets if t.start_pos is not None]
        positioned.sort(key=lambda t: t.start_pos)

        # First pass: select which targets to include based on importance
        selected = []
        current_chars = 0

        # Sort by importance for selection
        by_importance = sorted(positioned, key=lambda t: -t.importance)

        for target in by_importance:
            target_len = target.char_count

            # Always include high-importance targets (9-10)
            if target.importance >= 9:
                selected.append(target)
                current_chars += target_len
                continue

            # Include if we haven't reached ideal density
            if current_chars + target_len <= ideal_cloze_chars:
                selected.append(target)
                current_chars += target_len
                continue

            # If we're between ideal and max, include based on importance threshold
            if current_chars < max_cloze_chars:
                # Higher importance = more likely to include even above ideal
                if target.importance >= 7 or current_chars < min_cloze_chars:
                    if current_chars + target_len <= max_cloze_chars:
                        selected.append(target)
                        current_chars += target_len

        # Check if we need more targets to meet minimum density
        if current_chars < min_cloze_chars:
            remaining = [t for t in positioned if t not in selected]
            remaining.sort(key=lambda t: -t.importance)

            for target in remaining:
                if current_chars >= min_cloze_chars:
                    break
                if current_chars + target.char_count <= max_cloze_chars:
                    selected.append(target)
                    current_chars += target.char_count

        # Second pass: apply selected targets in position order, handling overlaps
        selected.sort(key=lambda t: t.start_pos)

        used = []
        skipped = []
        result_parts = []
        last_end = 0

        for target in selected:
            # Skip if overlaps with previous
            if target.start_pos < last_end:
                skipped.append(target)
                continue

            # Apply this target
            result_parts.append(text[last_end:target.start_pos])
            result_parts.append(target.to_cloze())
            last_end = target.end_pos
            used.append(target)

        # Add remaining text
        result_parts.append(text[last_end:])

        # Mark unselected as skipped
        skipped.extend([t for t in positioned if t not in used and t not in skipped])

        return ''.join(result_parts), used, skipped

    def _calculate_density(self, text: str) -> float:
        """Calculate cloze density in text."""
        cloze_pattern = re.compile(r'\{\{c\d+::(.+?)\}\}')
        clozes = cloze_pattern.findall(text)

        cloze_chars = sum(len(c) for c in clozes)
        plain_text = cloze_pattern.sub(r'\1', text)
        text_length = len(plain_text)

        return cloze_chars / text_length if text_length > 0 else 0.0

    def parse_llm_targets(self, llm_output: str) -> list[ClozeTarget]:
        """Parse cloze targets from LLM output with importance scores.

        Expected JSON format:
        [
          {
            "text": "Dasein",
            "type": "KEY_TERM",
            "importance": 10,
            "reason": "Core concept of Heideggerian philosophy",
            "group": 1
          },
          ...
        ]
        """
        targets = []

        try:
            # Find JSON array in output
            match = re.search(r'\[[\s\S]*\]', llm_output)
            if not match:
                logger.warning("No JSON array found in LLM output")
                return targets

            data = json.loads(match.group())

            for item in data:
                if not isinstance(item, dict):
                    continue

                text = item.get("text", item.get("term", ""))
                if not text:
                    continue

                # Parse type
                type_str = item.get("type", "KEY_TERM").upper()
                type_map = {
                    "KEY_TERM": ClozeTargetType.KEY_TERM,
                    "DEFINITION": ClozeTargetType.DEFINITION,
                    "FOREIGN": ClozeTargetType.FOREIGN_PHRASE,
                    "FOREIGN_PHRASE": ClozeTargetType.FOREIGN_PHRASE,
                    "PHRASE": ClozeTargetType.FULL_PHRASE,
                    "FULL_PHRASE": ClozeTargetType.FULL_PHRASE,
                    "CONCEPT": ClozeTargetType.CONCEPT,
                }
                target_type = type_map.get(type_str, ClozeTargetType.KEY_TERM)

                # Parse importance (default 5, range 1-10)
                importance = item.get("importance", 5)
                if isinstance(importance, str):
                    try:
                        importance = int(importance)
                    except ValueError:
                        importance = 5
                importance = max(1, min(10, importance))

                # Parse reason and group
                reason = item.get("reason", "")
                group = item.get("group", 1)
                group = max(1, min(self.max_groups, group))

                targets.append(ClozeTarget(
                    text=text,
                    target_type=target_type,
                    importance=importance,
                    reason=reason,
                    cloze_group=group,
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM targets JSON: {e}")

        return targets
