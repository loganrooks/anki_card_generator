"""Hybrid cloze deletion engine.

This engine combines LLM-identified targets with rule-based application
to achieve precise control over cloze deletion density.

The key insight is:
1. Use the LLM to IDENTIFY what should be cloze deleted (semantic understanding)
2. Use rules to APPLY the clozes with density control (precision)

This solves the problem of getting 7-9 words instead of 30% target.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..core.models import ClozeTarget, ClozeTargetType, GenerationSettings

logger = logging.getLogger(__name__)


@dataclass
class ClozeApplication:
    """Result of applying cloze deletions."""
    text: str                       # Text with clozes applied
    targets_used: list[ClozeTarget] # Targets that were applied
    targets_skipped: list[ClozeTarget]  # Targets skipped (density limit)
    actual_density: float
    target_density: float


class ClozeEngine:
    """Engine for applying cloze deletions with density control.

    This engine takes identified cloze targets and applies them
    while respecting density constraints.
    """

    def __init__(self, settings: GenerationSettings):
        self.settings = settings
        self.target_density = settings.target_density
        self.density_tolerance = settings.density_tolerance
        self.max_groups = settings.max_cloze_groups

    def apply(
        self,
        text: str,
        targets: list[ClozeTarget],
    ) -> ClozeApplication:
        """Apply cloze deletions to text with density control.

        Args:
            text: Original text
            targets: Identified cloze targets (from LLM or rules)

        Returns:
            ClozeApplication with modified text and stats
        """
        if not targets:
            return ClozeApplication(
                text=text,
                targets_used=[],
                targets_skipped=[],
                actual_density=0.0,
                target_density=self.target_density,
            )

        # Sort targets by priority
        sorted_targets = self._prioritize_targets(targets)

        # Find positions of each target in text
        positioned_targets = self._find_positions(text, sorted_targets)

        # Apply targets up to density limit
        result_text, used, skipped = self._apply_with_density_limit(
            text, positioned_targets
        )

        # Calculate actual density
        actual_density = self._calculate_density(result_text)

        return ClozeApplication(
            text=result_text,
            targets_used=used,
            targets_skipped=skipped,
            actual_density=actual_density,
            target_density=self.target_density,
        )

    def _prioritize_targets(self, targets: list[ClozeTarget]) -> list[ClozeTarget]:
        """Sort targets by priority for application."""
        # Priority order:
        # 1. Foreign phrases (always should be cloze deleted)
        # 2. Key terms / definitions
        # 3. Full phrases
        # 4. Concepts

        priority_map = {
            ClozeTargetType.FOREIGN_PHRASE: 0,
            ClozeTargetType.KEY_TERM: 1,
            ClozeTargetType.DEFINITION: 1,
            ClozeTargetType.FULL_PHRASE: 2,
            ClozeTargetType.CONCEPT: 3,
        }

        return sorted(
            targets,
            key=lambda t: (priority_map.get(t.target_type, 99), -t.confidence)
        )

    def _find_positions(
        self,
        text: str,
        targets: list[ClozeTarget],
    ) -> list[ClozeTarget]:
        """Find start/end positions of targets in text."""
        positioned = []

        for target in targets:
            # Skip if already has position
            if target.start_pos is not None:
                positioned.append(target)
                continue

            # Find target in text (case-insensitive)
            pattern = re.escape(target.text)
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                target.start_pos = match.start()
                target.end_pos = match.end()
                positioned.append(target)
            else:
                logger.debug(f"Target not found in text: '{target.text[:50]}...'")

        # Sort by position
        return sorted(positioned, key=lambda t: t.start_pos or 0)

    def _apply_with_density_limit(
        self,
        text: str,
        targets: list[ClozeTarget],
    ) -> tuple[str, list[ClozeTarget], list[ClozeTarget]]:
        """Apply targets respecting density limit."""
        text_length = len(text)
        max_cloze_chars = int(text_length * (self.target_density + self.density_tolerance))

        used = []
        skipped = []
        current_cloze_chars = 0

        # Build result by replacing targets
        result_parts = []
        last_end = 0

        for target in targets:
            if target.start_pos is None:
                continue

            target_len = len(target.text)

            # Check if we can fit this target
            if current_cloze_chars + target_len > max_cloze_chars:
                # Check if we're below minimum density
                current_density = current_cloze_chars / text_length
                if current_density >= (self.target_density - self.density_tolerance):
                    skipped.append(target)
                    continue

            # Check for overlap with previous
            if target.start_pos < last_end:
                skipped.append(target)
                continue

            # Apply this target
            result_parts.append(text[last_end:target.start_pos])
            result_parts.append(target.to_cloze())
            last_end = target.end_pos

            current_cloze_chars += target_len
            used.append(target)

        # Add remaining text
        result_parts.append(text[last_end:])

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
        """Parse cloze targets from LLM output.

        Expects output like:
        - KEY_TERM: "Dasein"
        - FOREIGN: "être-au-monde"
        - PHRASE: "being-in-the-world"

        Or JSON format.
        """
        targets = []

        # Try JSON format first
        try:
            import json
            data = json.loads(llm_output)
            if isinstance(data, list):
                for item in data:
                    target_type = ClozeTargetType.KEY_TERM
                    if isinstance(item, dict):
                        text = item.get("text", item.get("term", ""))
                        type_str = item.get("type", "key_term").lower()
                        if "foreign" in type_str:
                            target_type = ClozeTargetType.FOREIGN_PHRASE
                        elif "phrase" in type_str:
                            target_type = ClozeTargetType.FULL_PHRASE
                        elif "definition" in type_str:
                            target_type = ClozeTargetType.DEFINITION
                        group = item.get("group", 1)
                    else:
                        text = str(item)
                        group = 1

                    if text:
                        targets.append(ClozeTarget(
                            text=text,
                            target_type=target_type,
                            cloze_group=min(group, self.max_groups),
                        ))
                return targets
        except json.JSONDecodeError:
            pass

        # Try line-based format
        for line in llm_output.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse "TYPE: text" format
            if ':' in line:
                type_part, text_part = line.split(':', 1)
                type_str = type_part.strip().lower()
                text = text_part.strip().strip('"\'')

                target_type = ClozeTargetType.KEY_TERM
                if 'foreign' in type_str:
                    target_type = ClozeTargetType.FOREIGN_PHRASE
                elif 'phrase' in type_str:
                    target_type = ClozeTargetType.FULL_PHRASE
                elif 'definition' in type_str:
                    target_type = ClozeTargetType.DEFINITION
                elif 'concept' in type_str:
                    target_type = ClozeTargetType.CONCEPT

                if text:
                    targets.append(ClozeTarget(
                        text=text,
                        target_type=target_type,
                        cloze_group=1,
                    ))
            else:
                # Just text, assume key term
                text = line.strip('- "\'')
                if text:
                    targets.append(ClozeTarget(
                        text=text,
                        target_type=ClozeTargetType.KEY_TERM,
                        cloze_group=1,
                    ))

        return targets
