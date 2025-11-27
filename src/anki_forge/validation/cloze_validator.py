"""Cloze deletion syntax validation."""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Regex pattern for Anki cloze deletions
CLOZE_PATTERN = re.compile(r'\{\{c(\d+)::(.+?)\}\}')


@dataclass
class ClozeStats:
    """Statistics about cloze deletions in text."""
    total_clozes: int
    unique_groups: set[int]
    cloze_texts: list[str]
    cloze_char_count: int
    plain_text_length: int
    density: float

    @property
    def avg_cloze_length(self) -> float:
        if not self.cloze_texts:
            return 0.0
        return sum(len(t) for t in self.cloze_texts) / len(self.cloze_texts)


def validate_cloze_syntax(text: str) -> tuple[bool, list[str]]:
    """Validate cloze deletion syntax in text.

    Args:
        text: Text potentially containing cloze deletions

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check for unbalanced braces
    open_braces = text.count('{{')
    close_braces = text.count('}}')
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")

    # Check for valid cloze format
    clozes = CLOZE_PATTERN.findall(text)
    if not clozes:
        # Check if there are malformed clozes
        if '{{c' in text or '::' in text:
            errors.append("Contains cloze-like syntax but no valid cloze deletions found")

    # Check cloze numbering
    cloze_nums = set(int(c[0]) for c in clozes)
    if cloze_nums:
        if min(cloze_nums) != 1:
            errors.append(f"Cloze numbering should start at 1, found: {sorted(cloze_nums)}")
        if max(cloze_nums) > 3:
            errors.append(f"Cloze numbers should be 1-3, found: {max(cloze_nums)}")

    # Check for empty cloze content
    for num, content in clozes:
        if not content.strip():
            errors.append(f"Empty cloze deletion found: {{{{c{num}::}}}}")

    return len(errors) == 0, errors


def extract_cloze_stats(text: str) -> ClozeStats:
    """Extract statistics about cloze deletions in text."""
    clozes = CLOZE_PATTERN.findall(text)

    cloze_texts = [content for _, content in clozes]
    cloze_char_count = sum(len(t) for t in cloze_texts)

    # Get plain text (with cloze markers removed)
    plain_text = CLOZE_PATTERN.sub(r'\2', text)
    plain_text_length = len(plain_text)

    density = cloze_char_count / plain_text_length if plain_text_length > 0 else 0.0

    return ClozeStats(
        total_clozes=len(clozes),
        unique_groups=set(int(num) for num, _ in clozes),
        cloze_texts=cloze_texts,
        cloze_char_count=cloze_char_count,
        plain_text_length=plain_text_length,
        density=density,
    )


class ClozeValidator:
    """Validates and repairs cloze deletions in cards."""

    def __init__(
        self,
        min_density: float = 0.10,
        max_density: float = 0.50,
        min_cloze_count: int = 1,
        max_cloze_groups: int = 3,
    ):
        self.min_density = min_density
        self.max_density = max_density
        self.min_cloze_count = min_cloze_count
        self.max_cloze_groups = max_cloze_groups

    def validate(self, text: str) -> tuple[bool, list[str]]:
        """Validate card text.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Syntax validation
        syntax_valid, syntax_errors = validate_cloze_syntax(text)
        issues.extend(syntax_errors)

        if not syntax_valid:
            return False, issues

        # Stats validation
        stats = extract_cloze_stats(text)

        if stats.total_clozes < self.min_cloze_count:
            issues.append(
                f"Too few clozes: {stats.total_clozes} < {self.min_cloze_count}"
            )

        if stats.density < self.min_density:
            issues.append(
                f"Density too low: {stats.density:.1%} < {self.min_density:.1%}"
            )

        if stats.density > self.max_density:
            issues.append(
                f"Density too high: {stats.density:.1%} > {self.max_density:.1%}"
            )

        if len(stats.unique_groups) > self.max_cloze_groups:
            issues.append(
                f"Too many cloze groups: {len(stats.unique_groups)} > {self.max_cloze_groups}"
            )

        return len(issues) == 0, issues

    def repair_numbering(self, text: str) -> str:
        """Repair cloze numbering to be 1, 2, 3."""
        clozes = CLOZE_PATTERN.findall(text)
        if not clozes:
            return text

        # Get unique numbers and create mapping
        unique_nums = sorted(set(int(num) for num, _ in clozes))
        mapping = {old: new for new, old in enumerate(unique_nums, start=1)}

        # Replace with corrected numbers
        def replace_num(match):
            old_num = int(match.group(1))
            content = match.group(2)
            new_num = mapping.get(old_num, old_num)
            return f"{{{{c{new_num}::{content}}}}}"

        return CLOZE_PATTERN.sub(replace_num, text)
