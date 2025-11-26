"""Quality gates for card generation.

Quality gates allow automatic re-generation of cards that don't
meet quality thresholds.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

from ..core.models import Card, GenerationSettings
from .cloze_validator import extract_cloze_stats

logger = logging.getLogger(__name__)


class QualityGate(ABC):
    """Abstract base class for quality gates."""

    @abstractmethod
    def check(self, card: Card) -> tuple[bool, str]:
        """Check if card passes this gate.

        Args:
            card: Card to check

        Returns:
            Tuple of (passes, reason_if_failed)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Gate name for logging."""
        pass


class DensityGate(QualityGate):
    """Gate that checks cloze deletion density."""

    def __init__(
        self,
        min_density: float = 0.15,
        max_density: float = 0.50,
    ):
        self.min_density = min_density
        self.max_density = max_density

    @property
    def name(self) -> str:
        return "DensityGate"

    def check(self, card: Card) -> tuple[bool, str]:
        if card.cloze_density < self.min_density:
            return False, f"Density {card.cloze_density:.1%} below minimum {self.min_density:.1%}"

        if card.cloze_density > self.max_density:
            return False, f"Density {card.cloze_density:.1%} above maximum {self.max_density:.1%}"

        return True, ""


class LengthGate(QualityGate):
    """Gate that checks card length."""

    def __init__(
        self,
        min_words: int = 20,
        max_words: int = 200,
        min_sentences: int = 2,
        max_sentences: int = 10,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    @property
    def name(self) -> str:
        return "LengthGate"

    def check(self, card: Card) -> tuple[bool, str]:
        if card.word_count < self.min_words:
            return False, f"Too few words: {card.word_count} < {self.min_words}"

        if card.word_count > self.max_words:
            return False, f"Too many words: {card.word_count} > {self.max_words}"

        if card.sentence_count < self.min_sentences:
            return False, f"Too few sentences: {card.sentence_count} < {self.min_sentences}"

        if card.sentence_count > self.max_sentences:
            return False, f"Too many sentences: {card.sentence_count} > {self.max_sentences}"

        return True, ""


class ClozeCountGate(QualityGate):
    """Gate that checks number of cloze deletions."""

    def __init__(self, min_clozes: int = 3, max_clozes: int = 50):
        self.min_clozes = min_clozes
        self.max_clozes = max_clozes

    @property
    def name(self) -> str:
        return "ClozeCountGate"

    def check(self, card: Card) -> tuple[bool, str]:
        if card.cloze_count < self.min_clozes:
            return False, f"Too few clozes: {card.cloze_count} < {self.min_clozes}"

        if card.cloze_count > self.max_clozes:
            return False, f"Too many clozes: {card.cloze_count} > {self.max_clozes}"

        return True, ""


class QualityChecker:
    """Runs multiple quality gates on cards."""

    def __init__(self, gates: list[QualityGate] = None):
        self.gates = gates or []

    @classmethod
    def from_settings(cls, settings: GenerationSettings) -> "QualityChecker":
        """Create checker from generation settings."""
        gates = [
            DensityGate(
                min_density=settings.target_density - settings.density_tolerance,
                max_density=settings.target_density + settings.density_tolerance,
            ),
            LengthGate(
                min_sentences=settings.min_sentences,
                max_sentences=settings.max_sentences,
            ),
            ClozeCountGate(min_clozes=3),
        ]
        return cls(gates)

    def check(self, card: Card) -> tuple[bool, list[str]]:
        """Run all gates on a card.

        Returns:
            Tuple of (all_passed, list_of_failures)
        """
        failures = []

        for gate in self.gates:
            passed, reason = gate.check(card)
            if not passed:
                failures.append(f"{gate.name}: {reason}")
                logger.debug(f"Card failed {gate.name}: {reason}")

        return len(failures) == 0, failures

    def check_batch(self, cards: list[Card]) -> dict:
        """Check a batch of cards and return statistics."""
        passed = 0
        failed = 0
        failure_reasons = {}

        for card in cards:
            ok, reasons = self.check(card)
            if ok:
                passed += 1
            else:
                failed += 1
                for reason in reasons:
                    gate_name = reason.split(':')[0]
                    failure_reasons[gate_name] = failure_reasons.get(gate_name, 0) + 1

        return {
            "total": len(cards),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(cards) if cards else 0,
            "failure_breakdown": failure_reasons,
        }
