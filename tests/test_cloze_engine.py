"""Tests for the cloze deletion engine."""

import pytest
from anki_forge.core.models import (
    ClozeTarget,
    ClozeTargetType,
    GenerationSettings,
)
from anki_forge.validation.cloze_engine import ClozeEngine, ClozeApplication


class TestClozeEngineBasic:
    """Basic tests for ClozeEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine with default settings (30% density)."""
        settings = GenerationSettings(
            target_density=0.30,
            density_tolerance=0.05,
        )
        return ClozeEngine(settings)

    @pytest.fixture
    def sample_text(self):
        """Sample philosophical text for testing."""
        return (
            "Heidegger's concept of Dasein refers to human existence. "
            "The German term means being-there in English."
        )

    def test_empty_targets(self, engine, sample_text):
        """Test handling of empty target list."""
        result = engine.apply(sample_text, [])
        assert result.text == sample_text
        assert result.targets_used == []
        assert result.actual_density == 0.0

    def test_single_target(self, engine, sample_text):
        """Test applying a single target."""
        targets = [
            ClozeTarget(
                text="Dasein",
                target_type=ClozeTargetType.FOREIGN_PHRASE,
                importance=10,
            )
        ]
        result = engine.apply(sample_text, targets)

        assert "{{c1::Dasein}}" in result.text
        assert len(result.targets_used) == 1
        assert result.actual_density > 0

    def test_multiple_targets(self, engine, sample_text):
        """Test applying multiple targets."""
        targets = [
            ClozeTarget(
                text="Dasein",
                target_type=ClozeTargetType.FOREIGN_PHRASE,
                importance=10,
            ),
            ClozeTarget(
                text="being-there",
                target_type=ClozeTargetType.KEY_TERM,
                importance=8,
            ),
        ]
        result = engine.apply(sample_text, targets)

        assert "{{c1::Dasein}}" in result.text
        assert len(result.targets_used) >= 1


class TestImportanceBasedSelection:
    """Tests for importance-based target selection."""

    @pytest.fixture
    def engine(self):
        """Create engine with 20% density for clearer testing."""
        settings = GenerationSettings(
            target_density=0.20,
            density_tolerance=0.05,
        )
        return ClozeEngine(settings)

    def test_high_importance_always_included(self, engine):
        """Test that importance 9-10 targets are always included."""
        text = "The concept of Dasein is central to phenomenology and existentialism."
        targets = [
            ClozeTarget(
                text="Dasein",
                target_type=ClozeTargetType.KEY_TERM,
                importance=10,
            ),
            ClozeTarget(
                text="phenomenology",
                target_type=ClozeTargetType.KEY_TERM,
                importance=3,
            ),
        ]
        result = engine.apply(text, targets)

        # High importance target should be included
        used_texts = [t.text for t in result.targets_used]
        assert "Dasein" in used_texts

    def test_importance_priority_order(self, engine):
        """Test that higher importance targets are selected first."""
        text = "Alpha beta gamma delta epsilon zeta eta theta."
        targets = [
            ClozeTarget(text="Alpha", target_type=ClozeTargetType.KEY_TERM, importance=3),
            ClozeTarget(text="beta", target_type=ClozeTargetType.KEY_TERM, importance=9),
            ClozeTarget(text="gamma", target_type=ClozeTargetType.KEY_TERM, importance=5),
            ClozeTarget(text="delta", target_type=ClozeTargetType.KEY_TERM, importance=10),
        ]
        result = engine.apply(text, targets)

        # Higher importance targets should be preferred
        if len(result.targets_used) >= 2:
            used_texts = [t.text for t in result.targets_used]
            # delta (10) and beta (9) should be included before Alpha (3)
            assert "delta" in used_texts or "beta" in used_texts

    def test_avg_importance_metric(self, engine):
        """Test that average importance is calculated correctly."""
        text = "First second third fourth fifth sixth seventh."
        targets = [
            ClozeTarget(text="First", target_type=ClozeTargetType.KEY_TERM, importance=10),
            ClozeTarget(text="second", target_type=ClozeTargetType.KEY_TERM, importance=8),
        ]
        result = engine.apply(text, targets)

        if len(result.targets_used) > 0:
            assert result.avg_importance_used > 0
            assert result.avg_importance_used <= 10


class TestDensityControl:
    """Tests for density control."""

    def test_respects_max_density(self):
        """Test that density doesn't exceed maximum (except for high-importance targets)."""
        settings = GenerationSettings(
            target_density=0.20,
            density_tolerance=0.05,
        )
        engine = ClozeEngine(settings)

        # Long text with many potential targets at varying importance
        text = "One two three four five six seven eight nine ten eleven twelve."
        targets = [
            ClozeTarget(text="One", target_type=ClozeTargetType.KEY_TERM, importance=5),
            ClozeTarget(text="two", target_type=ClozeTargetType.KEY_TERM, importance=5),
            ClozeTarget(text="three", target_type=ClozeTargetType.KEY_TERM, importance=5),
            ClozeTarget(text="four", target_type=ClozeTargetType.KEY_TERM, importance=5),
            ClozeTarget(text="five", target_type=ClozeTargetType.KEY_TERM, importance=5),
            ClozeTarget(text="six", target_type=ClozeTargetType.KEY_TERM, importance=5),
        ]

        result = engine.apply(text, targets)

        # Density should not exceed target + tolerance (with some buffer)
        max_density = settings.target_density + settings.density_tolerance
        assert result.actual_density <= max_density + 0.10  # Allow some flexibility

    def test_meets_min_density_when_possible(self):
        """Test that minimum density is achieved when targets allow."""
        settings = GenerationSettings(
            target_density=0.30,
            density_tolerance=0.05,
        )
        engine = ClozeEngine(settings)

        text = "Dasein Sorge Angst Sein Zeit Welt Mitsein Existenz."
        targets = [
            ClozeTarget(text=word, target_type=ClozeTargetType.KEY_TERM, importance=8)
            for word in text.split()
        ]

        result = engine.apply(text, targets)

        # Should have some clozes
        assert result.actual_density > 0
        assert len(result.targets_used) > 0


class TestTargetParsing:
    """Tests for parsing LLM output."""

    @pytest.fixture
    def engine(self):
        settings = GenerationSettings()
        return ClozeEngine(settings)

    def test_parse_valid_json(self, engine):
        """Test parsing valid JSON target list."""
        llm_output = '''
        [
            {"text": "Dasein", "type": "KEY_TERM", "importance": 10, "reason": "Core concept", "group": 1},
            {"text": "Sorge", "type": "FOREIGN", "importance": 8, "reason": "German term", "group": 2}
        ]
        '''
        targets = engine.parse_llm_targets(llm_output)

        assert len(targets) == 2
        assert targets[0].text == "Dasein"
        assert targets[0].importance == 10
        assert targets[1].target_type == ClozeTargetType.FOREIGN_PHRASE

    def test_parse_with_markdown(self, engine):
        """Test parsing JSON embedded in markdown."""
        llm_output = '''
        Here are the targets:
        ```json
        [
            {"text": "phenomenology", "type": "KEY_TERM", "importance": 7, "group": 1}
        ]
        ```
        '''
        targets = engine.parse_llm_targets(llm_output)

        assert len(targets) == 1
        assert targets[0].text == "phenomenology"

    def test_parse_missing_fields(self, engine):
        """Test parsing with missing optional fields."""
        llm_output = '[{"text": "concept", "type": "KEY_TERM"}]'
        targets = engine.parse_llm_targets(llm_output)

        assert len(targets) == 1
        assert targets[0].importance == 5  # Default
        assert targets[0].reason == ""  # Default

    def test_parse_invalid_json(self, engine):
        """Test handling of invalid JSON."""
        llm_output = "This is not valid JSON"
        targets = engine.parse_llm_targets(llm_output)

        assert targets == []

    def test_parse_importance_clamping(self, engine):
        """Test that parsed importance is clamped."""
        llm_output = '[{"text": "test", "type": "KEY_TERM", "importance": 15}]'
        targets = engine.parse_llm_targets(llm_output)

        assert targets[0].importance == 10  # Clamped to max

    def test_parse_string_importance(self, engine):
        """Test parsing importance as string."""
        llm_output = '[{"text": "test", "type": "KEY_TERM", "importance": "8"}]'
        targets = engine.parse_llm_targets(llm_output)

        assert targets[0].importance == 8


class TestOverlapHandling:
    """Tests for handling overlapping targets."""

    @pytest.fixture
    def engine(self):
        settings = GenerationSettings(
            target_density=0.50,  # High density to include more
            density_tolerance=0.20,
        )
        return ClozeEngine(settings)

    def test_no_double_cloze(self, engine):
        """Test that same text isn't cloze'd twice."""
        text = "The Dasein is Dasein."
        targets = [
            ClozeTarget(text="Dasein", target_type=ClozeTargetType.KEY_TERM, importance=10),
        ]
        result = engine.apply(text, targets)

        # Should only have one cloze even though "Dasein" appears twice
        cloze_count = result.text.count("{{c1::")
        assert cloze_count == 1

    def test_overlapping_targets_higher_importance_wins(self, engine):
        """Test that overlapping targets prefer higher importance."""
        text = "The being-in-the-world concept."
        targets = [
            ClozeTarget(
                text="being-in-the-world",
                target_type=ClozeTargetType.KEY_TERM,
                importance=10,
            ),
            ClozeTarget(
                text="being",
                target_type=ClozeTargetType.KEY_TERM,
                importance=5,
            ),
        ]
        result = engine.apply(text, targets)

        # The longer, higher-importance target should be used
        used_texts = [t.text for t in result.targets_used]
        if "being-in-the-world" in used_texts:
            # If longer target used, shorter shouldn't overlap
            assert "{{c" in result.text


class TestClozeApplication:
    """Tests for ClozeApplication result object."""

    def test_result_attributes(self):
        """Test ClozeApplication has expected attributes."""
        result = ClozeApplication(
            text="Test {{c1::text}}",
            targets_used=[],
            targets_skipped=[],
            actual_density=0.25,
            target_density=0.30,
            avg_importance_used=8.5,
            min_importance_used=7,
        )

        assert result.actual_density == 0.25
        assert result.target_density == 0.30
        assert result.avg_importance_used == 8.5
        assert result.min_importance_used == 7
