"""Tests for core data models."""

import pytest
from anki_forge.core.models import (
    Card,
    Chunk,
    Document,
    ClozeTarget,
    ClozeTargetType,
    GenerationSettings,
    Difficulty,
    ChunkingStrategy,
)


class TestClozeTarget:
    """Tests for ClozeTarget model."""

    def test_basic_creation(self):
        """Test creating a basic cloze target."""
        target = ClozeTarget(
            text="Dasein",
            target_type=ClozeTargetType.KEY_TERM,
        )
        assert target.text == "Dasein"
        assert target.target_type == ClozeTargetType.KEY_TERM
        assert target.importance == 5  # Default
        assert target.cloze_group == 1  # Default

    def test_importance_clamping(self):
        """Test that importance is clamped to 1-10 range."""
        # Too high
        target = ClozeTarget(
            text="test",
            target_type=ClozeTargetType.KEY_TERM,
            importance=15,
        )
        assert target.importance == 10

        # Too low
        target = ClozeTarget(
            text="test",
            target_type=ClozeTargetType.KEY_TERM,
            importance=-5,
        )
        assert target.importance == 1

    def test_to_cloze_format(self):
        """Test conversion to Anki cloze format."""
        target = ClozeTarget(
            text="being-there",
            target_type=ClozeTargetType.KEY_TERM,
            cloze_group=2,
        )
        assert target.to_cloze() == "{{c2::being-there}}"

    def test_char_count(self):
        """Test character count property."""
        target = ClozeTarget(
            text="phenomenology",
            target_type=ClozeTargetType.KEY_TERM,
        )
        assert target.char_count == 13

    def test_with_reason(self):
        """Test target with reason field."""
        target = ClozeTarget(
            text="Sorge",
            target_type=ClozeTargetType.FOREIGN_PHRASE,
            importance=9,
            reason="German term for care/concern",
        )
        assert target.reason == "German term for care/concern"
        assert target.importance == 9


class TestCard:
    """Tests for Card model."""

    def test_basic_card(self):
        """Test creating a basic card."""
        card = Card(
            text="The {{c1::Dasein}} is central to phenomenology.",
            citation="Heidegger, Being and Time",
        )
        assert card.cloze_count == 1
        assert card.citation == "Heidegger, Being and Time"

    def test_cloze_count(self):
        """Test counting cloze deletions."""
        card = Card(
            text="{{c1::Existence}} precedes {{c2::essence}}.",
            citation="Sartre",
        )
        assert card.cloze_count == 2

    def test_cloze_density_calculation(self):
        """Test cloze density is calculated correctly."""
        # "Dasein" (6 chars) out of ~40 chars = ~15%
        card = Card(
            text="The concept of {{c1::Dasein}} is important.",
            citation="Test",
        )
        assert 0.10 < card.cloze_density < 0.20

    def test_multiple_cloze_groups(self):
        """Test card with multiple cloze groups."""
        card = Card(
            text="{{c1::Heidegger}} wrote about {{c2::Dasein}} and {{c3::Sorge}}.",
            citation="Test",
        )
        assert card.cloze_count == 3

    def test_word_count(self):
        """Test word count calculation."""
        card = Card(
            text="The {{c1::concept}} of being is fundamental.",
            citation="Test",
        )
        assert card.word_count == 6

    def test_get_plain_text(self):
        """Test extracting plain text without cloze markers."""
        card = Card(
            text="The {{c1::Dasein}} is {{c2::being-there}}.",
            citation="Test",
        )
        assert card.get_plain_text() == "The Dasein is being-there."

    def test_to_anki_row(self):
        """Test conversion to Anki import format."""
        card = Card(
            text="{{c1::Test}} card.",
            citation="Source",
        )
        row = card.to_anki_row()
        assert row["Text"] == "{{c1::Test}} card."
        assert row["Citation"] == "Source"


class TestChunk:
    """Tests for Chunk model."""

    def test_basic_chunk(self):
        """Test creating a basic chunk."""
        chunk = Chunk(
            text="Some philosophical text here.",
            index=0,
            citation="Book, p. 1",
        )
        assert chunk.index == 0
        assert chunk.citation == "Book, p. 1"

    def test_char_and_word_count(self):
        """Test automatic char and word count calculation."""
        chunk = Chunk(
            text="One two three four five.",
            index=0,
            citation="Test",
        )
        assert chunk.word_count == 5
        assert chunk.char_count == 24

    def test_content_hash(self):
        """Test content hash generation."""
        chunk1 = Chunk(text="Same text", index=0, citation="A")
        chunk2 = Chunk(text="Same text", index=1, citation="B")
        chunk3 = Chunk(text="Different text", index=0, citation="A")

        # Same text = same hash
        assert chunk1.content_hash == chunk2.content_hash
        # Different text = different hash
        assert chunk1.content_hash != chunk3.content_hash

    def test_with_metadata(self):
        """Test chunk with full metadata."""
        chunk = Chunk(
            text="Text content",
            index=5,
            citation="Book, Ch. 3",
            start_page=42,
            end_page=43,
            section_title="Chapter 3: Being",
        )
        assert chunk.start_page == 42
        assert chunk.section_title == "Chapter 3: Being"


class TestDocument:
    """Tests for Document model."""

    def test_basic_document(self):
        """Test creating a basic document."""
        doc = Document(
            title="Being and Time",
            author="Martin Heidegger",
            file_path="/path/to/book.epub",
        )
        assert doc.title == "Being and Time"
        assert doc.author == "Martin Heidegger"
        assert doc.chunks == []

    def test_with_chunks(self):
        """Test document with chunks."""
        chunks = [
            Chunk(text="First section.", index=0, citation="p. 1"),
            Chunk(text="Second section.", index=1, citation="p. 2"),
        ]
        doc = Document(
            title="Test",
            author="Author",
            file_path="/test.epub",
            chunks=chunks,
        )
        assert len(doc.chunks) == 2

    def test_total_words(self):
        """Test total word count across chunks."""
        doc = Document(
            title="Test",
            author="Author",
            file_path="/test.epub",
            chunks=[
                Chunk(text="One two three.", index=0, citation="A"),
                Chunk(text="Four five.", index=1, citation="B"),
            ],
        )
        assert doc.total_words == 5

    def test_total_chars(self):
        """Test total character count across chunks."""
        doc = Document(
            title="Test",
            author="Author",
            file_path="/test.epub",
            chunks=[
                Chunk(text="Hello", index=0, citation="A"),
                Chunk(text="World", index=1, citation="B"),
            ],
        )
        assert doc.total_chars == 10

    def test_content_hash(self):
        """Test document content hash."""
        doc1 = Document(
            title="Test",
            author="Author",
            file_path="/test.epub",
            chunks=[Chunk(text="Content", index=0, citation="A")],
        )
        doc2 = Document(
            title="Different",
            author="Other",
            file_path="/other.epub",
            chunks=[Chunk(text="Content", index=0, citation="B")],
        )
        # Same content = same hash (title/author don't affect it)
        assert doc1.content_hash == doc2.content_hash


class TestGenerationSettings:
    """Tests for GenerationSettings model."""

    def test_default_settings(self):
        """Test default generation settings."""
        settings = GenerationSettings()
        assert settings.target_density == 0.30
        assert settings.density_tolerance == 0.05
        assert settings.max_cloze_groups == 3

    def test_custom_density(self):
        """Test custom density settings."""
        settings = GenerationSettings(
            target_density=0.40,
            density_tolerance=0.10,
        )
        assert settings.target_density == 0.40
        assert settings.density_tolerance == 0.10

    def test_difficulty_preset_easy(self):
        """Test easy difficulty preset."""
        settings = GenerationSettings(difficulty=Difficulty.EASY)
        assert settings.target_density == 0.15
        assert settings.target_full_phrases is False

    def test_difficulty_preset_hard(self):
        """Test hard difficulty preset."""
        settings = GenerationSettings(difficulty=Difficulty.HARD)
        assert settings.target_density == 0.35
        assert settings.target_full_phrases is True

    def test_difficulty_preset_expert(self):
        """Test expert difficulty preset."""
        settings = GenerationSettings(difficulty=Difficulty.EXPERT)
        assert settings.target_density == 0.45


class TestEnums:
    """Tests for enum types."""

    def test_difficulty_values(self):
        """Test Difficulty enum values."""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"
        assert Difficulty.EXPERT.value == "expert"

    def test_chunking_strategy_values(self):
        """Test ChunkingStrategy enum values."""
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
        assert ChunkingStrategy.SLIDING_WINDOW.value == "sliding_window"

    def test_cloze_target_type_values(self):
        """Test ClozeTargetType enum values."""
        assert ClozeTargetType.KEY_TERM.value == "key_term"
        assert ClozeTargetType.FOREIGN_PHRASE.value == "foreign"
        assert ClozeTargetType.DEFINITION.value == "definition"
