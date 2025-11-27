#!/usr/bin/env python3
"""Test script to verify pipeline components work correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from anki_forge.core.models import (
        Card, Chunk, Document, GenerationSettings,
        ClozeTarget, ClozeTargetType, Difficulty
    )
    from anki_forge.validation.cloze_engine import ClozeEngine
    from anki_forge.parsers import get_parser, EpubParser, PdfParser
    from anki_forge.chunking import get_chunker
    from anki_forge.generation import (
        get_provider, CardGenerator, BatchProcessor, BatchConfig
    )
    from anki_forge.output import AnkiCsvWriter

    print("  ✓ All imports successful")
    return True


def test_cloze_engine():
    """Test the cloze engine with mock targets."""
    print("\nTesting ClozeEngine...")

    from anki_forge.core.models import (
        GenerationSettings, ClozeTarget, ClozeTargetType
    )
    from anki_forge.validation.cloze_engine import ClozeEngine

    # Create settings (30% density)
    settings = GenerationSettings(
        target_density=0.30,
        density_tolerance=0.05,
    )
    engine = ClozeEngine(settings)

    # Sample text (~100 chars)
    text = (
        "Heidegger's concept of Dasein refers to human existence. "
        "The German term means 'being-there' in English."
    )
    print(f"  Input text ({len(text)} chars): {text[:50]}...")

    # Mock targets with importance scores
    targets = [
        ClozeTarget(
            text="Dasein",
            target_type=ClozeTargetType.FOREIGN_PHRASE,
            importance=10,
            reason="Core Heideggerian concept",
            cloze_group=1,
        ),
        ClozeTarget(
            text="being-there",
            target_type=ClozeTargetType.KEY_TERM,
            importance=8,
            reason="English translation",
            cloze_group=1,
        ),
        ClozeTarget(
            text="human existence",
            target_type=ClozeTargetType.DEFINITION,
            importance=6,
            reason="Definition of Dasein",
            cloze_group=2,
        ),
    ]

    # Apply clozes
    result = engine.apply(text, targets)

    print(f"  Targets provided: {len(targets)}")
    print(f"  Targets used: {len(result.targets_used)}")
    print(f"  Targets skipped: {len(result.targets_skipped)}")
    print(f"  Actual density: {result.actual_density:.1%}")
    print(f"  Target density: {result.target_density:.1%}")
    print(f"  Avg importance used: {result.avg_importance_used:.1f}")
    print(f"  Output text: {result.text[:60]}...")

    # Verify clozes are present
    assert "{{c1::Dasein}}" in result.text or "{{c1::being-there}}" in result.text
    print("  ✓ ClozeEngine working correctly")
    return True


def test_card_model():
    """Test the Card model calculates stats correctly."""
    print("\nTesting Card model...")

    from anki_forge.core.models import Card

    # Create a card with clozes
    text = (
        "The concept of {{c1::Dasein}} means {{c2::being-there}}. "
        "It is central to {{c1::phenomenology}}."
    )
    card = Card(text=text, citation="Heidegger, Being and Time")

    print(f"  Cloze count: {card.cloze_count}")
    print(f"  Cloze density: {card.cloze_density:.1%}")
    print(f"  Word count: {card.word_count}")

    assert card.cloze_count == 3
    assert card.cloze_density > 0
    print("  ✓ Card model working correctly")
    return True


def test_chunking():
    """Test chunking strategies."""
    print("\nTesting chunking...")

    from anki_forge.core.models import Document, Chunk, ChunkingStrategy
    from anki_forge.core.config import ChunkingConfig
    from anki_forge.chunking import get_chunker

    # Create a mock document
    doc = Document(
        title="Test Document",
        author="Test Author",
        file_path="/tmp/test.epub",
        chunks=[
            Chunk(
                text="First paragraph with some philosophical content. " * 10,
                index=0,
                citation="Test, p. 1",
            ),
            Chunk(
                text="Second paragraph discussing Heidegger's Dasein concept. " * 10,
                index=1,
                citation="Test, p. 2",
            ),
        ]
    )

    # Test paragraph chunker (use enum, not string)
    config = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH, target_length=200)
    chunker = get_chunker(config)
    chunks = chunker.chunk(doc)

    print(f"  Input sections: {len(doc.chunks)}")
    print(f"  Output chunks: {len(chunks)}")
    print(f"  First chunk size: {chunks[0].char_count} chars")

    assert len(chunks) > 0
    print("  ✓ Chunking working correctly")
    return True


def test_csv_output():
    """Test CSV output generation."""
    print("\nTesting CSV output...")

    from anki_forge.core.models import Card
    from anki_forge.output import AnkiCsvWriter
    import tempfile
    import os

    # Create test cards
    cards = [
        Card(
            text="The {{c1::Dasein}} is central to phenomenology.",
            citation="Heidegger, Being and Time",
        ),
        Card(
            text="{{c1::Existence}} precedes {{c2::essence}}.",
            citation="Sartre, Being and Nothingness",
        ),
    ]

    # Write to temp file
    writer = AnkiCsvWriter(tags=["test", "philosophy"])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name

    writer.write(cards, temp_path)

    # Verify file exists and has content
    assert os.path.exists(temp_path)
    with open(temp_path) as f:
        content = f.read()

    print(f"  Cards written: {len(cards)}")
    print(f"  File size: {len(content)} bytes")
    print(f"  Preview: {content[:100]}...")

    os.unlink(temp_path)
    print("  ✓ CSV output working correctly")
    return True


def test_cli_help():
    """Test CLI is accessible."""
    print("\nTesting CLI...")

    import subprocess
    result = subprocess.run(
        ["anki-forge", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Generate Anki flashcards" in result.stdout
    print("  ✓ CLI accessible")

    # Test providers command
    result = subprocess.run(
        ["anki-forge", "providers", "--free"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "gemini" in result.stdout.lower()
    print("  ✓ Providers command working")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Anki Forge Pipeline Component Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_card_model,
        test_cloze_engine,
        test_chunking,
        test_csv_output,
        test_cli_help,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

    print("\n✓ All pipeline components working!")
    print("\nTo run full end-to-end test with LLM:")
    print("  export GOOGLE_API_KEY=your-key")
    print("  anki-forge generate book.epub --mode hybrid")


if __name__ == "__main__":
    main()
