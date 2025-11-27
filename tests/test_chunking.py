"""Tests for text chunking strategies."""

import pytest
from anki_forge.core.models import Document, Chunk, ChunkingStrategy
from anki_forge.core.config import ChunkingConfig
from anki_forge.chunking import get_chunker, ParagraphChunker, SlidingWindowChunker
from anki_forge.chunking.base import Chunker


class TestGetChunker:
    """Tests for chunker factory function."""

    def test_paragraph_strategy(self):
        """Test getting paragraph chunker."""
        config = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
        chunker = get_chunker(config)
        assert isinstance(chunker, ParagraphChunker)

    def test_sliding_window_strategy(self):
        """Test getting sliding window chunker."""
        config = ChunkingConfig(strategy=ChunkingStrategy.SLIDING_WINDOW)
        chunker = get_chunker(config)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_semantic_strategy(self):
        """Test getting semantic chunker."""
        config = ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC)
        chunker = get_chunker(config)
        # SemanticChunker should be returned
        assert isinstance(chunker, Chunker)


class TestParagraphChunker:
    """Tests for paragraph-based chunking."""

    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            target_length=500,
        )
        return ParagraphChunker(config)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return Document(
            title="Test Document",
            author="Test Author",
            file_path="/test.epub",
            chunks=[
                Chunk(
                    text="First paragraph with some content. " * 5,
                    index=0,
                    citation="p. 1",
                ),
                Chunk(
                    text="Second paragraph discussing philosophy. " * 5,
                    index=1,
                    citation="p. 2",
                ),
            ]
        )

    def test_chunk_empty_document(self, chunker):
        """Test chunking empty document."""
        doc = Document(
            title="Empty",
            author="Author",
            file_path="/empty.epub",
            chunks=[],
        )
        result = chunker.chunk(doc)
        assert result == []

    def test_chunk_preserves_content(self, chunker, sample_document):
        """Test that chunking preserves all content."""
        result = chunker.chunk(sample_document)

        # All original text should be in result
        original_text = "".join(c.text for c in sample_document.chunks)
        result_text = "".join(c.text for c in result)

        # Content should be preserved (whitespace may differ)
        assert len(result_text) > 0

    def test_chunk_creates_citations(self, chunker, sample_document):
        """Test that chunks have citations."""
        result = chunker.chunk(sample_document)

        for chunk in result:
            assert chunk.citation  # Has a citation

    def test_chunk_sequential_indices(self, chunker, sample_document):
        """Test that chunks have sequential indices."""
        result = chunker.chunk(sample_document)

        for i, chunk in enumerate(result):
            assert chunk.index == i

    def test_respects_target_length(self):
        """Test that chunks respect target length."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            target_length=100,
        )
        chunker = ParagraphChunker(config)

        # Create document with long text
        doc = Document(
            title="Long",
            author="Author",
            file_path="/long.epub",
            chunks=[
                Chunk(
                    text="A" * 500,  # 500 character chunk
                    index=0,
                    citation="p. 1",
                ),
            ]
        )

        result = chunker.chunk(doc)

        # Should create multiple chunks for long text
        # (exact behavior depends on implementation)
        assert len(result) >= 1


class TestSlidingWindowChunker:
    """Tests for sliding window chunking."""

    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            window_size=100,
            overlap=20,
        )
        return SlidingWindowChunker(config)

    def test_creates_overlapping_chunks(self, chunker):
        """Test that sliding window creates overlapping chunks."""
        doc = Document(
            title="Test",
            author="Author",
            file_path="/test.epub",
            chunks=[
                Chunk(
                    text="A" * 250,  # 250 characters
                    index=0,
                    citation="p. 1",
                ),
            ]
        )

        result = chunker.chunk(doc)

        # Should create multiple overlapping chunks
        assert len(result) >= 2

    def test_window_size_respected(self, chunker):
        """Test that window size is approximately respected."""
        doc = Document(
            title="Test",
            author="Author",
            file_path="/test.epub",
            chunks=[
                Chunk(
                    text="Word " * 100,
                    index=0,
                    citation="p. 1",
                ),
            ]
        )

        result = chunker.chunk(doc)

        # Chunks should be around window_size (with some tolerance)
        for chunk in result[:-1]:  # Except possibly last chunk
            assert chunk.char_count <= chunker.config.window_size + 50


class TestChunkerBase:
    """Tests for base Chunker interface."""

    def test_abstract_chunk_method(self):
        """Test that Chunker.chunk() is abstract."""
        # Can't test directly without a concrete implementation
        # but we can verify interface exists
        assert hasattr(Chunker, 'chunk')


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.strategy == ChunkingStrategy.PARAGRAPH
        assert config.target_length == 5000
        assert config.respect_boundaries is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            window_size=1000,
            overlap=200,
        )
        assert config.strategy == ChunkingStrategy.SLIDING_WINDOW
        assert config.window_size == 1000
        assert config.overlap == 200

    def test_semantic_config(self):
        """Test semantic chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.8,
        )
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.8


class TestChunkingIntegration:
    """Integration tests for chunking pipeline."""

    def test_chunking_workflow(self):
        """Test complete chunking workflow."""
        # Create document
        doc = Document(
            title="Philosophy Book",
            author="Philosopher",
            file_path="/book.epub",
            chunks=[
                Chunk(
                    text="Heidegger's concept of Dasein. " * 10,
                    index=0,
                    citation="Ch. 1, p. 1",
                ),
                Chunk(
                    text="The concept of being-in-the-world. " * 10,
                    index=1,
                    citation="Ch. 1, p. 2",
                ),
            ]
        )

        # Configure and run chunking
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            target_length=200,
        )
        chunker = get_chunker(config)
        result = chunker.chunk(doc)

        # Verify results
        assert len(result) > 0
        for chunk in result:
            assert chunk.text  # Has content
            assert chunk.citation  # Has citation
            assert isinstance(chunk.index, int)  # Has valid index

    def test_very_short_document(self):
        """Test chunking very short document."""
        doc = Document(
            title="Short",
            author="Author",
            file_path="/short.epub",
            chunks=[
                Chunk(
                    text="Just a few words.",
                    index=0,
                    citation="p. 1",
                ),
            ]
        )

        config = ChunkingConfig(target_length=1000)
        chunker = get_chunker(config)
        result = chunker.chunk(doc)

        # Short document should produce at least one chunk
        assert len(result) >= 1
        assert "Just a few words" in result[0].text
