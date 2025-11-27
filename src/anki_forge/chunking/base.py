"""Abstract base class for text chunkers."""

from abc import ABC, abstractmethod
import logging

from ..core.models import Document, Chunk
from ..core.config import ChunkingConfig

logger = logging.getLogger(__name__)


class Chunker(ABC):
    """Abstract base class for text chunking strategies.

    Chunkers take a Document with raw sections and split them into
    appropriately-sized chunks for LLM processing.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks.

        Args:
            document: Document with raw sections

        Returns:
            List of Chunk objects ready for LLM processing
        """
        pass

    def _create_chunk(
        self,
        text: str,
        index: int,
        citation: str,
        **kwargs
    ) -> Chunk:
        """Helper to create a Chunk with consistent formatting."""
        return Chunk(
            text=text.strip(),
            index=index,
            citation=citation,
            **kwargs
        )

    def get_stats(self, chunks: list[Chunk]) -> dict:
        """Get statistics about chunking results."""
        if not chunks:
            return {"count": 0}

        char_counts = [c.char_count for c in chunks]
        word_counts = [c.word_count for c in chunks]

        return {
            "count": len(chunks),
            "total_chars": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chars": sum(char_counts) / len(chunks),
            "avg_words": sum(word_counts) / len(chunks),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
        }
