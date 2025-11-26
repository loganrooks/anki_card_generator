"""Text chunking strategies."""

from .base import Chunker
from .paragraph import ParagraphChunker
from .semantic import SemanticChunker
from .sliding_window import SlidingWindowChunker

from ..core.models import ChunkingStrategy
from ..core.config import ChunkingConfig

__all__ = [
    "Chunker",
    "ParagraphChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    "get_chunker",
]


def get_chunker(config: ChunkingConfig) -> Chunker:
    """Get appropriate chunker based on strategy."""
    chunkers = {
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
    }

    chunker_class = chunkers.get(config.strategy)
    if not chunker_class:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")

    return chunker_class(config)
