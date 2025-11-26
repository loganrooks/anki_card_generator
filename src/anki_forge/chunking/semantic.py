"""Semantic text chunker using sentence embeddings.

This chunker groups semantically similar content together,
which can produce more coherent cards than purely length-based chunking.
"""

import logging
from typing import Optional

from .base import Chunker
from ..core.models import Document, Chunk
from ..core.config import ChunkingConfig
from ..core.exceptions import ChunkingError

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """Chunk text based on semantic similarity.

    Uses sentence-transformers to embed sentences and groups
    similar content together. Falls back to paragraph chunking
    if sentence-transformers is not available.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._model = None
        self._available = None

    def _check_availability(self) -> bool:
        """Check if sentence-transformers is available."""
        if self._available is not None:
            return self._available

        try:
            from sentence_transformers import SentenceTransformer
            self._available = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._available = False

        return self._available

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            if not self._check_availability():
                raise ChunkingError(
                    "Semantic chunking requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into semantically coherent chunks."""
        if not self._check_availability():
            # Fall back to paragraph chunking
            logger.warning("Falling back to paragraph chunking")
            from .paragraph import ParagraphChunker
            fallback = ParagraphChunker(self.config)
            return fallback.chunk(document)

        # Extract all sentences with metadata
        sentences = self._extract_sentences(document)

        if not sentences:
            logger.warning(f"No sentences found in document: {document.title}")
            return []

        # Get embeddings for all sentences
        model = self._get_model()
        texts = [s["text"] for s in sentences]
        embeddings = model.encode(texts, show_progress_bar=False)

        # Group sentences by semantic similarity
        groups = self._group_by_similarity(
            sentences,
            embeddings,
            similarity_threshold=self.config.similarity_threshold,
            max_chunk_size=self.config.target_length,
        )

        # Convert groups to chunks
        chunks = []
        for idx, group in enumerate(groups):
            text = " ".join(s["text"] for s in group)
            citations = list(set(s["citation"] for s in group))
            citation = citations[0] if len(citations) == 1 else f"{citations[0]} - {citations[-1]}"

            chunks.append(Chunk(
                text=text,
                index=idx,
                citation=citation,
            ))

        logger.info(
            f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences"
        )

        return chunks

    def _extract_sentences(self, document: Document) -> list[dict]:
        """Extract sentences from document with metadata."""
        import re

        sentences = []
        for section in document.chunks:
            # Split into sentences
            section_sentences = re.split(r'(?<=[.!?])\s+', section.text)

            for sent in section_sentences:
                sent = sent.strip()
                if sent and len(sent) > 10:  # Filter very short fragments
                    sentences.append({
                        "text": sent,
                        "citation": section.citation,
                        "section_index": section.index,
                    })

        return sentences

    def _group_by_similarity(
        self,
        sentences: list[dict],
        embeddings,
        similarity_threshold: float,
        max_chunk_size: int,
    ) -> list[list[dict]]:
        """Group sentences by semantic similarity."""
        import numpy as np

        groups = []
        current_group = []
        current_embedding = None
        current_length = 0

        for i, sent in enumerate(sentences):
            sent_length = len(sent["text"])
            sent_embedding = embeddings[i]

            # Start new group if needed
            if not current_group:
                current_group.append(sent)
                current_embedding = sent_embedding
                current_length = sent_length
                continue

            # Check similarity to current group
            similarity = np.dot(current_embedding, sent_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(sent_embedding)
            )

            # Check if we should add to current group
            would_exceed_size = current_length + sent_length > max_chunk_size
            is_similar = similarity >= similarity_threshold

            if is_similar and not would_exceed_size:
                current_group.append(sent)
                # Update running average embedding
                current_embedding = (
                    current_embedding * len(current_group) + sent_embedding
                ) / (len(current_group) + 1)
                current_length += sent_length
            else:
                # Start new group
                groups.append(current_group)
                current_group = [sent]
                current_embedding = sent_embedding
                current_length = sent_length

        # Don't forget last group
        if current_group:
            groups.append(current_group)

        return groups
