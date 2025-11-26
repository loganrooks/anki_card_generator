"""Paragraph-aware text chunker.

This chunker respects paragraph boundaries while targeting a specific
chunk size. It's based on the original anki_card_generator logic but
refactored for modularity.
"""

import logging
import re
from typing import Optional

from .base import Chunker
from ..core.models import Document, Chunk
from ..core.config import ChunkingConfig

logger = logging.getLogger(__name__)


class ParagraphChunker(Chunker):
    """Chunk text while respecting paragraph boundaries.

    This chunker:
    - Targets a specific character length per chunk
    - Keeps paragraphs together (doesn't split mid-paragraph)
    - Tracks citation/section changes across chunks
    - Can optionally allow mid-paragraph splits for very long paragraphs
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into paragraph-aware chunks."""
        # First, flatten all sections into paragraphs with citations
        flat_paragraphs = self._flatten_to_paragraphs(document)

        if not flat_paragraphs:
            logger.warning(f"No paragraphs found in document: {document.title}")
            return []

        # Group paragraphs into chunks targeting the configured length
        chunks = self._group_into_chunks(
            flat_paragraphs,
            target_length=self.config.target_length,
            respect_boundaries=self.config.respect_boundaries,
        )

        logger.info(
            f"Created {len(chunks)} chunks from {len(flat_paragraphs)} paragraphs "
            f"(target: {self.config.target_length} chars)"
        )

        return chunks

    def _flatten_to_paragraphs(self, document: Document) -> list[dict]:
        """Flatten document sections into list of paragraphs with metadata."""
        paragraphs = []

        for section in document.chunks:
            # Split section text into paragraphs
            section_paragraphs = [
                p.strip() for p in section.text.split("\n\n")
                if p.strip()
            ]

            for para_idx, para_text in enumerate(section_paragraphs):
                paragraphs.append({
                    "text": para_text,
                    "citation": section.citation,
                    "section_title": section.section_title,
                    "section_index": section.index,
                    "paragraph_index": para_idx,
                })

        return paragraphs

    def _group_into_chunks(
        self,
        paragraphs: list[dict],
        target_length: int,
        respect_boundaries: bool = True,
    ) -> list[Chunk]:
        """Group paragraphs into chunks targeting specific length."""
        chunks = []
        current_texts = []
        current_length = 0
        current_citation = None
        chunk_citations = []

        for i, para in enumerate(paragraphs):
            para_text = para["text"]
            para_length = len(para_text)
            para_citation = para["citation"]

            # Check if adding this paragraph gets us closer to target
            new_length = current_length + para_length + 2  # +2 for \n\n

            if respect_boundaries:
                # Decision: does adding this paragraph get us closer to target?
                distance_without = abs(current_length - target_length)
                distance_with = abs(new_length - target_length)

                if current_texts and distance_without < distance_with:
                    # Current chunk is closer to target - finalize it
                    chunks.append(self._finalize_chunk(
                        current_texts,
                        chunk_citations,
                        len(chunks),
                    ))
                    current_texts = []
                    current_length = 0
                    chunk_citations = []

            else:
                # Allow mid-paragraph splits if paragraph is too long
                if para_length > target_length:
                    # Split this paragraph
                    split_texts = self._split_long_paragraph(para_text, target_length)
                    for split_text in split_texts:
                        if current_length + len(split_text) > target_length and current_texts:
                            chunks.append(self._finalize_chunk(
                                current_texts,
                                chunk_citations,
                                len(chunks),
                            ))
                            current_texts = []
                            current_length = 0
                            chunk_citations = []

                        current_texts.append(split_text)
                        current_length += len(split_text) + 2
                        if para_citation not in chunk_citations:
                            chunk_citations.append(para_citation)
                    continue

            # Add citation marker if section changed
            if para_citation != current_citation:
                current_citation = para_citation
                if para_citation not in chunk_citations:
                    chunk_citations.append(para_citation)

            # Add paragraph to current chunk
            current_texts.append(para_text)
            current_length = new_length

        # Don't forget the last chunk
        if current_texts:
            chunks.append(self._finalize_chunk(
                current_texts,
                chunk_citations,
                len(chunks),
            ))

        return chunks

    def _finalize_chunk(
        self,
        texts: list[str],
        citations: list[str],
        index: int,
    ) -> Chunk:
        """Create a Chunk from accumulated texts."""
        # Combine texts with paragraph breaks
        full_text = "\n\n".join(texts)

        # Create combined citation
        if len(citations) == 1:
            citation = citations[0]
        else:
            citation = f"{citations[0]} - {citations[-1]}"

        return Chunk(
            text=full_text,
            index=index,
            citation=citation,
            paragraph_indices=list(range(len(texts))),
        )

    def _split_long_paragraph(self, text: str, max_length: int) -> list[str]:
        """Split a long paragraph into smaller pieces at sentence boundaries."""
        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > max_length and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks
