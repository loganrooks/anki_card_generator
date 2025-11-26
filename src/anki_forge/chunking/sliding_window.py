"""Sliding window text chunker.

Simple chunking strategy that creates fixed-size chunks with overlap.
Useful when you want consistent chunk sizes regardless of content structure.
"""

import logging
from typing import Optional

from .base import Chunker
from ..core.models import Document, Chunk
from ..core.config import ChunkingConfig

logger = logging.getLogger(__name__)


class SlidingWindowChunker(Chunker):
    """Chunk text using a sliding window approach.

    Creates chunks of fixed size with configurable overlap.
    Simple and predictable, but may split content awkwardly.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size overlapping chunks."""
        # Combine all text
        full_text = "\n\n".join(section.text for section in document.chunks)

        if not full_text.strip():
            logger.warning(f"No text found in document: {document.title}")
            return []

        window_size = self.config.window_size
        overlap = self.config.overlap

        # Validate settings
        if overlap >= window_size:
            logger.warning(
                f"Overlap ({overlap}) >= window size ({window_size}), "
                f"setting overlap to {window_size // 4}"
            )
            overlap = window_size // 4

        chunks = []
        pos = 0
        chunk_idx = 0

        while pos < len(full_text):
            # Get chunk text
            end_pos = min(pos + window_size, len(full_text))
            chunk_text = full_text[pos:end_pos]

            # Try to end at a sentence boundary if not at end of text
            if end_pos < len(full_text):
                # Look for sentence boundary in last 20% of chunk
                search_start = int(len(chunk_text) * 0.8)
                search_region = chunk_text[search_start:]

                for boundary in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    last_boundary = search_region.rfind(boundary)
                    if last_boundary != -1:
                        # Adjust end position to sentence boundary
                        actual_end = search_start + last_boundary + len(boundary)
                        chunk_text = chunk_text[:actual_end]
                        end_pos = pos + actual_end
                        break

            # Create chunk
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=chunk_idx,
                citation=f"[{document.title}: chars {pos}-{end_pos}]",
            ))

            # Move position, accounting for overlap
            pos = end_pos - overlap
            chunk_idx += 1

            # Safety check to prevent infinite loop
            if pos <= 0 and chunk_idx > 1:
                break

        logger.info(
            f"Created {len(chunks)} sliding window chunks "
            f"(window: {window_size}, overlap: {overlap})"
        )

        return chunks
