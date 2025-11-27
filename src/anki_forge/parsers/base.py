"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

from ..core.models import Document

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """Configuration for document parsing."""
    # Page filtering
    start_page: int = 0
    end_page: Optional[int] = None

    # Content filtering
    remove_footnotes: bool = True
    remove_toc: bool = True
    remove_copyright: bool = True

    # Tag configuration (for EPUB/HTML)
    footnote_selector: Optional[dict] = None
    toc_selector: Optional[dict] = None
    copyright_selector: Optional[dict] = None
    header_selectors: Optional[list[dict]] = None
    paragraph_selectors: Optional[list[dict]] = None


class DocumentParser(ABC):
    """Abstract base class for document parsers.

    Subclasses must implement:
    - parse(file_path, config) -> Document
    - get_supported_extensions() -> list[str]
    """

    @abstractmethod
    def parse(self, file_path: str, config: Optional[ParserConfig] = None) -> Document:
        """Parse a document file and return structured content.

        Args:
            file_path: Path to the document file
            config: Optional parser configuration

        Returns:
            Document object with extracted content

        Raises:
            ParserError: If parsing fails
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        ext = file_path.lower().split(".")[-1]
        return ext in self.get_supported_extensions()
