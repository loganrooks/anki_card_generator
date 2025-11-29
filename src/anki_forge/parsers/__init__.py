"""Document parsers for various file formats."""

from .base import DocumentParser
from .epub import EpubParser
from .pdf import PdfParser
from .mobi import MobiParser

__all__ = ["DocumentParser", "EpubParser", "PdfParser", "MobiParser"]


def get_parser(file_path: str) -> DocumentParser:
    """Get appropriate parser based on file extension.

    Supported formats:
    - .epub: EPUB ebooks
    - .pdf: PDF documents (requires pymupdf)
    - .mobi, .azw, .azw3: Kindle formats (requires mobi)
    """
    ext = file_path.lower().split(".")[-1]

    parsers = {
        "epub": EpubParser,
        "pdf": PdfParser,
        "mobi": MobiParser,
        "azw": MobiParser,
        "azw3": MobiParser,
    }

    parser_class = parsers.get(ext)
    if not parser_class:
        supported = ", ".join(sorted(set(parsers.keys())))
        raise ValueError(f"Unsupported file type: .{ext}. Supported: {supported}")

    return parser_class()
