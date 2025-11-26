"""Document parsers for various file formats."""

from .base import DocumentParser
from .epub import EpubParser
from .pdf import PdfParser

__all__ = ["DocumentParser", "EpubParser", "PdfParser"]


def get_parser(file_path: str) -> DocumentParser:
    """Get appropriate parser based on file extension."""
    ext = file_path.lower().split(".")[-1]

    parsers = {
        "epub": EpubParser,
        "pdf": PdfParser,
        # Future: "mobi": MobiParser,
        # Future: "txt": TextParser,
    }

    parser_class = parsers.get(ext)
    if not parser_class:
        supported = ", ".join(parsers.keys())
        raise ValueError(f"Unsupported file type: .{ext}. Supported: {supported}")

    return parser_class()
