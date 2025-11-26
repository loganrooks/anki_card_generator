"""Document parsers for various file formats."""

from .base import DocumentParser
from .epub import EpubParser

__all__ = ["DocumentParser", "EpubParser"]


def get_parser(file_path: str) -> DocumentParser:
    """Get appropriate parser based on file extension."""
    ext = file_path.lower().split(".")[-1]

    parsers = {
        "epub": EpubParser,
        # Future: "mobi": MobiParser,
        # Future: "pdf": PdfParser,
    }

    parser_class = parsers.get(ext)
    if not parser_class:
        supported = ", ".join(parsers.keys())
        raise ValueError(f"Unsupported file type: .{ext}. Supported: {supported}")

    return parser_class()
