"""Anki-compatible CSV output.

Anki expects a simple CSV format with fields separated by tabs or commas.
The first row can optionally contain field names.

For cloze deletion notes, the minimum fields are:
- Text (with {{c1::...}} cloze deletions)
- Extra (optional, for additional info like citations)

See: https://docs.ankiweb.net/importing/text-files.html
"""

import csv
import logging
from pathlib import Path
from typing import Optional, TextIO

from ..core.models import Card

logger = logging.getLogger(__name__)


class AnkiCsvWriter:
    """Writer for Anki-compatible CSV files.

    Produces a simple two-column CSV:
    - Column 1: Card text with cloze deletions
    - Column 2: Citation/source

    This format can be directly imported into Anki as a "Cloze" note type.
    """

    def __init__(
        self,
        include_header: bool = True,
        delimiter: str = "\t",  # Tab-separated is Anki's default
        tags: list[str] = None,
    ):
        self.include_header = include_header
        self.delimiter = delimiter
        self.tags = tags or []

    def write(self, cards: list[Card], output_path: str) -> int:
        """Write cards to CSV file.

        Args:
            cards: List of Card objects
            output_path: Path to output CSV file

        Returns:
            Number of cards written
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            return self._write_to_file(cards, f)

    def _write_to_file(self, cards: list[Card], f: TextIO) -> int:
        """Write cards to file handle."""
        writer = csv.writer(f, delimiter=self.delimiter, quoting=csv.QUOTE_MINIMAL)

        # Optionally write header
        if self.include_header:
            headers = ["Text", "Citation"]
            if self.tags:
                headers.append("Tags")
            writer.writerow(headers)

        # Write cards
        count = 0
        for card in cards:
            row = [card.text, card.citation]
            if self.tags:
                row.append(" ".join(self.tags))
            writer.writerow(row)
            count += 1

        logger.info(f"Wrote {count} cards to CSV")
        return count

    def write_with_metadata(
        self,
        cards: list[Card],
        output_path: str,
        deck_name: Optional[str] = None,
    ) -> int:
        """Write cards with Anki import metadata.

        Adds special comment lines that Anki recognizes:
        - #separator:tab
        - #deck:DeckName
        - #notetype:Cloze
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Write Anki metadata comments
            sep_name = "tab" if self.delimiter == "\t" else "comma"
            f.write(f"#separator:{sep_name}\n")
            f.write("#html:true\n")
            f.write("#notetype:Cloze\n")

            if deck_name:
                f.write(f"#deck:{deck_name}\n")

            if self.tags:
                f.write(f"#tags:{' '.join(self.tags)}\n")

            f.write("\n")

            return self._write_to_file(cards, f)


def cards_to_csv_string(cards: list[Card], delimiter: str = "\t") -> str:
    """Convert cards to CSV string (for clipboard, etc.)."""
    import io

    output = io.StringIO()
    writer = AnkiCsvWriter(include_header=False, delimiter=delimiter)
    writer._write_to_file(cards, output)
    return output.getvalue()
