"""JSON export for debugging and analysis."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..core.models import Card, Document, GenerationRun

logger = logging.getLogger(__name__)


class JsonExporter:
    """Export cards and runs to JSON format.

    JSON export is useful for:
    - Debugging card generation
    - Analyzing cloze statistics
    - Archiving runs with full metadata
    """

    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def export_cards(self, cards: list[Card], output_path: str) -> None:
        """Export cards to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "text": card.text,
                "citation": card.citation,
                "stats": {
                    "cloze_count": card.cloze_count,
                    "cloze_density": round(card.cloze_density, 3),
                    "word_count": card.word_count,
                    "sentence_count": card.sentence_count,
                },
            }
            for card in cards
        ]

        self._write_json(data, output_path)
        logger.info(f"Exported {len(cards)} cards to {output_path}")

    def export_run(
        self,
        run: GenerationRun,
        document: Document,
        output_path: str,
    ) -> None:
        """Export full run with metadata."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": run.run_id,
            "document": {
                "title": document.title,
                "author": document.author,
                "file_path": document.file_path,
                "total_pages": document.total_pages,
                "total_words": document.total_words,
                "content_hash": document.content_hash,
            },
            "settings": {
                "target_density": run.settings.target_density,
                "difficulty": run.settings.difficulty.value if run.settings.difficulty else None,
                "min_sentences": run.settings.min_sentences,
                "max_sentences": run.settings.max_sentences,
            },
            "provider": {
                "name": run.provider_name,
                "model": run.model_name,
            },
            "stats": {
                "total_cards": len(run.cards),
                "total_tokens": run.total_tokens,
                "estimated_cost": round(run.estimated_cost, 4),
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            },
            "cards": [
                {
                    "index": i,
                    "text": card.text,
                    "citation": card.citation,
                    "cloze_count": card.cloze_count,
                    "cloze_density": round(card.cloze_density, 3),
                }
                for i, card in enumerate(run.cards)
            ],
            "errors": run.errors,
        }

        self._write_json(data, output_path)
        logger.info(f"Exported run {run.run_id} to {output_path}")

    def _write_json(self, data: dict, output_path: str) -> None:
        """Write JSON data to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            if self.pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)


def load_cards_from_json(json_path: str) -> list[Card]:
    """Load cards from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    cards = []
    items = data if isinstance(data, list) else data.get("cards", [])

    for item in items:
        cards.append(Card(
            text=item.get("text", item.get("Text", "")),
            citation=item.get("citation", item.get("Citation", "")),
        ))

    return cards
