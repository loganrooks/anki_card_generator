"""Core data models using Pydantic for validation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import hashlib
import json


class Difficulty(Enum):
    """Card difficulty presets."""
    EASY = "easy"           # Key terms only, low density (~15%)
    MEDIUM = "medium"       # Terms + some phrases (~25%)
    HARD = "hard"           # Terms + phrases, high density (~35%)
    EXPERT = "expert"       # Everything, maximum density (~45%)


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""
    PARAGRAPH = "paragraph"         # Respect paragraph boundaries
    SEMANTIC = "semantic"           # Use embeddings for similarity
    SLIDING_WINDOW = "sliding_window"  # Fixed size with overlap
    SENTENCE = "sentence"           # Sentence-based chunking


class ClozeTargetType(Enum):
    """Types of content to target for cloze deletion."""
    KEY_TERM = "key_term"           # Technical/philosophical terms
    DEFINITION = "definition"       # "X is defined as..." patterns
    FOREIGN_PHRASE = "foreign"      # German, Greek, Latin, French
    FULL_PHRASE = "phrase"          # Longer meaningful phrases
    CONCEPT = "concept"             # Abstract concepts


@dataclass
class ClozeTarget:
    """A target identified for cloze deletion."""
    text: str
    target_type: ClozeTargetType
    cloze_group: int = 1  # c1, c2, or c3
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    confidence: float = 1.0

    def to_cloze(self) -> str:
        """Convert to Anki cloze format."""
        return f"{{{{c{self.cloze_group}::{self.text}}}}}"


@dataclass
class Chunk:
    """A chunk of text extracted from a document."""
    text: str
    index: int
    citation: str

    # Metadata
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_title: Optional[str] = None
    paragraph_indices: list[int] = field(default_factory=list)

    # Processing state
    char_count: int = field(init=False)
    word_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())

    @property
    def content_hash(self) -> str:
        """Generate hash for caching/deduplication."""
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Card:
    """A single Anki flashcard."""
    text: str                       # Card text with {{c1::...}} cloze deletions
    citation: str                   # Source citation

    # Metadata
    chunk_index: Optional[int] = None
    card_index: Optional[int] = None

    # Cloze statistics
    cloze_count: int = field(init=False)
    cloze_density: float = field(init=False)

    # Quality metrics
    word_count: int = field(init=False)
    sentence_count: int = field(init=False)

    def __post_init__(self):
        import re
        # Count cloze deletions
        cloze_pattern = r'\{\{c\d+::(.+?)\}\}'
        clozes = re.findall(cloze_pattern, self.text)
        self.cloze_count = len(clozes)

        # Calculate density (cloze chars / total chars)
        cloze_chars = sum(len(c) for c in clozes)
        # Get plain text without cloze markers
        plain_text = re.sub(r'\{\{c\d+::(.+?)\}\}', r'\1', self.text)
        self.cloze_density = cloze_chars / len(plain_text) if plain_text else 0

        self.word_count = len(plain_text.split())
        self.sentence_count = len(re.findall(r'[.!?]+', plain_text))

    def to_anki_row(self) -> dict:
        """Convert to Anki import format."""
        return {
            "Text": self.text,
            "Citation": self.citation,
        }

    def get_plain_text(self) -> str:
        """Get text with cloze markers removed."""
        import re
        return re.sub(r'\{\{c\d+::(.+?)\}\}', r'\1', self.text)


@dataclass
class Document:
    """A parsed document (EPUB, PDF, etc.)."""
    title: str
    author: str
    file_path: str

    # Content
    chunks: list[Chunk] = field(default_factory=list)

    # Metadata
    total_pages: Optional[int] = None
    language: str = "en"

    # Processing info
    parsed_at: datetime = field(default_factory=datetime.now)

    @property
    def content_hash(self) -> str:
        """Generate hash of all content for caching."""
        all_text = "".join(c.text for c in self.chunks)
        return hashlib.sha256(all_text.encode()).hexdigest()[:16]

    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.chunks)

    @property
    def total_chars(self) -> int:
        return sum(c.char_count for c in self.chunks)


@dataclass
class GenerationSettings:
    """Settings for card generation."""
    # Cloze density control
    target_density: float = 0.30        # Target 30% of text
    density_tolerance: float = 0.05     # Allow +/- 5%

    # Card length constraints
    min_sentences: int = 3
    max_sentences: int = 8
    target_words: int = 100

    # What to target for cloze
    target_key_terms: bool = True
    target_definitions: bool = True
    target_foreign_phrases: bool = True
    target_full_phrases: bool = True

    # Cloze grouping
    max_cloze_groups: int = 3           # c1, c2, c3
    group_thematically: bool = True

    # Difficulty preset (overrides individual settings)
    difficulty: Optional[Difficulty] = None

    def __post_init__(self):
        if self.difficulty:
            self._apply_difficulty_preset()

    def _apply_difficulty_preset(self):
        """Apply difficulty preset to settings."""
        presets = {
            Difficulty.EASY: {
                "target_density": 0.15,
                "target_full_phrases": False,
                "max_cloze_groups": 2,
            },
            Difficulty.MEDIUM: {
                "target_density": 0.25,
                "target_full_phrases": False,
                "max_cloze_groups": 3,
            },
            Difficulty.HARD: {
                "target_density": 0.35,
                "target_full_phrases": True,
                "max_cloze_groups": 3,
            },
            Difficulty.EXPERT: {
                "target_density": 0.45,
                "target_full_phrases": True,
                "max_cloze_groups": 3,
            },
        }
        preset = presets.get(self.difficulty, {})
        for key, value in preset.items():
            setattr(self, key, value)


@dataclass
class GenerationRun:
    """Metadata for a card generation run."""
    run_id: str
    document_hash: str
    settings: GenerationSettings

    # Results
    cards: list[Card] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Cost tracking
    total_tokens: int = 0
    estimated_cost: float = 0.0

    # Provider info
    provider_name: str = ""
    model_name: str = ""

    @classmethod
    def create(cls, document: Document, settings: GenerationSettings) -> "GenerationRun":
        """Create a new generation run."""
        run_data = {
            "document_hash": document.content_hash,
            "settings": settings.__dict__,
            "timestamp": datetime.now().isoformat(),
        }
        run_id = hashlib.sha256(
            json.dumps(run_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        return cls(
            run_id=run_id,
            document_hash=document.content_hash,
            settings=settings,
        )
