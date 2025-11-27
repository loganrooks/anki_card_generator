"""Prompt template management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# Built-in prompt templates directory
TEMPLATES_DIR = Path(__file__).parent / "builtin"


@dataclass
class PromptTemplate:
    """A prompt template for card generation."""
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str

    # Optional metadata
    domain: str = "general"
    version: str = "1.0"
    author: Optional[str] = None

    # Placeholder variables
    variables: list[str] = field(default_factory=list)

    def format_system(self, **kwargs) -> str:
        """Format system prompt with variables."""
        return self.system_prompt.format(**kwargs)

    def format_user(self, text_chunk: str, **kwargs) -> str:
        """Format user prompt with text chunk and variables."""
        return self.user_prompt_template.format(text_chunk=text_chunk, **kwargs)


def load_prompt(path: str) -> PromptTemplate:
    """Load a prompt template from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return PromptTemplate(
        name=data.get("name", Path(path).stem),
        description=data.get("description", ""),
        system_prompt=data.get("system_prompt", ""),
        user_prompt_template=data.get("user_prompt_template", "{text_chunk}"),
        domain=data.get("domain", "general"),
        version=data.get("version", "1.0"),
        author=data.get("author"),
        variables=data.get("variables", []),
    )


def get_builtin_prompts() -> dict[str, PromptTemplate]:
    """Get all built-in prompt templates."""
    prompts = {}

    # Add hardcoded defaults first
    prompts["default"] = get_default_prompt()
    prompts["philosophy"] = get_philosophy_prompt()
    prompts["vocabulary"] = get_vocabulary_prompt()

    # Load any YAML templates
    if TEMPLATES_DIR.exists():
        for yaml_file in TEMPLATES_DIR.glob("*.yaml"):
            try:
                prompt = load_prompt(yaml_file)
                prompts[prompt.name] = prompt
            except Exception as e:
                logger.warning(f"Failed to load prompt {yaml_file}: {e}")

    return prompts


def get_default_prompt() -> PromptTemplate:
    """Get the default prompt template."""
    return PromptTemplate(
        name="default",
        description="General-purpose cloze deletion prompt",
        domain="general",
        system_prompt="""You are an expert at creating Anki flashcards with cloze deletions.

Your task is to create flashcards from the given text that will help with memorization and understanding.

Guidelines:
- Create cards that test key concepts, definitions, and relationships
- Use {{c1::...}}, {{c2::...}}, {{c3::...}} for cloze deletions
- Group related deletions under the same cloze number
- Each card should have 3-8 sentences for context
- Target approximately {target_density:.0%} of the text for cloze deletions
- Ensure cards are self-contained with enough context

Output ONLY valid JSON in this format:
[
  {{"Text": "The {{{{c1::concept}}}} is defined as...", "Citation": "[source]"}},
  ...
]""",
        user_prompt_template="""Create cloze deletion flashcards from this text:

{text_chunk}

Citation to use: {citation}

Remember: Output ONLY the JSON array, no other text.""",
        variables=["target_density", "citation"],
    )


def get_philosophy_prompt() -> PromptTemplate:
    """Get philosophy-focused prompt template."""
    return PromptTemplate(
        name="philosophy",
        description="Optimized for philosophical texts with technical terminology",
        domain="philosophy",
        system_prompt="""You are a philosophy professor creating Anki flashcards for advanced study.

Your task is to create cloze deletion cards that help students master:
- Key philosophical terminology and their definitions
- Important arguments and their structure
- Relationships between concepts
- Foreign terms (German, Greek, Latin, French)

Guidelines:
- Use {{c1::...}}, {{c2::...}}, {{c3::...}} for cloze deletions
- Group thematically: c1 for main concepts, c2 for related terms, c3 for foreign phrases
- Target approximately {target_density:.0%} of text for deletions
- ALWAYS cloze delete: technical terms, defined concepts, foreign phrases
- Each card needs 4-8 sentences for proper context
- Ensure every foreign term (German, Greek, Latin) is cloze deleted with c3

Output ONLY valid JSON:
[
  {{"Text": "For {{{{c3::Heidegger}}}}, {{{{c1::Dasein}}}} refers to...", "Citation": "[source]"}},
  ...
]""",
        user_prompt_template="""Convert this philosophical text into cloze deletion flashcards:

{text_chunk}

Citation: {citation}
Author: {author}

Ensure all technical terminology and foreign phrases are cloze deleted.
Output ONLY the JSON array.""",
        variables=["target_density", "citation", "author"],
    )


def get_vocabulary_prompt() -> PromptTemplate:
    """Get vocabulary-focused prompt template."""
    return PromptTemplate(
        name="vocabulary",
        description="For learning vocabulary and terminology",
        domain="vocabulary",
        system_prompt="""You are creating vocabulary flashcards using cloze deletions.

Focus on:
- Key terms and their definitions
- Word roots and etymology when present
- Usage in context

Guidelines:
- Use {{c1::term}}: {{c2::definition}} pattern
- Keep cards focused on single terms
- Include enough context for understanding
- Target {target_density:.0%} deletion density

Output ONLY valid JSON:
[
  {{"Text": "{{{{c1::Term}}}}: {{{{c2::definition here}}}}", "Citation": "[source]"}},
  ...
]""",
        user_prompt_template="""Extract vocabulary terms and definitions from this text:

{text_chunk}

Citation: {citation}

Output ONLY the JSON array.""",
        variables=["target_density", "citation"],
    )
