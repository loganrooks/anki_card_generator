"""Command-line interface for Anki Forge."""

import logging
import sys
from pathlib import Path
from datetime import datetime

import click

from .core.config import Config, load_config, ProviderConfig, ChunkingConfig
from .core.models import Difficulty, ChunkingStrategy, GenerationSettings, GenerationRun
from .parsers import get_parser
from .chunking import get_chunker
from .generation import get_provider, ResponseCache
from .output import AnkiCsvWriter, JsonExporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Anki Forge - Generate Anki flashcards from documents."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output CSV file path')
@click.option('-c', '--config', type=click.Path(exists=True), help='Config file path')
@click.option('--provider', type=click.Choice(['openai', 'ollama']), default='openai',
              help='LLM provider to use')
@click.option('--model', type=str, help='Model name (e.g., gpt-4o-mini, llama3.1:8b)')
@click.option('--difficulty', type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              default='medium', help='Difficulty preset')
@click.option('--density', type=float, default=0.25, help='Target cloze density (0.0-1.0)')
@click.option('--chunking', type=click.Choice(['paragraph', 'semantic', 'sliding_window']),
              default='paragraph', help='Chunking strategy')
@click.option('--chunk-size', type=int, default=5000, help='Target chunk size in characters')
@click.option('--no-cache', is_flag=True, help='Disable response caching')
@click.option('--json', 'json_output', is_flag=True, help='Also output JSON for debugging')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
def generate(
    input_file: str,
    output: str,
    config: str,
    provider: str,
    model: str,
    difficulty: str,
    density: float,
    chunking: str,
    chunk_size: int,
    no_cache: bool,
    json_output: bool,
    verbose: bool,
):
    """Generate Anki cards from a document.

    Example:
        anki-forge generate book.epub -o cards.csv --difficulty hard
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    cfg = load_config(config) if config else Config()

    # Override with CLI options
    if provider:
        cfg.provider.name = provider
    if model:
        cfg.provider.model = model

    # Set up generation settings
    diff = Difficulty(difficulty) if difficulty else None
    cfg.generation = GenerationSettings(
        target_density=density,
        difficulty=diff,
    )

    # Set up chunking
    cfg.chunking.strategy = ChunkingStrategy(chunking)
    cfg.chunking.target_length = chunk_size

    # Determine output path
    if not output:
        output = Path(input_file).stem + "_cards.csv"

    click.echo(f"📚 Processing: {input_file}")
    click.echo(f"⚙️  Provider: {cfg.provider.name} ({cfg.provider.model})")
    click.echo(f"📊 Difficulty: {difficulty}, Density: {density:.0%}")

    try:
        # Parse document
        click.echo("📖 Parsing document...")
        parser = get_parser(input_file)
        document = parser.parse(input_file)
        click.echo(f"   Found {len(document.chunks)} sections, {document.total_words} words")

        # Chunk document
        click.echo(f"✂️  Chunking ({chunking})...")
        chunker = get_chunker(cfg.chunking)
        chunks = chunker.chunk(document)
        click.echo(f"   Created {len(chunks)} chunks")

        # Set up provider with optional caching
        cache = None if no_cache else ResponseCache(cfg.cache_dir)
        llm = get_provider(cfg.provider, cache)

        # Generate cards
        click.echo(f"🤖 Generating cards...")
        cards = []
        run = GenerationRun.create(document, cfg.generation)
        run.provider_name = cfg.provider.name
        run.model_name = cfg.provider.model

        # This is a simplified generation loop
        # A full implementation would use the prompt templates and cloze engine
        from .generation.prompts.templates import get_philosophy_prompt
        prompt_template = get_philosophy_prompt()

        with click.progressbar(chunks, label="   Processing chunks") as bar:
            for chunk in bar:
                try:
                    system_prompt = prompt_template.format_system(
                        target_density=cfg.generation.target_density
                    )
                    user_prompt = prompt_template.format_user(
                        text_chunk=chunk.text,
                        citation=chunk.citation,
                        author=document.author,
                    )

                    response = llm.generate_with_retry(user_prompt, system_prompt)
                    run.total_tokens += response.input_tokens + response.output_tokens
                    run.estimated_cost += response.estimated_cost

                    # Parse cards from response
                    parsed = _parse_cards_from_response(response.content, chunk.citation)
                    cards.extend(parsed)

                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk.index}: {e}")
                    run.errors.append({"chunk": chunk.index, "error": str(e)})

        run.cards = cards
        run.completed_at = datetime.now()

        # Write output
        click.echo(f"💾 Writing {len(cards)} cards to {output}")
        writer = AnkiCsvWriter(tags=["anki-forge", difficulty])
        writer.write_with_metadata(cards, output, deck_name=document.title)

        if json_output:
            json_path = Path(output).with_suffix('.json')
            exporter = JsonExporter()
            exporter.export_run(run, document, str(json_path))
            click.echo(f"   Also wrote JSON to {json_path}")

        # Show stats
        stats = llm.get_stats()
        click.echo("\n📈 Statistics:")
        click.echo(f"   Cards generated: {len(cards)}")
        click.echo(f"   Total tokens: {run.total_tokens:,}")
        click.echo(f"   Estimated cost: ${run.estimated_cost:.4f}")

        if cache and not no_cache:
            cache_stats = cache.get_stats()
            click.echo(f"   Cache hit rate: {cache_stats['hit_rate']:.0%}")

        click.echo(f"\n✅ Done! Import {output} into Anki as a Cloze note type.")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--provider', type=click.Choice(['openai', 'ollama']), default='ollama')
def check(provider: str):
    """Check if LLM provider is available."""
    click.echo(f"Checking {provider} availability...")

    try:
        if provider == 'ollama':
            from .generation.ollama import OllamaProvider
            cfg = ProviderConfig(name="ollama", model="llama3.1:8b")
            p = OllamaProvider(cfg)
            if p._check_availability():
                models = p.list_models()
                click.echo(f"✅ Ollama is running")
                click.echo(f"   Available models: {', '.join(models) or 'None (run: ollama pull llama3.1:8b)'}")
            else:
                click.echo("❌ Ollama is not running")
                click.echo("   Install from: https://ollama.ai/download")
        else:
            import os
            if os.environ.get("OPENAI_API_KEY"):
                click.echo("✅ OpenAI API key found")
            else:
                click.echo("❌ OPENAI_API_KEY not set")
                click.echo("   Set with: export OPENAI_API_KEY=your-key")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.argument('output_path', type=click.Path())
def init_config(output_path: str):
    """Create a sample configuration file."""
    sample_config = """# Anki Forge Configuration

provider:
  name: ollama           # or: openai
  model: llama3.1:8b     # or: gpt-4o-mini
  # api_key: ...         # optional, uses env var if not set

chunking:
  strategy: paragraph    # or: semantic, sliding_window
  target_length: 5000
  respect_boundaries: true

generation:
  target_density: 0.25   # 25% of text cloze deleted
  density_tolerance: 0.05
  difficulty: medium     # easy, medium, hard, expert
  min_sentences: 3
  max_sentences: 8
  target_key_terms: true
  target_definitions: true
  target_foreign_phrases: true
  target_full_phrases: true

output:
  output_dir: ./output
  json_output: true
  csv_output: true
  tags:
    - anki-forge
    - philosophy

enable_cache: true
cache_dir: ./.cache
log_level: INFO
"""
    with open(output_path, 'w') as f:
        f.write(sample_config)

    click.echo(f"✅ Created config file: {output_path}")
    click.echo("   Edit this file and run: anki-forge generate book.epub -c config.yaml")


def _parse_cards_from_response(content: str, default_citation: str) -> list:
    """Parse cards from LLM JSON response."""
    import json
    import re
    from .core.models import Card

    cards = []

    # Try to extract JSON array
    try:
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            data = json.loads(match.group())
            for item in data:
                text = item.get("Text", item.get("text", ""))
                citation = item.get("Citation", item.get("citation", default_citation))
                if text:
                    cards.append(Card(text=text, citation=citation))
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")

    return cards


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
