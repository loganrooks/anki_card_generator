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
from .generation import (
    get_provider,
    list_providers,
    get_free_providers,
    ResponseCache,
    CardGenerator,
    BatchConfig,
)
from .output import AnkiCsvWriter, JsonExporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Available providers
PROVIDER_CHOICES = ['openai', 'ollama', 'gemini', 'openrouter']
MODE_CHOICES = ['direct', 'hybrid', 'hybrid_batched']


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """Anki Forge - Generate Anki flashcards from documents.

    Supports EPUB, PDF, and text files. Uses LLMs to identify key concepts
    and create cloze deletion flashcards.

    Quick start:
        anki-forge generate book.epub --provider gemini --mode hybrid

    For free usage:
        anki-forge providers --free
        anki-forge generate book.epub --provider gemini  # Free tier: 15 RPM
    """
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output CSV file path')
@click.option('-c', '--config', type=click.Path(exists=True), help='Config file path')
@click.option('--provider', type=click.Choice(PROVIDER_CHOICES), default='gemini',
              help='LLM provider (default: gemini - free tier available)')
@click.option('--model', type=str, help='Model name (e.g., gemini-1.5-flash, gpt-4o-mini)')
@click.option('--mode', type=click.Choice(MODE_CHOICES), default='hybrid',
              help='Generation mode: direct (LLM generates cards), hybrid (precise density), '
                   'hybrid_batched (batches for rate-limited providers like Gemini)')
@click.option('--difficulty', type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              default='medium', help='Difficulty preset')
@click.option('--density', type=float, default=0.30, help='Target cloze density (0.0-1.0)')
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
    mode: str,
    difficulty: str,
    density: float,
    chunking: str,
    chunk_size: int,
    no_cache: bool,
    json_output: bool,
    verbose: bool,
):
    """Generate Anki cards from a document.

    Examples:
        # Basic usage with Gemini (free tier)
        anki-forge generate book.epub -o cards.csv

        # Precise density control with hybrid mode
        anki-forge generate book.epub --mode hybrid --density 0.30

        # Batched mode for Gemini (handles rate limits automatically)
        anki-forge generate book.epub --mode hybrid_batched --provider gemini

        # Use local Ollama (completely free, no rate limits)
        anki-forge generate book.epub --provider ollama --model llama3.1:8b
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    cfg = load_config(config) if config else Config()

    # Override with CLI options
    cfg.provider.name = provider
    if model:
        cfg.provider.model = model

    # Set up generation settings
    diff = Difficulty(difficulty) if difficulty else None
    settings = GenerationSettings(
        target_density=density,
        difficulty=diff,
    )
    cfg.generation = settings

    # Set up chunking
    cfg.chunking.strategy = ChunkingStrategy(chunking)
    cfg.chunking.target_length = chunk_size

    # Determine output path
    if not output:
        output = Path(input_file).stem + "_cards.csv"

    # Auto-select mode for Gemini (recommend batched due to rate limits)
    if provider == 'gemini' and mode == 'hybrid':
        click.echo("Tip: Consider using --mode hybrid_batched for Gemini (handles 15 RPM limit)")

    click.echo(f"Processing: {input_file}")
    click.echo(f"Provider: {provider} ({cfg.provider.model or 'default model'})")
    click.echo(f"Mode: {mode}")
    click.echo(f"Difficulty: {difficulty}, Density: {density:.0%}")

    try:
        # Parse document
        click.echo("\nParsing document...")
        parser = get_parser(input_file)
        document = parser.parse(input_file)
        click.echo(f"  Found {len(document.chunks)} sections, {document.total_words:,} words")

        # Chunk document
        click.echo(f"Chunking ({chunking})...")
        chunker = get_chunker(cfg.chunking)
        chunks = chunker.chunk(document)
        click.echo(f"  Created {len(chunks)} chunks")

        # Set up provider with optional caching
        cache = None if no_cache else ResponseCache(cfg.cache_dir)
        llm = get_provider(cfg.provider, cache)

        # Set up card generator with selected mode
        batch_config = None
        if mode == 'hybrid_batched':
            batch_config = BatchConfig()  # Auto-configures based on provider

        generator = CardGenerator(
            provider=llm,
            settings=settings,
            mode=mode,
            batch_config=batch_config,
        )

        # Generate cards
        click.echo(f"Generating cards ({mode} mode)...")

        # Create generation run for tracking
        run = GenerationRun.create(document, settings)
        run.provider_name = provider
        run.model_name = cfg.provider.model or "default"

        # Generate with progress indication
        if mode == 'hybrid_batched':
            click.echo(f"  Processing {len(chunks)} chunks in batches...")
            result = generator.generate(chunks, document.author)
            click.echo(f"  Used {result.batches_used} batches "
                      f"(avg {result.chunks_per_batch_avg:.1f} chunks/batch)")
        else:
            with click.progressbar(length=len(chunks), label="  Processing") as bar:
                # Process in smaller batches for progress updates
                all_cards = []
                batch_size = 5
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    result = generator.generate(batch, document.author)
                    all_cards.extend(result.cards)
                    bar.update(len(batch))

                # Create final result
                result = generator.generate([], document.author)  # Empty to get stats
                result.cards = all_cards
                result.chunks_processed = len(chunks)

        # Update run with results
        run.cards = result.cards
        run.total_tokens = result.total_tokens
        run.estimated_cost = result.estimated_cost
        run.completed_at = datetime.now()

        # Write output
        click.echo(f"\nWriting {len(result.cards)} cards to {output}")
        writer = AnkiCsvWriter(tags=["anki-forge", difficulty])
        writer.write_with_metadata(result.cards, output, deck_name=document.title)

        if json_output:
            json_path = Path(output).with_suffix('.json')
            exporter = JsonExporter()
            exporter.export_run(run, document, str(json_path))
            click.echo(f"  Also wrote JSON to {json_path}")

        # Show stats
        click.echo("\nStatistics:")
        click.echo(f"  Cards generated: {len(result.cards)}")
        click.echo(f"  Chunks processed: {result.chunks_processed}")
        if result.chunks_failed > 0:
            click.echo(f"  Chunks failed: {result.chunks_failed}")
        click.echo(f"  Average density: {result.avg_density:.1%}")
        click.echo(f"  Cards in density range: {result.density_in_range}/{len(result.cards)}")
        if result.avg_importance_used > 0:
            click.echo(f"  Average importance: {result.avg_importance_used:.1f}/10")
        click.echo(f"  Total tokens: {result.total_tokens:,}")
        click.echo(f"  Estimated cost: ${result.estimated_cost:.4f}")

        if cache and not no_cache:
            cache_stats = cache.get_stats()
            click.echo(f"  Cache hit rate: {cache_stats['hit_rate']:.0%}")

        click.echo(f"\nDone! Import {output} into Anki as a Cloze note type.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--free', is_flag=True, help='Show only free providers')
def providers(free: bool):
    """List available LLM providers and their status."""
    click.echo("Available LLM Providers:\n")

    all_providers = list_providers()
    free_list = get_free_providers()

    for name, info in all_providers.items():
        if free and not info['free']:
            continue

        free_badge = " [FREE]" if info['free'] else ""
        click.echo(f"  {name}{free_badge}")
        click.echo(f"    Default model: {info['default_model']}")
        if info['env_var']:
            click.echo(f"    Requires: {info['env_var']}")
        else:
            click.echo("    Requires: Local installation")
        click.echo()

    if not free:
        click.echo("Tip: Use --free to show only free providers")
    else:
        click.echo("Free providers shown. Set up:")
        click.echo("  gemini: Get key at https://makersuite.google.com/app/apikey")
        click.echo("  openrouter: Get key at https://openrouter.ai/keys")
        click.echo("  ollama: Install from https://ollama.ai/download")


@cli.command()
@click.option('--provider', type=click.Choice(PROVIDER_CHOICES), default='gemini')
def check(provider: str):
    """Check if an LLM provider is available and configured."""
    click.echo(f"Checking {provider} availability...\n")

    try:
        if provider == 'ollama':
            from .generation.ollama import OllamaProvider
            cfg = ProviderConfig(name="ollama", model="llama3.1:8b")
            p = OllamaProvider(cfg)
            if p._check_availability():
                models = p.list_models()
                click.echo(f"Ollama is running")
                click.echo(f"Available models: {', '.join(models) or 'None'}")
                if not models:
                    click.echo("\nTo get started: ollama pull llama3.1:8b")
            else:
                click.echo("Ollama is not running")
                click.echo("Install from: https://ollama.ai/download")

        elif provider == 'gemini':
            import os
            if os.environ.get("GOOGLE_API_KEY"):
                click.echo("Gemini API key found (GOOGLE_API_KEY)")
                click.echo("Free tier: 15 RPM, 1M tokens/day")
            else:
                click.echo("GOOGLE_API_KEY not set")
                click.echo("Get a free key at: https://makersuite.google.com/app/apikey")
                click.echo("Then: export GOOGLE_API_KEY=your-key")

        elif provider == 'openrouter':
            import os
            if os.environ.get("OPENROUTER_API_KEY"):
                click.echo("OpenRouter API key found (OPENROUTER_API_KEY)")
                click.echo("Free models available (e.g., llama-3.1-8b-instruct:free)")
            else:
                click.echo("OPENROUTER_API_KEY not set")
                click.echo("Get a key at: https://openrouter.ai/keys")
                click.echo("Then: export OPENROUTER_API_KEY=your-key")

        elif provider == 'openai':
            import os
            if os.environ.get("OPENAI_API_KEY"):
                click.echo("OpenAI API key found (OPENAI_API_KEY)")
            else:
                click.echo("OPENAI_API_KEY not set")
                click.echo("Set with: export OPENAI_API_KEY=your-key")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('output_path', type=click.Path())
def init_config(output_path: str):
    """Create a sample configuration file."""
    sample_config = """# Anki Forge Configuration

provider:
  name: gemini           # openai, ollama, gemini, openrouter
  model: gemini-1.5-flash # or: gpt-4o-mini, llama3.1:8b
  # api_key: ...         # optional, uses env var if not set

chunking:
  strategy: paragraph    # paragraph, semantic, sliding_window
  target_length: 5000
  respect_boundaries: true

generation:
  target_density: 0.30   # 30% of text cloze deleted
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

# Generation mode: direct, hybrid, or hybrid_batched
# - direct: LLM generates complete cards (simpler, less precise)
# - hybrid: LLM identifies targets, rules apply clozes (precise density)
# - hybrid_batched: Like hybrid but batches chunks (best for Gemini)
mode: hybrid

enable_cache: true
cache_dir: ./.cache
log_level: INFO
"""
    with open(output_path, 'w') as f:
        f.write(sample_config)

    click.echo(f"Created config file: {output_path}")
    click.echo("Edit this file and run: anki-forge generate book.epub -c config.yaml")


@cli.command()
def models():
    """Show recommended models for each provider."""
    click.echo("Recommended Models:\n")

    click.echo("FREE Options:")
    click.echo("  gemini:")
    click.echo("    gemini-1.5-flash (default) - Fast, 1M context, 15 RPM free")
    click.echo("    gemini-1.0-pro - Stable, smaller context")
    click.echo()
    click.echo("  openrouter (free tier):")
    click.echo("    meta-llama/llama-3.1-8b-instruct:free")
    click.echo("    mistralai/mistral-7b-instruct:free")
    click.echo("    google/gemma-2-9b-it:free")
    click.echo()
    click.echo("  ollama (local, no limits):")
    click.echo("    llama3.1:8b - Good balance of speed/quality")
    click.echo("    mistral - Fast, good for simple texts")
    click.echo()

    click.echo("PAID Options:")
    click.echo("  openai:")
    click.echo("    gpt-4o-mini - Best value, very capable")
    click.echo("    gpt-4o - Highest quality")
    click.echo()
    click.echo("  openrouter (paid):")
    click.echo("    anthropic/claude-3-haiku - Fast and cheap")
    click.echo("    anthropic/claude-3-sonnet - Good balance")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
