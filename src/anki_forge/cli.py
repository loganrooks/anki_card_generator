"""Command-line interface for Anki Forge."""

import logging
import sys
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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

# Rich console for enhanced output
console = Console()

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
@click.version_option(version="0.3.0")
def cli():
    """Anki Forge - Generate Anki flashcards from documents.

    Supports EPUB and PDF files. Uses LLMs to identify key concepts
    and create cloze deletion flashcards.

    \b
    QUICK START:
        anki-forge generate book.epub --provider gemini --mode hybrid

    \b
    INTERACTIVE MODES:
        anki-forge wizard    # Step-by-step guided setup
        anki-forge tui       # Full dashboard interface

    \b
    FREE USAGE:
        anki-forge providers --free
        anki-forge generate book.epub --provider gemini  # 15 RPM free
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
        console.print("[yellow]Tip:[/yellow] Consider using --mode hybrid_batched for Gemini (handles 15 RPM limit)")

    # Display configuration panel
    console.print(Panel.fit(
        f"[bold]File:[/bold] {input_file}\n"
        f"[bold]Provider:[/bold] {provider} ({cfg.provider.model or 'default model'})\n"
        f"[bold]Mode:[/bold] {mode}\n"
        f"[bold]Difficulty:[/bold] {difficulty}, [bold]Density:[/bold] {density:.0%}",
        title="[bold cyan]Anki Forge[/bold cyan]",
        border_style="cyan"
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Parse document
            parse_task = progress.add_task("[cyan]Parsing document...", total=1)
            parser = get_parser(input_file)
            document = parser.parse(input_file)
            progress.update(parse_task, completed=1, description=f"[green]✓ Parsed: {len(document.chunks)} sections, {document.total_words:,} words")

            # Chunk document
            chunk_task = progress.add_task(f"[cyan]Chunking ({chunking})...", total=1)
            chunker = get_chunker(cfg.chunking)
            chunks = chunker.chunk(document)
            progress.update(chunk_task, completed=1, description=f"[green]✓ Created {len(chunks)} chunks")

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

            # Create generation run for tracking
            run = GenerationRun.create(document, settings)
            run.provider_name = provider
            run.model_name = cfg.provider.model or "default"

            # Generate with progress indication
            gen_task = progress.add_task(f"[cyan]Generating cards ({mode})...", total=len(chunks))

            if mode == 'hybrid_batched':
                result = generator.generate(chunks, document.author)
                progress.update(gen_task, completed=len(chunks),
                               description=f"[green]✓ Generated with {result.batches_used} batches (avg {result.chunks_per_batch_avg:.1f} chunks/batch)")
            else:
                # Process in smaller batches for progress updates
                all_cards = []
                batch_size = 5
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    result = generator.generate(batch, document.author)
                    all_cards.extend(result.cards)
                    progress.update(gen_task, advance=len(batch))

                # Create final result
                result = generator.generate([], document.author)  # Empty to get stats
                result.cards = all_cards
                result.chunks_processed = len(chunks)
                progress.update(gen_task, description=f"[green]✓ Generated {len(all_cards)} cards")

        # Update run with results
        run.cards = result.cards
        run.total_tokens = result.total_tokens
        run.estimated_cost = result.estimated_cost
        run.completed_at = datetime.now()

        # Write output
        console.print(f"\n[bold]Writing {len(result.cards)} cards to {output}[/bold]")
        writer = AnkiCsvWriter(tags=["anki-forge", difficulty])
        writer.write_with_metadata(result.cards, output, deck_name=document.title)

        if json_output:
            json_path = Path(output).with_suffix('.json')
            exporter = JsonExporter()
            exporter.export_run(run, document, str(json_path))
            console.print(f"  Also wrote JSON to {json_path}")

        # Show stats table
        stats_table = Table(title="Generation Statistics", show_header=False, box=None)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Cards generated", str(len(result.cards)))
        stats_table.add_row("Chunks processed", str(result.chunks_processed))
        if result.chunks_failed > 0:
            stats_table.add_row("Chunks failed", f"[red]{result.chunks_failed}[/red]")
        stats_table.add_row("Average density", f"{result.avg_density:.1%}")
        stats_table.add_row("Cards in density range", f"{result.density_in_range}/{len(result.cards)}")
        if result.avg_importance_used > 0:
            stats_table.add_row("Average importance", f"{result.avg_importance_used:.1f}/10")
        stats_table.add_row("Total tokens", f"{result.total_tokens:,}")
        stats_table.add_row("Estimated cost", f"${result.estimated_cost:.4f}")

        if cache and not no_cache:
            cache_stats = cache.get_stats()
            stats_table.add_row("Cache hit rate", f"{cache_stats['hit_rate']:.0%}")

        console.print()
        console.print(stats_table)

        console.print(Panel.fit(
            f"[bold green]✓ Done![/bold green]\n\n"
            f"Import [cyan]{output}[/cyan] into Anki as a [bold]Cloze[/bold] note type.",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--free', is_flag=True, help='Show only free providers')
def providers(free: bool):
    """List available LLM providers and their status."""
    import os

    all_providers = list_providers()

    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Free", style="green")
    table.add_column("Default Model", style="white")
    table.add_column("Status", style="yellow")

    env_vars = {
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": None,
    }

    for name, info in all_providers.items():
        if free and not info['free']:
            continue

        free_badge = "✓" if info['free'] else ""
        env_var = env_vars.get(name)

        if env_var:
            status = "✓ Key set" if os.environ.get(env_var) else f"Set {env_var}"
        else:
            status = "Local install"

        table.add_row(name, free_badge, info['default_model'], status)

    console.print(table)
    console.print()

    if free:
        console.print(Panel.fit(
            "[bold]Setup Instructions[/bold]\n\n"
            "[cyan]gemini:[/cyan] https://makersuite.google.com/app/apikey\n"
            "[cyan]openrouter:[/cyan] https://openrouter.ai/keys\n"
            "[cyan]ollama:[/cyan] https://ollama.ai/download",
            border_style="green"
        ))
    else:
        console.print("Tip: Use --free to show only free providers")


@cli.command()
@click.option('--provider', type=click.Choice(PROVIDER_CHOICES), default='gemini')
def check(provider: str):
    """Check if an LLM provider is available and configured."""
    console.print(f"[bold]Checking {provider} availability...[/bold]\n")

    try:
        if provider == 'ollama':
            from .generation.ollama import OllamaProvider
            cfg = ProviderConfig(name="ollama", model="llama3.1:8b")
            p = OllamaProvider(cfg)
            if p._check_availability():
                models = p.list_models()
                console.print("[green]✓[/green] Ollama is running")
                console.print(f"  Available models: {', '.join(models) or 'None'}")
                if not models:
                    console.print("\n  To get started: [cyan]ollama pull llama3.1:8b[/cyan]")
            else:
                console.print("[red]✗[/red] Ollama is not running")
                console.print("  Install from: [cyan]https://ollama.ai/download[/cyan]")

        elif provider == 'gemini':
            import os
            if os.environ.get("GOOGLE_API_KEY"):
                console.print("[green]✓[/green] Gemini API key found (GOOGLE_API_KEY)")
                console.print("  Free tier: 15 RPM, 1M tokens/day")
            else:
                console.print("[yellow]![/yellow] GOOGLE_API_KEY not set")
                console.print("  Get a free key at: [cyan]https://makersuite.google.com/app/apikey[/cyan]")
                console.print("  Then: [dim]export GOOGLE_API_KEY=your-key[/dim]")

        elif provider == 'openrouter':
            import os
            if os.environ.get("OPENROUTER_API_KEY"):
                console.print("[green]✓[/green] OpenRouter API key found (OPENROUTER_API_KEY)")
                console.print("  Free models available (e.g., llama-3.1-8b-instruct:free)")
            else:
                console.print("[yellow]![/yellow] OPENROUTER_API_KEY not set")
                console.print("  Get a key at: [cyan]https://openrouter.ai/keys[/cyan]")
                console.print("  Then: [dim]export OPENROUTER_API_KEY=your-key[/dim]")

        elif provider == 'openai':
            import os
            if os.environ.get("OPENAI_API_KEY"):
                console.print("[green]✓[/green] OpenAI API key found (OPENAI_API_KEY)")
            else:
                console.print("[yellow]![/yellow] OPENAI_API_KEY not set")
                console.print("  Set with: [dim]export OPENAI_API_KEY=your-key[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


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

    console.print(f"[green]✓[/green] Created config file: [cyan]{output_path}[/cyan]")
    console.print("  Edit this file and run: [dim]anki-forge generate book.epub -c config.yaml[/dim]")


@cli.command()
def models():
    """Show recommended models for each provider."""
    # Free models table
    free_table = Table(title="[bold green]FREE Options[/bold green]")
    free_table.add_column("Provider", style="cyan")
    free_table.add_column("Model", style="white")
    free_table.add_column("Notes", style="dim")

    free_table.add_row("gemini", "gemini-1.5-flash", "Fast, 1M context, 15 RPM free (default)")
    free_table.add_row("gemini", "gemini-1.0-pro", "Stable, smaller context")
    free_table.add_row("openrouter", "meta-llama/llama-3.1-8b-instruct:free", "Good general purpose")
    free_table.add_row("openrouter", "mistralai/mistral-7b-instruct:free", "Fast")
    free_table.add_row("openrouter", "google/gemma-2-9b-it:free", "Good quality")
    free_table.add_row("ollama", "llama3.1:8b", "Local, no limits, good balance")
    free_table.add_row("ollama", "mistral", "Local, fast")

    console.print(free_table)
    console.print()

    # Paid models table
    paid_table = Table(title="[bold yellow]PAID Options[/bold yellow]")
    paid_table.add_column("Provider", style="cyan")
    paid_table.add_column("Model", style="white")
    paid_table.add_column("Notes", style="dim")

    paid_table.add_row("openai", "gpt-4o-mini", "Best value, very capable")
    paid_table.add_row("openai", "gpt-4o", "Highest quality")
    paid_table.add_row("openrouter", "anthropic/claude-3-haiku", "Fast and cheap")
    paid_table.add_row("openrouter", "anthropic/claude-3-sonnet", "Good balance")

    console.print(paid_table)


@cli.command()
def wizard():
    """Interactive wizard for guided card generation.

    Walk through file selection, provider setup, and configuration
    with helpful prompts and validation.

    Requires: pip install anki-forge[tui]
    """
    try:
        from .tui import run_wizard
    except ImportError:
        console.print("[red]Wizard requires TUI dependencies.[/red]")
        console.print("Install with: pip install anki-forge[tui]")
        sys.exit(1)

    result = run_wizard()

    if result:
        # Run generation with wizard results
        console.print("\n[bold]Starting generation...[/bold]\n")

        ctx = click.Context(generate)
        ctx.invoke(
            generate,
            input_file=result["input_file"],
            output=result["output"],
            config=None,
            provider=result["provider"],
            model=None,
            mode=result["mode"],
            difficulty=result["difficulty"],
            density=result["density"],
            chunking=result["chunking"],
            chunk_size=result["chunk_size"],
            no_cache=False,
            json_output=False,
            verbose=False,
        )


@cli.command()
def tui():
    """Launch the full dashboard TUI.

    A complete graphical interface with:
    - File browser for document selection
    - Settings panel for configuration
    - Real-time generation progress
    - Output preview

    Requires: pip install anki-forge[tui]
    """
    try:
        from .tui import run_dashboard
    except ImportError:
        console.print("[red]TUI requires textual.[/red]")
        console.print("Install with: pip install anki-forge[tui]")
        sys.exit(1)

    result = run_dashboard()

    # If user chose wizard mode from TUI
    if result == "wizard":
        ctx = click.Context(wizard)
        ctx.invoke(wizard)


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
