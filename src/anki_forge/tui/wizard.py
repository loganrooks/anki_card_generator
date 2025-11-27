"""Interactive wizard for guided card generation."""

import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_questionary():
    """Check if questionary is installed."""
    try:
        import questionary
        return True
    except ImportError:
        return False


def run_wizard() -> Optional[dict]:
    """Run the interactive wizard for card generation.

    Returns:
        Dict with configuration options, or None if cancelled
    """
    if not check_questionary():
        console.print(
            "[red]Wizard requires questionary. Install with:[/red]\n"
            "  pip install anki-forge[tui]"
        )
        return None

    import questionary
    from questionary import Style

    # Custom style for the wizard
    custom_style = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:green'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan'),
        ('selected', 'fg:green'),
    ])

    console.print(Panel.fit(
        "[bold cyan]Anki Forge Wizard[/bold cyan]\n"
        "Generate flashcards from your documents",
        border_style="cyan"
    ))
    console.print()

    # Step 1: Select input file
    input_file = questionary.path(
        "Select document to process:",
        style=custom_style,
        validate=lambda p: Path(p).exists() or "File not found",
    ).ask()

    if not input_file:
        console.print("[yellow]Cancelled.[/yellow]")
        return None

    # Check file type
    ext = Path(input_file).suffix.lower()
    if ext not in ['.epub', '.pdf']:
        console.print(f"[red]Unsupported file type: {ext}[/red]")
        console.print("Supported: .epub, .pdf")
        return None

    console.print(f"  [green]✓[/green] Selected: {input_file}")

    # Step 2: Select provider
    providers = [
        {"name": "Gemini (free tier)", "value": "gemini"},
        {"name": "OpenRouter (free models available)", "value": "openrouter"},
        {"name": "Ollama (local, no limits)", "value": "ollama"},
        {"name": "OpenAI (paid)", "value": "openai"},
    ]

    provider = questionary.select(
        "Select LLM provider:",
        choices=[questionary.Choice(p["name"], value=p["value"]) for p in providers],
        style=custom_style,
    ).ask()

    if not provider:
        return None

    console.print(f"  [green]✓[/green] Provider: {provider}")

    # Check API key
    env_vars = {
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    if provider in env_vars:
        env_var = env_vars[provider]
        if not os.environ.get(env_var):
            console.print(f"  [yellow]⚠ {env_var} not set[/yellow]")

            set_key = questionary.confirm(
                f"Would you like to enter your API key now?",
                style=custom_style,
                default=True,
            ).ask()

            if set_key:
                api_key = questionary.password(
                    f"Enter {env_var}:",
                    style=custom_style,
                ).ask()
                if api_key:
                    os.environ[env_var] = api_key
                    console.print(f"  [green]✓[/green] API key set")
                else:
                    console.print("[yellow]Skipped API key.[/yellow]")

    # Step 3: Select mode
    modes = [
        {"name": "Hybrid (recommended - precise density control)", "value": "hybrid"},
        {"name": "Hybrid Batched (best for Gemini rate limits)", "value": "hybrid_batched"},
        {"name": "Direct (simple, density not guaranteed)", "value": "direct"},
    ]

    mode = questionary.select(
        "Select generation mode:",
        choices=[questionary.Choice(m["name"], value=m["value"]) for m in modes],
        style=custom_style,
    ).ask()

    if not mode:
        return None

    console.print(f"  [green]✓[/green] Mode: {mode}")

    # Step 4: Select difficulty
    difficulties = [
        {"name": "Easy (15% density - key terms only)", "value": "easy"},
        {"name": "Medium (25% density - terms + phrases)", "value": "medium"},
        {"name": "Hard (35% density - high coverage)", "value": "hard"},
        {"name": "Expert (45% density - maximum)", "value": "expert"},
    ]

    difficulty = questionary.select(
        "Select difficulty level:",
        choices=[questionary.Choice(d["name"], value=d["value"]) for d in difficulties],
        style=custom_style,
        default="medium",
    ).ask()

    if not difficulty:
        return None

    console.print(f"  [green]✓[/green] Difficulty: {difficulty}")

    # Step 5: Advanced options
    show_advanced = questionary.confirm(
        "Configure advanced options?",
        style=custom_style,
        default=False,
    ).ask()

    density = 0.30
    chunking = "paragraph"
    chunk_size = 5000

    if show_advanced:
        # Custom density
        density_str = questionary.text(
            "Target cloze density (0.0-1.0):",
            style=custom_style,
            default="0.30",
            validate=lambda x: _validate_density(x),
        ).ask()
        density = float(density_str) if density_str else 0.30

        # Chunking strategy
        chunking_options = [
            {"name": "Paragraph (respects natural boundaries)", "value": "paragraph"},
            {"name": "Sliding Window (fixed size with overlap)", "value": "sliding_window"},
            {"name": "Semantic (similarity-based)", "value": "semantic"},
        ]
        chunking = questionary.select(
            "Chunking strategy:",
            choices=[questionary.Choice(c["name"], value=c["value"]) for c in chunking_options],
            style=custom_style,
        ).ask() or "paragraph"

        # Chunk size
        chunk_size_str = questionary.text(
            "Target chunk size (characters):",
            style=custom_style,
            default="5000",
            validate=lambda x: x.isdigit() and int(x) > 0 or "Must be positive integer",
        ).ask()
        chunk_size = int(chunk_size_str) if chunk_size_str else 5000

    # Step 6: Output path
    default_output = Path(input_file).stem + "_cards.csv"
    output_file = questionary.text(
        "Output file path:",
        style=custom_style,
        default=default_output,
    ).ask()

    if not output_file:
        output_file = default_output

    console.print(f"  [green]✓[/green] Output: {output_file}")

    # Summary
    console.print()
    console.print(Panel.fit(
        "[bold]Configuration Summary[/bold]",
        border_style="green"
    ))

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Input", input_file)
    table.add_row("Output", output_file)
    table.add_row("Provider", provider)
    table.add_row("Mode", mode)
    table.add_row("Difficulty", difficulty)
    table.add_row("Density", f"{density:.0%}")
    table.add_row("Chunking", chunking)
    console.print(table)
    console.print()

    # Confirm
    proceed = questionary.confirm(
        "Start generation?",
        style=custom_style,
        default=True,
    ).ask()

    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        return None

    return {
        "input_file": input_file,
        "output": output_file,
        "provider": provider,
        "mode": mode,
        "difficulty": difficulty,
        "density": density,
        "chunking": chunking,
        "chunk_size": chunk_size,
    }


def _validate_density(value: str) -> bool | str:
    """Validate density input."""
    try:
        d = float(value)
        if 0.0 <= d <= 1.0:
            return True
        return "Must be between 0.0 and 1.0"
    except ValueError:
        return "Must be a number"
