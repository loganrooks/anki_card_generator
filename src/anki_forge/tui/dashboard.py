"""Full Textual TUI dashboard for Anki Forge."""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def check_textual():
    """Check if textual is installed."""
    try:
        import textual
        return True
    except ImportError:
        return False


def run_dashboard():
    """Run the full Textual dashboard TUI."""
    if not check_textual():
        console.print(
            "[red]Dashboard requires textual. Install with:[/red]\n"
            "  pip install anki-forge[tui]"
        )
        return

    # Import here to avoid import errors if textual not installed
    from .dashboard_app import AnkiForgeApp

    app = AnkiForgeApp()
    app.run()


# Separate file to keep imports clean
# This will be dashboard_app.py content below
