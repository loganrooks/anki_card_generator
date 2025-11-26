"""Textual TUI Application for Anki Forge."""

from pathlib import Path
from typing import Optional
import os

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Header, Footer, Static, Button, Input, Select,
        ProgressBar, Label, DirectoryTree, Log, Rule
    )
    from textual.screen import Screen
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


if TEXTUAL_AVAILABLE:

    class SettingsPanel(Static):
        """Settings panel widget."""

        def compose(self) -> ComposeResult:
            yield Label("Provider:", classes="setting-label")
            yield Select(
                [(p, p) for p in ["gemini", "openrouter", "ollama", "openai"]],
                id="provider-select",
                value="gemini",
            )

            yield Label("Mode:", classes="setting-label")
            yield Select(
                [
                    ("Hybrid (precise density)", "hybrid"),
                    ("Hybrid Batched (rate limits)", "hybrid_batched"),
                    ("Direct (simple)", "direct"),
                ],
                id="mode-select",
                value="hybrid",
            )

            yield Label("Difficulty:", classes="setting-label")
            yield Select(
                [
                    ("Easy (15%)", "easy"),
                    ("Medium (25%)", "medium"),
                    ("Hard (35%)", "hard"),
                    ("Expert (45%)", "expert"),
                ],
                id="difficulty-select",
                value="medium",
            )

            yield Label("Density:", classes="setting-label")
            yield Input(value="0.30", id="density-input", placeholder="0.0-1.0")

            yield Label("Chunk Size:", classes="setting-label")
            yield Input(value="5000", id="chunk-size-input", placeholder="characters")


    class FilePanel(Static):
        """File selection panel."""

        def compose(self) -> ComposeResult:
            yield Label("Select Document:", classes="panel-title")
            yield Input(placeholder="Enter file path...", id="file-input")
            yield Label("Or browse:", classes="hint")
            yield DirectoryTree(".", id="file-tree")


    class OutputPanel(Static):
        """Output and progress panel."""

        def compose(self) -> ComposeResult:
            yield Label("Generation Output", classes="panel-title")
            yield Log(id="output-log", highlight=True, max_lines=100)
            yield Rule()
            yield Label("Progress:", classes="setting-label")
            yield ProgressBar(id="progress-bar", show_eta=True)
            yield Label("Ready", id="status-label")


    class AnkiForgeApp(App):
        """Main Textual application for Anki Forge."""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 3 2;
            grid-columns: 1fr 1fr 2fr;
            grid-rows: 1fr auto;
        }

        #file-panel {
            column-span: 1;
            row-span: 1;
            border: solid green;
            padding: 1;
        }

        #settings-panel {
            column-span: 1;
            row-span: 1;
            border: solid cyan;
            padding: 1;
        }

        #output-panel {
            column-span: 1;
            row-span: 2;
            border: solid yellow;
            padding: 1;
        }

        #button-bar {
            column-span: 2;
            row-span: 1;
            height: auto;
            padding: 1;
            align: center middle;
        }

        .panel-title {
            text-style: bold;
            color: cyan;
            margin-bottom: 1;
        }

        .setting-label {
            margin-top: 1;
            color: white;
        }

        .hint {
            color: gray;
            text-style: italic;
        }

        Button {
            margin: 1;
        }

        #generate-btn {
            background: green;
        }

        #quit-btn {
            background: red;
        }

        DirectoryTree {
            height: 15;
        }

        Log {
            height: 20;
        }

        ProgressBar {
            margin: 1 0;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("g", "generate", "Generate"),
            Binding("w", "wizard", "Wizard"),
        ]

        def __init__(self):
            super().__init__()
            self.selected_file: Optional[str] = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)

            yield Container(
                FilePanel(id="file-panel"),
                id="file-container",
            )

            yield Container(
                SettingsPanel(id="settings-panel"),
                id="settings-container",
            )

            yield Container(
                OutputPanel(id="output-panel"),
                id="output-container",
            )

            yield Horizontal(
                Button("Generate", id="generate-btn", variant="success"),
                Button("Wizard Mode", id="wizard-btn", variant="primary"),
                Button("Quit", id="quit-btn", variant="error"),
                id="button-bar",
            )

            yield Footer()

        def on_mount(self) -> None:
            """Called when app is mounted."""
            self.title = "Anki Forge"
            self.sub_title = "Generate Anki Flashcards"
            self._log("Welcome to Anki Forge!")
            self._log("Select a document and configure settings, then press Generate.")
            self._check_api_keys()

        def _check_api_keys(self) -> None:
            """Check for API keys and log status."""
            keys = {
                "GOOGLE_API_KEY": "Gemini",
                "OPENROUTER_API_KEY": "OpenRouter",
                "OPENAI_API_KEY": "OpenAI",
            }
            found = []
            for key, name in keys.items():
                if os.environ.get(key):
                    found.append(name)

            if found:
                self._log(f"API keys found: {', '.join(found)}")
            else:
                self._log("[yellow]No API keys found. Set GOOGLE_API_KEY for Gemini.[/yellow]")

        def _log(self, message: str) -> None:
            """Log a message to the output panel."""
            log = self.query_one("#output-log", Log)
            log.write_line(message)

        def _set_status(self, message: str) -> None:
            """Set the status label."""
            label = self.query_one("#status-label", Label)
            label.update(message)

        def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
            """Handle file selection from tree."""
            path = str(event.path)
            if path.endswith(('.epub', '.pdf')):
                self.selected_file = path
                self.query_one("#file-input", Input).value = path
                self._log(f"Selected: {path}")
            else:
                self._log(f"[red]Unsupported: {path}[/red]")

        def on_input_changed(self, event: Input.Changed) -> None:
            """Handle input changes."""
            if event.input.id == "file-input":
                path = event.value
                if Path(path).exists() and path.endswith(('.epub', '.pdf')):
                    self.selected_file = path

        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            if event.button.id == "generate-btn":
                self.action_generate()
            elif event.button.id == "wizard-btn":
                self.action_wizard()
            elif event.button.id == "quit-btn":
                self.action_quit()

        def action_generate(self) -> None:
            """Start card generation."""
            if not self.selected_file:
                self._log("[red]Please select a file first.[/red]")
                return

            if not Path(self.selected_file).exists():
                self._log(f"[red]File not found: {self.selected_file}[/red]")
                return

            # Get settings
            provider = self.query_one("#provider-select", Select).value
            mode = self.query_one("#mode-select", Select).value
            difficulty = self.query_one("#difficulty-select", Select).value
            density = self.query_one("#density-input", Input).value
            chunk_size = self.query_one("#chunk-size-input", Input).value

            self._log("")
            self._log(f"[bold]Starting generation...[/bold]")
            self._log(f"  File: {self.selected_file}")
            self._log(f"  Provider: {provider}")
            self._log(f"  Mode: {mode}")
            self._log(f"  Difficulty: {difficulty}")

            self._set_status("Generating...")

            # Run generation in background
            self.run_worker(
                self._run_generation(
                    self.selected_file, provider, mode, difficulty, density, chunk_size
                ),
                name="generation",
            )

        async def _run_generation(
            self, file_path: str, provider: str, mode: str,
            difficulty: str, density: str, chunk_size: str
        ) -> None:
            """Run the generation process."""
            try:
                from ..core.config import Config, ProviderConfig, ChunkingConfig
                from ..core.models import GenerationSettings, Difficulty, ChunkingStrategy
                from ..parsers import get_parser
                from ..chunking import get_chunker
                from ..generation import get_provider, CardGenerator, BatchConfig
                from ..output import AnkiCsvWriter

                progress_bar = self.query_one("#progress-bar", ProgressBar)

                # Parse document
                self._log("Parsing document...")
                parser = get_parser(file_path)
                document = parser.parse(file_path)
                self._log(f"  Found {len(document.chunks)} sections")

                # Chunk document
                self._log("Chunking...")
                chunking_config = ChunkingConfig(
                    strategy=ChunkingStrategy.PARAGRAPH,
                    target_length=int(chunk_size),
                )
                chunker = get_chunker(chunking_config)
                chunks = chunker.chunk(document)
                self._log(f"  Created {len(chunks)} chunks")

                # Set up provider
                self._log(f"Setting up {provider} provider...")
                provider_config = ProviderConfig(name=provider, model="")
                llm = get_provider(provider_config)

                # Set up generator
                settings = GenerationSettings(
                    target_density=float(density),
                    difficulty=Difficulty(difficulty),
                )

                batch_config = BatchConfig() if mode == "hybrid_batched" else None
                generator = CardGenerator(llm, settings, mode=mode, batch_config=batch_config)

                # Generate cards
                self._log(f"Generating cards ({mode} mode)...")
                progress_bar.update(total=len(chunks), progress=0)

                result = generator.generate(chunks, document.author)

                progress_bar.update(progress=len(chunks))

                # Write output
                output_path = Path(file_path).stem + "_cards.csv"
                writer = AnkiCsvWriter(tags=["anki-forge", difficulty])
                writer.write(result.cards, output_path)

                self._log("")
                self._log(f"[green]✓ Generated {len(result.cards)} cards![/green]")
                self._log(f"  Output: {output_path}")
                self._log(f"  Avg density: {result.avg_density:.1%}")
                self._log(f"  Tokens used: {result.total_tokens:,}")

                self._set_status(f"Done! {len(result.cards)} cards generated")

            except Exception as e:
                self._log(f"[red]Error: {e}[/red]")
                self._set_status("Error")

        def action_wizard(self) -> None:
            """Switch to wizard mode."""
            self._log("Launching wizard mode...")
            # Exit the app and run wizard
            self.exit(result="wizard")

        def action_quit(self) -> None:
            """Quit the application."""
            self.exit()


else:
    # Fallback if textual not installed
    class AnkiForgeApp:
        def run(self):
            print("Textual not installed. Run: pip install anki-forge[tui]")
