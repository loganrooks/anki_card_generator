"""Calibre UI action for Anki Forge."""

import os
import tempfile
from functools import partial

from calibre.gui2 import error_dialog, info_dialog, question_dialog
from calibre.gui2.actions import InterfaceAction
from calibre.utils.config import JSONConfig

from qt.core import QMenu, QProgressDialog, QThread, pyqtSignal

# Plugin configuration
prefs = JSONConfig("plugins/anki_forge")
prefs.defaults["provider"] = "gemini"
prefs.defaults["mode"] = "hybrid"
prefs.defaults["density"] = 0.30
prefs.defaults["difficulty"] = "medium"
prefs.defaults["output_dir"] = ""


class GenerateWorker(QThread):
    """Background worker for card generation."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, int)  # output_path, card_count
    error = pyqtSignal(str)

    def __init__(self, book_path, output_path, settings):
        super().__init__()
        self.book_path = book_path
        self.output_path = output_path
        self.settings = settings

    def run(self):
        """Run card generation in background thread."""
        try:
            # Import anki_forge modules
            from anki_forge.parsers import get_parser
            from anki_forge.chunking import get_chunker
            from anki_forge.core.config import Config, ChunkingConfig, ProviderConfig
            from anki_forge.core.models import GenerationSettings, ChunkingStrategy, Difficulty
            from anki_forge.generation import get_provider, CardGenerator
            from anki_forge.output import AnkiCsvWriter

            self.progress.emit("Parsing document...")

            # Parse document
            parser = get_parser(self.book_path)
            document = parser.parse(self.book_path)

            self.progress.emit(f"Found {len(document.chunks)} sections")

            # Create config
            cfg = Config()
            cfg.provider = ProviderConfig(
                name=self.settings["provider"],
                api_key=os.environ.get(self._get_api_key_env()),
            )
            cfg.chunking = ChunkingConfig(
                strategy=ChunkingStrategy.PARAGRAPH,
                target_length=5000,
            )

            # Chunk document
            self.progress.emit("Chunking content...")
            chunker = get_chunker(cfg.chunking)
            chunks = chunker.chunk(document)

            self.progress.emit(f"Created {len(chunks)} chunks")

            # Set up generation
            settings = GenerationSettings(
                target_density=self.settings["density"],
                difficulty=Difficulty(self.settings["difficulty"]),
            )

            provider = get_provider(cfg.provider)
            generator = CardGenerator(
                provider=provider,
                settings=settings,
                mode=self.settings["mode"],
            )

            # Generate cards
            self.progress.emit("Generating cards (this may take a while)...")
            result = generator.generate(chunks, document.author)

            self.progress.emit(f"Generated {len(result.cards)} cards")

            # Write output
            self.progress.emit("Writing cards to file...")
            writer = AnkiCsvWriter(tags=["anki-forge", "calibre"])
            writer.write_with_metadata(
                result.cards,
                self.output_path,
                deck_name=document.title,
            )

            self.finished.emit(self.output_path, len(result.cards))

        except ImportError as e:
            self.error.emit(
                f"Anki Forge not installed. Install with:\n"
                f"pip install anki-forge[all]\n\n"
                f"Error: {e}"
            )
        except Exception as e:
            self.error.emit(str(e))

    def _get_api_key_env(self):
        """Get environment variable name for API key."""
        provider_keys = {
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": None,
        }
        return provider_keys.get(self.settings["provider"], "GOOGLE_API_KEY")


class AnkiForgeAction(InterfaceAction):
    """Calibre toolbar action for generating Anki cards."""

    name = "Anki Forge"
    action_spec = ("Anki Forge", None, "Generate Anki flashcards from selected books", None)
    popup_type = QMenu.MenuButtonPopup
    action_add_menu = True

    def genesis(self):
        """Initialize the action."""
        self.menu = QMenu(self.gui)
        self.create_menu_actions()
        self.qaction.setMenu(self.menu)
        self.qaction.triggered.connect(self.generate_cards)

        # Set icon
        icon = get_icons("images/icon.png", "Anki Forge")
        self.qaction.setIcon(icon)

    def create_menu_actions(self):
        """Create menu items."""
        self.menu.clear()

        # Main action
        self.create_menu_item_ex(
            self.menu,
            "Generate Cards from Selected",
            unique_name="Generate Cards",
            triggered=self.generate_cards,
        )

        self.menu.addSeparator()

        # Configuration
        self.create_menu_item_ex(
            self.menu,
            "Configure",
            unique_name="Configure Anki Forge",
            triggered=self.show_config,
        )

    def generate_cards(self):
        """Generate Anki cards from selected books."""
        rows = self.gui.library_view.selectionModel().selectedRows()
        if not rows:
            error_dialog(
                self.gui,
                "No books selected",
                "Please select one or more books to generate cards from.",
                show=True,
            )
            return

        # Get book paths
        book_ids = list(map(self.gui.library_view.model().id, rows))
        db = self.gui.current_db.new_api

        for book_id in book_ids:
            # Get book formats
            formats = db.formats(book_id, verify_formats=True)

            if not formats:
                error_dialog(
                    self.gui,
                    "No formats",
                    f"Book has no supported formats.",
                    show=True,
                )
                continue

            # Prefer EPUB, then PDF, then MOBI
            preferred = ["EPUB", "PDF", "MOBI", "AZW3", "AZW"]
            selected_format = None
            for fmt in preferred:
                if fmt in formats:
                    selected_format = fmt
                    break

            if not selected_format:
                selected_format = formats[0]

            # Get path to format
            book_path = db.format_abspath(book_id, selected_format)
            if not book_path:
                continue

            # Get output path
            title = db.field_for("title", book_id)
            output_dir = prefs["output_dir"] or tempfile.gettempdir()
            output_path = os.path.join(
                output_dir,
                f"{title}_anki_cards.csv".replace("/", "_"),
            )

            # Confirm with user
            if not question_dialog(
                self.gui,
                "Generate Anki Cards",
                f"Generate flashcards from:\n{title}\n\n"
                f"Using provider: {prefs['provider']}\n"
                f"Output: {output_path}\n\n"
                f"Continue?",
            ):
                continue

            # Start generation
            self._run_generation(book_path, output_path)

    def _run_generation(self, book_path, output_path):
        """Run card generation with progress dialog."""
        settings = {
            "provider": prefs["provider"],
            "mode": prefs["mode"],
            "density": prefs["density"],
            "difficulty": prefs["difficulty"],
        }

        # Create progress dialog
        progress = QProgressDialog(
            "Initializing...",
            "Cancel",
            0,
            0,
            self.gui,
        )
        progress.setWindowTitle("Generating Anki Cards")
        progress.setMinimumDuration(0)

        # Create worker
        worker = GenerateWorker(book_path, output_path, settings)

        def on_progress(msg):
            progress.setLabelText(msg)

        def on_finished(path, count):
            progress.close()
            info_dialog(
                self.gui,
                "Cards Generated",
                f"Successfully generated {count} cards!\n\n"
                f"Output: {path}\n\n"
                f"Import this file into Anki as a Cloze note type.",
                show=True,
            )

        def on_error(msg):
            progress.close()
            error_dialog(
                self.gui,
                "Generation Failed",
                msg,
                show=True,
            )

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)

        progress.canceled.connect(worker.terminate)

        worker.start()
        progress.exec_()

    def show_config(self):
        """Show configuration dialog."""
        self.interface_action_base_plugin.do_user_config(self.gui)


def get_icons(path, name):
    """Get icon for plugin."""
    from calibre.gui2 import get_icons as calibre_get_icons
    try:
        return calibre_get_icons(path)
    except Exception:
        from qt.core import QIcon
        return QIcon()
