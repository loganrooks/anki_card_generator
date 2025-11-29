"""Configuration widget for Anki Forge Calibre plugin."""

from qt.core import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QPushButton, QFileDialog,
)
from calibre.utils.config import JSONConfig

# Load preferences
prefs = JSONConfig("plugins/anki_forge")
prefs.defaults["provider"] = "gemini"
prefs.defaults["mode"] = "hybrid"
prefs.defaults["density"] = 0.30
prefs.defaults["difficulty"] = "medium"
prefs.defaults["output_dir"] = ""
prefs.defaults["api_key_gemini"] = ""
prefs.defaults["api_key_openrouter"] = ""
prefs.defaults["api_key_openai"] = ""


class ConfigWidget(QWidget):
    """Configuration widget for plugin settings."""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Provider settings
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout()
        provider_group.setLayout(provider_layout)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["gemini", "openrouter", "openai", "ollama"])
        provider_layout.addRow("Provider:", self.provider_combo)

        self.api_key_gemini = QLineEdit()
        self.api_key_gemini.setPlaceholderText("GOOGLE_API_KEY (or set env var)")
        self.api_key_gemini.setEchoMode(QLineEdit.Password)
        provider_layout.addRow("Gemini Key:", self.api_key_gemini)

        self.api_key_openrouter = QLineEdit()
        self.api_key_openrouter.setPlaceholderText("OPENROUTER_API_KEY (or set env var)")
        self.api_key_openrouter.setEchoMode(QLineEdit.Password)
        provider_layout.addRow("OpenRouter Key:", self.api_key_openrouter)

        self.api_key_openai = QLineEdit()
        self.api_key_openai.setPlaceholderText("OPENAI_API_KEY (or set env var)")
        self.api_key_openai.setEchoMode(QLineEdit.Password)
        provider_layout.addRow("OpenAI Key:", self.api_key_openai)

        layout.addWidget(provider_group)

        # Generation settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QFormLayout()
        gen_group.setLayout(gen_layout)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["hybrid", "direct", "hybrid_batched"])
        gen_layout.addRow("Mode:", self.mode_combo)

        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["easy", "medium", "hard", "expert"])
        gen_layout.addRow("Difficulty:", self.difficulty_combo)

        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.10, 0.50)
        self.density_spin.setSingleStep(0.05)
        self.density_spin.setSuffix(" (cloze density)")
        gen_layout.addRow("Density:", self.density_spin)

        layout.addWidget(gen_group)

        # Output settings
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        output_group.setLayout(output_layout)

        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Default: temp directory")
        output_layout.addWidget(self.output_dir)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(browse_btn)

        layout.addWidget(output_group)

        # Help text
        help_label = QLabel(
            "<p><b>Setup:</b></p>"
            "<ol>"
            "<li>Get a free Gemini API key from "
            "<a href='https://makersuite.google.com/app/apikey'>Google AI Studio</a></li>"
            "<li>Enter the key above or set GOOGLE_API_KEY environment variable</li>"
            "<li>Select books in Calibre and click the Anki Forge button</li>"
            "</ol>"
            "<p><b>Tip:</b> Use 'hybrid' mode for best cloze density control.</p>"
        )
        help_label.setOpenExternalLinks(True)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        layout.addStretch()

    def browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir.text(),
        )
        if path:
            self.output_dir.setText(path)

    def load_settings(self):
        """Load settings from preferences."""
        self.provider_combo.setCurrentText(prefs["provider"])
        self.mode_combo.setCurrentText(prefs["mode"])
        self.difficulty_combo.setCurrentText(prefs["difficulty"])
        self.density_spin.setValue(prefs["density"])
        self.output_dir.setText(prefs["output_dir"])
        self.api_key_gemini.setText(prefs["api_key_gemini"])
        self.api_key_openrouter.setText(prefs["api_key_openrouter"])
        self.api_key_openai.setText(prefs["api_key_openai"])

    def save_settings(self):
        """Save settings to preferences."""
        prefs["provider"] = self.provider_combo.currentText()
        prefs["mode"] = self.mode_combo.currentText()
        prefs["difficulty"] = self.difficulty_combo.currentText()
        prefs["density"] = self.density_spin.value()
        prefs["output_dir"] = self.output_dir.text()
        prefs["api_key_gemini"] = self.api_key_gemini.text()
        prefs["api_key_openrouter"] = self.api_key_openrouter.text()
        prefs["api_key_openai"] = self.api_key_openai.text()
