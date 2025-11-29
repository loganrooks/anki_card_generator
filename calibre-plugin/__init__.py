"""Anki Forge Calibre Plugin.

Adds a toolbar button to generate Anki flashcards from selected ebooks.
"""

from calibre.customize import InterfaceActionBase

PLUGIN_NAME = "Anki Forge"
PLUGIN_VERSION = (0, 3, 0)
PLUGIN_AUTHORS = "Logan Rooks"
PLUGIN_DESCRIPTION = "Generate Anki flashcards from ebooks using LLMs"


class AnkiForgePlugin(InterfaceActionBase):
    """Calibre plugin for Anki Forge integration."""

    name = PLUGIN_NAME
    description = PLUGIN_DESCRIPTION
    supported_platforms = ["windows", "osx", "linux"]
    author = PLUGIN_AUTHORS
    version = PLUGIN_VERSION
    minimum_calibre_version = (5, 0, 0)

    # The actual plugin that will be loaded
    actual_plugin = "calibre_plugins.anki_forge.ui:AnkiForgeAction"

    def is_customizable(self):
        """Allow plugin customization."""
        return True

    def config_widget(self):
        """Return the configuration widget."""
        from calibre_plugins.anki_forge.config import ConfigWidget
        return ConfigWidget()

    def save_settings(self, config_widget):
        """Save plugin settings."""
        config_widget.save_settings()

    def customization_help(self, gui=False):
        """Return customization help."""
        return "Configure LLM provider, API keys, and card generation settings."
