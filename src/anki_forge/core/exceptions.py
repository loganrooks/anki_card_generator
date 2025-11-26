"""Custom exceptions for Anki Forge."""


class AnkiForgeError(Exception):
    """Base exception for all Anki Forge errors."""
    pass


class ParserError(AnkiForgeError):
    """Error during document parsing."""

    def __init__(self, message: str, file_path: str = None, details: str = None):
        self.file_path = file_path
        self.details = details
        super().__init__(message)


class ChunkingError(AnkiForgeError):
    """Error during text chunking."""

    def __init__(self, message: str, chunk_index: int = None):
        self.chunk_index = chunk_index
        super().__init__(message)


class GenerationError(AnkiForgeError):
    """Error during card generation."""

    def __init__(self, message: str, chunk_index: int = None, provider: str = None):
        self.chunk_index = chunk_index
        self.provider = provider
        super().__init__(message)


class ValidationError(AnkiForgeError):
    """Error during card validation."""

    def __init__(self, message: str, card_index: int = None, field: str = None):
        self.card_index = card_index
        self.field = field
        super().__init__(message)


class ProviderError(AnkiForgeError):
    """Error from LLM provider."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        status_code: int = None,
        retryable: bool = False
    ):
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(message)


class ConfigError(AnkiForgeError):
    """Error in configuration."""

    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message)


class CacheError(AnkiForgeError):
    """Error with caching system."""
    pass
