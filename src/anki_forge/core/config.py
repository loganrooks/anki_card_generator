"""Configuration management for Anki Forge."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import os
import yaml

from .models import Difficulty, ChunkingStrategy, GenerationSettings
from .exceptions import ConfigError


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.9

    # Cost tracking
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH

    # Paragraph strategy settings
    target_length: int = 5000
    respect_boundaries: bool = True

    # Semantic strategy settings
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7

    # Sliding window settings
    window_size: int = 3000
    overlap: int = 500


@dataclass
class OutputConfig:
    """Configuration for output formats."""
    output_dir: str = "./output"
    json_output: bool = True
    csv_output: bool = True

    # Anki-specific
    deck_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Main configuration for Anki Forge."""
    # Provider settings
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        name="openai",
        model="gpt-4o-mini"
    ))

    # Chunking settings
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Generation settings
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    # Output settings
    output: OutputConfig = field(default_factory=OutputConfig)

    # Caching
    enable_cache: bool = True
    cache_dir: str = "./.cache"

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        config = cls()

        # Provider config
        if "provider" in data:
            provider_data = data["provider"]
            config.provider = ProviderConfig(
                name=provider_data.get("name", "openai"),
                model=provider_data.get("model", "gpt-4o-mini"),
                api_key=provider_data.get("api_key") or os.environ.get(
                    f"{provider_data.get('name', 'openai').upper()}_API_KEY"
                ),
                api_base=provider_data.get("api_base"),
                max_tokens=provider_data.get("max_tokens", 3000),
                temperature=provider_data.get("temperature", 0.7),
                top_p=provider_data.get("top_p", 0.9),
                cost_per_1k_input=provider_data.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=provider_data.get("cost_per_1k_output", 0.0),
            )

        # Chunking config
        if "chunking" in data:
            chunk_data = data["chunking"]
            strategy_str = chunk_data.get("strategy", "paragraph")
            config.chunking = ChunkingConfig(
                strategy=ChunkingStrategy(strategy_str),
                target_length=chunk_data.get("target_length", 5000),
                respect_boundaries=chunk_data.get("respect_boundaries", True),
                embedding_model=chunk_data.get("embedding_model", "all-MiniLM-L6-v2"),
                similarity_threshold=chunk_data.get("similarity_threshold", 0.7),
                window_size=chunk_data.get("window_size", 3000),
                overlap=chunk_data.get("overlap", 500),
            )

        # Generation settings
        if "generation" in data:
            gen_data = data["generation"]
            difficulty = gen_data.get("difficulty")
            if difficulty:
                difficulty = Difficulty(difficulty)
            config.generation = GenerationSettings(
                target_density=gen_data.get("target_density", 0.30),
                density_tolerance=gen_data.get("density_tolerance", 0.05),
                min_sentences=gen_data.get("min_sentences", 3),
                max_sentences=gen_data.get("max_sentences", 8),
                target_words=gen_data.get("target_words", 100),
                target_key_terms=gen_data.get("target_key_terms", True),
                target_definitions=gen_data.get("target_definitions", True),
                target_foreign_phrases=gen_data.get("target_foreign_phrases", True),
                target_full_phrases=gen_data.get("target_full_phrases", True),
                max_cloze_groups=gen_data.get("max_cloze_groups", 3),
                group_thematically=gen_data.get("group_thematically", True),
                difficulty=difficulty,
            )

        # Output config
        if "output" in data:
            out_data = data["output"]
            config.output = OutputConfig(
                output_dir=out_data.get("output_dir", "./output"),
                json_output=out_data.get("json_output", True),
                csv_output=out_data.get("csv_output", True),
                deck_name=out_data.get("deck_name"),
                tags=out_data.get("tags", []),
            )

        # Other settings
        config.enable_cache = data.get("enable_cache", True)
        config.cache_dir = data.get("cache_dir", "./.cache")
        config.log_level = data.get("log_level", "INFO")
        config.verbose = data.get("verbose", False)

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "provider": {
                "name": self.provider.name,
                "model": self.provider.model,
                "api_base": self.provider.api_base,
                "max_tokens": self.provider.max_tokens,
                "temperature": self.provider.temperature,
                "top_p": self.provider.top_p,
            },
            "chunking": {
                "strategy": self.chunking.strategy.value,
                "target_length": self.chunking.target_length,
                "respect_boundaries": self.chunking.respect_boundaries,
            },
            "generation": {
                "target_density": self.generation.target_density,
                "density_tolerance": self.generation.density_tolerance,
                "difficulty": self.generation.difficulty.value if self.generation.difficulty else None,
            },
            "output": {
                "output_dir": self.output.output_dir,
                "json_output": self.output.json_output,
                "csv_output": self.output.csv_output,
            },
            "enable_cache": self.enable_cache,
            "log_level": self.log_level,
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file or use defaults.

    Searches for config in:
    1. Provided path
    2. ./anki_forge.yaml
    3. ~/.config/anki_forge/config.yaml
    4. Falls back to defaults
    """
    search_paths = []

    if config_path:
        search_paths.append(Path(config_path))

    search_paths.extend([
        Path("./anki_forge.yaml"),
        Path("./config.yaml"),
        Path.home() / ".config" / "anki_forge" / "config.yaml",
    ])

    for path in search_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                    return Config.from_dict(data or {})
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in config file: {e}")
            except Exception as e:
                raise ConfigError(f"Error loading config from {path}: {e}")

    # Return default config
    return Config()


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
