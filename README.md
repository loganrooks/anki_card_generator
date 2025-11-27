# Anki Forge

Generate high-quality Anki flashcards with cloze deletions from EPUB and PDF documents using LLMs.

## Features

- **Multiple LLM Providers**: Gemini (free), OpenRouter (free models), Ollama (local), OpenAI
- **Precise Density Control**: Hybrid mode guarantees target cloze density (e.g., 30% ± 5%)
- **Importance-Based Selection**: LLM scores targets 1-10, highest priority included first
- **Document Support**: EPUB and PDF parsing with metadata extraction
- **Batch Processing**: Efficient batching for rate-limited providers like Gemini
- **Response Caching**: Save money by caching LLM responses

## Installation

```bash
# Basic installation
pip install anki-forge

# With PDF support
pip install anki-forge[pdf]

# With Gemini provider
pip install anki-forge[gemini]

# All features
pip install anki-forge[all]

# Development
pip install -e ".[dev]"
```

## Quick Start

```bash
# Set up a free provider (Gemini)
export GOOGLE_API_KEY=your-key  # Get from https://makersuite.google.com/app/apikey

# Generate cards from a book
anki-forge generate book.epub -o cards.csv

# Use hybrid mode for precise density control
anki-forge generate book.pdf --mode hybrid --density 0.30

# Use batched mode for Gemini (handles 15 RPM rate limit)
anki-forge generate book.epub --mode hybrid_batched
```

## Generation Modes

### Direct Mode (default: `--mode direct`)
LLM generates complete cards with cloze deletions. Simple but density not guaranteed.

### Hybrid Mode (`--mode hybrid`)
1. LLM identifies what to cloze (semantic task) with importance scores
2. Rules engine applies clozes with precise density control
3. Guarantees target density is met (e.g., 30% ± 5%)

### Hybrid Batched Mode (`--mode hybrid_batched`)
Like hybrid, but batches multiple chunks per API call. Best for:
- Gemini (15 RPM free tier, 1M token context)
- Other rate-limited providers with large context windows

## Providers

| Provider | Free Tier | Setup |
|----------|-----------|-------|
| Gemini | 15 RPM, 1M tokens/day | `export GOOGLE_API_KEY=...` |
| OpenRouter | Free models available | `export OPENROUTER_API_KEY=...` |
| Ollama | Unlimited (local) | Install from ollama.ai |
| OpenAI | Paid only | `export OPENAI_API_KEY=...` |

```bash
# List available providers
anki-forge providers --free

# Check provider status
anki-forge check --provider gemini
```

## Configuration

Create a config file:
```bash
anki-forge init-config config.yaml
```

Example `config.yaml`:
```yaml
provider:
  name: gemini
  model: gemini-1.5-flash

chunking:
  strategy: paragraph
  target_length: 5000

generation:
  target_density: 0.30
  density_tolerance: 0.05
  difficulty: medium

mode: hybrid
enable_cache: true
```

Run with config:
```bash
anki-forge generate book.epub -c config.yaml
```

## CLI Reference

```
anki-forge generate [OPTIONS] INPUT_FILE

Options:
  -o, --output PATH          Output CSV file path
  -c, --config PATH          Config file path
  --provider [openai|ollama|gemini|openrouter]
                             LLM provider (default: gemini)
  --model TEXT               Model name
  --mode [direct|hybrid|hybrid_batched]
                             Generation mode (default: hybrid)
  --difficulty [easy|medium|hard|expert]
                             Difficulty preset (default: medium)
  --density FLOAT            Target cloze density 0.0-1.0 (default: 0.30)
  --chunking [paragraph|semantic|sliding_window]
                             Chunking strategy (default: paragraph)
  --chunk-size INTEGER       Target chunk size in chars (default: 5000)
  --no-cache                 Disable response caching
  --json                     Also output JSON for debugging
  -v, --verbose              Verbose output
```

## Difficulty Presets

| Preset | Density | Description |
|--------|---------|-------------|
| easy | 15% | Key terms only |
| medium | 25% | Terms + some phrases |
| hard | 35% | Terms + phrases, high density |
| expert | 45% | Maximum density |

## Architecture

```
src/anki_forge/
├── core/           # Data models, config, exceptions
├── parsers/        # EPUB, PDF document parsers
├── chunking/       # Paragraph, semantic, sliding window
├── generation/     # LLM providers, card generator, batching
├── validation/     # Cloze engine, validators
├── output/         # Anki CSV, JSON export
└── cli.py          # Command-line interface
```

### Key Components

**ClozeEngine**: Applies cloze deletions with importance-based selection
- Targets scored 1-10 by LLM
- High importance (9-10) always included
- Fills to density with remaining targets
- Prevents overlapping clozes

**BatchProcessor**: Combines chunks for efficient API usage
- Auto-configures based on provider context limits
- Respects rate limits with automatic delays
- Parses batch responses back to individual results

**CardGenerator**: Orchestrates the generation pipeline
- Supports direct, hybrid, and hybrid_batched modes
- Tracks quality metrics (density, importance)
- Handles errors gracefully with fallbacks

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## Output Format

Generated CSV is compatible with Anki's import:
1. Open Anki → File → Import
2. Select the CSV file
3. Set note type to "Cloze"
4. Map fields: Text, Citation, Tags

Example card:
```
Text: Heidegger's concept of {{c1::Dasein}} refers to {{c2::human existence}}.
Citation: Being and Time, p. 42
Tags: anki-forge philosophy
```

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
