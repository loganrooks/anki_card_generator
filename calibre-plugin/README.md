# Anki Forge Calibre Plugin

Generate Anki flashcards directly from your Calibre library.

## Installation

### Prerequisites

1. Install Anki Forge:
   ```bash
   pip install anki-forge[all]
   ```

2. Get a free API key from one of:
   - [Google AI Studio](https://makersuite.google.com/app/apikey) (Gemini - recommended)
   - [OpenRouter](https://openrouter.ai/keys)

### Install Plugin

1. Build the plugin:
   ```bash
   cd calibre-plugin
   ./build.sh
   ```

2. In Calibre:
   - Go to **Preferences** > **Plugins**
   - Click **Load plugin from file**
   - Select `dist/AnkiForge-v0.3.0.zip`
   - Restart Calibre

## Usage

1. Select one or more books in your Calibre library
2. Click the **Anki Forge** button in the toolbar
3. Confirm the generation settings
4. Wait for card generation to complete
5. Import the generated CSV into Anki as a **Cloze** note type

## Configuration

Click **Anki Forge** > **Configure** to set:

- **Provider**: LLM provider (gemini, openrouter, openai, ollama)
- **API Key**: Your API key (or set environment variable)
- **Mode**: Generation mode (hybrid recommended)
- **Difficulty**: Card difficulty preset
- **Density**: Target cloze deletion density (0.10-0.50)
- **Output Directory**: Where to save generated cards

## Supported Formats

- EPUB
- PDF (requires `pip install anki-forge[pdf]`)
- MOBI/AZW/AZW3 (requires `pip install anki-forge[kindle]`)

## Troubleshooting

### "Anki Forge not installed"
Install the package in the same Python environment Calibre uses:
```bash
calibre-debug -c "import sys; print(sys.executable)"
# Then use that Python to install:
/path/to/python -m pip install anki-forge[all]
```

### API Key Issues
Set environment variables before starting Calibre:
```bash
export GOOGLE_API_KEY=your-key-here
calibre
```

Or configure keys in the plugin settings.
