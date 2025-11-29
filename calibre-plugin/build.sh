#!/bin/bash
# Build Calibre plugin zip file

PLUGIN_NAME="AnkiForge"
VERSION="0.3.0"

# Navigate to plugin directory
cd "$(dirname "$0")"

# Create build directory
mkdir -p ../dist

# Create zip file
zip -r "../dist/${PLUGIN_NAME}-v${VERSION}.zip" \
    __init__.py \
    ui.py \
    config.py \
    plugin-import-name-anki_forge.txt \
    -x "*.pyc" -x "__pycache__/*" -x "*.pyo"

echo "Built: dist/${PLUGIN_NAME}-v${VERSION}.zip"
echo ""
echo "To install:"
echo "  1. Open Calibre"
echo "  2. Go to Preferences > Plugins"
echo "  3. Click 'Load plugin from file'"
echo "  4. Select dist/${PLUGIN_NAME}-v${VERSION}.zip"
echo "  5. Restart Calibre"
echo ""
echo "Note: You must also install anki-forge:"
echo "  pip install anki-forge[all]"
