#!/bin/bash
# VLM-Patrol setup script — works on Ubuntu, macOS, Windows (Git Bash)

set -e

echo "=== VLM-Patrol Setup ==="

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy config if not exists
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo "Created config.yaml from template — edit it to configure your setup"
fi

echo ""
echo "=== Setup complete ==="
echo "To start:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Then open http://localhost:8765 in your browser"
