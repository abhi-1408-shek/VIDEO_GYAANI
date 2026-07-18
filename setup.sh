#!/bin/bash
# setup.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot setup script for the Automated Video Dubbing System.
# Run this once to create a virtual environment and install all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        🎬  VIDEO DUBBING SYSTEM – SETUP  🎬                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Check prerequisites ──────────────────────────────────────────────────────
echo "▶ Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.10+."
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "✗ ffmpeg not found."
    echo "  → macOS:  brew install ffmpeg"
    echo "  → Ubuntu: sudo apt install ffmpeg"
    echo "  → Windows: https://ffmpeg.org/download.html"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PYTHON_VERSION found"
echo "✓ ffmpeg found"

# ── Create virtual environment ───────────────────────────────────────────────
echo ""
echo "▶ Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "  Virtual environment already exists, skipping creation."
else
    python3 -m venv .venv
    echo "✓ Virtual environment created at .venv/"
fi

# Activate
source .venv/bin/activate

# ── Upgrade pip ──────────────────────────────────────────────────────────────
echo ""
echo "▶ Upgrading pip..."
pip install --upgrade pip --quiet

# ── Install PyTorch (CPU-only build) ────────────────────────────────────────
echo ""
echo "▶ Installing PyTorch (CPU version)..."
echo "  (If you have a CUDA GPU, replace this with the appropriate torch install command)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
echo "✓ PyTorch installed"

# ── Install all dependencies ─────────────────────────────────────────────────
echo ""
echo "▶ Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "✓ Core dependencies installed"

# ── Install IndicTransToolkit (best translator for Indian languages) ──────────
echo ""
echo "▶ Installing IndicTransToolkit (IndicTrans2 for Indian languages)..."
pip install git+https://github.com/VarunGumma/IndicTransToolkit --quiet && \
    echo "✓ IndicTransToolkit installed" || \
    echo "⚠  IndicTransToolkit install failed — will fall back to MarianMT"

# ── Install soundfile (needed for numpy audio export) ────────────────────────
echo ""
echo "▶ Installing soundfile..."
pip install soundfile --quiet
echo "✓ soundfile installed"

# ── Verify key imports ───────────────────────────────────────────────────────
echo ""
echo "▶ Verifying imports..."
python3 -c "import yt_dlp; print('✓ yt_dlp')"
python3 -c "import faster_whisper; print('✓ faster_whisper')"
python3 -c "import edge_tts; print('✓ edge_tts')"
python3 -c "import pydub; print('✓ pydub')"
python3 -c "import transformers; print('✓ transformers')"
python3 -c "import rich; print('✓ rich')"
python3 -c "import click; print('✓ click')"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  Setup Complete!                                         ║"
echo "║                                                              ║"
echo "║  Activate the environment:                                   ║"
echo "║    source .venv/bin/activate                                 ║"
echo "║                                                              ║"
echo "║  Run the dubber:                                             ║"
echo "║    python main.py \"<YOUTUBE_URL>\"                            ║"
echo "║    python main.py \"<URL>\" --voice female --model large-v2    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
