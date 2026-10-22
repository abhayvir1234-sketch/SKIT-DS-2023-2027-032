#!/usr/bin/env bash
# ============================================================
#  run.sh — Launch the ALPR Streamlit application on Linux/macOS
#  Usage: bash run.sh
#  Author: Karam Singh Bana (Team Member 2)
# ============================================================

set -e

echo ""
echo "====================================================="
echo "  Vehicle Number Plate Recognition System"
echo "  AI-powered ALPR — College Engineering Project"
echo "====================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ and retry."
    exit 1
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "[INFO] Installing/verifying dependencies..."
pip install -r requirements.txt --quiet

# Copy .env.example
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example — set ALPR_MODEL_PATH before running."
fi

echo ""
echo "[INFO] Starting Streamlit application..."
echo "[INFO] Open http://localhost:8501 in your browser."
echo "[INFO] Press Ctrl+C to stop."
echo ""

streamlit run app.py
