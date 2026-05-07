#!/usr/bin/env bash
# ============================================================
# setup.sh — Bootstrap script for AI Travel Planner on RunPod
#
# Usage:
#   1. Upload this repo to the pod (git clone or via RunPod UI)
#   2. cd AI_travel_planner
#   3. bash setup.sh
#   4. Open the public URL shown in RunPod for port 5000
# ============================================================

set -e   # exit on any error

# ── Colours ──────────────────────────────────────────────────
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; RESET="\033[0m"
info()  { echo -e "${GREEN}[setup]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[setup]${RESET} $*"; }
error() { echo -e "${RED}[setup]${RESET} $*" >&2; }

# ── 1. Move to repo root ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Working directory: $PWD"

# ── 2. Find Python 3.10 and create a venv ────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

# Prefer python3.10 explicitly, fall back to python3 / python
if   command -v python3.10 &>/dev/null; then BASE_PYTHON=$(command -v python3.10)
elif command -v python3    &>/dev/null; then BASE_PYTHON=$(command -v python3)
elif command -v python     &>/dev/null; then BASE_PYTHON=$(command -v python)
else
    error "No Python interpreter found. Use a RunPod template that includes Python 3.10."
    exit 1
fi
info "Base interpreter: $($BASE_PYTHON --version)"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in .venv …"
    $BASE_PYTHON -m venv "$VENV_DIR"
else
    info "Virtual environment .venv already exists, skipping creation."
fi

info "Activating virtual environment…"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTHON=$(command -v python)
info "Using Python: $($PYTHON --version)"

# ── 3. Create .env if missing ────────────────────────────────
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warn ".env not found — created from .env.example."
        warn "Fill in your credentials in .env before the app starts."
    else
        warn "No .env file found. Create one with HF_TOKEN, AMADEUS_API_KEY, AMADEUS_API_SECRET."
    fi
fi

# ── 4. Read credentials from .env and validate ───────────────
set -a; source .env 2>/dev/null || true; set +a

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    error "HF_TOKEN is not set in .env — the LLM will not load."
    error "Edit .env and re-run this script."
    exit 1
fi

if [ -z "$AMADEUS_API_KEY" ] || [ "$AMADEUS_API_KEY" = "your_amadeus_api_key_here" ]; then
    warn "AMADEUS_API_KEY is not set — hotel/activity search will be disabled."
fi

info "Credentials OK."

# ── 5. Install Python dependencies ───────────────────────────
info "Installing Python dependencies…"

PIP="$PYTHON -m pip"
$PIP install --upgrade pip --quiet

# Install PyTorch with CUDA 12.1 (covers most RunPod GPU templates).
# Skip if torch is already installed with CUDA support.
if ! $PYTHON -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    info "Installing PyTorch with CUDA 12.1…"
    $PIP install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 \
        --quiet
else
    info "PyTorch + CUDA already available, skipping."
fi

# Install remaining deps from requirements.txt
$PIP install -r requirements.txt --quiet
info "All dependencies installed."

# ── 6. (Optional) Pre-download the model ────────────────────
# Uncomment the lines below to download Llama weights before
# starting the server. Recommended on slow-startup pods.
#
# info "Pre-downloading model weights (this takes several minutes)…"
# $PYTHON -c "
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import os
# model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
# token = os.environ.get('HF_TOKEN')
# AutoTokenizer.from_pretrained(model_id, token=token)
# AutoModelForCausalLM.from_pretrained(model_id, token=token)
# print('Download complete.')
# "

# ── 7. Start the Flask app ───────────────────────────────────
info "Starting AI Travel Planner on port 5000…"
info "Open the RunPod HTTP endpoint for port 5000 in your browser."
echo ""

exec $PYTHON app.py
