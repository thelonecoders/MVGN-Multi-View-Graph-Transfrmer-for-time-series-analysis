#!/usr/bin/env bash
#==============================================================================
# create_venv.sh — Just the Python venv (skip apt install)
#==============================================================================
#
# Useful when:
#   - You already have system packages installed (curl, build-essential, git)
#   - You're on a non-Ubuntu system that has python3 + venv
#   - You want to skip the apt-get step of run_step0_install.sh
#
# Usage:
#   chmod +x setup/create_venv.sh
#   ./setup/create_venv.sh
#
# Output: code/.venv/ with PyTorch (CUDA 12.1) + transformers + ranger21 + ...
#
# Time: 5-8 minutes (just the pip installs, no apt)
# Disk: ~3.3 GB (venv only, no apt cache)
# RAM peak: 1.2 GB
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"
VENV_DIR="${CODE_DIR}/.venv"

echo "--- Creating virtualenv at ${VENV_DIR} ---"

# Verify python3
if ! command -v python3 >/dev/null 2>&1; then
    echo "  [ERROR] python3 not found."
    exit 1
fi
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "  Python: ${PY_VER}"
if [[ "${PY_VER}" < "3.10" ]]; then
    echo "  [ERROR] need Python 3.10+; have ${PY_VER}"
    exit 1
fi

# Create venv
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools 2>&1 | tail -2

# Install PyTorch with CUDA 12.1
echo ""
echo "  Installing PyTorch (CUDA 12.1) — ~2.0 GB download"
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

# Install other deps
echo ""
echo "  Installing other Python deps"
pip install --quiet \
    "pyyaml>=6.0" \
    "transformers>=4.30.0" \
    "ranger21>=2.1.0" \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "tqdm>=4.65" \
    "matplotlib>=3.7" \
    "wandb>=0.15" 2>&1 | tail -3

if [[ -f "${CODE_DIR}/requirements.txt" ]]; then
    pip install --quiet -r "${CODE_DIR}/requirements.txt" 2>&1 | tail -2 || true
fi

# Verify
echo ""
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "--- Done. venv at ${VENV_DIR} ---"
echo "  To activate: source ${VENV_DIR}/bin/activate"
exit 0
