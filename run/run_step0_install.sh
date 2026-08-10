#!/usr/bin/env bash
#==============================================================================
# STEP 0: Environment check + install
#==============================================================================
#
# Verifies Ubuntu + NVIDIA GPU + Python 3.10+, then:
#   1. Installs system packages (curl, build-essential, git)
#   2. Creates a Python virtual environment at code/.venv/
#   3. Installs PyTorch with CUDA 12.1 (cu121 wheel — works on RTX 3080 Ti sm_86)
#   4. Installs all other Python deps from code/requirements.txt
#   5. Verifies PyTorch can see the GPU
#
# Idempotent: if code/.venv/ already exists, skips venv creation but still
# activates it and runs pip install (which is itself idempotent).
#
# Time: 8-12 minutes (dominated by PyTorch cu121 wheel download, ~2 GB)
# Disk: ~3.5 GB (apt + venv + pip cache)
# RAM peak: 1.5 GB (during pip install)
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"
VENV_DIR="${CODE_DIR}/.venv"

echo "--- Step 0: Environment check ---"

# --- OS ---
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    echo "  OS: ${PRETTY_NAME:-unknown}"
    if [[ "${VERSION_ID:-}" != "24.04" && "${VERSION_ID:-}" != "22.04" ]]; then
        echo "  [warn] Ubuntu 24.04 or 22.04 recommended; found ${VERSION_ID:-unknown}"
    fi
else
    echo "  [warn] /etc/os-release not found; cannot verify OS"
fi

# --- GPU ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  [ERROR] nvidia-smi not found. Install NVIDIA driver first:"
    echo "          sudo apt install nvidia-driver-535"
    exit 1
fi
echo "  GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'

GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
echo "  Detected VRAM: ${GPU_MEM_MB} MB"
if [[ "${GPU_MEM_MB}" -lt 11000 ]]; then
    echo "  [ERROR] GPU has less than 11 GB VRAM; need at least 11 GB (RTX 3080 Ti has 12 GB)"
    exit 1
fi

# --- Disk ---
FREE_GB=$(df -BG "${BUNDLE_ROOT}" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
echo "  Free disk: ${FREE_GB} GB"
if [[ "${FREE_GB}" -lt 50 ]]; then
    echo "  [ERROR] need at least 50 GB free; have ${FREE_GB} GB"
    exit 1
fi

# --- Python ---
if ! command -v python3 >/dev/null 2>&1; then
    echo "  [ERROR] python3 not found. Install: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "  Python: ${PY_VER}"
if [[ "${PY_VER}" < "3.10" ]]; then
    echo "  [ERROR] need Python 3.10+; have ${PY_VER}"
    exit 1
fi

echo "  Environment OK."

# --- Install system packages ---
echo ""
echo "--- Step 0: Installing system packages ---"
sudo apt-get update -qq || true
sudo apt-get install -y -qq curl build-essential git python3-venv python3-pip 2>&1 | tail -3

# --- Create venv ---
echo ""
echo "--- Step 0: Creating virtualenv at ${VENV_DIR} ---"
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- Upgrade pip ---
echo ""
echo "--- Step 0: Upgrading pip ---"
pip install --upgrade pip wheel setuptools 2>&1 | tail -2

# --- Install PyTorch with CUDA 12.1 ---
echo ""
echo "--- Step 0: Installing PyTorch (CUDA 12.1) ---"
echo "  Downloading ~2.0 GB wheel. This may take 4-6 minutes on a 5-10 MB/s connection."
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

# --- Install other Python deps ---
echo ""
echo "--- Step 0: Installing other Python deps ---"
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

# Also install requirements.txt (in case anything is missing)
if [[ -f "${CODE_DIR}/requirements.txt" ]]; then
    pip install --quiet -r "${CODE_DIR}/requirements.txt" 2>&1 | tail -2 || true
fi

# --- Verify GPU is visible to PyTorch ---
echo ""
echo "--- Step 0: Verifying PyTorch + CUDA ---"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  [warn] CUDA not available — training will be very slow on CPU.')
"

# --- Done ---
echo ""
echo "--- Step 0: Done. venv at ${VENV_DIR} ---"
echo "  To activate manually: source ${VENV_DIR}/bin/activate"
exit 0
