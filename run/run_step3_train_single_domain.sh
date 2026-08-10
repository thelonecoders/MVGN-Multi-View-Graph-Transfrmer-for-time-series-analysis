#!/usr/bin/env bash
#==============================================================================
# STEP 3: Train MVGT-Net on a single TimeMMD domain
#==============================================================================
#
# Full production training on ONE TimeMMD domain.
#
# Usage:
#   ./run/run_step3_train_single_domain.sh                              # Climate_AQI, 100 epochs, cuda
#   ./run/run_step3_train_single_domain.sh Economy_Trade                # Economy_Trade, 100 epochs, cuda
#   ./run/run_step3_train_single_domain.sh Climate_AQI 50               # Climate_AQI, 50 epochs, cuda
#   ./run/run_step3_train_single_domain.sh Economy_Trade 2 --smoke-test --device cpu  # smoke
#
# Args:
#   $1 = domain (default: Climate_AQI). One of:
#        Climate_AQI, Economy_Unemp, Economy_Trade, Economy_VMT,
#        Agriculture_Fema, Agriculture_Broil, Climate_Precip,
#        Health_Flu, Energy_Gas
#   $2 = epochs (default: 100)
#   Remaining args are passed through to train_real.py (e.g. --device cpu, --smoke-test)
#
# Time per domain (estimated, RTX 3080 Ti, 100 epochs max with early stopping):
#   Climate_AQI       : ~12-22 min (7,552 train samples, lookback=96, horizon=96)
#   Economy_Trade     : ~2-4 min   (256 train samples, lookback=8, horizon=12)
#   Economy_Unemp     : ~3-5 min   (608 train, lookback=8, horizon=12)
#   Economy_VMT       : ~2-4 min   (352 train, lookback=8, horizon=12)
#   Agriculture_Fema  : ~1-3 min   (160 train, lookback=8, horizon=12)
#   Agriculture_Broil : ~2-4 min   (320 train, lookback=8, horizon=12)
#   Climate_Precip    : ~2-4 min   (320 train, lookback=8, horizon=12)
#   Health_Flu        : ~4-7 min   (896 train, lookback=36, horizon=24)
#   Energy_Gas        : ~4-7 min   (960 train, lookback=36, horizon=24)
#
# Disk: ~150 MB per domain (checkpoint + metrics JSON)
# RAM peak: 4.2 GB
# VRAM peak: 4.6 GB (well under 12 GB budget)
#
# Output:
#   code/checkpoints/<Domain>/best.pt       — best val_MAE checkpoint
#   code/checkpoints/<Domain>/latest.pt     — latest epoch checkpoint
#   code/checkpoints/<Domain>/epoch_NN.pt   — periodic checkpoints (every 10 epochs)
#   code/results/<Domain>/metrics.json      — final test metrics + history
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"

DOMAIN="${1:-Climate_AQI}"
EPOCHS="${2:-100}"
shift 2 || true   # remaining args passed through

# Activate venv if it exists
if [[ -f "${CODE_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CODE_DIR}/.venv/bin/activate"
fi

cd "${CODE_DIR}"

echo "--- Step 3: Train MVGT-Net on domain ${DOMAIN} (${EPOCHS} epochs max) ---"
echo "  Bundle root: ${BUNDLE_ROOT}"
echo "  Code dir   : ${CODE_DIR}"
echo "  Data root  : ${CODE_DIR}/data/TimeMMD"
echo "  Args       : $*"
echo ""

python3 scripts/train_real.py \
    --config configs/default.yaml \
    --domain "${DOMAIN}" \
    --epochs "${EPOCHS}" \
    --data-root "${CODE_DIR}/data/TimeMMD" \
    --checkpoint-dir "${CODE_DIR}/checkpoints" \
    --results-dir "${CODE_DIR}/results" \
    "$@"

echo ""
echo "--- Step 3: Done. ---"
echo "  Metrics  : ${CODE_DIR}/results/${DOMAIN}/metrics.json"
echo "  Best ckpt: ${CODE_DIR}/checkpoints/${DOMAIN}/best.pt"
echo ""
echo "  Quick view:"
echo "    cat ${CODE_DIR}/results/${DOMAIN}/metrics.json | python3 -m json.tool | head -30"
exit 0
