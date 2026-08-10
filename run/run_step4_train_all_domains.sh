#!/usr/bin/env bash
#==============================================================================
# STEP 4: Train MVGT-Net on ALL 9 TimeMMD domains sequentially
#==============================================================================
#
# Repeats Step 3 for all 9 TimeMMD domains:
#   Climate_AQI, Economy_Unemp, Economy_Trade, Economy_VMT,
#   Agriculture_Fema, Agriculture_Broil, Climate_Precip,
#   Health_Flu, Energy_Gas
#
# Usage:
#   ./run/run_step4_train_all_domains.sh                     # 100 epochs, cuda
#   ./run/run_step4_train_all_domains.sh 50                  # 50 epochs, cuda
#   ./run/run_step4_train_all_domains.sh 100 --device cpu    # force CPU (slow)
#
# Args:
#   $1 = epochs (default: 100)
#   Remaining args are passed through to train_real.py (e.g. --device cpu)
#
# Time: 32-60 minutes total on RTX 3080 Ti (sum of per-domain times,
#       with early stopping typically reducing average by 20-30%)
# Disk: ~1.4 GB total (9 checkpoints + 9 metrics JSONs)
# RAM peak: 4.2 GB
# VRAM peak: 4.6 GB (well under 12 GB budget)
#
# Output:
#   code/checkpoints/<Domain>/best.pt       — per domain
#   code/results/<Domain>/metrics.json      — per domain
#   code/results/all_domains_summary.json   — cross-domain summary
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"

EPOCHS="${1:-100}"
shift || true   # remaining args passed through

# Activate venv if it exists
if [[ -f "${CODE_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CODE_DIR}/.venv/bin/activate"
fi

cd "${CODE_DIR}"

echo "--- Step 4: Train MVGT-Net on ALL 9 TimeMMD domains (${EPOCHS} epochs max each) ---"
echo "  Bundle root: ${BUNDLE_ROOT}"
echo "  Code dir   : ${CODE_DIR}"
echo "  Data root  : ${CODE_DIR}/data/TimeMMD"
echo "  Args       : $*"
echo ""

python3 scripts/train_real.py \
    --config configs/default.yaml \
    --all-domains \
    --epochs "${EPOCHS}" \
    --data-root "${CODE_DIR}/data/TimeMMD" \
    --checkpoint-dir "${CODE_DIR}/checkpoints" \
    --results-dir "${CODE_DIR}/results" \
    "$@"

echo ""
echo "--- Step 4: Done. ---"
echo "  Cross-domain summary: ${CODE_DIR}/results/all_domains_summary.json"
echo ""
echo "  Quick view:"
echo "    cat ${CODE_DIR}/results/all_domains_summary.json | python3 -m json.tool | head -50"
exit 0
