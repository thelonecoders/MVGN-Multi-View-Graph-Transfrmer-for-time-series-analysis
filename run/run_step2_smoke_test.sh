#!/usr/bin/env bash
#==============================================================================
# STEP 2: Smoke tests + 2-epoch smoke training
#==============================================================================
#
# Runs two smoke checks:
#   1. The 11 unit tests in code/tests/test_smoke.py (exercises every class
#      in the mvgt_net package with random tensors and verifies forward-pass
#      shapes end-to-end). Expected: 11/11 PASS.
#   2. A 2-epoch smoke training run on Economy_Trade with batch_size=4
#      (verifies the full training pipeline: data load, model build,
#      optimizer, loss, AMP, checkpoint, test eval).
#
# Time: 30s-2min (CPU smoke) or 5-15s (GPU smoke)
# Disk: ~50 MB (smoke checkpoint + smoke metrics)
# RAM peak: 2.5 GB
# VRAM peak: 1.5 GB
#
# Output: code/results/Economy_Trade/metrics.json (smoke metrics)
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"

# Activate venv if it exists
if [[ -f "${CODE_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CODE_DIR}/.venv/bin/activate"
fi

cd "${CODE_DIR}"

echo "--- Step 2a: Run 11 unit tests (code/tests/test_smoke.py) ---"
python3 tests/test_smoke.py
echo ""

echo "--- Step 2b: Run 2-epoch smoke training (Economy_Trade) ---"
echo "  This verifies the full training pipeline on real data."
echo ""

# Use the smoke-test flag of train_real.py (2 epochs, batch_size=4, CPU OK)
python3 scripts/train_real.py \
    --config configs/default.yaml \
    --domain Economy_Trade \
    --smoke-test \
    --device "${DEVICE:-cpu}" \
    --data-root "${CODE_DIR}/data/TimeMMD" \
    --checkpoint-dir "${CODE_DIR}/checkpoints" \
    --results-dir "${CODE_DIR}/results"

echo ""
echo "--- Step 2: Done. Smoke metrics at code/results/Economy_Trade/metrics.json ---"
echo ""
echo "  Quick view:"
echo "    cat ${CODE_DIR}/results/Economy_Trade/metrics.json | python3 -m json.tool | head -25"
exit 0
