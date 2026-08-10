#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"

if [[ -f "${CODE_DIR}/.venv/bin/activate" ]]; then
    source "${CODE_DIR}/.venv/bin/activate"
fi

cd "${CODE_DIR}"

echo "=============================================================================="
echo "  STEP J: Climate_AQI Training with Mitigations M1-M4"
echo "=============================================================================="
echo "  Bundle root: ${BUNDLE_ROOT}"
echo "  Code dir   : ${CODE_DIR}"
echo "  Data root  : ${CODE_DIR}/data/TimeMMD"
echo "  Mitigations:"
echo "    M1: early_stopping_patience = 30 (was 15)"
echo "    M2: warmup_iters = 50 (fixed, was auto-calculated)"
echo "    M3: batch_size = 8 (was 32, quadruples gradient updates)"
echo "    M4: normalization = minmax (was zscore)"
echo "=============================================================================="
echo ""

DATA_DIR="data/TimeMMD/Climate_AQI"
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: Climate_AQI dataset not found at ${CODE_DIR}/${DATA_DIR}/train.jsonl"
    echo "Please run Step 1 (dataset download) first."
    exit 1
fi

echo "Dataset verified: ${CODE_DIR}/${DATA_DIR}/train.jsonl exists"
echo ""

RESULTS_BASE="${CODE_DIR}/results_mitigated"
CKPT_BASE="${CODE_DIR}/checkpoints_mitigated"
mkdir -p "$RESULTS_BASE" "$CKPT_BASE"

echo "Starting training..."
echo "  Results dir : ${RESULTS_BASE}/Climate_AQI/metrics.json"
echo "  Checkpoint  : ${CKPT_BASE}/Climate_AQI/best.pt"
echo "  Log file    : ${RESULTS_BASE}/train_Climate_AQI_mitigated.log"
echo ""

python scripts/train_real.py \
    --config configs/default.yaml \
    --domain Climate_AQI \
    --epochs 100 \
    --data-root "${CODE_DIR}/data/TimeMMD" \
    --checkpoint-dir "$CKPT_BASE" \
    --results-dir "$RESULTS_BASE" \
    --batch-size 8 \
    --patience 30 \
    --warmup-iters 50 \
    --normalization minmax \
    --lr 0.001 \
    --seed 42 \
    2>&1 | tee "${RESULTS_BASE}/train_Climate_AQI_mitigated.log"

echo ""
echo "=============================================================================="
echo "  Step J Complete"
echo "=============================================================================="
echo "Results saved to:"
echo "  - ${RESULTS_BASE}/Climate_AQI/metrics.json"
echo "  - ${RESULTS_BASE}/train_Climate_AQI_mitigated.log"
echo "  - ${CKPT_BASE}/Climate_AQI/best.pt"
echo ""
echo "Quick view:"
echo "  cat ${RESULTS_BASE}/Climate_AQI/metrics.json | python3 -m json.tool | head -40"
echo ""
echo "Next: compare R2 vs Energy_Gas (+0.6077), then proceed to Step K (SHAP)."
