#!/usr/bin/env bash
#==============================================================================
# STEP 5: Phase F engineering analyses (5 scripts)
#==============================================================================
#
# Runs all 5 Phase F diagnostic scripts in sequence:
#
#   1. latency_carbon.py        — inference latency benchmark + carbon footprint
#   2. robustness_analysis.py   — accuracy vs injected Gaussian noise
#   3. scaling_analysis.py      — accuracy + wall-clock vs training-set fraction
#   4. cross_domain_transfer.py — 9x9 cross-domain transfer matrix
#   5. (carbon is split out of latency_carbon.py — both run in #1)
#
# Outputs go to: code/14_engineering_analyses/<analysis>/
#
# IMPORTANT — DIAGNOSTIC label:
#   The Phase F scripts run on a synthetic mini-batch by default, NOT on
#   the full trained model. Their outputs are valid for sanity-checking
#   the analysis pipelines and the plotting code, but they are NOT
#   thesis-grade performance numbers. To produce thesis-grade numbers,
#   point each script at the trained checkpoints from Step 4 (see each
#   script's --checkpoint flag).
#
# Time: 15-35 minutes total
# Disk: ~50 MB total (CSVs + PNGs + SVGs + JSONs)
# RAM peak: 3.8 GB
# VRAM peak: 4.6 GB
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"
ANALYSES_DIR="${CODE_DIR}/14_engineering_analyses"

# Activate venv if it exists
if [[ -f "${CODE_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CODE_DIR}/.venv/bin/activate"
fi

cd "${CODE_DIR}"

mkdir -p "${ANALYSES_DIR}"

# Write a README in the analyses dir explaining the DIAGNOSTIC label
cat > "${ANALYSES_DIR}/README.md" << 'EOF'
# 14 — Engineering Analyses (Phase F outputs)

This directory holds the outputs of the 5 Phase F engineering analysis scripts:

| Subdir | Source script | What it measures |
|--------|---------------|------------------|
| `latency/` | `scripts/latency_carbon.py` (Part 1) | Inference latency per batch / per sample + throughput + peak GPU memory, across batch sizes {1,4,8,16,32,64,128,256} |
| `carbon/` | `scripts/latency_carbon.py` (Part 2) | kgCO2e estimate for the full 432-experiment Chapter 18 training suite (Patterson et al. 2021 methodology) |
| `robustness/` | `scripts/robustness_analysis.py` | Accuracy degradation under injected Gaussian noise (σ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}) |
| `scaling/` | `scripts/scaling_analysis.py` | Accuracy + wall-clock as a function of training-set fraction (10%, 25%, 50%, 75%, 100%) |
| `transfer/` | `scripts/cross_domain_transfer.py` | 9×9 cross-domain transfer matrix (model trained on domain A, fine-tuned + evaluated on domain B) |

## DIAGNOSTIC label (important)

The outputs in this directory were generated on a **synthetic mini-batch**
(16 samples, 8 nodes, lookback=12, horizon=12), not on the full trained
model on the real TimeMMD dataset.

This means the outputs are valid for:

- Sanity-checking that the analysis pipelines run end-to-end without errors
- Sanity-checking that the plotting code produces well-formed figures
- Sanity-checking that the CSV/JSON schemas are correct
- Sanity-checking that the carbon-footprint calculation runs

But the outputs are NOT valid as:

- Thesis-grade performance numbers
- Comparable to any published baseline
- Indicative of model performance on the real TimeMMD dataset

To produce thesis-grade numbers, point each script at the trained
checkpoints from `code/checkpoints/<Domain>/best.pt` using each script's
`--checkpoint` flag (see the script docstrings for details).

## How to regenerate

```bash
cd code
source .venv/bin/activate
python scripts/latency_carbon.py
python scripts/robustness_analysis.py
python scripts/scaling_analysis.py
python scripts/cross_domain_transfer.py
```

Each script overwrites its own output subdirectory, so re-running is safe.
EOF

echo "--- Step 5a: Latency benchmark + Carbon footprint (scripts/latency_carbon.py) ---"
python3 scripts/latency_carbon.py
echo ""

echo "--- Step 5b: Robustness analysis (scripts/robustness_analysis.py) ---"
python3 scripts/robustness_analysis.py
echo ""

echo "--- Step 5c: Scaling analysis (scripts/scaling_analysis.py) ---"
python3 scripts/scaling_analysis.py
echo ""

echo "--- Step 5d: Cross-domain transfer (scripts/cross_domain_transfer.py) ---"
python3 scripts/cross_domain_transfer.py
echo ""

# Summary
echo "--- Step 5: Done. ---"
echo "  All Phase F outputs in: ${ANALYSES_DIR}/"
echo ""
echo "  Files created:"
find "${ANALYSES_DIR}" -type f | sort | sed 's/^/    /'
echo ""
exit 0
