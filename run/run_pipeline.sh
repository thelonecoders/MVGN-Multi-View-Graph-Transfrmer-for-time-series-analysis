#!/usr/bin/env bash
#==============================================================================
# MASTER PIPELINE: ST-LLM-Plus VPS Code Bundle
#==============================================================================
#
# Runs the entire pipeline from step 0 (fresh Ubuntu 24.04 VPS) to:
#   - Installed Python venv with PyTorch + CUDA
#   - Downloaded REAL TimeMMD dataset (9 domains, 322 MB, SHA-256 verified)
#   - Cached BERT-base-uncased tokenizer + model
#   - Trained MVGT-Net on all 9 TimeMMD domains (with early stopping)
#   - All 5 Phase F engineering analyses
#   - Final per-domain summary table
#
# Usage:
#   ./run/run_pipeline.sh                     # full run on all 9 domains
#   ./run/run_pipeline.sh --domain Climate_AQI  # single domain
#   ./run/run_pipeline.sh --smoke-test          # 2-epoch test on Economy_Trade
#   ./run/run_pipeline.sh --skip-install        # skip apt + venv + pip
#   ./run/run_pipeline.sh --skip-download       # skip dataset re-download
#   ./run/run_pipeline.sh --skip-smoke          # skip smoke test
#   ./run/run_pipeline.sh --skip-train          # skip training
#   ./run/run_pipeline.sh --skip-analyses       # skip Phase F analyses
#   ./run/run_pipeline.sh --epochs 50           # override max epochs
#   ./run/run_pipeline.sh --device cpu          # force CPU (slow)
#   ./run/run_pipeline.sh --step install        # run only step 0 (install)
#   ./run/run_pipeline.sh --step download       # run only step 1 (download)
#   ./run/run_pipeline.sh --step smoke          # run only step 2 (smoke)
#   ./run/run_pipeline.sh --step train          # run only step 3 (train)
#   ./run/run_pipeline.sh --step analyses       # run only step 4 (analyses)
#
# Exit codes:
#   0 = success
#   1 = environment check failed
#   2 = dataset download failed
#   3 = smoke test failed
#   4 = training failed
#   5 = Phase F analyses failed
#
#==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve bundle root (parent of run/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_DIR="${BUNDLE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Tee all output to the log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "===================================================================="
echo "  ST-LLM-Plus VPS Code Bundle — Master Pipeline"
echo "  Bundle root: ${BUNDLE_ROOT}"
echo "  Log file   : ${LOG_FILE}"
echo "  Started    : $(date)"
echo "===================================================================="

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
SKIP_SMOKE=0
SKIP_TRAIN=0
SKIP_ANALYSES=0
SMOKE_TEST=0
DOMAIN=""
EPOCHS=""
DEVICE="cuda"
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)    SKIP_INSTALL=1;    shift ;;
        --skip-download)   SKIP_DOWNLOAD=1;   shift ;;
        --skip-smoke)      SKIP_SMOKE=1;      shift ;;
        --skip-train)      SKIP_TRAIN=1;      shift ;;
        --skip-analyses)   SKIP_ANALYSES=1;   shift ;;
        --smoke-test)      SMOKE_TEST=1;      shift ;;
        --domain)          DOMAIN="$2";       shift 2 ;;
        --epochs)          EPOCHS="$2";       shift 2 ;;
        --device)          DEVICE="$2";       shift 2 ;;
        --step)
            ONLY_STEP="$2"; shift 2
            case "${ONLY_STEP}" in
                install|download|smoke|train|analyses) ;;
                *) echo "ERROR: --step must be one of: install|download|smoke|train|analyses"; exit 1 ;;
            esac
            ;;
        --help|-h)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown arg '$1' (try --help)"
            exit 1
            ;;
    esac
done

# If --step was given, only run that one step
if [[ -n "${ONLY_STEP}" ]]; then
    case "${ONLY_STEP}" in
        install)   "${SCRIPT_DIR}/run_step0_install.sh"        || exit 1 ;;
        download)  "${SCRIPT_DIR}/run_step1_download_dataset.sh" || exit 2 ;;
        smoke)     "${SCRIPT_DIR}/run_step2_smoke_test.sh"      || exit 3 ;;
        train)
            if [[ -n "${DOMAIN}" ]]; then
                "${SCRIPT_DIR}/run_step3_train_single_domain.sh" "${DOMAIN}" "${EPOCHS:-100}" || exit 4
            else
                "${SCRIPT_DIR}/run_step4_train_all_domains.sh" "${EPOCHS:-100}" || exit 4
            fi
            ;;
        analyses)  "${SCRIPT_DIR}/run_step5_analyses.sh"       || exit 5 ;;
    esac
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 0: Environment check + install
# ---------------------------------------------------------------------------
if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
    echo ""
    echo "=================================================================="
    echo "  STEP 0: Environment check + install"
    echo "=================================================================="
    "${SCRIPT_DIR}/run_step0_install.sh" || {
        echo "  [FATAL] Step 0 (install) failed (exit $?)"
        exit 1
    }
else
    echo ""
    echo "=== STEP 0: SKIPPED (--skip-install) ==="
fi

# ---------------------------------------------------------------------------
# Step 1: Download the REAL TimeMMD dataset
# ---------------------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
    echo ""
    echo "=================================================================="
    echo "  STEP 1: Download REAL TimeMMD dataset (9 domains, 322 MB)"
    echo "=================================================================="
    "${SCRIPT_DIR}/run_step1_download_dataset.sh" || {
        echo "  [FATAL] Step 1 (dataset download) failed (exit $?)"
        exit 2
    }
else
    echo ""
    echo "=== STEP 1: SKIPPED (--skip-download) ==="
fi

# ---------------------------------------------------------------------------
# Step 2: Smoke test
# ---------------------------------------------------------------------------
if [[ "${SKIP_SMOKE}" -eq 0 && "${SMOKE_TEST}" -eq 0 ]]; then
    echo ""
    echo "=================================================================="
    echo "  STEP 2: Smoke tests (11 unit tests + 2-epoch training)"
    echo "=================================================================="
    "${SCRIPT_DIR}/run_step2_smoke_test.sh" || {
        echo "  [FATAL] Step 2 (smoke test) failed (exit $?)"
        exit 3
    }
else
    echo ""
    echo "=== STEP 2: SKIPPED (--skip-smoke or --smoke-test) ==="
fi

# ---------------------------------------------------------------------------
# Step 3/4: Train
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
    echo ""
    echo "=================================================================="
    if [[ "${SMOKE_TEST}" -eq 1 ]]; then
        echo "  STEP 3: SMOKE TRAIN (2 epochs, Economy_Trade, device=${DEVICE})"
        echo "=================================================================="
        "${SCRIPT_DIR}/run_step3_train_single_domain.sh" Economy_Trade 2 --smoke-test --device "${DEVICE}" || {
            echo "  [FATAL] Step 3 (smoke train) failed (exit $?)"
            exit 4
        }
    elif [[ -n "${DOMAIN}" ]]; then
        echo "  STEP 3: TRAIN single domain = ${DOMAIN}, epochs=${EPOCHS:-100}, device=${DEVICE}"
        echo "=================================================================="
        "${SCRIPT_DIR}/run_step3_train_single_domain.sh" "${DOMAIN}" "${EPOCHS:-100}" --device "${DEVICE}" || {
            echo "  [FATAL] Step 3 (train ${DOMAIN}) failed (exit $?)"
            exit 4
        }
    else
        echo "  STEP 4: TRAIN all 9 domains, epochs=${EPOCHS:-100}, device=${DEVICE}"
        echo "=================================================================="
        "${SCRIPT_DIR}/run_step4_train_all_domains.sh" "${EPOCHS:-100}" --device "${DEVICE}" || {
            echo "  [FATAL] Step 4 (train all) failed (exit $?)"
            exit 4
        }
    fi
else
    echo ""
    echo "=== STEP 3/4: SKIPPED (--skip-train) ==="
fi

# ---------------------------------------------------------------------------
# Step 5: Phase F analyses
# ---------------------------------------------------------------------------
if [[ "${SKIP_ANALYSES}" -eq 0 && "${SMOKE_TEST}" -eq 0 ]]; then
    echo ""
    echo "=================================================================="
    echo "  STEP 5: Phase F engineering analyses (5 scripts)"
    echo "=================================================================="
    "${SCRIPT_DIR}/run_step5_analyses.sh" || {
        echo "  [FATAL] Step 5 (analyses) failed (exit $?)"
        exit 5
    }
else
    echo ""
    echo "=== STEP 5: SKIPPED (--skip-analyses or --smoke-test) ==="
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "===================================================================="
echo "  PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "  Log file : ${LOG_FILE}"
echo "===================================================================="
echo ""
echo "  Results directory: ${BUNDLE_ROOT}/code/results/"
echo "  Checkpoints     : ${BUNDLE_ROOT}/code/checkpoints/"
echo "  Phase F outputs : ${BUNDLE_ROOT}/code/14_engineering_analyses/"
echo ""

# Print per-domain summary if it exists
SUMMARY="${BUNDLE_ROOT}/code/results/all_domains_summary.json"
if [[ -f "${SUMMARY}" ]]; then
    echo "  Per-domain test metrics:"
    # Activate venv for the python3 call (uses venv's python)
    if [[ -f "${BUNDLE_ROOT}/code/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${BUNDLE_ROOT}/code/.venv/bin/activate"
    fi
    python3 -c "
import json
with open('${SUMMARY}') as f:
    results = json.load(f)
print(f\"  {'Domain':<22} {'MAE':>10} {'RMSE':>10} {'WAPE':>10} {'epochs':>8} {'min':>8}\")
print(f\"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}\")
for r in results:
    if 'error' in r:
        print(f\"  {r['domain']:<22}  ERROR: {r['error'][:50]}\")
        continue
    m = r['test_metrics']
    print(f\"  {r['domain']:<22} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['WAPE']:>10.4f} {r['epochs_completed']:>8} {r['total_train_time_s']/60:>8.1f}\")
"
fi

echo ""
echo "  To inspect per-domain metrics in detail:"
echo "    cat ${BUNDLE_ROOT}/code/results/<Domain>/metrics.json | python3 -m json.tool"
echo ""
exit 0
