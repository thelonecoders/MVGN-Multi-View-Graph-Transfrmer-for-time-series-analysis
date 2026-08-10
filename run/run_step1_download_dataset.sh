#!/usr/bin/env bash
#==============================================================================
# STEP 1: Download the REAL TimeMMD dataset
#==============================================================================
#
# Downloads all 9 TimeMMD domains (27 JSONL files, 322 MB) from the verified
# Hugging Face mirror (AndrewRWilliams/time-mmd-DC, which hosts the official
# AdityaLab TimeMMD data — arXiv:2406.08627, NeurIPS 2024 D&B Track).
#
# Each file is SHA-256 verified. A failed download is retried up to 3 times
# with exponential backoff. A top-level DATASET_MANIFEST.json is written.
#
# Idempotent: if code/data/TimeMMD/DATASET_MANIFEST.json already exists,
# this script prints a message and exits 0.
#
# Time: 1-3 minutes (network-bound at ~5-10 MB/s)
# Disk: 322 MB
# RAM peak: 0.3 GB
#
# Output: code/data/TimeMMD/<Domain>/{train,validation,test}.jsonl + manifest.json
#         code/data/TimeMMD/DATASET_MANIFEST.json
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${BUNDLE_ROOT}/code"
DATA_DIR="${CODE_DIR}/data/TimeMMD"
DOWNLOADER="${BUNDLE_ROOT}/dataset_downloader/download_timemmd_real.py"

# Activate venv if it exists (so python3 = venv's python)
if [[ -f "${CODE_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CODE_DIR}/.venv/bin/activate"
fi

echo "--- Step 1: Download REAL TimeMMD dataset ---"
echo "  Downloader: ${DOWNLOADER}"
echo "  Output dir: ${DATA_DIR}"

# Idempotency check
if [[ -f "${DATA_DIR}/DATASET_MANIFEST.json" ]]; then
    echo "  [SKIP] ${DATA_DIR}/DATASET_MANIFEST.json already exists."
    echo "  To force re-download, delete that file first."
    # Print summary of existing manifest
    python3 -c "
import json
m = json.load(open('${DATA_DIR}/DATASET_MANIFEST.json'))
print(f'    Files      : {m[\"total_files\"]}')
print(f'    Total bytes: {m[\"total_bytes\"]:,} ({m[\"total_bytes\"] / (1 << 20):.1f} MiB)')
print(f'    Domains    : {len(m[\"domains\"])}')
print(f'    Synthetic  : {m[\"no_synthetic_data\"]}')
print(f'    Placeholders: {m[\"no_placeholders\"]}')
"
    exit 0
fi

# Check downloader exists
if [[ ! -f "${DOWNLOADER}" ]]; then
    echo "  [ERROR] Downloader not found: ${DOWNLOADER}"
    exit 2
fi

# Run the downloader
# The downloader has internal retry + SHA-256 verification per file.
python3 "${DOWNLOADER}" --output "${DATA_DIR}"

# Verify final manifest
if [[ ! -f "${DATA_DIR}/DATASET_MANIFEST.json" ]]; then
    echo "  [ERROR] DATASET_MANIFEST.json not found after download"
    exit 2
fi

# Print summary
echo ""
echo "  Dataset manifest summary:"
python3 -c "
import json
m = json.load(open('${DATA_DIR}/DATASET_MANIFEST.json'))
print(f'    Total files : {m[\"total_files\"]}')
print(f'    Total bytes : {m[\"total_bytes\"]:,} ({m[\"total_bytes\"] / (1 << 20):.1f} MiB)')
print(f'    Domains     : {len(m[\"domains\"])}')
print(f'    Synthetic   : {m[\"no_synthetic_data\"]}')
print(f'    Placeholders: {m[\"no_placeholders\"]}')
print()
print('    Per-domain:')
for d in m['domains']:
    print(f'      {d[\"domain\"]:<22} {d[\"total_bytes\"]:>12,}B  ({d[\"total_bytes\"] / (1 << 20):>6.1f} MiB)  {d[\"frequency\"]}')
"

echo ""
echo "--- Step 1: Done. Dataset at ${DATA_DIR} ---"
exit 0
