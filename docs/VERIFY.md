# Verification Commands (zero hallucinations)

Every claim in the README and in `RESOURCE_ESTIMATE.md` is verifiable. This
document gives the exact command to run after each pipeline step to confirm
the expected outcome. If a verification fails, the cause is almost always
one of: (a) the step didn't actually run, (b) the step ran but failed
silently, or (c) the step ran on a different machine than where the
verification is being run.

---

## §1. After Step 0 — environment install

### 1.1 Python venv exists

```bash
ls -la code/.venv/bin/python3
# Expected: a symlink to /usr/bin/python3.x (typically 3.10 or 3.11)
```

### 1.2 PyTorch is installed and sees the GPU

```bash
source code/.venv/bin/activate
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version   : {torch.version.cuda}')
    print(f'GPU name       : {torch.cuda.get_device_name(0)}')
    print(f'Compute caps  : {torch.cuda.get_device_capability(0)}')
    print(f'VRAM (GB)      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}')
"
```

**Expected output:**
```
PyTorch version: 2.x.x+cu121
CUDA available : True
CUDA version   : 12.1
GPU name       : NVIDIA GeForce RTX 3080 Ti
Compute caps  : (8, 6)
VRAM (GB)      : 11.77 (or 12.00 if driver reports reserved-headroom differently)
```

### 1.3 All required Python packages are installed

```bash
source code/.venv/bin/activate
pip list 2>/dev/null | grep -E "^(torch|transformers|ranger21|numpy|pandas|scikit-learn|tqdm|wandb|pyyaml|matplotlib) "
```

**Expected output** (versions may be slightly newer):
```
matplotlib          3.x.x
numpy               1.x.x
pandas              2.x.x
pyyaml              6.x
ranger21            2.1.x
scikit-learn        1.x.x
torch               2.x.x+cu121
tqdm                4.x.x
transformers        4.x.x
wandb               0.x.x
```

### 1.4 Disk usage matches estimate (~3.5 GB)

```bash
du -sh code/.venv
# Expected: ~3.0-3.5 GB (PyTorch cu121 wheel is ~2.0 GB by itself)
```

---

## §2. After Step 1 — dataset download

### 2.1 Top-level manifest exists and has correct counts

```bash
python3 -c "
import json
m = json.load(open('code/data/TimeMMD/DATASET_MANIFEST.json'))
print(f'Files       : {m[\"total_files\"]}')
print(f'Total bytes : {m[\"total_bytes\"]:,}')
print(f'Total MiB   : {m[\"total_bytes\"] / (1 << 20):.1f}')
print(f'Domains     : {len(m[\"domains\"])}')
print(f'Synthetic   : {m[\"no_synthetic_data\"]}')
print(f'Placeholders: {m[\"no_placeholders\"]}')
"
```

**Expected output:**
```
Files       : 27
Total bytes : 322,278,053
Total MiB   : 307.4
Domains     : 9
Synthetic   : False
Placeholders: False
```

### 2.2 All 27 JSONL files exist on disk

```bash
find code/data/TimeMMD -name "*.jsonl" | wc -l
# Expected: 27
```

### 2.3 Per-domain manifests exist

```bash
find code/data/TimeMMD -name "manifest.json" -not -path "*/DATASET_MANIFEST.json" | wc -l
# Expected: 9
```

### 2.4 SHA-256 of a sample file matches the manifest

```bash
DOMAIN=Climate_AQI
SPLIT=train
FILE="code/data/TimeMMD/${DOMAIN}/${SPLIT}.jsonl"

# Get the SHA-256 from the per-domain manifest
EXPECTED=$(python3 -c "
import json
m = json.load(open('code/data/TimeMMD/${DOMAIN}/manifest.json'))
for f in m['files']:
    if f['split'] == '${SPLIT}':
        print(f['sha256'])
")

# Compute the actual SHA-256
ACTUAL=$(sha256sum "${FILE}" | awk '{print $1}')

echo "Expected: ${EXPECTED}"
echo "Actual  : ${ACTUAL}"
[[ "${EXPECTED}" == "${ACTUAL}" ]] && echo "MATCH ✓" || echo "MISMATCH ✗"
```

### 2.5 Record count for a sample file

```bash
wc -l code/data/TimeMMD/Climate_AQI/train.jsonl
# Expected: 7552 (per DOMAIN_REGISTRY in code/mvgt_net/data.py)
```

### 2.6 First record of a sample file has the expected keys

```bash
head -1 code/data/TimeMMD/Climate_AQI/train.jsonl | python3 -c "
import json, sys
rec = json.loads(sys.stdin.read())
expected = ['batch_x', 'batch_y', 'batch_x_timestamps_datetime64_ns',
            'batch_y_timestamps_datetime64_ns', 'batch_x_mark', 'batch_y_mark',
            'batch_text']
for k in expected:
    present = k in rec
    print(f'  {k:<40} {\"present\" if present else \"MISSING\"}')
"
```

**Expected output** (note: validation split legitimately lacks `batch_y_timestamps_datetime64_ns` — this is an upstream quirk, documented in `code/mvgt_net/data.py`):

```
  batch_x                                 present
  batch_y                                 present
  batch_x_timestamps_datetime64_ns        present
  batch_y_timestamps_datetime64_ns        present
  batch_x_mark                            present
  batch_y_mark                            present
  batch_text                              present
```

### 2.7 Disk usage matches estimate (322 MB)

```bash
du -sh code/data/TimeMMD
# Expected: ~308 MB (matches the manifest's total_bytes / 1<<20)
```

---

## §3. After Step 2 — smoke test + 2-epoch smoke training

### 3.1 All 11 unit tests pass

```bash
cd code
source .venv/bin/activate
python3 tests/test_smoke.py
```

**Expected final output:**
```
Results: 11/11 tests passed
ALL TESTS PASSED
```

### 3.2 Smoke metrics JSON exists

```bash
ls -la code/results/Economy_Trade/metrics.json
# Expected: file exists, ~2-5 KB
```

### 3.3 Smoke metrics has the expected schema

```bash
python3 -c "
import json
m = json.load(open('code/results/Economy_Trade/metrics.json'))
expected_keys = ['domain', 'model', 'config', 'training_config', 'stats',
                 'history', 'test_metrics', 'best_val_MAE',
                 'trainable_parameters', 'total_parameters',
                 'trainable_percentage', 'total_train_time_s',
                 'epochs_completed', 'seed', 'timestamp']
for k in expected_keys:
    present = k in m
    print(f'  {k:<30} {\"present\" if present else \"MISSING\"}')
print()
print(f'  test_metrics: {m[\"test_metrics\"]}')
print(f'  epochs_completed: {m[\"epochs_completed\"]}')
print(f'  total_train_time_s: {m[\"total_train_time_s\"]}')
"
```

**Expected output:**
```
  domain                         present
  model                          present
  ...
  timestamp                      present

  test_metrics: {'MAE': ..., 'MSE': ..., 'RMSE': ..., 'WAPE': ..., 'MAPE': ..., 'sMAPE': ..., 'R2': ...}
  epochs_completed: 2
  total_train_time_s: <a small number, typically 5-30 seconds>
```

### 3.4 Smoke checkpoint exists

```bash
ls -la code/checkpoints/Economy_Trade/best.pt
ls -la code/checkpoints/Economy_Trade/latest.pt
# Expected: both files exist, each ~5-50 MB
```

---

## §4. After Step 3 (single domain) or Step 4 (all 9 domains) — training

### 4.1 Per-domain metrics JSON exists for each trained domain

```bash
# Single domain (Step 3)
ls -la code/results/Climate_AQI/metrics.json

# All 9 domains (Step 4)
for d in Climate_AQI Economy_Unemp Economy_Trade Economy_VMT \
         Agriculture_Fema Agriculture_Broil Climate_Precip \
         Health_Flu Energy_Gas; do
    if [[ -f "code/results/${d}/metrics.json" ]]; then
        echo "  ✓ ${d}"
    else
        echo "  ✗ ${d} — MISSING"
    fi
done
```

### 4.2 Cross-domain summary JSON exists (Step 4 only)

```bash
ls -la code/results/all_domains_summary.json
# Expected: file exists, ~5-50 KB (depends on history length)
```

### 4.3 Per-domain summary table

```bash
python3 -c "
import json
results = json.load(open('code/results/all_domains_summary.json'))
print(f\"{'Domain':<22} {'MAE':>10} {'RMSE':>10} {'WAPE':>10} {'epochs':>8} {'min':>8}\")
print(f\"{'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}\")
total_min = 0.0
for r in results:
    if 'error' in r:
        print(f\"  {r['domain']:<22}  ERROR: {r['error'][:50]}\")
        continue
    m = r['test_metrics']
    min_train = r['total_train_time_s'] / 60
    total_min += min_train
    print(f\"  {r['domain']:<22} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['WAPE']:>10.4f} {r['epochs_completed']:>8} {min_train:>8.1f}\")
print(f\"{'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}\")
print(f\"  {'TOTAL':<22} {'':>10} {'':>10} {'':>10} {'':>8} {total_min:>8.1f}\")
"
```

### 4.4 Best checkpoint exists for each trained domain

```bash
for d in Climate_AQI Economy_Unemp Economy_Trade Economy_VMT \
         Agriculture_Fema Agriculture_Broil Climate_Precip \
         Health_Flu Energy_Gas; do
    if [[ -f "code/checkpoints/${d}/best.pt" ]]; then
        SIZE=$(du -h "code/checkpoints/${d}/best.pt" | awk '{print $1}')
        echo "  ✓ ${d}/best.pt (${SIZE})"
    else
        echo "  ✗ ${d}/best.pt — MISSING"
    fi
done
```

### 4.5 Disk usage matches estimate (~150 MB per domain, ~1.4 GB total)

```bash
du -sh code/checkpoints
du -sh code/results
# Expected: ~1.4 GB checkpoints, ~5-50 MB results (JSON is small)
```

### 4.6 VRAM usage stays under 12 GB budget (during training)

```bash
# In a separate terminal while training is running:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
# Expected: memory.used < 6000 MiB (well under 12288 MiB on RTX 3080 Ti)
```

### 4.7 CPU usage matches 4-core estimate (during training)

```bash
# In a separate terminal while training is running:
top -bn1 -p $(pgrep -f train_real.py | head -1) | tail -2
# Expected: %CPU ~ 400 (4 cores × 100%)
```

---

## §5. After Step 5 — Phase F analyses

### 5.1 All 5 analysis subdirectories exist

```bash
for d in latency carbon robustness scaling transfer; do
    if [[ -d "code/14_engineering_analyses/${d}" ]]; then
        echo "  ✓ ${d}/"
    else
        echo "  ✗ ${d}/ — MISSING"
    fi
done
```

### 5.2 Each analysis has its expected output files

```bash
# Latency
ls code/14_engineering_analyses/latency/
# Expected: latency_table.csv, latency_curves.png, latency_curves.svg

# Carbon
ls code/14_engineering_analyses/carbon/
# Expected: carbon_report.json, carbon_breakdown.png, carbon_breakdown.svg

# Robustness
ls code/14_engineering_analyses/robustness/
# Expected: robustness_table.csv, robustness_curves.png, robustness_curves.svg

# Scaling
ls code/14_engineering_analyses/scaling/
# Expected: scaling_table.csv, scaling_curves.png, scaling_curves.svg

# Transfer
ls code/14_engineering_analyses/transfer/
# Expected: transfer_table.csv, transfer_heatmap.png, transfer_heatmap.svg, transfer_summary.json
```

### 5.3 Latency CSV has the expected schema

```bash
head -3 code/14_engineering_analyses/latency/latency_table.csv
# Expected:
#   batch_size,latency_ms_per_batch,latency_ms_per_sample,throughput_samples_per_s,peak_gpu_memory_mb
#   1,...,...,...,...
#   4,...,...,...,...
```

### 5.4 Carbon report has the expected schema

```bash
python3 -c "
import json
r = json.load(open('code/14_engineering_analyses/carbon/carbon_report.json'))
print(json.dumps(r, indent=2))
"
# Expected: a dict with keys like 'total_kg_co2e', 'per_experiment_kg_co2e',
# 'methodology', 'assumptions', etc.
```

### 5.5 DIAGNOSTIC README is present

```bash
cat code/14_engineering_analyses/README.md | head -20
# Expected: explains the DIAGNOSTIC label (synthetic mini-batch, not thesis-grade)
```

---

## §6. End-to-end verification (after the full pipeline)

### 6.1 All 6 expected directories exist

```bash
for d in code/.venv code/data/TimeMMD code/checkpoints code/results code/14_engineering_analyses logs; do
    if [[ -d "${d}" ]]; then
        echo "  ✓ ${d}/"
    else
        echo "  ✗ ${d}/ — MISSING"
    fi
done
```

### 6.2 Total disk usage matches estimate (~8.7 GB)

```bash
du -sh code/.venv code/data code/checkpoints code/results code/14_engineering_analyses logs 2>/dev/null
echo "---"
du -sh code 2>/dev/null
# Expected: code/.venv ~3.3 GB, code/data ~322 MB, code/checkpoints ~1.4 GB,
#           code/results ~5-50 MB, code/14_engineering_analyses ~50 MB,
#           logs ~1-10 MB
# Total code/ ~5-6 GB (the rest is the source code itself, ~10 MB)
```

### 6.3 Pipeline log file exists and ends with "PIPELINE COMPLETE"

```bash
LATEST_LOG=$(ls -t logs/pipeline_*.log | head -1)
echo "Latest log: ${LATEST_LOG}"
tail -20 "${LATEST_LOG}"
# Expected: a "PIPELINE COMPLETE" banner followed by the per-domain summary table
```

### 6.4 All exit codes are 0

The master script `run/run_pipeline.sh` exits 0 only if every step succeeded.
If you see a non-zero exit code, check the log file for the failing step.

| Exit code | Meaning |
|-----------|---------|
| 0 | Success — all steps completed |
| 1 | Environment check failed (GPU/CUDA/disk/Python) |
| 2 | Dataset download failed |
| 3 | Smoke test failed |
| 4 | Training failed |
| 5 | Phase F analyses failed |

---

## §7. Re-running individual steps

Each step script is idempotent (safe to re-run). Use these commands to
re-run a single step:

```bash
# Re-run only step 0 (install)
./run/run_pipeline.sh --step install

# Re-run only step 1 (download) — will skip if manifest exists; delete the manifest to force
./run/run_pipeline.sh --step download

# Re-run only step 2 (smoke)
./run/run_pipeline.sh --step smoke

# Re-run only step 3 (train a single domain)
./run/run_pipeline.sh --step train --domain Climate_AQI --epochs 100

# Re-run only step 4 (train all 9 domains)
./run/run_pipeline.sh --step train --epochs 100

# Re-run only step 5 (Phase F analyses)
./run/run_pipeline.sh --step analyses
```

Or call the step scripts directly:

```bash
./run/run_step0_install.sh
./run/run_step1_download_dataset.sh
./run/run_step2_smoke_test.sh
./run/run_step3_train_single_domain.sh Climate_AQI 100
./run/run_step4_train_all_domains.sh 100
./run/run_step5_analyses.sh
```

---

## §8. What to do if verification fails

If any verification command fails:

1. **Read the actual error message.** The verification commands are
   designed to print exactly what's missing. The first failing line is
   almost always the root cause.
2. **Check the pipeline log.** `logs/pipeline_<timestamp>.log` has the
   full stdout/stderr of every step.
3. **Re-run the failing step in isolation.** Use the `--step` flag
   (see §7 above) to re-run only the failing step.
4. **If the step still fails, read the source code of the failing
   script.** Every script has a comprehensive docstring explaining what
   it does, what it expects, and what it produces. The code does not lie.

If you cannot resolve the failure after these steps, the most likely
causes are (a) insufficient hardware (GPU < 11 GB VRAM, RAM < 8 GB,
disk < 50 GB), (b) network issues (Hugging Face mirror unreachable,
PyTorch wheel download interrupted), or (c) OS incompatibility (not
Ubuntu 22.04/24.04, missing NVIDIA driver, missing CUDA 12.1+).
