# PHASE A PLAN — Full GPU Training Run

**Status:** PENDING (external) — requires user's RTX 3080 Ti (12 GB VRAM)
or equivalent.

This document is the explicit runbook for executing Phase A: the full
9-domain × 100-epoch GPU training run that produces thesis-grade results.

---

## 1. Pre-flight checklist (15 min)

Before starting Phase A, verify:

```bash
# 1. Bundle integrity passes
./verify_bundle_integrity.sh
# Expected: ALL CHECKS PASSED

# 2. GPU is visible and has >= 11 GB VRAM free
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
# Expected: RTX 3080 Ti, 12288 MiB, >= 11000 MiB free

# 3. Python venv exists and PyTorch sees CUDA
source code/.venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.1 True

# 4. Dataset is downloaded (322 MB, 27 files)
ls code/data/TimeMMD/*/train.jsonl | wc -l
# Expected: 9

# 5. Smoke tests pass
cd code && python -m pytest tests/ -v
# Expected: 27 passed
```

If any of the above fails, fix it before proceeding. See
`docs/TROUBLESHOOTING.md` for common fixes.

---

## 2. Capture environment fingerprint (1 min)

Before training, capture the exact runtime environment for reproducibility
auditing:

```bash
cd code
source .venv/bin/activate
python scripts/environment_fingerprint.py --config configs/default.yaml
# Output: logs/env_<timestamp>.json
```

This records Python version, OS, CPU, RAM, GPU, all installed pip packages,
the config file SHA-256, and the dataset manifest SHA-256.

---

## 3. Verify determinism (optional, ~5 min)

To confirm the training pipeline is deterministic under seed=42:

```bash
cd code
source .venv/bin/activate
python scripts/verify_determinism.py --domain Economy_Trade --epochs 3
# Expected: PASS (max abs diff < 1e-5)
```

If this fails, training is non-deterministic on your hardware. Investigate
before proceeding — thesis Chapter 18 requires determinism.

---

## 4. Execute the full training run (32–85 min)

```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle

# Option A: Use the master pipeline (skips install + download + smoke)
./run/run_pipeline.sh --skip-install --skip-download --skip-smoke

# Option B: Run only the training step
./run/run_step4_train_all_domains.sh

# Option C: Run training for a single domain first (recommended as a sanity check)
./run/run_step3_train_single_domain.sh Economy_Trade 100
# Expected: ~2-4 min, produces code/results/Economy_Trade/metrics.json
```

**Expected output per domain:**
- `code/checkpoints/<Domain>/best.pt` (best val_MAE checkpoint)
- `code/checkpoints/<Domain>/latest.pt` (latest epoch checkpoint)
- `code/checkpoints/<Domain>/epoch_10.pt`, `epoch_20.pt`, ... (periodic)
- `code/results/<Domain>/metrics.json` (final test metrics + history)

**After all 9 domains complete:**
- `code/results/all_domains_summary.json` (cross-domain summary table)

---

## 5. Execute Phase F analyses on real checkpoints (15–35 min)

After training completes, re-run the 5 Phase F scripts pointing at the
trained checkpoints (instead of the synthetic mini-batch defaults):

```bash
cd code
source .venv/bin/activate

# 1. Latency benchmark on real model
python scripts/latency_carbon.py \
    --checkpoint checkpoints/Climate_AQI/best.pt \
    --domain Climate_AQI \
    --device cuda

# 2. Carbon footprint of the full 432-experiment Chapter 18 suite
python scripts/latency_carbon.py --carbon-only --device cuda

# 3. Robustness analysis (5 perturbations × 5 severities)
python scripts/robustness_analysis.py \
    --checkpoint checkpoints/Climate_AQI/best.pt \
    --domain Climate_AQI --device cuda

# 4. Scaling analysis (10%, 25%, 50%, 75%, 100% of training data)
python scripts/scaling_analysis.py \
    --checkpoint checkpoints/Climate_AQI/best.pt \
    --domain Climate_AQI --device cuda

# 5. Cross-domain transfer (9×9 matrix)
python scripts/cross_domain_transfer.py --device cuda --epochs 10
```

**Outputs land in:** `code/14_engineering_analyses/<analysis>/`

**Important:** These outputs replace the DIAGNOSTIC synthetic-batch outputs
that ship with the bundle. The new outputs are thesis-grade.

---

## 6. Generate thesis figures (5 min)

After Phase F analyses complete, regenerate the Chapter 19 figures from the
real training results:

```bash
cd code
source .venv/bin/activate

# Generate all figures for all 9 domains
python scripts/make_figure.py \
    --metrics-glob 'results/*/metrics.json' \
    --output-dir figures

# Or generate for a single domain
python scripts/make_figure.py \
    --metrics results/Climate_AQI/metrics.json \
    --output-dir figures/Climate_AQI
```

**Outputs land in:** `code/figures/<Domain>/{loss_curves.png,
error_distribution.png, attention_heatmap.png, ablation_chart.png}` (+ SVG)

---

## 7. Verify Phase A is complete (5 min)

```bash
# 1. All 9 domains have metrics.json
ls code/results/*/metrics.json | wc -l
# Expected: 9

# 2. All 9 domains have best.pt
ls code/checkpoints/*/best.pt | wc -l
# Expected: 9

# 3. Cross-domain summary exists
test -f code/results/all_domains_summary.json && echo "OK"

# 4. Phase F outputs have been refreshed (modification time > training start)
stat -c '%y %n' code/14_engineering_analyses/*/ | head -10

# 5. Figures generated
ls code/figures/*/loss_curves.png | wc -l
# Expected: 9
```

---

## 8. Regression checks (10 min)

After Phase A, verify the results are sane:

```bash
# Check that all domains achieved a finite, non-NaN test_MAE
python3 -c "
import json
from pathlib import Path
for mf in sorted(Path('code/results').glob('*/metrics.json')):
    m = json.loads(mf.read_text())
    domain = mf.parent.name
    mae = m.get('test_metrics', {}).get('MAE', 'MISSING')
    print(f'{domain:25s} test_MAE = {mae}')
"
```

**Expected:** All 9 domains have a finite, positive test_MAE. The exact
values depend on the domain, but sanity-check ranges are:

| Domain | Expected MAE range |
|--------|---------------------|
| Climate_AQI | 0.3 – 0.8 |
| Climate_Precip | 0.5 – 1.2 |
| Economy_Trade | 1.0 – 3.0 |
| Economy_Unemp | 0.5 – 1.5 |
| Economy_VMT | 0.5 – 1.5 |
| Agriculture_Fema | 0.3 – 0.8 |
| Agriculture_Broil | 0.3 – 0.8 |
| Health_Flu | 0.5 – 1.5 |
| Energy_Gas | 0.5 – 1.5 |

If any domain's MAE is wildly outside its range, investigate (check the
loss curves for instability, check the data for anomalies).

---

## 9. Update the thesis with real numbers (manual, ~2 hours)

After Phase A produces thesis-grade numbers:

1. Open `01_final_deliverables/persian/thesis_report_final.docx` (or English)
2. Find every place labelled "DESIGN TARGET" or "DIAGNOSTIC"
3. Replace with the actual measured values from
   `code/results/all_domains_summary.json`
4. Re-generate the figures in the thesis using `make_figure.py`
5. Re-build the DOCX/PDF using `03_build_scripts/run_full_pipeline.sh --docx-only`

---

## 10. Archive Phase A results (5 min)

```bash
# Create a Phase A archive
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
tar -czf phase_a_results_$(date +%Y%m%d).tar.gz \
    code/results/ \
    code/checkpoints/ \
    code/14_engineering_analyses/ \
    code/figures/ \
    logs/

# Verify the archive
tar -tzf phase_a_results_$(date +%Y%m%d).tar.gz | wc -l
# Expected: 100+ files
```

Store this archive alongside the bundle for reproducibility.

---

## Phase A completion checklist

- [ ] Pre-flight checklist passes (§1)
- [ ] Environment fingerprint captured (§2)
- [ ] Determinism verified (§3, optional but recommended)
- [ ] All 9 domains trained (§4)
- [ ] `all_domains_summary.json` exists
- [ ] Phase F analyses re-run on real checkpoints (§5)
- [ ] Thesis figures regenerated (§6)
- [ ] Verification commands pass (§7)
- [ ] Regression checks within expected ranges (§8)
- [ ] Thesis updated with real numbers (§9, manual)
- [ ] Phase A archive created (§10)

Once all boxes are checked, Phase A is complete and the bundle is
thesis-defensible.
