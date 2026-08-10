# QUICKSTART — ST-LLM-Plus VPS Code Bundle

**Fastest path from zero to first result.** For full documentation, see
`README.md`.

---

## Prerequisites (5 minutes)

1. **Ubuntu 24.04 LTS VPS** with:
   - NVIDIA GPU (≥ 8 GB VRAM; 12 GB recommended)
   - 16+ vCPU
   - 32 GB RAM
   - 50 GB free disk

2. **Verify:**
```bash
cat /etc/os-release | grep VERSION_ID       # VERSION_ID="24.04"
nvidia-smi                                  # GPU visible
python3 --version                           # Python 3.10+
df -h .                                     # 50 GB free
```

---

## Option A: One-command (full pipeline, ~80–140 min)

```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
chmod +x run/run_pipeline.sh
./run/run_pipeline.sh
```

This runs:
1. Install dependencies (8–12 min)
2. Download dataset (1–3 min)
3. Smoke tests (30 sec – 2 min)
4. Train on all 9 domains (32–60 min)
5. Phase F analyses (15–35 min)

---

## Option B: Step-by-step (for inspection/learning)

### Step 0 — Install
```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
chmod +x run/*.sh setup/*.sh
./run/run_step0_install.sh
```
**Time:** 8–12 min | **Disk:** ~3.5 GB | **RAM:** 1.5 GB peak

### Step 1 — Download dataset
```bash
./run/run_step1_download_dataset.sh
```
**Time:** 1–3 min | **Disk:** 322 MB

### Step 2 — Smoke test
```bash
./run/run_step2_smoke_test.sh
```
**Time:** 30 sec – 2 min | **VRAM:** 1.5 GB peak

### Step 3 — Train one domain
```bash
# Train on Climate_AQI (the largest, ~12-22 min)
./run/run_step3_train_single_domain.sh Climate_AQI 100

# Or train on Economy_Trade (the smallest, ~2-4 min)
./run/run_step3_train_single_domain.sh Economy_Trade 100
```
**VRAM:** 4.6 GB peak | **RAM:** 4.2 GB peak

### Step 4 — Train all 9 domains
```bash
./run/run_step4_train_all_domains.sh
```
**Time:** 32–60 min | **Disk:** ~1.4 GB checkpoints

### Step 5 — Phase F analyses
```bash
./run/run_step5_analyses.sh
```
**Time:** 15–35 min | **Outputs:** `code/14_engineering_analyses/`

---

## Option C: Smoke test only (fastest, ~2 min)

```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
chmod +x run/*.sh
./run/run_pipeline.sh --smoke-test --skip-install --skip-download
```
Runs 2 epochs on Economy_Trade with CPU. No GPU required.

---

## Verify it worked

```bash
# Check final results:
cat code/results/all_domains_summary.json | python3 -m json.tool | head -30

# Check Phase F outputs:
ls code/14_engineering_analyses/

# Verify bundle integrity:
./verify_bundle_integrity.sh
```

---

## What's next?

| If you want to... | Read... |
|--------------------|---------|
| Understand the code architecture | `docs/ARCHITECTURE.md` |
| See the full time/disk/RAM estimate | `docs/RESOURCE_ESTIMATE.md` |
| Verify each step worked | `docs/VERIFY.md` |
| Troubleshoot issues | `docs/TROUBLESHOOTING.md` |
| See common questions | `docs/FAQ.md` |
| Plan the Phase A GPU run | `docs/PHASE_A_PLAN.md` |
| Read the full thesis | `../ST-LLM-Plus_Thesis_Bundle/01_final_deliverables/` |

---

## One-line status check

```bash
./verify_bundle_integrity.sh 2>&1 | tail -5
```

Expected output:
```
  Passed: 6
  Failed: 0
  Skipped: 2

==========================================
  ALL CHECKS PASSED
==========================================
```
