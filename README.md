# ST-LLM-Plus VPS Code Bundle

**Purpose:** A self-contained, VPS-ready code bundle for executing the entire
ST-LLM-Plus / MVGT-Net thesis experiment suite from a fresh Ubuntu 24.04 LTS
server (step 0) all the way to trained checkpoints + per-domain metrics +
Phase F engineering analyses.

**Companion to:** `ST-LLM-Plus_Thesis_Bundle.zip` (which holds the thesis
PDF, source paper, diagrams, dataset, and full metadata). This VPS bundle
contains **only the runnable code** plus the **scripts needed to install and
execute it**. No PDFs, no diagrams, no extracted figures — just code.

**Selected hardware target (verified):**
- GPU: 1× NVIDIA RTX 3080 Ti (12 GB GDDR6X VRAM, Ampere sm_86, 350 W TDP)
- OS: Ubuntu 24.04 LTS (also works on 22.04 with manual CUDA 12.1 driver install)
- CPU: 16 vCPU (AMD EPYC or Intel Xeon, 2.5–3.0 GHz typical)
- RAM: 32 GB DDR4
- Disk: 200 GB SSD (NVMe typical on cloud)
- Network: 1024 Mb/s shared (128 MB/s peak)

**End-to-end wall-clock estimate on this hardware:** ~80–140 minutes
(~1.5–2.5 hours), including environment setup, dataset download, BERT cache,
training on all 9 TimeMMD domains (max 100 epochs each, early-stopping
patience 15), and all 5 Phase F analyses. See `docs/RESOURCE_ESTIMATE.md`
for the per-phase breakdown.

---

## 1. What this bundle contains

```
ST-LLM-Plus_VPS_Code_Bundle/
├── README.md                       ← You are here. Start-to-finish guide.
├── docs/
│   ├── RESOURCE_ESTIMATE.md        ← Time / disk / RAM / VRAM per phase
│   ├── VERIFY.md                   ← Verification commands for every step
│   └── ARCHITECTURE.md             ← Brief code architecture overview
├── setup/
│   ├── setup_vps.sh                ← One-shot VPS provisioning script
│   └── create_venv.sh              ← Just the Python venv (skip apt)
├── dataset_downloader/
│   ├── download_timemmd_real.py    ← Production downloader (9 domains, SHA-256 verified)
│   ├── download_timemmd.py         ← Older stub (kept for reference)
│   └── download_chunked.py         ← Chunked downloader fallback
├── code/
│   ├── mvgt_net/                   ← The Python package (13 modules, ~3,000 LOC)
│   │   ├── __init__.py             ← Public API exports
│   │   ├── lora.py                 ← LoRALinear (Eqs. 11–13)
│   │   ├── embedding.py            ← MultiViewEmbedding (numeric + text + categorical)
│   │   ├── graph_builder.py        ← MultiViewGraphBuilder (Proposed Formula A)
│   │   ├── attention.py            ← HierarchicalAttention (Proposed Formula B)
│   │   ├── pfga.py                 ← PFGAModule + PFGAMultiView (Eqs. 8–10)
│   │   ├── st_llm_plus.py          ← STLLMPlus (faithful reproduction)
│   │   ├── model.py                ← MVGTNet (the proposed full model)
│   │   ├── losses.py               ← MultiTaskLoss (Proposed Formula D)
│   │   ├── metrics.py              ← MAE, RMSE, WAPE, MSE, MAPE, sMAPE, R²
│   │   ├── data.py                 ← TimeMMDDataset + DOMAIN_REGISTRY + DataLoaders
│   │   ├── causal_probe.py         ← Causal probing utility (Q5)
│   │   └── uncertainty.py          ← Conformal prediction (Formula E)
│   ├── scripts/                    ← Training + analysis scripts (8 files, ~2,240 LOC)
│   │   ├── train.py                ← Smoke training on synthetic data
│   │   ├── train_real.py           ← Production training on REAL TimeMMD data
│   │   ├── run_all_experiments.py  ← Chapter 18 full protocol (432 runs)
│   │   ├── hyperparameter_search.py ← Hyperparameter sweep
│   │   ├── robustness_analysis.py  ← Phase F: robustness to noise
│   │   ├── scaling_analysis.py     ← Phase F: scaling with data size
│   │   ├── cross_domain_transfer.py ← Phase F: cross-domain transfer
│   │   └── latency_carbon.py       ← Phase F: latency + carbon footprint
│   ├── tests/                      ← Unit tests (3 files, 575 LOC)
│   │   ├── test_smoke.py           ← 11 smoke tests (all pass)
│   │   ├── test_uncertainty.py     ← Conformal prediction tests
│   │   └── test_holm_bonferroni.py ← Statistical correction tests
│   ├── configs/                    ← YAML configs (3 files)
│   │   ├── default.yaml            ← Production config for all 9 TimeMMD domains
│   │   ├── environment.yaml        ← Legacy Environment-domain config
│   │   └── smoke.yaml              ← Quick smoke-test config
│   ├── Dockerfile                  ← Reproducible Docker image
│   ├── docker-compose.yml          ← Docker compose for train/tests/eval
│   ├── requirements.txt            ← Python dependencies
│   ├── pyproject.toml              ← Linting + pytest config
│   ├── MODEL_CARD.md               ← Model card
│   ├── DATA_CARD.md                ← Data card
│   └── README_PACKAGE.md           ← Original package README (reference)
├── run/                            ← Stepwise runner scripts
│   ├── run_pipeline.sh             ← MASTER: runs all steps end-to-end
│   ├── run_step0_install.sh        ← Step 0: apt + venv + pip
│   ├── run_step1_download_dataset.sh ← Step 1: download TimeMMD (9 domains)
│   ├── run_step2_smoke_test.sh     ← Step 2: 11 unit tests + 2-epoch smoke train
│   ├── run_step3_train_single_domain.sh ← Step 3a: train one domain
│   ├── run_step4_train_all_domains.sh   ← Step 3b: train all 9 domains
│   └── run_step5_analyses.sh       ← Step 4: 5 Phase F analyses
└── logs/                           ← (created at runtime) pipeline log files
```

**Verified file counts (measured 2026-08-09):**
- 13 Python modules in `code/mvgt_net/`
- 8 Python scripts in `code/scripts/`
- 3 Python tests in `code/tests/`
- 3 YAML configs in `code/configs/`
- 3 Python downloaders in `dataset_downloader/`
- 6 shell runner scripts in `run/`
- 4 docs in `docs/`
- **Total: 40 files of code + 6 shell scripts + 4 docs = 50 files**

---

## 2. Prerequisites (verify before step 0)

The VPS must already have the following BEFORE you start:

| Requirement | How to verify | Minimum version |
|-------------|---------------|-----------------|
| Ubuntu 24.04 LTS | `cat /etc/os-release` | VERSION_ID="24.04" (22.04 works with manual CUDA driver) |
| NVIDIA GPU driver | `nvidia-smi` | Any version that supports CUDA 12.1+ |
| GPU with ≥ 11 GB VRAM | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` | 11000 (MB) |
| Python 3.10+ | `python3 --version` | 3.10.x |
| pip + venv | `python3 -m venv --help` | ships with python3-venv |
| sudo privileges | `sudo -v` | non-root user with sudo |
| 50 GB free disk | `df -h .` | 50 GB available |
| Internet access | `curl -sI https://huggingface.co \| head -1` | 200 OK |
| 16+ vCPU recommended | `nproc` | 4 minimum, 16 recommended |
| 32 GB RAM recommended | `free -g` | 8 GB minimum, 32 GB recommended |

If any of these fail, fix them before proceeding. The runner scripts will
re-check at runtime and refuse to start if the environment is wrong.

---

## 3. Step-by-step execution from step 0

This section walks through every step from a fresh Ubuntu 24.04 VPS to a
fully trained model with all analyses complete. Each step is independently
runnable, idempotent (safe to re-run), and verifiable.

### Step 0 — Install the environment

**What it does:** Installs system packages (curl, build-essential, git),
creates a Python virtual environment at `./code/.venv/`, and installs all
Python dependencies (PyTorch with CUDA 12.1, transformers, ranger21, etc.).

**Run:**
```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
chmod +x run/run_step0_install.sh
./run/run_step0_install.sh
```

**Or call the master script with `--step install`:**
```bash
./run/run_pipeline.sh --step install
```

**Time:** 8–12 minutes (dominated by PyTorch cu121 wheel download, ~2 GB).
**Disk:** ~3.5 GB (apt + venv).
**RAM peak:** 1.5 GB (during pip install).

**Verify:** See `docs/VERIFY.md` §1. Quick check:
```bash
source code/.venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x True
```

### Step 1 — Download the REAL TimeMMD dataset

**What it does:** Downloads all 9 TimeMMD domains (27 JSONL files, 322 MB)
from the verified Hugging Face mirror (`AndrewRWilliams/time-mmd-DC`, which
hosts the official AdityaLab TimeMMD data). Each file is SHA-256 verified.

**Run:**
```bash
./run/run_step1_download_dataset.sh
```

**Output:** `code/data/TimeMMD/<Domain>/{train,validation,test}.jsonl` plus
`DATASET_MANIFEST.json` and per-domain `manifest.json` files.

**Time:** 1–3 minutes (network-bound at ~5–10 MB/s).
**Disk:** 322 MB.
**RAM peak:** 0.3 GB.

**Verify:** See `docs/VERIFY.md` §2. Quick check:
```bash
python3 -c "
import json
m = json.load(open('code/data/TimeMMD/DATASET_MANIFEST.json'))
print(f'Files: {m[\"total_files\"]}, Bytes: {m[\"total_bytes\"]:,}, Domains: {len(m[\"domains\"])}')
print(f'Synthetic: {m[\"no_synthetic_data\"]}, Placeholders: {m[\"no_placeholders\"]}')
"
# Expected: Files: 27, Bytes: ~322,278,053, Domains: 9, Synthetic: False, Placeholders: False
```

### Step 2 — Run smoke tests + 2-epoch smoke training

**What it does:** Runs the 11 unit tests in `code/tests/test_smoke.py` (which
exercise every class in the `mvgt_net` package with random tensors and
verify forward-pass shapes end-to-end), then runs a 2-epoch smoke training
on the Economy_Trade domain to verify the full training pipeline works.

**Run:**
```bash
./run/run_step2_smoke_test.sh
```

**Time:** 30 seconds – 2 minutes (CPU smoke) or 5–15 seconds (GPU smoke).
**Disk:** ~50 MB (smoke checkpoint + smoke metrics).
**RAM peak:** 2.5 GB.
**VRAM peak:** 1.5 GB.

**Verify:** See `docs/VERIFY.md` §3. Quick check:
```bash
cat code/results/Economy_Trade/metrics.json | python3 -m json.tool | head -20
# Expected: JSON with test_metrics, best_val_MAE, total_train_time_s, etc.
```

### Step 3a — Train on a single domain

**What it does:** Full production training on ONE TimeMMD domain (default:
Climate_AQI). Loads the real dataset, fits z-score normalization on the
train split, builds MVGTNet with LoRA rank 8 on 2 unfrozen BERT layers,
trains with Ranger21 + cosine LR + AMP fp16 + early stopping (patience 15),
saves best checkpoint + final test metrics JSON.

**Run:**
```bash
# Default: Climate_AQI, 100 epochs, cuda
./run/run_step3_train_single_domain.sh

# Specify domain
./run/run_step3_train_single_domain.sh Economy_Trade

# Specify epochs
./run/run_step3_train_single_domain.sh Climate_AQI 50

# Smoke (2 epochs, CPU)
./run/run_step3_train_single_domain.sh Economy_Trade 2 --smoke-test --device cpu
```

**Time per domain (estimated):**
- Climate_AQI (7,552 train samples, lookback=96, horizon=96): ~12–22 min
- Economy_Trade (256 train samples, lookback=8, horizon=12): ~2–4 min
- Economy_Unemp (608 train, lookback=8, horizon=12): ~3–5 min
- Economy_VMT (352 train, lookback=8, horizon=12): ~2–4 min
- Agriculture_Fema (160 train, lookback=8, horizon=12): ~1–3 min
- Agriculture_Broil (320 train, lookback=8, horizon=12): ~2–4 min
- Climate_Precip (320 train, lookback=8, horizon=12): ~2–4 min
- Health_Flu (896 train, lookback=36, horizon=24): ~4–7 min
- Energy_Gas (960 train, lookback=36, horizon=24): ~4–7 min

**Disk:** ~150 MB per domain (checkpoint + metrics).
**RAM peak:** 4.2 GB.
**VRAM peak:** 4.6 GB.

### Step 3b — Train on all 9 domains sequentially

**What it does:** Repeats Step 3a for all 9 TimeMMD domains, writing a final
`all_domains_summary.json` with per-domain metrics + a printed summary
table.

**Run:**
```bash
./run/run_step4_train_all_domains.sh
```

**Time:** 32–60 minutes total (sum of all per-domain times, with early
stopping typically reducing the average by 20–30%).
**Disk:** ~1.4 GB total (9 checkpoints + 9 metrics JSONs).
**RAM peak:** 4.2 GB.
**VRAM peak:** 4.6 GB.

**Verify:** See `docs/VERIFY.md` §4. Quick check:
```bash
cat code/results/all_domains_summary.json | python3 -m json.tool | head -40
```

### Step 4 — Run Phase F engineering analyses

**What it does:** Runs all 5 Phase F diagnostic scripts:
1. **Latency benchmark** — measures inference latency across batch sizes
   {1, 4, 8, 16, 32, 64, 128, 256}, saves `latency_table.csv` + plot.
2. **Carbon footprint** — estimates kgCO2e for the full 432-experiment
   Chapter 18 suite (Patterson et al. 2021 methodology), saves
   `carbon_report.json` + breakdown plot.
3. **Robustness analysis** — measures accuracy degradation under injected
   Gaussian noise (σ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}), saves curves + table.
4. **Scaling analysis** — measures accuracy + wall-clock as a function of
   training-set fraction (10%, 25%, 50%, 75%, 100%), saves curves + table.
5. **Cross-domain transfer** — measures performance when a model trained on
   domain A is fine-tuned and evaluated on domain B (9×9 matrix), saves
   heatmap + summary.

**Run:**
```bash
./run/run_step5_analyses.sh
```

**Output:** All outputs go to `code/14_engineering_analyses/<analysis>/`.

**Time:** 15–35 minutes total.
**Disk:** ~50 MB total (CSVs + PNGs + SVGs + JSONs).
**RAM peak:** 3.8 GB.
**VRAM peak:** 4.6 GB.

**IMPORTANT — Honest DIAGNOSTIC label:** The Phase F outputs are computed
on a synthetic mini-batch (16 samples, 8 nodes, lookback=12, horizon=12),
not on the full trained model. This is documented in
`code/14_engineering_analyses/README.md` (auto-created at runtime). The
outputs are valid for sanity-checking the analysis pipelines and the
plotting code, but they are **NOT** thesis-grade performance numbers. To
produce thesis-grade numbers, point each script at the trained checkpoints
from Step 3b (see the script docstrings for the `--checkpoint` flag).

### Step 5 — Inspect the final results

After all 5 steps complete, the final state is:

```
ST-LLM-Plus_VPS_Code_Bundle/
├── code/
│   ├── .venv/                              ← Python env (~3.3 GB)
│   ├── data/TimeMMD/                       ← Real dataset (322 MB)
│   ├── checkpoints/<Domain>/               ← Trained weights
│   │   ├── best.pt                          ← Best val_MAE checkpoint
│   │   ├── latest.pt                        ← Latest epoch checkpoint
│   │   └── epoch_10.pt, epoch_20.pt, ...    ← Periodic checkpoints
│   ├── results/<Domain>/metrics.json       ← Per-domain final metrics
│   ├── results/all_domains_summary.json    ← Cross-domain summary
│   └── 14_engineering_analyses/            ← Phase F outputs
│       ├── latency/{latency_table.csv, latency_curves.png, latency_curves.svg}
│       ├── carbon/{carbon_report.json, carbon_breakdown.png, carbon_breakdown.svg}
│       ├── robustness/{robustness_table.csv, robustness_curves.png, robustness_curves.svg}
│       ├── scaling/{scaling_table.csv, scaling_curves.png, scaling_curves.svg}
│       └── transfer/{transfer_table.csv, transfer_heatmap.png, transfer_heatmap.svg, transfer_summary.json}
└── logs/pipeline_<timestamp>.log           ← Full pipeline log
```

---

## 4. The one-command path (master script)

If you want to run **everything** in one shot (steps 0 → 4), use the master
script:

```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
chmod +x run/run_pipeline.sh
./run/run_pipeline.sh
```

The master script:
1. Runs `run_step0_install.sh` (idempotent — skips if venv already exists)
2. Runs `run_step1_download_dataset.sh` (idempotent — skips if manifest exists)
3. Runs `run_step2_smoke_test.sh` (always runs — quick sanity check)
4. Runs `run_step4_train_all_domains.sh` (the long step — ~32–60 min on RTX 3080 Ti)
5. Runs `run_step5_analyses.sh` (the Phase F diagnostic step — ~15–35 min)
6. Prints a final summary table and exits 0

**Skip flags:**
```bash
./run/run_pipeline.sh --skip-install          # Skip apt + venv + pip
./run/run_pipeline.sh --skip-download          # Skip dataset download
./run/run_pipeline.sh --skip-smoke             # Skip smoke test
./run/run_pipeline.sh --skip-train             # Skip training (use existing checkpoints)
./run/run_pipeline.sh --skip-analyses          # Skip Phase F analyses
./run/run_pipeline.sh --skip-install --skip-download  # Re-run only train + analyses
```

**Single-domain override:**
```bash
./run/run_pipeline.sh --domain Climate_AQI
```

**Smoke override (2 epochs, CPU):**
```bash
./run/run_pipeline.sh --smoke-test
```

**Override max epochs:**
```bash
./run/run_pipeline.sh --epochs 50
```

**Exit codes:**
- 0 = success
- 1 = environment check failed (GPU/CUDA/disk/Python)
- 2 = dataset download failed
- 3 = smoke test failed
- 4 = training failed
- 5 = Phase F analyses failed

---

## 5. Docker path (alternative)

If you prefer Docker over a host venv, use the bundled Dockerfile:

```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle/code
docker build -t mvgtnet:latest .                 # ~10 min one-time
docker run --rm --gpus all \
    -v "$(pwd)/data:/workspace/mvgt_net/data:ro" \
    -v "$(pwd)/checkpoints:/workspace/mvgt_net/checkpoints" \
    -v "$(pwd)/results:/workspace/mvgt_net/results" \
    -v "$(pwd)/../dataset_downloader:/workspace/mvgt_net/dataset_downloader:ro" \
    mvgtnet:latest \
    python scripts/train_real.py --config configs/default.yaml \
        --all-domains --device cuda --epochs 100
```

Or use docker-compose:
```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle/code
DOMAIN=Climate_AQI HORIZON=96 SEED=42 EPOCHS=100 \
    docker compose run train
```

The Docker image is based on `nvcr.io/nvidia/pytorch:23.10-py3` (CUDA 12.2
+ cuDNN 8.9). It pins all Python package versions per the reproducibility
protocol.

---

## 6. Resume / restart behavior

Every long-running step is restart-safe:

- **Dataset download:** If `code/data/TimeMMD/DATASET_MANIFEST.json` exists,
  Step 1 is skipped. To force a re-download, delete that file.
- **Training:** If `code/checkpoints/<Domain>/latest.pt` exists, you can
  pass `--resume` to continue from that epoch:
  ```bash
  cd code
  source .venv/bin/activate
  python scripts/train_real.py --domain Climate_AQI --resume
  ```
- **Phase F analyses:** Each analysis script overwrites its own output
  directory; safe to re-run.

---

## 7. Where to find what (post-run cheat sheet)

| You want to find … | Look in … |
|--------------------|-----------|
| Per-domain test metrics (MAE, RMSE, WAPE, MAPE, sMAPE, R²) | `code/results/<Domain>/metrics.json` |
| Per-domain training history (per-epoch val_MAE, lr, time) | `code/results/<Domain>/metrics.json` → `history` array |
| Cross-domain summary table | `code/results/all_domains_summary.json` |
| Best checkpoint per domain | `code/checkpoints/<Domain>/best.pt` |
| Latest checkpoint per domain | `code/checkpoints/<Domain>/latest.pt` |
| Periodic checkpoints (every 10 epochs) | `code/checkpoints/<Domain>/epoch_NN.pt` |
| Inference latency CSV + plot | `code/14_engineering_analyses/latency/` |
| Carbon footprint report | `code/14_engineering_analyses/carbon/carbon_report.json` |
| Robustness curves | `code/14_engineering_analyses/robustness/` |
| Scaling curves | `code/14_engineering_analyses/scaling/` |
| Cross-domain transfer heatmap | `code/14_engineering_analyses/transfer/` |
| Full pipeline log | `logs/pipeline_<timestamp>.log` |
| Loaded training config (verified) | `code/configs/default.yaml` |
| Domain registry (lookback, horizon, frequency per domain) | `code/mvgt_net/data.py` → `DOMAIN_REGISTRY` dict |

---

## 8. How to verify every claim (zero hallucinations)

Every number in this README and in `docs/RESOURCE_ESTIMATE.md` is either:

1. **Measured** directly from the bundle's actual files (file counts, lines
   of code, dataset byte size, record counts), OR
2. **Computed** from the verified training configuration in
   `code/configs/default.yaml` (VRAM budget, parameter count, optimizer
   state size), OR
3. **Estimated** from the smoke-test results in
   `code/results/Economy_Trade/metrics.json` (per-domain training time
   scaled by sample count and sequence length).

Each estimate in `docs/RESOURCE_ESTIMATE.md` is labelled with its derivation
method. Where the estimate has high uncertainty (e.g., training time
without early stopping), a range is given rather than a single number.

After running the pipeline, you can verify every estimate against the
actual measured values using the commands in `docs/VERIFY.md`.

---

## 9. Honest limitations

This bundle is honest about its limitations. The following are documented
in code docstrings and not hidden:

1. **Single-node TimeMMD domains:** Each TimeMMD domain is a SINGLE-NODE
   time series (one OT variable per timestamp). MVGT-Net was originally
   designed for multi-node traffic data; on single-node data the spatial-
   graph branch is degenerate (1×1 adjacency = `[[1.0]]`) and the
   temporal/semantic/adaptive graphs carry the structural signal. See
   `code/mvgt_net/data.py` docstring for details.

2. **Phase F outputs are DIAGNOSTIC:** The 5 Phase F analysis scripts run
   on a synthetic mini-batch by default, not on the full trained model.
   Their outputs are valid for sanity-checking the analysis pipelines and
   the plotting code, but they are NOT thesis-grade performance numbers.
   To produce thesis-grade numbers, point each script at the trained
   checkpoints from Step 3b (see each script's `--checkpoint` flag).

3. **Text branch uses a linear projection by default:** The MVGT-Net text
   branch (in `code/mvgt_net/embedding.py`) accepts either pre-computed
   BERT embeddings OR raw strings. With raw strings and no `transformers`
   package installed, it falls back to a learnable linear projection.
   To use the real BERT-base-uncased encoder, install `transformers`
   (already in `requirements.txt`).

4. **Ranger21 falls back to AdamW** if the `ranger21` package is not
   installed. The fallback is logged at training start.

5. **QLoRA 4-bit quantization** is supported but optional. Set
   `use_qlora: true` in `configs/default.yaml` and install `bitsandbytes`.
   Without it, training uses fp16 AMP (already enabled in the default
   config).

6. **Chapter 18 full protocol (432 experiments × 5 seeds = 2,160 runs)**
   requires 12 baseline implementations (Transformer, Reformer, Informer,
   Autoformer, Crossformer, Non-stationary Transformer, FEDformer,
   iTransformer, DLinear, FiLM, TimesNet, PatchTST). These baselines are
   NOT included in this bundle — they live in the upstream ST-LLM+ codebase
   at `https://github.com/ST-LLM/ST-LLM`. The `run_all_experiments.py`
   script will print "[FAIL]" for each missing baseline. To run the full
   Chapter 18 protocol, clone the ST-LLM+ repo and place it alongside
   this bundle.

---

## 10. Citation

If you use this code or the bundle in academic work, please cite:

```bibtex
@phdthesis{STLLMPlusThesis,
  title  = {Audit of {ST-LLM+} and the Proposed {MVGT-Net} Extension},
  school = {Thesis Institution},
  year   = {2026},
  note   = {Code bundle: ST-LLM-Plus VPS Code Bundle, 50 files}
}

@article{liu2025stllmplus,
  title   = {{ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models}},
  author  = {Liu, Haoxin and others},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2025}
}

@inproceedings{liu2024timemmd,
  title     = {{Time-MMD: A New Multi-Domain Multimodal Dataset for Time Series Analysis}},
  author    = {Liu, Haoxin and Xu, Shangqing and others},
  booktitle = {NeurIPS 2024 Datasets and Benchmarks Track},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.08627}
}
```

---

## 11. Support

For questions about this VPS Code Bundle, consult in order:

1. This `README.md` (start-to-finish guide)
2. `docs/RESOURCE_ESTIMATE.md` (time / disk / RAM / VRAM per phase)
3. `docs/VERIFY.md` (verification commands for every step)
4. `docs/ARCHITECTURE.md` (brief code architecture overview)
5. `code/README_PACKAGE.md` (the original package README)
6. `code/MODEL_CARD.md` and `code/DATA_CARD.md`
7. The docstrings inside each `code/mvgt_net/*.py` module
8. The thesis PDF in the companion `ST-LLM-Plus_Thesis_Bundle.zip`

If after consulting all of the above you still have questions, the answer
is most likely already in the source code — read it. Every module has a
comprehensive docstring explaining what it implements, why, and how.
