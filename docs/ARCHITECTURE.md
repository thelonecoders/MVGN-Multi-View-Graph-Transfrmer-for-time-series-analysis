# Code Architecture Overview

This document gives a brief architectural overview of the MVGT-Net code in
this bundle. For full implementation details, see the docstrings inside
each `code/mvgt_net/*.py` module.

---

## 1. High-level data flow

```
Raw TimeMMD JSONL files
        │
        ▼
┌─────────────────────────────┐
│ mvgt_net.data               │  TimeMMDDataset, fit_normalization,
│  - TimeMMDDataset           │  collate_fn, get_dataloaders
│  - get_dataloaders          │
└──────────────┬──────────────┘
               │  (x_numeric, y_target, x_text, x_cat, adj)
               ▼
┌─────────────────────────────┐
│ mvgt_net.model.MVGTNet      │  The full model
│                             │
│  ┌──────────────────────┐   │
│  │ MultiViewEmbedding   │   │  Numeric + Text + Categorical → D-dim
│  └──────────┬───────────┘   │
│             ▼               │
│  ┌──────────────────────┐   │
│  │ MultiViewGraphBuilder│   │  4 adjacency matrices (Proposed Formula A)
│  └──────────┬───────────┘   │
│             ▼               │
│  ┌──────────────────────┐   │
│  │ PFGAMultiView        │   │  F frozen + U unfrozen layers (Eqs. 8-10)
│  │  - PFGAModule × N    │   │  Each layer = HierarchicalAttention + FFN
│  │    - HierarchicalAtt │   │   - Time attention → View attention → Graph attention
│  │    - FFN             │   │     (Proposed Formula B)
│  │    - LoRALinear      │   │
│  └──────────┬───────────┘   │
│             ▼               │
│  ┌──────────────────────┐   │
│  │ Output head          │   │  numeric regression (+ optional cat, text-gen)
│  └──────────┬───────────┘   │
│             ▼               │
│      outputs["numeric"]     │  shape: (B, horizon, num_nodes, input_dim)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ mvgt_net.losses             │  MultiTaskLoss with dynamic weighting
│  MultiTaskLoss              │  (Proposed Formula D)
└──────────────┬──────────────┘
               │  total_loss (scalar)
               ▼
        backward + optimizer.step()
```

---

## 2. Module responsibilities

### `mvgt_net/__init__.py`
Public API exports. Re-exports the main classes and functions from the
submodules so callers can do `from mvgt_net import MVGTNet`.

### `mvgt_net/lora.py` — LoRALinear
Low-rank adaptation layer (Eqs. 11–13). Wraps a frozen `nn.Linear` and
adds a trainable low-rank update `α/r × M @ L @ x`. Default: r=8, α=16.
Initializes `L` via Kaiming uniform and `M` via zeros (so initial output
equals base output — no degradation at start of training).

### `mvgt_net/embedding.py` — MultiViewEmbedding
Implements the generalized form of ST-LLM+ Equations 2–7. Three branches:
- **Numeric:** pointwise 1×1 conv (PConv) over the input dim
- **Text:** linear projection from BERT 768-dim to D (BERT itself optional)
- **Categorical:** learnable embedding lookup
Output: `(B, N, 3D)` (concatenation of 3 views, each D-dim).

### `mvgt_net/graph_builder.py` — MultiViewGraphBuilder (Proposed Formula A)
Builds 4 adjacency matrices:
- **Spatial:** pre-computed by caller (geographic adjacency)
- **Temporal:** Pearson correlation of node time series
- **Semantic:** cosine similarity of node embeddings
- **Adaptive:** learnable `E @ E^T` (GraphWaveNet-style)
View weights via single softmax (mathematically sufficient; a second
softmax would be redundant). Top-k sparsification (default k=8) for
complexity control. Row normalization for spectral stability.

### `mvgt_net/attention.py` — HierarchicalAttention (Proposed Formula B)
Three levels, applied in order:
1. **Time attention:** multi-head self-attention over the time dimension
2. **View attention:** single-head attention over the 3 views (numeric, text, categorical)
3. **Graph attention:** multi-head attention with **additive** masking
   (corrected from the thesis's Hadamard formulation; see docstring for
   the correction note)
Returns attention weights from all 3 levels for interpretability (Q5).

### `mvgt_net/pfga.py` — PFGAModule + PFGAMultiView (Eqs. 8–10)
- `PFGAModule`: a single PFGA layer (either frozen or unfrozen).
  Frozen = MHA + FFN frozen, only LayerNorm trainable. Unfrozen = full
  fine-tuning, with LoRA on the linear projections.
- `PFGAMultiView`: stacks (F + U) PFGAModules. First F are frozen, last U
  are unfrozen. Default: F=6, U=2 (per ST-LLM+ protocol).

### `mvgt_net/st_llm_plus.py` — STLLMPlus
Faithful reproduction of the source ST-LLM+ model (from the IEEE TKDE 2025
paper). Kept for ablation comparison: MVGT-Net vs ST-LLM+.

### `mvgt_net/model.py` — MVGTNet (the proposed full model)
Combines all 5 components above into one model. Multi-task outputs:
- `numeric`: regression head (always)
- `categorical`: classification head (optional, if `num_categories > 0`)
- `text`: text generation head (optional, if `use_text_gen=True`)
The `parameter_efficiency()` method returns metrics comparable to ST-LLM+
Table II (total params, trainable params, %trainable, %frozen).

### `mvgt_net/losses.py` — MultiTaskLoss (Proposed Formula D)
Dynamic task weighting: `w_k = softmax(MLP(loss_history_k))`. Loss history
is a ring buffer (default length 5). Cites Kendall et al. (CVPR 2018) and
Chen et al. (NeurIPS 2018) as related prior art. Weights are detached
from autograd to avoid in-place buffer issues.

### `mvgt_net/metrics.py` — MAE, RMSE, WAPE, MSE, MAPE, sMAPE, R²
All metrics are masked (ignore NaN/Inf) for real-world data robustness.
`all_metrics(pred, target)` returns a dict of all 7 metrics in one call.

### `mvgt_net/data.py` — TimeMMDDataset + DOMAIN_REGISTRY + DataLoaders
- `DOMAIN_REGISTRY`: dict of 9 TimeMMD domains with their lookback, horizon,
  frequency, and variable count.
- `TimeMMDDataset`: PyTorch Dataset for one split (train/val/test) of one
  domain. Loads JSONL records, applies z-score normalization (using
  train-derived statistics), parses text facts.
- `fit_normalization`: computes per-variable mean and std from the TRAIN
  split only.
- `get_dataloaders`: builds train/val/test DataLoaders with the SAME
  train-derived statistics. Returns (train_loader, val_loader, test_loader,
  stats_dict).

### `mvgt_net/causal_probe.py` — Causal probing utility (Q5)
Implements the causal probing analysis used in Chapter 5 of the thesis
(Q5: interpretability). Not used by the main training pipeline.

### `mvgt_net/uncertainty.py` — Conformal prediction (Formula E)
Implements split conformal prediction for 90% (configurable) prediction
intervals. Calibrated on the validation set, evaluated on the test set.

---

## 3. Script responsibilities

### `scripts/train.py` — Smoke training on synthetic data
Quick smoke test of the training pipeline using synthetic random-walk
data. Used by `tests/test_smoke.py` and by the smoke-test step of the
pipeline. NOT for production training.

### `scripts/train_real.py` — Production training on REAL TimeMMD data
This is the main training script. Loads real TimeMMD JSONL data, builds
MVGTNet, trains with Ranger21 + cosine LR + AMP fp16 + early stopping,
saves best checkpoint + final test metrics. Supports `--all-domains` to
loop over all 9 domains sequentially. Supports `--resume` to continue
from the latest checkpoint.

### `scripts/run_all_experiments.py` — Chapter 18 full protocol (432 runs)
Orchestrates the full empirical validation: 9 domains × 4 horizons ×
12 baselines × 5 seeds = 2,160 runs. **Requires the 12 baselines to be
installed separately** (they live in the upstream ST-LLM+ repo). Without
the baselines, only the MVGT-Net runs succeed; the baseline runs print
"[FAIL]" but don't crash the pipeline. After all runs, aggregates results
into a main_table.csv and applies Holm-Bonferroni correction for
statistical significance testing.

### `scripts/hyperparameter_search.py` — Hyperparameter sweep
Grid search over `learning_rate`, `batch_size`, `lora_rank`, `topk`,
`frozen_layers`. Each combination runs a 10-epoch mini-train and reports
val_MAE.

### `scripts/latency_carbon.py` — Phase F: latency + carbon
Two analyses in one script:
1. Inference latency benchmark across batch sizes {1, 4, 8, 16, 32, 64,
   128, 256}. Reports per-batch and per-sample latency, throughput, peak
   GPU memory.
2. Carbon footprint estimate for the full 432-experiment Chapter 18
   training suite, using the Patterson et al. (2021) methodology.

### `scripts/robustness_analysis.py` — Phase F: robustness
Measures accuracy degradation under injected Gaussian noise
(σ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}). For each noise level, runs a 10-epoch
mini-train and reports val_MAE.

### `scripts/scaling_analysis.py` — Phase F: scaling
Measures accuracy + wall-clock as a function of training-set fraction
(10%, 25%, 50%, 75%, 100%). For each fraction, runs a 10-epoch mini-train
on the subset and reports val_MAE.

### `scripts/cross_domain_transfer.py` — Phase F: cross-domain transfer
9×9 matrix: model trained on domain A is fine-tuned (5 epochs) and
evaluated on domain B. Reports transfer accuracy heatmap.

---

## 4. Test responsibilities

### `tests/test_smoke.py` — 11 unit tests
Exercises every class in `mvgt_net` with random tensors and verifies
forward-pass shapes end-to-end. Expected: 11/11 PASS. Run with:
`python3 tests/test_smoke.py` or `pytest tests/test_smoke.py -v`.

### `tests/test_uncertainty.py` — Conformal prediction tests
Tests the conformal prediction module (Formula E). Verifies that
prediction intervals achieve the nominal coverage rate on synthetic data.

### `tests/test_holm_bonferroni.py` — Statistical correction tests
Tests the Holm-Bonferroni correction implementation in
`scripts/run_all_experiments.py`. Verifies that the corrected p-values
match the published algorithm of Holm (1979).

---

## 5. Config responsibilities

### `configs/default.yaml` — Production config (verified)
The verified training protocol for all 9 TimeMMD domains. Tuned for
RTX 3080 Ti (12 GB VRAM) on Ubuntu 24.04. Used by `scripts/train_real.py`.
Domain-specific overrides (lookback, horizon, num_nodes, input_dim) are
applied automatically by `train_real.py` based on the `--domain` flag and
the `DOMAIN_REGISTRY` in `mvgt_net/data.py`.

### `configs/environment.yaml` — Legacy Environment-domain config
Older config used by `scripts/train.py` for the synthetic smoke test.
Documents the original ST-LLM+ Environment domain assumptions. NOT used
by `train_real.py`.

### `configs/smoke.yaml` — Quick smoke-test config
Even smaller config for rapid iteration. 2 epochs, batch_size=4, CPU-only.
Used by some test paths but not the main pipeline.

---

## 6. Why this architecture works for TimeMMD

TimeMMD is a multi-domain multimodal time-series dataset with 9 domains
(Climate, Economy, Energy, Health, Agriculture), each with numeric +
text-fact data. MVGT-Net was originally designed for multi-node traffic
data, but on TimeMMD:

1. **Single-node adaptation:** Each TimeMMD domain is a single-node time
   series (one OT variable per timestamp). The spatial-graph branch is
   degenerate (1×1 adjacency = `[[1.0]]`), and the temporal, semantic,
   and adaptive graphs carry the structural signal. This is documented
   in `mvgt_net/data.py` and `code/DATA_CARD.md`.

2. **Text fusion:** The MultiViewEmbedding text branch accepts either
   pre-computed BERT embeddings OR raw strings. With raw strings and
   `transformers` installed, it uses the real BERT-base-uncased encoder
   (110M params, ~440 MB in fp16, fits in 12 GB VRAM with LoRA r=8).

3. **LoRA efficiency:** The 6 frozen + 2 unfrozen PFGA layers (per
   ST-LLM+ protocol) plus LoRA r=8 on the unfrozen layers keeps the
   trainable parameter count under 1% of total — comparable to ST-LLM+
   Table II.

4. **Mixed precision:** AMP fp16 + gradient clipping (norm 1.0) keeps
   training stable on the RTX 3080 Ti. Ranger21's lookahead mechanism
   further stabilizes the late-training phase.

5. **Early stopping:** Patience 15 epochs is sufficient for TimeMMD's
   small domains (most converge in 30–60 epochs). Without early stopping,
   the full 100-epoch budget would roughly double the training time.

---

## 7. Where to look for specific implementation details

| You want to know … | Look in … |
|--------------------|-----------|
| How is the multi-view adjacency built? | `mvgt_net/graph_builder.py` docstring |
| Why is the graph attention additive, not Hadamard? | `mvgt_net/attention.py` docstring (correction note) |
| How does the dynamic loss weighting work? | `mvgt_net/losses.py` docstring + Kendall et al. (CVPR 2018) citation |
| How are masked metrics computed? | `mvgt_net/metrics.py` docstring |
| How is the TimeMMD JSONL parsed? | `mvgt_net/data.py` docstring + `_parse_batch_text` function |
| What's the upstream validation-split quirk? | `mvgt_net/data.py` docstring (Honest limitations §3) |
| What's the NaN text marker in Economy_Unemp? | `mvgt_net/data.py` `_parse_batch_text` docstring |
| How is the conformal prediction interval calibrated? | `mvgt_net/uncertainty.py` docstring |
| What does the Ranger21 fallback look like? | `scripts/train_real.py` `build_optimizer` function |
| How is the early stopping implemented? | `scripts/train_real.py` `train_one_domain` function |
| What's the Holm-Bonferroni correction algorithm? | `scripts/run_all_experiments.py` `holm_bonferroni` function |
| What's the carbon footprint formula? | `scripts/latency_carbon.py` docstring + Patterson et al. (2021) citation |
