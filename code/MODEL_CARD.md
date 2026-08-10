# MVGT-Net Model Card

This model card follows the template proposed by Mitchell et al.
(Model Cards for Model Reporting, FAT* 2019, pp. 220-229) and is
intended to provide transparency about the MVGT-Net model's design,
intended use, limitations, and ethical considerations.

## 1. Model Details

- **Model name:** MVGT-Net (Multi-View Graph Transformer Network)
- **Model date:** 2026-08-09 (design document; empirical validation pending)
- **Model version:** 0.1.0
- **Model type:** Spatio-temporal multi-modal transformer for time-series
  forecasting with graph-structured dependencies
- **Owners:** MVGT-Net Project (this thesis)
- **Contact:** see thesis cover page
- **License:** MIT (code); the trained model weights, when available,
  will be released under CC-BY-4.0
- **Citation:** see thesis references, ref [1] for ST-LLM+ (the base
  architecture MVGT-Net extends)

## 2. Intended Use

- **Primary intended use:** Multi-domain time-series forecasting on the
  TimeMMD dataset (9 domains: Climate, Economy, Energy, Environment,
  Health_US, Health_AFR, Agriculture, Traffic, Bike).
- **Primary intended users:** Researchers in spatio-temporal machine
  learning, time-series foundation models, and multi-modal forecasting.
- **Out-of-scope uses:**
  - Real-time decision-making in safety-critical applications (Health,
    Climate) without a human-in-the-loop review of the conformal
    prediction intervals.
  - Domains outside the 9 TimeMMD domains without further evaluation.
  - Forecasting of financial markets for trading purposes.
  - Any deployment that does not account for distribution shift.

## 3. Factors

- **Model architecture factors:**
  - Number of frozen LLM layers F (default 6 for GPT-2, 8 for LLaMA-2-7B)
  - Number of unfrozen PFGA layers U
  - Hidden dimension d
  - Number of nodes V (varies per domain, 1 to 250+)
  - Top-k sparsification parameter k (default 8)
  - LoRA rank r (default 8)
- **Data factors:**
  - Domain (9 values; see Section 2)
  - Forecast horizon S (default 12; tested at 3, 6, 12, 24)
  - Lookback length P (default 12)
  - Sampling frequency (varies per domain: daily, weekly, monthly)
- **Evaluation factors:**
  - Random seed (5 seeds: 42, 43, 44, 45, 46)
  - Train/validation/test split (6:2:2 per the ST-LLM+ protocol)

## 4. Metrics

- **Point-forecast metrics:** MAE, MSE, RMSE, WAPE, MAPE, sMAPE, R-squared
- **Uncertainty metric:** empirical coverage of conformal prediction
  intervals at alpha = 0.1 (target: >= 90 percent) and alpha = 0.05
  (target: >= 95 percent)
- **Efficiency metrics:** training time per epoch, inference latency per
  batch, peak GPU memory, trainable parameter count
- **Current status:** All metrics are DESIGN TARGETS. No empirical
  results have been measured as of 2026-08-09 because the real TimeMMD
  dataset has not been downloaded and no full training run has been
  executed.

## 5. Evaluation Data

- **Training data:** TimeMMD dataset (9 domains, ~2 GB total). Hosted
  by AdityaLab at https://github.com/AdityaLab/TimeMMD under the
  Open Data Commons Attribution License v1.0.
- **Validation data:** 20 percent of each domain's training split,
  per the ST-LLM+ protocol.
- **Test data:** 20 percent of each domain, held out from training
  and validation.
- **Synthetic sample data:** 3 synthetic domains (Environment, Economy,
  Energy) generated via 12_dataset/synthetic_sample/generate_synthetic_
  sample.py for code-pipeline testing only. NOT for empirical evaluation.

## 6. Training Data

Same as Section 5 (training data). The model is trained per-domain
(no cross-domain pre-training); each domain has its own training run
with 5 random seeds.

## 7. Quantitative Analyses

### 7.1 Real TimeMMD Results (Phase A)

**PENDING.** No empirical results exist as of 2026-08-09. The full
evaluation protocol is specified in Chapter 18 of the thesis. When
the real TimeMMD dataset is downloaded and the 432-experiment suite
is executed, the results will be reported in three layers:
- Layer 1: main result table (mean and standard deviation across 5 seeds)
- Layer 2: statistical test table (paired t-test with Holm-Bonferroni
  correction (Holm 1979), Cohen's d effect sizes, bootstrap 95% CIs)
- Layer 3: qualitative analysis (attention heatmaps, SHAP feature
  attribution, failure case studies, conformal calibration plots)

### 7.2 Synthetic-Run Diagnostic Results (Phase F, executed 2026-08-09)

A 15-epoch training run on synthetic TimeMMD-format data (128 samples,
14 nodes, 12-step lookback, 3-step horizon) was executed to verify
the end-to-end pipeline. These results are DIAGNOSTIC ONLY and are
explicitly labeled as SYNTHETIC; they must NOT be interpreted as
performance claims for the real model.

| Metric | Value (synthetic, 15 epochs) |
|--------|------------------------------|
| Training loss (epoch 1) | 50.48 |
| Training loss (epoch 15) | 30.11 |
| Validation MAE (epoch 1) | 48.93 |
| Validation MAE (epoch 15) | 28.89 |
| Test MAE | 28.89 |
| Test MSE | 836.78 |
| Test RMSE | 28.93 |
| Test WAPE | 54.53% |
| Test MAPE | 54.50% |
| Test sMAPE | 74.94% |
| Test R² | -404.24 (negative; expected for un-pre-trained tiny model) |
| Total parameters | 292,716 |
| Trainable parameters | 163,116 (55.73%) |
| Conformal q_hat (alpha=0.1, n=26 calib) | computed at run time |
| Conformal empirical coverage on test | computed at run time |

The negative R² reflects that the tiny un-pre-trained model cannot
capture the strong upward trend in the synthetic data within 15
epochs. This is a DIAGNOSTIC of the synthetic test setup, NOT a
defect in the model architecture. The corresponding real-data
experiment (with pre-trained LLM checkpoint, full hidden dimension,
and 100+ epochs) is PENDING.

### 7.3 Phase F Engineering Analyses (executed 2026-08-09)

The following engineering analyses were executed on the synthetic
data and are honestly labeled. Each is documented in detail in
Chapter 19 of the thesis.

- **Robustness:** 5 perturbation types × 5 severity levels evaluated.
  Results flat on synthetic data (model converged to near-constant
  predictions). Real-data robustness study PENDING.
  Artifacts: results/robustness/robustness_table.csv,
  results/robustness/robustness_curves.png.

- **Scaling (V sweep):** V = 16, 32, 64, 128, 256. Forward-pass time
  increased from 3.92 ms (V=16) to 57.40 ms (V=256), consistent with
  the theoretical O(F·V²·(T+D+log V)) derived in Section 6-17.
  Artifacts: results/scaling/scaling_table.csv,
  results/scaling/scaling_curves.png.

- **Scaling (F sweep):** F = 1, 2, 4, 8, 16. Forward-pass time
  approximately flat (5.7 to 10.8 ms), entering the O(U·(k+1)·V·D)
  PFGA-term regime which is linear in F.

- **Scaling (T sweep):** T = 12, 24, 48, 96, 192. Forward-pass time
  approximately flat (5.8 to 10.6 ms), consistent with linear-in-T
  PFGA-term dominance.

- **Inference latency:** Batch sizes 1 to 256. Per-sample latency
  decreased from 1.72 ms (B=1) to 0.19 ms (B=256). Peak throughput
  5,346 samples/sec. Artifacts: results/latency/latency_table.csv,
  results/latency/latency_curves.png.

- **Carbon footprint:** Three scenarios computed for 72 A100-hours.
  Best case (renewable, PUE 1.1): 1.58 kgCO2e. World average
  (475 gCO2e/kWh, PUE 1.55): 21.20 kgCO2e. Worst case (coal grid,
  PUE 1.55): 36.60 kgCO2e. Artifacts: results/carbon/carbon_report.json,
  results/carbon/carbon_breakdown.png.

- **Cross-domain transfer:** 3×3 source-target matrix on synthetic
  domains (Environment, Economy, Energy). In-domain mean MAE 111.07,
  zero-shot mean MAE 111.07, transfer gap 0.00 (artifact of synthetic
  data). Real-data transfer across 9 TimeMMD domains PENDING.
  Artifacts: results/transfer/transfer_table.csv,
  results/transfer/transfer_heatmap.png.

## 8. Ethical Considerations

- **Bias and fairness:** TimeMMD contains data from multiple geographic
  regions (US, Africa, China, etc.). Per-subgroup forecasting
  performance has NOT been evaluated. If MVGT-Net is deployed in a
  decision-making context, per-subgroup coverage of the conformal
  intervals must be verified (Conditional Conformal Prediction,
  Gibbs and Candes 2021).
- **Privacy:** TimeMMD is aggregate public data; no individual-level
  data is used. The Health_US and Health_AFR domains contain
  population-level health statistics, not patient records.
- **Environmental impact:** Training the full 432-experiment suite
  is estimated to consume approximately 72 A100 GPU-hours, which
  corresponds to roughly 50 kWh of electricity (at 700W per A100).
  This is comparable to one transatlantic flight per passenger.
- **Misuse potential:** MVGT-Net is a forecasting tool, not a decision
  oracle. Predictions should not be used as the sole basis for
  policy decisions in Health, Climate, or Economy. The conformal
  prediction intervals (Formula E) provide uncertainty quantification
  but do not eliminate the risk of distribution shift.

## 9. Caveats and Recommendations

- The model has NOT been empirically validated. All performance claims
  in the thesis are design targets, not measured results.
- The conformal prediction guarantee is marginal (averaged over the
  calibration distribution), not conditional (per subgroup).
- The causal sensitivity probe (Section 6-19) is an observational
  approximation, not a causal claim.
- For non-stationary domains (Economy, Climate), the EnbPI bootstrap
  interval is recommended as an alternative to split conformal.
- For deployment, monitor the empirical coverage of the conformal
  intervals on a held-out test set; if coverage drops below 1 - alpha,
  re-calibrate or switch to EnbPI.

## 10. References

- Liu et al. ST-LLM+ (IEEE TKDE 2025, DOI 10.1109/TKDE.2025.3570705)
- Vovk, Gammerman, Shafer. Algorithmic Learning in a Random World (2005)
- Lei et al. Distribution-Free Prediction Bands (JASA 2018)
- Angelopoulos, Bates. Conformal Prediction Tutorial (FnTML 2023)
- Xu, Xie. Conformal Prediction for Dynamic Time-Series (ICML 2021)
- Mitchell et al. Model Cards for Model Reporting (FAT* 2019)
- Gebru et al. Datasheets for Datasets (FAccT 2021)
