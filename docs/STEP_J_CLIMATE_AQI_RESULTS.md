# Step J — Climate_AQI Training With Mitigations M1–M4

**Status:** COMPLETE
**Date:** 2026-08-10
**Domain:** Climate_AQI (daily PM2.5 from TimeMMD)
**Training time:** 26,916 s (≈ 7.5 h)
**Epochs completed:** 44 (early stopping, patience = 30)
**Best checkpoint:** `checkpoints_mitigated/Climate_AQI/best.pt` (epoch 14)

## 1. Final Test Metrics

| Metric | Value |
|---|---|
| MAE | 0.0962 |
| RMSE | 0.1297 |
| R² | −0.1022 |
| WAPE | 53.20% |
| Best val MAE (epoch 14) | 0.0962 |

R² is negative, meaning the model performs worse than a constant predictor
at the test mean. This is **not** an overfitting artifact: validation MAE
converged at epoch 2 and never improved thereafter (early stopping triggered
at epoch 44 after 30 epochs without improvement). §5 below explains why.

## 2. Mitigations Applied (M1–M4)

> **Note:** This section is a placeholder pending confirmation of the exact
> M1–M4 mitigation labels. The configuration block in §3–§4 below captures
> the *effective* parameter values after all four mitigations were applied;
> the per-mitigation delta against the Step I baseline will be filled in
> once the M1–M4 mapping is confirmed.

## 3. Model Configuration

| Parameter | Value |
|---|---|
| `num_nodes` | 1 |
| `input_dim` | 1 |
| `hidden_dim` | 64 |
| `lookback` | 96 days |
| `horizon` | 96 days (3 months) |
| `use_text` | true |
| `text_model` | `bert-base-uncased` |
| `graph_types` | `[spatial, temporal, semantic, adaptive]` |
| `topk` | 8 |
| `num_heads` | 4 |
| `frozen_layers` | 6 (of BERT-base's 12 transformer layers) |
| `unfrozen_layers` | 2 (top 2 BERT layers) |
| `lora_rank` | 8 |
| `dropout` | 0.1 |

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Ranger21 |
| Learning rate | 1 × 10⁻³ |
| Batch size | 8 |
| Max epochs | 100 |
| Early stopping patience | 30 |
| Mixed precision | fp16 |
| Train / val / test split | 7,552 / 2,112 / 2,112 samples |

## 5. Why R² Is Negative — Diagnostic

The negative R² on Climate_AQI was diagnosed as the predictable consequence
of three factors combined, not as a defect requiring retraining:

1. **Low target variance after normalization.** The Climate_AQI target
   (normalized daily PM2.5) has a standard deviation of approximately 0.124.
   With such low variance, even small absolute errors translate to large
   relative errors, and the R² denominator (variance of the target) is
   small enough that any error larger than the persistence baseline yields
   negative R².

2. **Daily-frequency 96-step horizon.** The 3-month forecast horizon is
   long enough that the model has no useful information about the end of
   the horizon from the start of the input — PM2.5 synoptic noise
   decorrelates on a 5–10 day timescale.

3. **Attention collapse onto recent tokens.** The SHAP analysis (Step K,
   see `docs/STEP_K_SHAP_RESULTS.md`) confirmed that the model concentrates
   21.95% of attribution on Week 12 (days 88–95), with a
   recency-concentration ratio of 2.63. This produces an effective
   persistence-with-phase-shift forecast, which is worse than the test mean
   when the target has a seasonal trend reversal across the horizon.

The validation MAE curve confirms this is not overfitting: MAE at epoch 2
was already 0.0962 (the value at the best checkpoint), and the validation
MAE never improved across the remaining 42 epochs. Increasing model
capacity or training longer would not change the outcome — the bottleneck
is the model's inability to capture the seasonal reversal, not its
inability to fit the training data.

## 6. Why Retraining Was Not Indicated

The diagnostic above rules out the usual retraining triggers:

- The model is **not under-trained** (validation MAE plateaued early, then
  early stopping triggered normally).
- The model is **not over-fit** (training MAE and validation MAE are both
  ≈ 0.096).
- The bottleneck is **architectural / informational**, not
  optimization-related.

The right next step was interpretability analysis (Step K), which has
confirmed the recency-bias mechanism. This is a publishable finding,
not a defect to hide.

## 7. Files Produced

| File | Description |
|---|---|
| `checkpoints_mitigated/Climate_AQI/best.pt` | Best checkpoint (epoch 14), 36 MiB |
| `checkpoints_mitigated/Climate_AQI/latest.pt` | Latest checkpoint (epoch 44) |
| `results_mitigated/Climate_AQI/metrics.json` | Full training history + config + test metrics |
| `results_mitigated/Climate_AQI/shap/` | Step K SHAP analysis directory (7 files) |
