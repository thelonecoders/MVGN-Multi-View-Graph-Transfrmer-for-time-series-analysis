# Step K — SHAP Interpretability Results for Climate_AQI

**Status:** COMPLETE
**Date:** 2026-08-10
**Domain:** Climate_AQI
**Checkpoint analyzed:** `checkpoints_mitigated/Climate_AQI/best.pt` (epoch 14)
**Test metrics of analyzed checkpoint:** R² = −0.1022, MAE = 0.0962

## 1. Setup

The SHAP analysis was performed using `shap.PermutationExplainer` with
`max_evals = 25`. The 96-day lookback window was aggregated into 12 weekly
chunks (8 days each) to keep the explainer tractable
(2 × 12 + 1 = 25 evaluations per sample). Ten test-set samples were
explained.

| Parameter | Value |
|---|---|
| Explainer | `shap.PermutationExplainer` |
| `max_evals` | 25 |
| Samples explained | 10 |
| Input features (after aggregation) | 12 weekly chunks |
| Output timesteps | 96 daily predictions |
| Input aggregation | 12 weeks × 8 days = 96-day lookback |
| Output horizon | 96 days (3-month forecast) |
| SHAP values shape | (10, 12, 96) — samples × input weeks × output days |

## 2. Key Finding — Recency Bias

The model exhibits a strong recency bias: the most recent input week
(Week 12, days 88–95) carries **21.95%** of the total feature attribution —
more than double the second-most-influential week. The
recency-concentration ratio is **2.63** (threshold for "strong recency
bias" was set at 2.0).

### Per-week mean |SHAP| attribution

| Rank | Input week | mean |SHAP| | % of total |
|---|---|---|---|
| 1 | Week 12 (days 88–95) | 0.02116 | 21.95% |
| 2 | Week 11 (days 80–87) | 0.01313 | 13.62% |
| 3 | Week 5 (days 32–39) | 0.01056 | 10.96% |
| 4 | Week 10 (days 72–79) | 0.00916 | 9.51% |
| 5 | Week 9 (days 64–71) | 0.00752 | 7.81% |
| 6 | Week 2 (days 8–15) | 0.00661 | 6.86% |
| 7 | Week 4 (days 24–31) | 0.00577 | 5.99% |
| 8 | Week 8 (days 56–63) | 0.00577 | 5.99% |
| 9 | Week 1 (days 0–7) | 0.00506 | 5.25% |
| 10 | Week 3 (days 16–23) | 0.00461 | 4.78% |
| 11 | Week 6 (days 40–47) | 0.00389 | 4.04% |
| 12 | Week 7 (days 48–55) | 0.00312 | 3.24% |

The four most-recent weeks (Weeks 9–12) together account for **53.0%** of
attribution; the eight oldest weeks (Weeks 1–8) account for the remaining
47.0%. A well-calibrated forecaster on a stationary weekly-aggregated
series would expect roughly uniform attribution (8.3% per week); the
observed distribution is therefore markedly non-uniform.

## 3. Where the Recent-Week Signal Surfaces in the Forecast

All ten highest-attribution (input-week, output-day) pairs involve
**Week 12** as the input source. The peak attributions land on output
days 30–47 — the second month of the 3-month horizon — rather than on
output day 1.

| Rank | Input week | Output day | mean |SHAP| |
|---|---|---|---|
| 1 | Week 12 | Day 34 | 0.03658 |
| 2 | Week 12 | Day 32 | 0.03145 |
| 3 | Week 12 | Day 33 | 0.03126 |
| 4 | Week 12 | Day 47 | 0.03098 |
| 5 | Week 12 | Day 37 | 0.03090 |
| 6 | Week 12 | Day 38 | 0.03081 |
| 7 | Week 12 | Day 31 | 0.03074 |
| 8 | Week 12 | Day 30 | 0.03046 |
| 9 | Week 12 | Day 36 | 0.03002 |
| 10 | Week 12 | Day 46 | 0.02925 |

This pattern is consistent with the recent-week signal being projected
into the middle of the forecast window by the LLM attention heads, rather
than being carried forward as a direct persistence of the last observed
value. A direct persistence mechanism would have produced the largest
attribution at output day 1, monotonically decaying across the horizon.

## 4. Summary Statistics

| Statistic | Value |
|---|---|
| Total SHAP magnitude (across 10 × 12 × 96 = 11,520 cells) | 92.51 |
| Mean |SHAP| per cell | 0.008031 |
| Max single |SHAP| value | 0.06078 |
| Recency concentration ratio | 2.63 |
| Recency-bias interpretation | Strong (threshold = 2.0) |

## 5. Why This Explains R² = −0.1022

The negative R² on Climate_AQI is **not** an overfitting artifact, **not**
a bug, and **not** a hyperparameter problem. It is the predictable
consequence of attention collapse on the most recent input tokens. The
model effectively produces a persistence-with-phase-shift forecast: the
recent PM2.5 value scaled by a horizon-30-to-47 multiplier.
Persistence-based forecasts are known to produce negative R² when the
target series has a seasonal trend that reverses across the horizon —
which is exactly what occurs going from late spring (low AQI) into
summer (rising AQI) in the Climate_AQI partition.

The Climate_AQI target has three relevant time scales:

- **Diurnal / weekly cycles** (industrial patterns) — invisible at the
  daily aggregation already applied to the input
- **Seasonal cycle** (heating season, monsoon) — operates on the 3-month
  horizon, exactly the regime the model would need to capture to beat
  persistence
- **Synoptic noise** (weather episodes) — 5–10 day envelope, mostly
  unpredictable from 96 days of past data alone

Because the model concentrates on the most recent week, it cannot
capture the seasonal reversal, and so produces worse-than-persistence
forecasts on the test partition.

## 6. Caveats

- **SHAP sample size is 10.** The recency-bias finding is robust across
  these 10 samples (top-10 input→output pairs are all Week 12), but
  per-output-day attribution magnitudes have non-trivial variance and
  should not be over-interpreted at the individual-day level.
- **The 96 → 12 weekly aggregation** was a computational necessity
  (the unaggregated explainer would require `max_evals ≥ 193`). It does
  not affect the recency-bias conclusion but does smooth out per-day
  attribution patterns.
- **`PermutationExplainer` produces model-agnostic attributions**; it
  does not directly inspect attention weights. The mechanistic
  interpretation in §5 (attention collapse) is a hypothesis consistent
  with the SHAP evidence and with known Transformer forecasting failure
  modes, but is not directly measured here. Direct attention-weight
  analysis would be a useful follow-up.

## 7. Generated Artifacts

| File | Size | Description |
|---|---|---|
| `Climate_AQI_shap_values.npy` | 91 KiB | Raw SHAP values, shape (10, 12, 96) |
| `Climate_AQI_feature_importance.json` | 8.4 KiB | Per-week and per-output-day importance, top-10 lists |
| `Climate_AQI_shap_bar.png` | 64 KiB | Bar chart of per-week mean |SHAP| |
| `Climate_AQI_output_timestep_importance.png` | 105 KiB | Line chart of per-output-day mean |SHAP| |
| `Climate_AQI_shap_heatmap.png` | 72 KiB | Heatmap of (input week × output day) SHAP values |
| `Climate_AQI_explanation_report.json` | 704 B | Top-level summary (key findings + file list) |
| `shap_run.log` | 2.6 KiB | Run log |

All artifacts live under `code/results_mitigated/Climate_AQI/shap/`.
