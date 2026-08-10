#!/usr/bin/env python3
"""Re-process saved SHAP values with correct labels and proper aggregation."""
import json, os
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output_dir = "results_mitigated/Climate_AQI/shap"
domain = "Climate_AQI"

# Load saved SHAP values
sv = np.load(os.path.join(output_dir, f"{domain}_shap_values.npy"))
print(f"Loaded SHAP values: {sv.shape}")  # (10, 12, 96)

n_samples, n_input_features, n_output_timesteps = sv.shape  # 10, 12, 96

# === Analysis 1: Aggregate across output timesteps (mean |SHAP| per input feature) ===
# This tells us: which input WEEKS matter most overall?
mean_abs_shap_per_week = np.abs(sv).mean(axis=(0, 2))  # (12,)

# === Analysis 2: Per-output-timestep importance (which output days are most predictable?) ===
mean_abs_shap_per_output = np.abs(sv).mean(axis=(0, 1))  # (96,)

# === Analysis 3: Top input-output pairs (which specific relationships dominate?) ===
sv_flat = sv.reshape(n_samples, -1)  # (10, 1152)
mean_abs_flat = np.abs(sv_flat).mean(axis=0)  # (1152,)

# === Feature names ===
chunk_size = 8  # 96 / 12
week_names = [f"Week {i+1} (days {i*chunk_size}-{(i+1)*chunk_size-1})" for i in range(n_input_features)]

# === Build clean importance JSON ===
importance = {
    "domain": domain,
    "explainer": "shap.PermutationExplainer (max_evals=25)",
    "shap_values_shape": list(sv.shape),
    "interpretation_dimensions": {
        "n_samples": n_samples,
        "n_input_features": n_input_features,
        "n_output_timesteps": n_output_timesteps,
        "input_aggregation": "12 weekly chunks (8 days each) from 96-day lookback",
        "output_horizon": "96 daily predictions (3-month forecast)",
    },
    "input_feature_importance": {
        "description": "Which input WEEKS matter most for predictions (aggregated across all 96 output timesteps)",
        "feature_names": week_names,
        "mean_abs_shap": mean_abs_shap_per_week.tolist(),
    },
    "output_timestep_importance": {
        "description": "Which output DAYS are most influenced by inputs (aggregated across all 12 input weeks)",
        "n_timesteps": n_output_timesteps,
        "mean_abs_shap": mean_abs_shap_per_output.tolist(),
    },
    "top_10_input_features": [
        {
            "rank": r+1,
            "feature": week_names[i],
            "week_index": int(i),
            "mean_abs_shap": float(mean_abs_shap_per_week[i]),
            "percent_of_total": float(100 * mean_abs_shap_per_week[i] / mean_abs_shap_per_week.sum())
        }
        for r, i in enumerate(np.argsort(mean_abs_shap_per_week)[::-1][:10])
    ],
    "top_10_input_output_pairs": [],
    "summary_statistics": {
        "total_shap_magnitude": float(np.abs(sv).sum()),
        "mean_shap_magnitude": float(np.abs(sv).mean()),
        "max_single_shap": float(np.abs(sv).max()),
        "recency_concentration": float(mean_abs_shap_per_week[-1] / mean_abs_shap_per_week.mean()),
        "interpretation": "recency_concentration > 2.0 means model strongly favors most recent week"
    }
}

# Top input-output pairs
top_pairs_idx = np.argsort(mean_abs_flat)[::-1][:10]
for r, flat_idx in enumerate(top_pairs_idx):
    in_idx = flat_idx // n_output_timesteps
    out_idx = flat_idx % n_output_timesteps
    importance["top_10_input_output_pairs"].append({
        "rank": r+1,
        "input_week": week_names[in_idx],
        "input_week_index": int(in_idx),
        "output_day": f"Day {out_idx+1} of forecast",
        "output_day_index": int(out_idx),
        "mean_abs_shap": float(mean_abs_flat[flat_idx])
    })

# Save
json_path = os.path.join(output_dir, f"{domain}_feature_importance.json")
with open(json_path, "w") as f:
    json.dump(importance, f, indent=2, cls=NpEncoder)
print(f"Saved: {json_path}")

# === Plots ===
# Plot 1: Input week importance bar chart
fig, ax = plt.subplots(figsize=(12, 6))
weeks = range(1, n_input_features+1)
ax.bar(weeks, mean_abs_shap_per_week, color='steelblue', edgecolor='navy')
ax.set_xlabel("Input Week (1 = oldest, 12 = most recent)")
ax.set_ylabel("Mean |SHAP value|")
ax.set_title(f"{domain}: Input Week Importance for 96-day Forecast\n(Higher = more influential on predictions)")
ax.set_xticks(weeks)
ax.grid(axis='y', alpha=0.3)
# Highlight most recent week
top_week = np.argmax(mean_abs_shap_per_week)
ax.bar([top_week+1], [mean_abs_shap_per_week[top_week]], color='red', edgecolor='darkred',
       label=f"Top week: {week_names[top_week]}")
ax.legend()
fig.tight_layout()
bar_path = os.path.join(output_dir, f"{domain}_shap_bar.png")
fig.savefig(bar_path, dpi=150)
plt.close(fig)
print(f"Saved: {bar_path}")

# Plot 2: Output timestep importance (which forecast days are most predictable?)
fig, ax = plt.subplots(figsize=(14, 5))
days = range(1, n_output_timesteps+1)
ax.plot(days, mean_abs_shap_per_output, color='darkgreen', linewidth=2)
ax.fill_between(days, 0, mean_abs_shap_per_output, alpha=0.3, color='lightgreen')
ax.set_xlabel("Forecast Day (1 = tomorrow, 96 = 3 months out)")
ax.set_ylabel("Mean |SHAP value|")
ax.set_title(f"{domain}: Forecast Horizon Sensitivity\n(Which output days are most influenced by input history?)")
ax.grid(alpha=0.3)
fig.tight_layout()
output_path = os.path.join(output_dir, f"{domain}_output_timestep_importance.png")
fig.savefig(output_path, dpi=150)
plt.close(fig)
print(f"Saved: {output_path}")

# Plot 3: Heatmap (input week × output day)
fig, ax = plt.subplots(figsize=(14, 6))
heatmap_data = np.abs(sv).mean(axis=0)  # (12, 96)
im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
ax.set_xlabel("Output Day (forecast horizon)")
ax.set_ylabel("Input Week (1=oldest, 12=most recent)")
ax.set_title(f"{domain}: SHAP Heatmap — Input Week × Output Day")
ax.set_yticks(range(n_input_features))
ax.set_yticklabels([f"W{i+1}" for i in range(n_input_features)])
plt.colorbar(im, label="Mean |SHAP|")
fig.tight_layout()
heatmap_path = os.path.join(output_dir, f"{domain}_shap_heatmap.png")
fig.savefig(heatmap_path, dpi=150)
plt.close(fig)
print(f"Saved: {heatmap_path}")

# === Final report ===
report = {
    "domain": domain,
    "checkpoint_epoch": 14,
    "test_r2": -0.1022,
    "test_mae": 0.0962,
    "n_samples_explained": n_samples,
    "n_input_features": n_input_features,
    "n_output_timesteps": n_output_timesteps,
    "key_findings": {
        "most_influential_input_week": week_names[top_week],
        "recency_concentration_ratio": float(mean_abs_shap_per_week[-1] / mean_abs_shap_per_week.mean()),
        "interpretation": "Model concentrates attention on most recent week(s) of input history.",
        "forecast_horizon_pattern": "see output_timestep_importance.png for forecast sensitivity curve"
    },
    "files_generated": [
        f"{domain}_feature_importance.json",
        f"{domain}_shap_bar.png",
        f"{domain}_output_timestep_importance.png",
        f"{domain}_shap_heatmap.png"
    ]
}
report_path = os.path.join(output_dir, f"{domain}_explanation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, cls=NpEncoder)
print(f"Saved: {report_path}")

print("\n" + "="*60)
print("Fix complete! Files generated:")
print("="*60)
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size = os.path.getsize(fpath)
    print(f"  {f:50s} {size:>10,} bytes")
