"""
Phase F: Robustness Analysis
=============================
Evaluates MVGT-Net's stability under realistic data-quality perturbations:

  1. Missing data:        randomly zero out X% of input timesteps
  2. Noisy text:          replace X% of input tokens with [UNK]
  3. Partial graph:       randomly remove X% of adjacency edges
  4. Gaussian noise:      add N(0, sigma) to numeric inputs
  5. Temporal shift:      shift input window by +/- k steps (distribution shift)

Each perturbation is run at 5 severity levels: {0, 10%, 25%, 50%, 75%}.
Reports MAE/MSE/RMSE/WAPE/MAPE/sMAPE/R2 at each level for both the
baseline (no perturbation) and the perturbed model.

Output:
    results/robustness/robustness_table.csv    - per-perturbation per-level metrics
    results/robustness/robustness_curves.png   - metric vs severity line plots

Usage:
    python scripts/robustness_analysis.py

References:
    Hendrycks & Dietterich (2019). "Benchmarking Neural Network Robustness
    to Common Corruptions and Perturbations", ICLR 2019.
    Zhao et al. (2022). "Data Augmentation for Spatio-Temporal Forecasting",
    KDD 2022.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from mvgt_net import MVGTNet, all_metrics

# VPS bundle: outputs go inside code/ (not sibling of code/)
RESULTS_DIR = Path(os.path.join(CODE_ROOT, "14_engineering_analyses", "robustness"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def make_synthetic_data(n_samples=128, num_nodes=14, lookback=12, horizon=3):
    """Generate synthetic TimeMMD-format data with structured signal."""
    t = torch.arange(0, lookback + horizon, dtype=torch.float32)
    base = 50.0 + 0.05 * t + 0.3 * torch.sin(2 * torch.pi * t / 7.0)
    node_offsets = torch.linspace(0, 5.0, num_nodes).unsqueeze(0)
    X, Y, CAT = [], [], []
    for s in range(n_samples):
        eps = torch.randn(lookback + horizon, num_nodes) * 0.1
        signal = base.unsqueeze(1) + node_offsets + eps
        X.append(signal[:lookback].unsqueeze(-1))
        Y.append(signal[lookback:].unsqueeze(-1))
        CAT.append(torch.randint(0, 5, (lookback, num_nodes)))
    x = torch.stack(X); y = torch.stack(Y)
    cat = torch.stack(CAT)
    text = {"fact": torch.randint(0, 30000, (n_samples, lookback, 32))}
    A = torch.ones(num_nodes, num_nodes) / num_nodes
    return x, y, text, cat, A


def train_baseline_model(epochs=10):
    """Train a small MVGT-Net for robustness evaluation."""
    x, y, text, cat, A = make_synthetic_data()
    cfg = {
        "num_nodes": 14, "input_dim": 1, "lookback": 12, "horizon": 3,
        "hidden_dim": 32, "num_heads": 2,
        "frozen_layers": 1, "unfrozen_layers": 1,
        "num_categories": 5, "lora_rank": 4, "lora_alpha": 8,
        "graph_types": ["spatial", "temporal", "semantic", "adaptive"],
        "vocab_size": 30000, "use_text": False, "use_categorical": True,
    }
    model = MVGTNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_train = 76
    train_ds = torch.utils.data.TensorDataset(x[:n_train], y[:n_train], text["fact"][:n_train], cat[:n_train])
    test_ds = torch.utils.data.TensorDataset(x[n_train:], y[n_train:], text["fact"][n_train:], cat[n_train:])
    def collate(batch):
        xs, ys, ts, cs = zip(*batch)
        return torch.stack(xs), torch.stack(ys), {"fact": torch.stack(ts)}, torch.stack(cs), A
    loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    print(f"Training baseline model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            xn, yt, xt, xc, a = batch
            opt.zero_grad()
            out = model(xn, xt, xc, adj_spatial=a)
            loss = torch.nn.functional.l1_loss(out["numeric"], yt)
            loss.backward(); opt.step()
    print(f"  Trained. Test MAE on clean data:")
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate)
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            xn, yt, xt, xc, a = batch
            out = model(xn, xt, xc, adj_spatial=a)
            preds.append(out["numeric"].cpu()); targets.append(yt.cpu())
    p = torch.cat(preds); t = torch.cat(targets)
    print(f"    MAE={torch.nn.functional.l1_loss(p, t).item():.4f}")
    return model, test_loader


# ====== Perturbation functions ======
def perturb_missing_data(x, severity):
    """Zero out `severity` fraction of input timesteps."""
    mask = torch.rand_like(x) > severity
    return x * mask.float()


def perturb_noisy_text(text, severity):
    """Replace `severity` fraction of text tokens with [UNK]=100."""
    new_text = {}
    for k, v in text.items():
        mask = torch.rand_like(v.float()) < severity
        v = v.clone()
        v[mask] = 100  # [UNK] token ID
        new_text[k] = v
    return new_text


def perturb_partial_graph(adj, severity):
    """Randomly remove `severity` fraction of edges."""
    mask = torch.rand_like(adj) > severity
    adj_pert = adj * mask.float()
    # Re-normalize rows
    row_sum = adj_pert.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return adj_pert / row_sum


def perturb_gaussian_noise(x, severity):
    """Add N(0, severity) noise to numeric inputs."""
    return x + torch.randn_like(x) * severity


def perturb_temporal_shift(x, severity):
    """Shift input window by +/- severity steps."""
    # severity here is an integer shift
    shift = int(severity)
    if shift == 0:
        return x
    x_new = x.clone()
    if shift > 0:
        x_new[:, shift:, :, :] = x[:, :-shift, :, :]
        x_new[:, :shift, :, :] = x[:, 0:1, :, :].expand(-1, shift, -1, -1)
    else:
        x_new[:, :shift, :, :] = x[:, -shift:, :, :]
        x_new[:, shift:, :, :] = x[:, -1:, :, :].expand(-1, -shift, -1, -1)
    return x_new


def evaluate_perturbation(model, test_loader, perturb_fn, severity, perturb_name):
    """Run inference with a given perturbation and return metrics."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            xn, yt, xt, xc, a = batch
            # Apply perturbation
            if perturb_name == "missing_data":
                xn_p = perturb_fn(xn, severity)
                a_p = a
                xt_p = xt
            elif perturb_name == "noisy_text":
                xn_p = xn
                a_p = a
                xt_p = perturb_fn(xt, severity)
            elif perturb_name == "partial_graph":
                xn_p = xn
                a_p = perturb_fn(a, severity)
                xt_p = xt
            elif perturb_name == "gaussian_noise":
                xn_p = perturb_fn(xn, severity)
                a_p = a
                xt_p = xt
            elif perturb_name == "temporal_shift":
                xn_p = perturb_fn(xn, severity)
                a_p = a
                xt_p = xt
            else:
                xn_p, a_p, xt_p = xn, a, xt
            out = model(xn_p, xt_p, xc, adj_spatial=a_p)
            preds.append(out["numeric"].cpu()); targets.append(yt.cpu())
    p = torch.cat(preds); t = torch.cat(targets)
    return all_metrics(p, t)


def main():
    parser = argparse.ArgumentParser(description="Phase F: Robustness analysis")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    model, test_loader = train_baseline_model(epochs=args.epochs)

    perturbations = {
        "missing_data":   [(0.0, "0%"), (0.1, "10%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%")],
        "noisy_text":     [(0.0, "0%"), (0.1, "10%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%")],
        "partial_graph":  [(0.0, "0%"), (0.1, "10%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%")],
        "gaussian_noise": [(0.0, "0.0"), (0.05, "0.05"), (0.1, "0.1"), (0.2, "0.2"), (0.5, "0.5")],
        "temporal_shift": [(0, "+0"), (1, "+1"), (2, "+2"), (3, "+3"), (5, "+5")],
    }
    perturb_fns = {
        "missing_data": perturb_missing_data,
        "noisy_text": perturb_noisy_text,
        "partial_graph": perturb_partial_graph,
        "gaussian_noise": perturb_gaussian_noise,
        "temporal_shift": perturb_temporal_shift,
    }

    import csv
    out_csv = RESULTS_DIR / "robustness_table.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["perturbation", "severity", "label", "MAE", "MSE", "RMSE", "WAPE", "MAPE", "sMAPE", "R2"])
        all_curves = {}
        for name, levels in perturbations.items():
            print(f"\n=== {name} ===")
            curves = {"MAE": [], "MSE": [], "RMSE": [], "labels": []}
            for sev, label in levels:
                m = evaluate_perturbation(model, test_loader, perturb_fns[name], sev, name)
                print(f"  {label:>5s}: MAE={m['MAE']:.4f} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f}")
                w.writerow([name, sev, label, f"{m['MAE']:.6f}", f"{m['MSE']:.6f}",
                            f"{m['RMSE']:.6f}", f"{m['WAPE']:.6f}", f"{m['MAPE']:.6f}",
                            f"{m['sMAPE']:.6f}", f"{m['R2']:.6f}"])
                curves["MAE"].append(float(m["MAE"]))
                curves["MSE"].append(float(m["MSE"]))
                curves["RMSE"].append(float(m["RMSE"]))
                curves["labels"].append(label)
            all_curves[name] = curves
    print(f"\nTable: {out_csv}")

    # Plot: 2x3 grid of metric-vs-severity curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    colors = {"missing_data": "#1f77b4", "noisy_text": "#ff7f0e",
              "partial_graph": "#2ca02c", "gaussian_noise": "#d62728",
              "temporal_shift": "#9467bd"}
    for i, metric in enumerate(["MAE", "MSE", "RMSE"]):
        ax = axes[i]
        for name, curves in all_curves.items():
            ax.plot(range(len(curves[metric])), curves[metric], "o-",
                    color=colors[name], label=name, linewidth=2, markersize=6)
        ax.set_xticks(range(len(curves["labels"])))
        ax.set_xticklabels(curves["labels"], rotation=30, ha="right")
        ax.set_xlabel("Severity level")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs perturbation severity")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    # Hide unused subplots
    for j in range(3, 6):
        axes[j].axis("off")
    fig.suptitle("Phase F: MVGT-Net Robustness Analysis (synthetic data)",
                 y=1.02, fontsize=13, fontweight="bold")
    out = RESULTS_DIR / "robustness_curves.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"Curves: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
