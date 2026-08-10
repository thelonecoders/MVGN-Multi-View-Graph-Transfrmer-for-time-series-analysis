"""
Phase F: Cross-Domain Transfer Study
=====================================
Evaluates how well a model trained on domain A performs on domain B
(without retraining). This quantifies the transferability of MVGT-Net's
learned representations across TimeMMD domains.

Three transfer settings:
  1. Zero-shot transfer:   train on A, evaluate on B (no fine-tuning)
  2. Few-shot transfer:    train on A, fine-tune on 10% of B, evaluate on B test
  3. Full fine-tune:       train on A, fine-tune on 100% of B, evaluate on B test

Domains (synthetic proxies for TimeMMD's 9 domains):
  - Environment:    daily frequency, low noise
  - Economy:        monthly frequency, high noise
  - Energy:         weekly frequency, medium noise

Output:
    results/transfer/transfer_table.csv     - per-(source, target) results
    results/transfer/transfer_heatmap.png   - source x target MAE matrix
    results/transfer/transfer_summary.json  - aggregated statistics

Usage:
    python scripts/cross_domain_transfer.py

References:
    Pan & Yang (2010). "A Survey on Transfer Learning", IEEE TKDD.
    Zhuang et al. (2020). "A Comprehensive Survey on Transfer Learning",
        Proceedings of the IEEE.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
RESULTS_DIR = Path(os.path.join(CODE_ROOT, "14_engineering_analyses", "transfer"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def generate_domain_data(domain_name, n_samples=128, num_nodes=14, lookback=12, horizon=3):
    """Generate synthetic data with domain-specific characteristics.

    TimeMMD's 9 domains have different frequencies, noise levels, and signal
    structures. We approximate 3 of them here as honest proxies.
    """
    t = torch.arange(0, lookback + horizon, dtype=torch.float32)
    if domain_name == "Environment":
        # Daily, low noise, weekly seasonality
        base = 50.0 + 0.05 * t + 0.3 * torch.sin(2 * torch.pi * t / 7.0)
        noise_sigma = 0.05
    elif domain_name == "Economy":
        # Monthly, high noise, annual seasonality
        base = 100.0 + 0.5 * t + 1.0 * torch.sin(2 * torch.pi * t / 12.0)
        noise_sigma = 0.5
    elif domain_name == "Energy":
        # Weekly, medium noise, daily + weekly seasonality
        base = 200.0 + 0.2 * t + 0.5 * torch.sin(2 * torch.pi * t / 7.0) + 0.3 * torch.sin(2 * torch.pi * t / 24.0)
        noise_sigma = 0.2
    else:
        base = 50.0 + 0.1 * t
        noise_sigma = 0.1
    node_offsets = torch.linspace(0, 5.0, num_nodes).unsqueeze(0)
    X, Y, CAT = [], [], []
    for s in range(n_samples):
        eps = torch.randn(lookback + horizon, num_nodes) * noise_sigma
        signal = base.unsqueeze(1) + node_offsets + eps
        X.append(signal[:lookback].unsqueeze(-1))
        Y.append(signal[lookback:].unsqueeze(-1))
        CAT.append(torch.randint(0, 5, (lookback, num_nodes)))
    x = torch.stack(X); y = torch.stack(Y)
    cat = torch.stack(CAT)
    text = {"fact": torch.randint(0, 30000, (n_samples, lookback, 32))}
    A = torch.ones(num_nodes, num_nodes) / num_nodes
    return x, y, text, cat, A


def train_model(domain_name, epochs=10, model=None):
    """Train MVGT-Net on the given domain."""
    x, y, text, cat, A = generate_domain_data(domain_name)
    if model is None:
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
    n_train = 96
    train_ds = torch.utils.data.TensorDataset(x[:n_train], y[:n_train], text["fact"][:n_train], cat[:n_train])
    test_ds = torch.utils.data.TensorDataset(x[n_train:], y[n_train:], text["fact"][n_train:], cat[n_train:])
    def collate(batch):
        xs, ys, ts, cs = zip(*batch)
        return torch.stack(xs), torch.stack(ys), {"fact": torch.stack(ts)}, torch.stack(cs), A
    loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate)
    print(f"  Training on {domain_name} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; nb = 0
        for batch in loader:
            xn, yt, xt, xc, a = batch
            opt.zero_grad()
            out = model(xn, xt, xc, adj_spatial=a)
            loss = torch.nn.functional.l1_loss(out["numeric"], yt)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
    return model, test_loader, A


def evaluate(model, test_loader, A):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            xn, yt, xt, xc, a = batch
            out = model(xn, xt, xc, adj_spatial=a)
            preds.append(out["numeric"].cpu()); targets.append(yt.cpu())
    p = torch.cat(preds); t = torch.cat(targets)
    return all_metrics(p, t)


def fine_tune(model, target_domain, fraction, epochs=3):
    """Fine-tune model on `fraction` of target domain data."""
    x, y, text, cat, A = generate_domain_data(target_domain)
    n_total = x.shape[0]
    n_ft = max(int(fraction * n_total), 4)
    n_train = 96
    x_ft = x[n_train:n_train+n_ft]
    y_ft = y[n_train:n_train+n_ft]
    text_ft = {"fact": text["fact"][n_train:n_train+n_ft]}
    cat_ft = cat[n_train:n_train+n_ft]
    if x_ft.shape[0] == 0:
        return model
    ft_ds = torch.utils.data.TensorDataset(x_ft, y_ft, text_ft["fact"], cat_ft)
    def collate(batch):
        xs, ys, ts, cs = zip(*batch)
        return torch.stack(xs), torch.stack(ys), {"fact": torch.stack(ts)}, torch.stack(cs), A
    loader = torch.utils.data.DataLoader(ft_ds, batch_size=4, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR for fine-tuning
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            xn, yt, xt, xc, a = batch
            opt.zero_grad()
            out = model(xn, xt, xc, adj_spatial=a)
            loss = torch.nn.functional.l1_loss(out["numeric"], yt)
            loss.backward(); opt.step()
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase F: Cross-domain transfer study")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    domains = ["Environment", "Economy", "Energy"]
    print("=== Phase F: Cross-Domain Transfer Study ===")
    print(f"Domains: {domains}")
    print()

    # Train a source model on each domain
    source_models = {}
    test_loaders = {}
    for src in domains:
        print(f"\n--- Training source model on {src} ---")
        model, tl, A = train_model(src, epochs=args.epochs)
        source_models[src] = (model, A)
        test_loaders[src] = tl
        m = evaluate(model, tl, A)
        print(f"  In-domain test MAE on {src}: {m['MAE']:.4f}")

    # Transfer matrix: source -> target
    print("\n=== Computing transfer matrix ===")
    out_csv = RESULTS_DIR / "transfer_table.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "setting", "MAE", "MSE", "RMSE", "WAPE", "MAPE", "sMAPE", "R2"])
        transfer_matrix = np.zeros((len(domains), len(domains)))
        for i, src in enumerate(domains):
            for j, tgt in enumerate(domains):
                src_model, src_A = source_models[src]
                tgt_loader = test_loaders[tgt]
                tgt_model, _ = source_models[tgt]
                tgt_A = tgt_model_A = None  # placeholder
                # Get target's A
                _, _, tgt_loader_with_A = train_model(tgt, epochs=0, model=None) if False else (None, None, None)
                # Re-get target A
                x_t, y_t, text_t, cat_t, tgt_A = generate_domain_data(tgt)
                # Zero-shot
                m_zero = evaluate(src_model, tgt_loader, tgt_A)
                transfer_matrix[i, j] = m_zero["MAE"]
                print(f"  {src} -> {tgt} (zero-shot): MAE={m_zero['MAE']:.4f}")
                w.writerow([src, tgt, "zero_shot", f"{m_zero['MAE']:.6f}",
                            f"{m_zero['MSE']:.6f}", f"{m_zero['RMSE']:.6f}",
                            f"{m_zero['WAPE']:.6f}", f"{m_zero['MAPE']:.6f}",
                            f"{m_zero['sMAPE']:.6f}", f"{m_zero['R2']:.6f}"])
                # Few-shot (10%)
                # Need a fresh copy of source model
                src_model_few, _, _ = train_model(src, epochs=args.epochs)
                src_model_few = fine_tune(src_model_few, tgt, fraction=0.1, epochs=3)
                m_few = evaluate(src_model_few, tgt_loader, tgt_A)
                print(f"  {src} -> {tgt} (few-shot 10%): MAE={m_few['MAE']:.4f}")
                w.writerow([src, tgt, "few_shot_10pct", f"{m_few['MAE']:.6f}",
                            f"{m_few['MSE']:.6f}", f"{m_few['RMSE']:.6f}",
                            f"{m_few['WAPE']:.6f}", f"{m_few['MAPE']:.6f}",
                            f"{m_few['sMAPE']:.6f}", f"{m_few['R2']:.6f}"])
                # Full fine-tune
                src_model_full, _, _ = train_model(src, epochs=args.epochs)
                src_model_full = fine_tune(src_model_full, tgt, fraction=1.0, epochs=5)
                m_full = evaluate(src_model_full, tgt_loader, tgt_A)
                print(f"  {src} -> {tgt} (full fine-tune): MAE={m_full['MAE']:.4f}")
                w.writerow([src, tgt, "full_finetune", f"{m_full['MAE']:.6f}",
                            f"{m_full['MSE']:.6f}", f"{m_full['RMSE']:.6f}",
                            f"{m_full['WAPE']:.6f}", f"{m_full['MAPE']:.6f}",
                            f"{m_full['sMAPE']:.6f}", f"{m_full['R2']:.6f}"])
    print(f"\nTable: {out_csv}")

    # Heatmap of zero-shot MAE
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    im = ax.imshow(transfer_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(len(domains))); ax.set_yticklabels(domains)
    ax.set_xlabel("Target domain"); ax.set_ylabel("Source domain")
    ax.set_title("Phase F: Cross-Domain Transfer (zero-shot MAE, synthetic)\n"
                 "Diagonal = in-domain; off-diagonal = transfer")
    for i in range(len(domains)):
        for j in range(len(domains)):
            ax.text(j, i, f"{transfer_matrix[i, j]:.2f}", ha="center", va="center",
                    color="black" if transfer_matrix[i, j] < transfer_matrix.max() / 2 else "white",
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="MAE (lower is better)")
    out = RESULTS_DIR / "transfer_heatmap.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"Heatmap: {out}")

    # Summary statistics
    diag = np.diag(transfer_matrix)
    off_diag = transfer_matrix[~np.eye(len(domains), dtype=bool)]
    summary = {
        "domains": domains,
        "in_domain_mean_mae": float(diag.mean()),
        "in_domain_std_mae": float(diag.std()),
        "zero_shot_mean_mae": float(off_diag.mean()),
        "zero_shot_std_mae": float(off_diag.std()),
        "transfer_gap_mean": float(off_diag.mean() - diag.mean()),
        "transfer_gap_std": float(off_diag.std() - diag.std() if len(off_diag) > 1 else 0.0),
        "n_pairs": int(len(off_diag)),
        "note": "All results are on SYNTHETIC data; real TimeMMD transfer performance is future work.",
    }
    out = RESULTS_DIR / "transfer_summary.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary: {out}")
    print(f"  In-domain mean MAE: {summary['in_domain_mean_mae']:.4f}")
    print(f"  Zero-shot mean MAE: {summary['zero_shot_mean_mae']:.4f}")
    print(f"  Transfer gap: {summary['transfer_gap_mean']:.4f} (positive = transfer hurts)")
    print("Done.")


if __name__ == "__main__":
    main()
