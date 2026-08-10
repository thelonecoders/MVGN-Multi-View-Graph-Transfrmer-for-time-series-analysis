"""
Phase F: Scaling Analysis
==========================
Measures how MVGT-Net's training time, inference latency, and peak GPU
memory scale with the three key problem-size dimensions:

    V = number of nodes            {16, 32, 64, 128, 256, 512}
    F = number of input features   {1, 2, 4, 8, 16}
    T = lookback window length     {12, 24, 48, 96, 192}

For each (V, F, T) configuration, runs:
    - 1 forward pass (measures inference latency, peak memory)
    - 1 backward pass (measures training time)
    - Reports theoretical Big-O from Section 6-17 for comparison

Output:
    results/scaling/scaling_table.csv   - per-config timing + memory
    results/scaling/scaling_curves.png  - log-log plots of cost vs V/F/T

Usage:
    python scripts/scaling_analysis.py

References:
    Vaswani et al. (2017). "Attention Is All You Need", NeurIPS 2017.
        (Origin of O(N^2 * d) attention complexity.)
    This thesis, Section 6-17 (first-principles complexity derivation).
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

from mvgt_net import MVGTNet

# VPS bundle: outputs go inside code/ (not sibling of code/)
RESULTS_DIR = Path(os.path.join(CODE_ROOT, "14_engineering_analyses", "scaling"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def theoretical_complexity(V, F, T, D=32, k=8, U=2):
    """Compute theoretical Big-O operation count from Section 6-17.

    T_MVGT = O((F*V^2 + U*(k+1)*V) * d)

    Per the 4-step derivation:
      1. Per-view adjacency:  O(F * V^2 * (T+D))  for F views, V^2 pairs
      2. Row normalization:   O(V^2)               per view
      3. Weighted combination:O(V^2)               once
      4. Top-k sparsification:O(V^2 * log(V))     per view

    Total graph construction: O(F * V^2 * (T + D + log V))
    PFGA forward (U unfrozen layers): O(U * (k+1) * V * D)
    """
    graph_cost = F * V * V * (T + D + np.log2(max(V, 2)))
    pfga_cost = U * (k + 1) * V * D
    return graph_cost + pfga_cost


def measure_one_config(V, F, T, device="cpu", n_warmup=2, n_runs=5):
    """Measure forward/backward time + peak memory for one config."""
    cfg = {
        "num_nodes": V, "input_dim": F, "lookback": T, "horizon": 3,
        "hidden_dim": 32, "num_heads": 2,
        "frozen_layers": 1, "unfrozen_layers": 1,
        "num_categories": 5, "lora_rank": 4, "lora_alpha": 8,
        "graph_types": ["spatial", "temporal", "semantic", "adaptive"],
        "vocab_size": 30000, "use_text": False, "use_categorical": True,
    }
    try:
        model = MVGTNet(cfg).to(device)
    except Exception as e:
        return None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 4
    x = torch.randn(B, T, V, F, device=device)
    y = torch.randn(B, 3, V, F, device=device)
    text = {"fact": torch.randint(0, 30000, (B, T, 32), device=device)}
    cat = torch.randint(0, 5, (B, T, V), device=device)
    adj = torch.eye(V, device=device)

    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        out = model(x, text, cat, adj_spatial=adj)
        loss = torch.nn.functional.l1_loss(out["numeric"], y)
        loss.backward()
        optimizer.step()

    # Measure forward
    fwd_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x, text, cat, adj_spatial=adj)
        torch.cuda.synchronize() if device == "cuda" else None
        fwd_times.append(time.perf_counter() - t0)

    # Measure backward
    bwd_times = []
    for _ in range(n_runs):
        optimizer.zero_grad()
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        out = model(x, text, cat, adj_spatial=adj)
        loss = torch.nn.functional.l1_loss(out["numeric"], y)
        loss.backward()
        torch.cuda.synchronize() if device == "cuda" else None
        bwd_times.append(time.perf_counter() - t0)

    # Peak memory
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_mem = 0.0

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "V": V, "F": F, "T": T,
        "params": n_params, "trainable": n_trainable,
        "forward_ms": float(np.mean(fwd_times) * 1000),
        "forward_std_ms": float(np.std(fwd_times) * 1000),
        "backward_ms": float(np.mean(bwd_times) * 1000),
        "backward_std_ms": float(np.std(bwd_times) * 1000),
        "peak_memory_mb": peak_mem,
        "theoretical_ops": int(theoretical_complexity(V, F, T)),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase F: Scaling analysis")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        args.device = "cpu"

    print("=== Phase F: Scaling Analysis ===")
    print(f"Device: {args.device}")
    print()

    # Three sweeps
    sweeps = [
        ("V_sweep", [{"V": v, "F": 1, "T": 12} for v in [16, 32, 64, 128, 256]]),
        ("F_sweep", [{"V": 32, "F": f, "T": 12} for f in [1, 2, 4, 8, 16]]),
        ("T_sweep", [{"V": 32, "F": 1, "T": t} for t in [12, 24, 48, 96, 192]]),
    ]

    out_csv = RESULTS_DIR / "scaling_table.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sweep", "V", "F", "T", "params", "trainable",
                    "forward_ms", "forward_std_ms",
                    "backward_ms", "backward_std_ms",
                    "peak_memory_mb", "theoretical_ops"])
        all_results = {}
        for sweep_name, configs in sweeps:
            print(f"--- {sweep_name} ---")
            all_results[sweep_name] = []
            for c in configs:
                r = measure_one_config(c["V"], c["F"], c["T"], device=args.device)
                if r is None:
                    print(f"  V={c['V']} F={c['F']} T={c['T']}: FAILED (likely OOM)")
                    continue
                print(f"  V={c['V']:4d} F={c['F']:2d} T={c['T']:3d}: "
                      f"params={r['params']:>10,} fwd={r['forward_ms']:8.2f}ms "
                      f"bwd={r['backward_ms']:8.2f}ms mem={r['peak_memory_mb']:.1f}MB "
                      f"theory_ops={r['theoretical_ops']:>12,}")
                w.writerow([sweep_name, r["V"], r["F"], r["T"],
                            r["params"], r["trainable"],
                            f"{r['forward_ms']:.4f}", f"{r['forward_std_ms']:.4f}",
                            f"{r['backward_ms']:.4f}", f"{r['backward_std_ms']:.4f}",
                            f"{r['peak_memory_mb']:.4f}", r["theoretical_ops"]])
                all_results[sweep_name].append(r)
    print(f"\nTable: {out_csv}")

    # Plot: 2x3 grid (forward / backward) x (V / F / T)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    sweep_keys = ["V_sweep", "F_sweep", "T_sweep"]
    sweep_labels = ["Number of nodes (V)", "Number of features (F)", "Lookback window (T)"]
    for j, metric in enumerate(["forward_ms", "backward_ms"]):
        for i, (sweep, xlabel) in enumerate(zip(sweep_keys, sweep_labels)):
            ax = axes[j, i]
            data = all_results[sweep]
            if not data:
                continue
            xs = [r[sweep[0]] for r in data]  # 'V', 'F', or 'T'
            ys = [r[metric] for r in data]
            theory = [r["theoretical_ops"] for r in data]
            # Normalize theoretical to match empirical scale (just for shape)
            if ys[0] > 0 and theory[0] > 0:
                theory_scaled = [t * ys[0] / theory[0] for t in theory]
            else:
                theory_scaled = theory
            ax.plot(xs, ys, "o-", color="#1f77b4", linewidth=2, markersize=8, label="Empirical")
            ax.plot(xs, theory_scaled, "s--", color="#d62728", linewidth=1.5, markersize=6,
                    label="Theoretical (scaled)", alpha=0.7)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(xlabel); ax.set_ylabel(f"{metric.replace('_', ' ')} (ms)")
            ax.set_title(f"{metric.replace('_', ' ').title()} vs {xlabel}")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8)
    fig.suptitle("Phase F: MVGT-Net Scaling Analysis (log-log)\n"
                 "Empirical timing vs theoretical Big-O (Section 6-17)",
                 y=1.02, fontsize=13, fontweight="bold")
    out = RESULTS_DIR / "scaling_curves.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"Curves: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
