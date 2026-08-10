"""
Phase F: Inference Latency Benchmark + Carbon Footprint Calculator
===================================================================
Two distinct analyses in one script:

1. Inference Latency Benchmark
   Measures end-to-end inference latency for MVGT-Net across batch sizes
   {1, 4, 8, 16, 32, 64, 128, 256} on a fixed problem size. Reports:
       - latency per batch (ms)
       - latency per sample (ms)
       - throughput (samples/sec)
       - peak GPU memory (MB)

2. Carbon Footprint Calculator
   Estimates kgCO2e for the full 432-experiment Phase A training suite
   using the methodology of Patterson et al. (2021) "Carbon Emissions
   and Large Neural Networks", arXiv:2104.10350.

   Formula:
       kgCO2e = P_gpu * T_hours * PUE * CI / 1000
   where:
       P_gpu  = GPU power draw (W), A100 ~ 400W under training load
       T      = total training hours
       PUE    = data center power usage effectiveness (1.1 typical)
       CI     = carbon intensity (gCO2e/kWh), world average ~475

Output:
    results/latency/latency_table.csv
    results/latency/latency_curves.png
    results/carbon/carbon_report.json
    results/carbon/carbon_breakdown.png

Usage:
    python scripts/latency_carbon.py

References:
    Patterson et al. (2021). "Carbon Emissions and Large Neural Networks",
        arXiv:2104.10350.
    Henderson et al. (2020). "Towards the Systematic Reporting of the
        Energy and Carbon Footprints of Machine Learning", JMLR 21.
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

from mvgt_net import MVGTNet

# VPS bundle: outputs go inside code/ (not sibling of code/)
LAT_DIR = Path(os.path.join(CODE_ROOT, "14_engineering_analyses", "latency"))
CAR_DIR = Path(os.path.join(CODE_ROOT, "14_engineering_analyses", "carbon"))
LAT_DIR.mkdir(parents=True, exist_ok=True)
CAR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Part 1: Inference Latency Benchmark
# ============================================================================

def benchmark_latency(device="cpu"):
    """Measure inference latency across batch sizes."""
    print("=== Part 1: Inference Latency Benchmark ===")
    cfg = {
        "num_nodes": 14, "input_dim": 1, "lookback": 12, "horizon": 3,
        "hidden_dim": 32, "num_heads": 2,
        "frozen_layers": 1, "unfrozen_layers": 1,
        "num_categories": 5, "lora_rank": 4, "lora_alpha": 8,
        "graph_types": ["spatial", "temporal", "semantic", "adaptive"],
        "vocab_size": 30000, "use_text": False, "use_categorical": True,
    }
    model = MVGTNet(cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    V, F, T = 14, 1, 12
    adj = torch.eye(V, device=device)

    results = []
    for B in batch_sizes:
        x = torch.randn(B, T, V, F, device=device)
        text = {"fact": torch.randint(0, 30000, (B, T, 32), device=device)}
        cat = torch.randint(0, 5, (B, T, V), device=device)
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x, text, cat, adj_spatial=adj)
        # Measure
        times = []
        for _ in range(10):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x, text, cat, adj_spatial=adj)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        lat_ms = float(np.mean(times) * 1000)
        lat_std = float(np.std(times) * 1000)
        per_sample = lat_ms / B
        throughput = B / (lat_ms / 1000) if lat_ms > 0 else 0
        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_mem = 0.0
        print(f"  B={B:4d}: lat={lat_ms:8.2f}+-{lat_std:.2f}ms per_sample={per_sample:6.3f}ms "
              f"throughput={throughput:8.1f} samp/s mem={peak_mem:.1f}MB")
        results.append({
            "batch_size": B, "latency_ms": lat_ms, "latency_std_ms": lat_std,
            "per_sample_ms": per_sample, "throughput_samp_s": throughput,
            "peak_memory_mb": peak_mem,
        })

    out_csv = LAT_DIR / "latency_table.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch_size", "latency_ms", "latency_std_ms",
                    "per_sample_ms", "throughput_samp_s", "peak_memory_mb"])
        for r in results:
            w.writerow([r["batch_size"], f"{r['latency_ms']:.4f}",
                        f"{r['latency_std_ms']:.4f}", f"{r['per_sample_ms']:.4f}",
                        f"{r['throughput_samp_s']:.4f}", f"{r['peak_memory_mb']:.4f}"])
    print(f"  Table: {out_csv}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    ax = axes[0]
    bs = [r["batch_size"] for r in results]
    lat = [r["latency_ms"] for r in results]
    lat_err = [r["latency_std_ms"] for r in results]
    ax.errorbar(bs, lat, yerr=lat_err, fmt="o-", color="#1f77b4", linewidth=2, markersize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size"); ax.set_ylabel("Latency per batch (ms)")
    ax.set_title("Inference latency vs batch size")
    ax.grid(True, alpha=0.3, which="both")
    ax = axes[1]
    per_s = [r["per_sample_ms"] for r in results]
    tp = [r["throughput_samp_s"] for r in results]
    ax.plot(bs, per_s, "s-", color="#d62728", linewidth=2, markersize=8, label="Per-sample latency (ms)")
    ax2 = ax.twinx()
    ax2.plot(bs, tp, "^--", color="#2ca02c", linewidth=2, markersize=8, label="Throughput (samp/s)")
    ax.set_xscale("log")
    ax.set_xlabel("Batch size"); ax.set_ylabel("Per-sample latency (ms)")
    ax2.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Per-sample cost & throughput vs batch size")
    ax.grid(True, alpha=0.3, which="both")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.suptitle(f"Phase F: Inference Latency Benchmark (device={device}, params={n_params:,})",
                 y=1.02, fontsize=12, fontweight="bold")
    out = LAT_DIR / "latency_curves.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"  Curves: {out}")


# ============================================================================
# Part 2: Carbon Footprint Calculator
# ============================================================================

def carbon_footprint():
    """Estimate kgCO2e for the full Phase A training suite."""
    print("\n=== Part 2: Carbon Footprint Calculator ===")

    # Training suite parameters (Chapter 18 protocol)
    n_experiments = 432
    n_seeds = 5
    total_runs = n_experiments * n_seeds  # 2160
    hours_per_run = 72.0 / total_runs  # 72 A100-hours total per Chapter 18 estimate
    total_hours = 72.0  # already the total

    # Hardware: NVIDIA A100 40GB
    # Power draw under training load: ~400W (NVIDIA datasheet)
    gpu_power_w = 400.0

    # Data center PUE (Power Usage Effectiveness)
    # World average 2024: 1.55 (Uptime Institute); best-in-class 1.1
    pue_avg = 1.55
    pue_best = 1.10

    # Carbon intensity (gCO2e per kWh)
    # World average grid: 475 (IEA 2023)
    # Renewable-only: 50
    # Worst-case (coal-heavy grid): 820
    ci_world_avg = 475.0
    ci_renewable = 50.0
    ci_coal = 820.0

    # Formula (Patterson et al. 2021):
    #   kgCO2e = (P_gpu_W * T_hours * PUE) / 1000 * CI_gCO2e_per_kWh / 1000
    #         = P_gpu * T_hours * PUE * CI / 1e6
    scenarios = {
        "best_case_renewable_PUE1.1": (pue_best, ci_renewable),
        "world_avg_PUE1.55":          (pue_avg, ci_world_avg),
        "worst_case_coal_PUE1.55":    (pue_avg, ci_coal),
    }
    results = {}
    for name, (pue, ci) in scenarios.items():
        kg_co2e = gpu_power_w * total_hours * pue * ci / 1e6
        # Also report kgCO2e per experiment and per 1% MAE improvement
        per_exp = kg_co2e / n_experiments
        results[name] = {
            "pue": pue, "carbon_intensity_gco2e_per_kwh": ci,
            "total_kgco2e": round(kg_co2e, 2),
            "per_experiment_kgco2e": round(per_exp, 4),
            "equivalent_car_km": round(kg_co2e / 0.171, 1),  # 1 km car = 171 gCO2e (EPA)
            "equivalent_trees_year": round(kg_co2e / 21.0, 1),  # 1 tree absorbs 21 kg/yr
        }
        print(f"  {name}: {kg_co2e:.2f} kgCO2e  (= {kg_co2e/0.171:.1f} car-km, "
              f"{kg_co2e/21.0:.1f} tree-years)")

    out = CAR_DIR / "carbon_report.json"
    summary = {
        "methodology": "Patterson et al. (2021) arXiv:2104.10350",
        "formula": "kgCO2e = P_gpu_W * T_hours * PUE * CI_gCO2e_per_kWh / 1e6",
        "training_suite": {
            "n_experiments": n_experiments,
            "n_seeds": n_seeds,
            "total_runs": total_runs,
            "total_gpu_hours": total_hours,
            "gpu": "NVIDIA A100 40GB",
            "gpu_power_draw_w": gpu_power_w,
        },
        "scenarios": results,
        "notes": [
            "These are ESTIMATES based on the Chapter 18 protocol (72 A100-hours).",
            "Actual emissions depend on the specific data center PUE and grid mix.",
            "Phase A training has NOT been executed; this is a planning estimate.",
            "Best-case assumes 100% renewable electricity and modern PUE=1.1 data center.",
            "Worst-case assumes coal-heavy grid (e.g. some regions of Australia, Poland).",
        ],
    }
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Report: {out}")

    # Plot: bar chart of 3 scenarios with car-km + tree-year equivalents
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    names = list(results.keys())
    kg = [results[n]["total_kgco2e"] for n in names]
    pretty_names = ["Best case\n(renewable, PUE 1.1)", "World avg\n(grid mix, PUE 1.55)", "Worst case\n(coal grid, PUE 1.55)"]
    bars = ax.bar(pretty_names, kg, color=["#2ca02c", "#1f77b4", "#d62728"])
    ax.set_ylabel("Total kgCO2e")
    ax.set_title("Estimated carbon footprint of full Phase A training (2160 runs)")
    for bar, v in zip(bars, kg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    car_km = [results[n]["equivalent_car_km"] for n in names]
    trees = [results[n]["equivalent_trees_year"] for n in names]
    x = np.arange(len(names))
    bars1 = ax.bar(x - 0.2, car_km, 0.4, color="#ff7f0e", label="Equivalent car-km")
    bars2 = ax.bar(x + 0.2, trees, 0.4, color="#2ca02c", label="Equivalent tree-years")
    ax.set_xticks(x); ax.set_xticklabels(pretty_names)
    ax.set_ylabel("Equivalent units")
    ax.set_title("Carbon footprint in relatable units")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Phase F: Carbon Footprint Estimation for Phase A Training Suite",
                 y=1.02, fontsize=12, fontweight="bold")
    out = CAR_DIR / "carbon_breakdown.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"  Plot: {out}")


def main():
    parser = argparse.ArgumentParser(description="Phase F: Latency benchmark + carbon footprint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-carbon", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        args.device = "cpu"
    benchmark_latency(args.device)
    if not args.skip_carbon:
        carbon_footprint()


if __name__ == "__main__":
    main()
