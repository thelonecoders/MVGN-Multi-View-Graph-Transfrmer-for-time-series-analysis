#!/usr/bin/env python3
"""
make_figure.py — Generate thesis figures (Chapter 19) from training-run results.

Reads a per-domain metrics.json file produced by train_real.py and generates
the standard set of thesis figures as PNG + SVG.

Figures generated (per domain):
  1. loss_curves.png/svg         — train + val loss over epochs
  2. error_distribution.png/svg  — histogram of prediction errors
  3. attention_heatmap.png/svg   — attention weight heatmap (if saved)
  4. ablation_chart.png/svg      — ablation bar chart (if ablation data present)

Usage:
  # Generate all figures for one domain
  python scripts/make_figure.py \\
      --metrics results/Economy_Trade/metrics.json \\
      --output-dir figures/Economy_Trade

  # Generate only the loss curve
  python scripts/make_figure.py \\
      --metrics results/Economy_Trade/metrics.json \\
      --output-dir figures/Economy_Trade \\
      --only loss_curves

  # Generate figures for all domains (using a glob)
  python scripts/make_figure.py \\
      --metrics-glob 'results/*/metrics.json' \\
      --output-dir figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Use a non-interactive backend + constrained_layout for clean spacing.
plt.rcParams["figure.constrained_layout.use"] = True


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fig_loss_curves(metrics: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate train + val loss curves."""
    history = metrics.get("history", [])
    if not history:
        print("  [SKIP] loss_curves: no history in metrics", file=sys.stderr)
        return []
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", h.get("train_MAE", np.nan)) for h in history]
    val_loss = [h.get("val_loss", h.get("val_MAE", np.nan)) for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train", color="#1f77b4", linewidth=2)
    ax.plot(epochs, val_loss, label="Validation", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Training & Validation Loss — {metrics.get('domain', '?')}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    outputs = []
    for ext in ("png", "svg"):
        out = output_dir / f"loss_curves.{ext}"
        fig.savefig(out, dpi=150)
        outputs.append(out)
    plt.close(fig)
    return outputs


def fig_error_distribution(
    metrics: Dict[str, Any], output_dir: Path
) -> List[Path]:
    """Generate a histogram of prediction errors (if predictions are saved)."""
    # Look for a predictions file alongside the metrics.
    metrics_path = Path(metrics.get("_metrics_path", ""))
    pred_path = metrics_path.parent / "predictions.json"
    if not pred_path.exists():
        print(
            f"  [SKIP] error_distribution: no predictions.json at {pred_path}",
            file=sys.stderr,
        )
        return []

    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    errors: List[float] = []
    for p in preds:
        if "prediction" in p and "ground_truth" in p:
            pred_arr = np.array(p["prediction"])
            gt_arr = np.array(p["ground_truth"])
            if pred_arr.shape == gt_arr.shape:
                errors.extend((pred_arr - gt_arr).flatten().tolist())

    if not errors:
        print("  [SKIP] error_distribution: no comparable predictions",
              file=sys.stderr)
        return []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero error")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Error Distribution — {metrics.get('domain', '?')}"
    )
    ax.legend()

    outputs = []
    for ext in ("png", "svg"):
        out = output_dir / f"error_distribution.{ext}"
        fig.savefig(out, dpi=150)
        outputs.append(out)
    plt.close(fig)
    return outputs


def fig_attention_heatmap(
    metrics: Dict[str, Any], output_dir: Path
) -> List[Path]:
    """Generate an attention-weight heatmap (if attention weights are saved)."""
    metrics_path = Path(metrics.get("_metrics_path", ""))
    attn_path = metrics_path.parent / "attention.npy"
    if not attn_path.exists():
        print(
            f"  [SKIP] attention_heatmap: no attention.npy at {attn_path}",
            file=sys.stderr,
        )
        return []

    try:
        attn = np.load(attn_path)
    except Exception as e:
        print(f"  [SKIP] attention_heatmap: cannot load attention.npy: {e}",
              file=sys.stderr)
        return []

    # attn shape: (..., heads, seq, seq) — average over heads.
    while attn.ndim > 3:
        attn = attn.mean(axis=0)
    if attn.ndim == 3:
        attn = attn.mean(axis=0)  # (seq, seq)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(
        f"Attention Heatmap — {metrics.get('domain', '?')}"
    )
    fig.colorbar(im, ax=ax, label="Attention weight")

    outputs = []
    for ext in ("png", "svg"):
        out = output_dir / f"attention_heatmap.{ext}"
        fig.savefig(out, dpi=150)
        outputs.append(out)
    plt.close(fig)
    return outputs


def fig_ablation_chart(
    metrics: Dict[str, Any], output_dir: Path
) -> List[Path]:
    """Generate an ablation bar chart (if ablation data is present)."""
    ablation = metrics.get("ablation", None)
    if not ablation:
        print("  [SKIP] ablation_chart: no ablation data in metrics",
              file=sys.stderr)
        return []

    components = list(ablation.keys())
    maes = [ablation[c].get("test_MAE", np.nan) for c in components]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(components, maes, color="#9467bd", edgecolor="black")
    ax.set_xlabel("Component removed")
    ax.set_ylabel("Test MAE")
    ax.set_title(
        f"Ablation Study — {metrics.get('domain', '?')}"
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    # Annotate bars
    for bar, mae in zip(bars, maes):
        if not np.isnan(mae):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{mae:.3f}", ha="center", va="bottom", fontsize=9)

    outputs = []
    for ext in ("png", "svg"):
        out = output_dir / f"ablation_chart.{ext}"
        fig.savefig(out, dpi=150)
        outputs.append(out)
    plt.close(fig)
    return outputs


FIGURE_GENERATORS = {
    "loss_curves": fig_loss_curves,
    "error_distribution": fig_error_distribution,
    "attention_heatmap": fig_attention_heatmap,
    "ablation_chart": fig_ablation_chart,
}


def process_metrics(
    metrics_path: Path,
    output_dir: Path,
    only: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """Generate figures for a single metrics.json file."""
    metrics = load_metrics(metrics_path)
    metrics["_metrics_path"] = str(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = FIGURE_GENERATORS
    if only:
        generators = {k: v for k, v in generators.items() if k in only}

    generated: Dict[str, List[Path]] = {}
    for name, gen in generators.items():
        print(f"  Generating {name}...", file=sys.stderr)
        generated[name] = gen(metrics, output_dir)
    return generated


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics", type=Path, default=None,
        help="Path to a single metrics.json file.",
    )
    p.add_argument(
        "--metrics-glob", type=str, default=None,
        help="Glob pattern for multiple metrics.json files "
             "(e.g., 'results/*/metrics.json').",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for figures.",
    )
    p.add_argument(
        "--only", nargs="+", choices=list(FIGURE_GENERATORS.keys()),
        default=None,
        help="Only generate these figures (default: all).",
    )
    args = p.parse_args()

    if not args.metrics and not args.metrics_glob:
        p.error("either --metrics or --metrics-glob must be given")

    metrics_files: List[Path] = []
    if args.metrics:
        metrics_files.append(args.metrics)
    if args.metrics_glob:
        import glob
        metrics_files.extend(Path(p) for p in glob.glob(args.metrics_glob))

    if not metrics_files:
        print("No metrics files found.", file=sys.stderr)
        return 1

    total_generated = 0
    for mf in metrics_files:
        print(f"\nProcessing: {mf}", file=sys.stderr)
        # If multiple files, put each in a sub-dir named after the domain.
        if len(metrics_files) > 1:
            domain_dir = mf.parent.name
            out = args.output_dir / domain_dir
        else:
            out = args.output_dir
        generated = process_metrics(mf, out, args.only)
        for name, files in generated.items():
            total_generated += len(files)
            for f in files:
                print(f"  -> {f}")

    print(f"\nDone. Generated {total_generated} figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
