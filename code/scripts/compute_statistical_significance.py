#!/usr/bin/env python3
"""
compute_statistical_significance.py — Apply Holm-Bonferroni correction to
the real Phase A results to produce the final significance table for
thesis Chapter 10.

Reads per-domain metrics from results/<Domain>/metrics.json and a list of
baseline metrics (from JSON), then computes:
  - Pairwise t-tests (MVGT-Net vs each baseline) per domain per metric
  - Holm-Bonferroni correction across all (n_domains × n_baselines × n_metrics)
    comparisons
  - Outputs a CSV + JSON with corrected p-values and significance stars

Usage:
  python scripts/compute_statistical_significance.py \\
      --results-dir results/ \\
      --baselines baselines.json \\
      --output-dir statistical_analysis/

Output:
  statistical_analysis/significance_table.csv
  statistical_analysis/significance_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """
    Apply Holm (1979) step-down correction to a list of p-values.

    Returns a list of dicts with the original index, raw p-value, corrected
    p-value, and reject flag.
    """
    indexed = list(enumerate(p_values))
    # Sort by p-value ascending
    indexed.sort(key=lambda x: x[1])
    n = len(indexed)
    results = []
    rejected_so_far = True
    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm correction: p_adj = p * (n - rank)
        corrected = p * (n - rank)
        # Cap at 1.0
        corrected = min(corrected, 1.0)
        # Holm step-down: once we fail to reject, all subsequent also fail
        reject = rejected_so_far and (corrected < alpha)
        if not reject:
            rejected_so_far = False
        results.append({
            "original_index": orig_idx,
            "raw_p": p,
            "corrected_p": corrected,
            "reject_h0": reject,
            "rank": rank + 1,
        })
    # Sort back to original order
    results.sort(key=lambda x: x["original_index"])
    return results


def stars(p: float) -> str:
    """Convert a p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def paired_t_test(a: List[float], b: List[float]) -> float:
    """
    Two-sided paired t-test on per-epoch losses.
    Returns the p-value.
    """
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    diffs = np.array(a) - np.array(b)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return 1.0 if mean_diff == 0 else 0.0
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs)))
    df = len(diffs) - 1
    # Approximate two-sided p-value from t-distribution
    # (using scipy if available, else normal approximation)
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    except ImportError:
        # Normal approximation (less accurate for small df)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    return float(p)


def load_per_epoch_losses(metrics_path: Path) -> Dict[str, List[float]]:
    """Load per-epoch train + val losses from a metrics.json file."""
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = m.get("history", [])
    return {
        "train_loss": [h.get("train_loss", h.get("train_MAE", np.nan)) for h in history],
        "val_loss": [h.get("val_loss", h.get("val_MAE", np.nan)) for h in history],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path, default=CODE_ROOT / "results",
        help="Directory containing per-domain results/<Domain>/metrics.json.",
    )
    p.add_argument(
        "--baselines", type=Path, default=CODE_ROOT / "baselines.json",
        help="JSON file with baseline metrics. Format: "
             '{"baselines": {"Transformer": {"Economy_Trade": '
             '{"val_loss_history": [...]}}}}',
    )
    p.add_argument(
        "--output-dir", type=Path, default=CODE_ROOT / "statistical_analysis",
        help="Output directory.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load MVGT-Net results
    mvgt_results: Dict[str, Dict[str, List[float]]] = {}
    for domain_dir in sorted(args.results_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        metrics_path = domain_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        mvgt_results[domain_dir.name] = load_per_epoch_losses(metrics_path)

    if not mvgt_results:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        return 1

    print(f"Loaded MVGT-Net results for {len(mvgt_results)} domains",
          file=sys.stderr)

    # Load baselines
    baselines: Dict[str, Any] = {}
    if args.baselines.exists():
        bdata = json.loads(args.baselines.read_text(encoding="utf-8"))
        baselines = bdata.get("baselines", {})
        print(f"Loaded {len(baselines)} baselines from {args.baselines}",
              file=sys.stderr)
    else:
        print(f"No baselines file at {args.baselines}; will only compute "
              f"MVGT-Net vs. itself (sanity check)", file=sys.stderr)

    # Compute pairwise comparisons
    comparisons: List[Dict[str, Any]] = []
    for domain, mvgt_losses in mvgt_results.items():
        for baseline_name, baseline_data in baselines.items() or [("MVGT-Net", mvgt_results)]:
            if baseline_name == "MVGT-Net":
                baseline_losses = mvgt_losses
            else:
                bd = baseline_data.get(domain, {})
                baseline_losses = {
                    "train_loss": bd.get("train_loss_history", []),
                    "val_loss": bd.get("val_loss_history", []),
                }
            for metric_name in ("train_loss", "val_loss"):
                a = mvgt_losses.get(metric_name, [])
                b = baseline_losses.get(metric_name, [])
                if not a or not b:
                    continue
                min_len = min(len(a), len(b))
                p_value = paired_t_test(a[:min_len], b[:min_len])
                comparisons.append({
                    "domain": domain,
                    "baseline": baseline_name,
                    "metric": metric_name,
                    "mvgt_mean": float(np.mean(a[:min_len])),
                    "baseline_mean": float(np.mean(b[:min_len])),
                    "p_value": p_value,
                })

    # Apply Holm-Bonferroni
    p_values = [c["p_value"] for c in comparisons]
    corrected = holm_bonferroni(p_values, alpha=args.alpha)

    for c, corr in zip(comparisons, corrected):
        c["corrected_p"] = corr["corrected_p"]
        c["reject_h0"] = corr["reject_h0"]
        c["stars"] = stars(corr["corrected_p"])

    # Write CSV
    csv_path = args.output_dir / "significance_table.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("domain,baseline,metric,mvgt_mean,baseline_mean,p_value,corrected_p,reject_h0,stars\n")
        for c in comparisons:
            f.write(f"{c['domain']},{c['baseline']},{c['metric']},"
                    f"{c['mvgt_mean']:.6f},{c['baseline_mean']:.6f},"
                    f"{c['p_value']:.6e},{c['corrected_p']:.6e},"
                    f"{c['reject_h0']},{c['stars']}\n")

    # Write JSON summary
    summary = {
        "alpha": args.alpha,
        "n_comparisons": len(comparisons),
        "n_significant": sum(1 for c in comparisons if c["reject_h0"]),
        "n_domains": len(mvgt_results),
        "n_baselines": len(baselines) if baselines else 1,
        "comparisons": comparisons,
    }
    json_path = args.output_dir / "significance_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults:", file=sys.stderr)
    print(f"  Comparisons:    {summary['n_comparisons']}", file=sys.stderr)
    print(f"  Significant:    {summary['n_significant']} (at α={args.alpha})",
          file=sys.stderr)
    print(f"  CSV:            {csv_path}", file=sys.stderr)
    print(f"  JSON:           {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
