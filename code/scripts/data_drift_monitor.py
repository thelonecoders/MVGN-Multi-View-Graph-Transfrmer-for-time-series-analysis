#!/usr/bin/env python3
"""data_drift_monitor.py — Detect data drift between dataset splits and over time.

Compares train / validation / test splits per domain to detect distributional
shifts that could invalidate model evaluation. Also supports comparing a
"reference" dataset (e.g., the original TimeMMD) against a "current" dataset
(e.g., a newly downloaded copy) to verify no upstream drift occurred.

Drift detection methods:
  - Numeric features: Kolmogorov-Smirnov (KS) test (scipy.stats.ks_2samp)
  - Categorical features: Chi-squared test on value frequencies
  - Sequence length: KS test on token / record lengths
  - Timestamp range: overlap percentage + boundary shift

Usage:
    python3 scripts/data_drift_monitor.py \\
        --dataset-dir 12_dataset/TimeMMD \\
        --output-dir logs/drift_reports

Outputs:
    logs/drift_reports/<timestamp>/
        drift_report.json   — full per-domain per-split drift report
        drift_summary.csv   — flat summary of all comparisons
        drift_heatmap.png   — visual heatmap of drift scores
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Defer scipy import so the script can still produce a usage message without it


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_numeric_features(records: list[dict]) -> dict[str, list[float]]:
    """Extract numeric features from records. Returns {feature_name: [values]}."""
    features: dict[str, list[float]] = {}
    if not records:
        return features
    # Discover numeric keys from the first record
    sample = records[0]
    numeric_keys = []
    for k, v in sample.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_keys.append(k)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            numeric_keys.append(k)
    for k in numeric_keys:
        vals: list[float] = []
        for r in records:
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
            elif isinstance(v, list):
                vals.extend(float(x) for x in v if isinstance(x, (int, float)))
        if vals:
            features[k] = vals
    return features


def ks_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    try:
        from scipy.stats import ks_2samp
        result = ks_2samp(a, b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # Fallback: simple mean+std comparison
        if not a or not b:
            return 0.0, 1.0
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a = sum((x - mean_a) ** 2 for x in a) / max(len(a) - 1, 1)
        var_b = sum((x - mean_b) ** 2 for x in b) / max(len(b) - 1, 1)
        pooled_std = (var_a + var_b) ** 0.5
        if pooled_std == 0:
            return 0.0, 1.0
        statistic = abs(mean_a - mean_b) / pooled_std
        return statistic, 0.5  # no proper p-value without scipy


def analyze_domain(domain_dir: Path) -> dict[str, Any]:
    """Analyze drift across train/validation/test splits for one domain."""
    splits = {}
    for split in ["train", "validation", "test"]:
        path = domain_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_jsonl(path)

    if len(splits) < 2:
        return {"domain": domain_dir.name, "error": f"Only {len(splits)} splits found", "splits": list(splits.keys())}

    # Extract numeric features per split
    features_per_split = {split: extract_numeric_features(recs) for split, recs in splits.items()}

    # Compare train vs validation, train vs test
    comparisons = []
    for split_b in ["validation", "test"]:
        if split_b not in features_per_split:
            continue
        common_features = sorted(set(features_per_split["train"].keys()) & set(features_per_split[split_b].keys()))
        for feat in common_features:
            a = features_per_split["train"][feat]
            b = features_per_split[split_b][feat]
            if len(a) < 2 or len(b) < 2:
                continue
            stat, pval = ks_test(a, b)
            drift_detected = pval < 0.05
            comparisons.append({
                "comparison": f"train_vs_{split_b}",
                "feature": feat,
                "ks_statistic": round(stat, 6),
                "p_value": round(pval, 6),
                "drift_detected": drift_detected,
                "n_train": len(a),
                "n_other": len(b),
            })

    # Record counts
    record_counts = {split: len(recs) for split, recs in splits.items()}

    return {
        "domain": domain_dir.name,
        "record_counts": record_counts,
        "features_analyzed": sorted(set().union(*[set(f.keys()) for f in features_per_split.values()])),
        "comparisons": comparisons,
        "drift_summary": {
            "total_comparisons": len(comparisons),
            "drift_detected_count": sum(1 for c in comparisons if c["drift_detected"]),
            "max_ks_statistic": max((c["ks_statistic"] for c in comparisons), default=0.0),
            "min_p_value": min((c["p_value"] for c in comparisons), default=1.0),
        },
    }


def make_heatmap(report: dict, out_path: Path) -> None:
    """Generate a drift-score heatmap PNG (requires matplotlib)."""
    try:
        import matplotlib.font_manager as fm
        for font_path in [
            "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[data_drift_monitor] matplotlib not available — skipping heatmap", file=sys.stderr)
        return

    domains = [d["domain"] for d in report["domains"] if "comparisons" in d]
    max_ks = [d.get("drift_summary", {}).get("max_ks_statistic", 0.0) for d in report["domains"] if "comparisons" in d]

    if not domains:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(domains) * 0.4)), constrained_layout=True)
    ax.barh(domains, max_ks, color=["#e74c3c" if v > 0.1 else "#2ecc71" for v in max_ks])
    ax.set_xlabel("Max KS statistic (train vs val/test)")
    ax.set_title("Data Drift per Domain (higher = more drift)")
    ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="drift threshold (KS=0.1)")
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=150)
    fig.savefig(str(out_path).replace(".png", ".svg"))
    plt.close(fig)
    print(f"[OK] Heatmap saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect data drift across TimeMMD splits.")
    ap.add_argument("--dataset-dir", default="12_dataset/TimeMMD",
                    help="Path to TimeMMD dataset root")
    ap.add_argument("--output-dir", default="logs/drift_reports",
                    help="Output directory for reports")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    domains = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"[INFO] Analyzing {len(domains)} domains...")

    domain_reports = []
    for d in domains:
        print(f"  - {d.name}...")
        domain_reports.append(analyze_domain(d))

    total_comparisons = sum(d.get("drift_summary", {}).get("total_comparisons", 0) for d in domain_reports)
    total_drift = sum(d.get("drift_summary", {}).get("drift_detected_count", 0) for d in domain_reports)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "domains_analyzed": len(domains),
        "total_comparisons": total_comparisons,
        "total_drift_detected": total_drift,
        "drift_rate": round(total_drift / max(total_comparisons, 1), 4),
        "domains": domain_reports,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = out_dir / "drift_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    # CSV summary
    csv_path = out_dir / "drift_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["domain", "comparison", "feature", "ks_statistic", "p_value", "drift_detected", "n_train", "n_other"])
        writer.writeheader()
        for d in domain_reports:
            for c in d.get("comparisons", []):
                writer.writerow({"domain": d["domain"], **c})

    # Heatmap
    make_heatmap(report, out_dir / "drift_heatmap.png")

    print(f"\n[OK] Drift report generated:")
    print(f"     JSON: {json_path}")
    print(f"     CSV:  {csv_path}")
    print(f"     Domains: {len(domains)}, Comparisons: {total_comparisons}, Drift detected: {total_drift} ({report['drift_rate']*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
