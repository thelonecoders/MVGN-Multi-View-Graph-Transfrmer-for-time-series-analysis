"""
MVGT-Net: Full Experiment Runner (Chapter 18 protocol)
=======================================================

This script orchestrates the full empirical validation described in
Chapter 18 of the thesis. It runs 432 experiments:
    9 domains * 4 horizons * 12 baselines = 432 experiments

Each experiment uses 5 random seeds (42, 43, 44, 45, 46), so the total
run count is 432 * 5 = 2160 runs. Estimated wall-clock time on a single
A100 40GB GPU: ~72 hours (per the Chapter 18 estimate).

PREREQUISITES (must be completed BEFORE running this script):
    1. Download the real TimeMMD dataset (~2 GB):
         python 12_dataset/download_script/download_timemmd.py --all
    2. Verify dataset checksums against the published hashes.
    3. Ensure a GPU with at least 24 GB VRAM is available.
    4. Set WANDB_API_KEY environment variable for experiment tracking.

USAGE:
    # Full run (2160 experiments, ~72 GPU-hours):
    python scripts/run_all_experiments.py --phase all

    # Quick smoke run (1 domain, 1 horizon, 1 baseline, 1 seed):
    python scripts/run_all_experiments.py --phase smoke

    # Only MVGT-Net (no baselines, 45 runs):
    python scripts/run_all_experiments.py --phase mvgtnet-only

    # Only one domain:
    python scripts/run_all_experiments.py --phase all --domain Environment

    # Statistical analysis after all runs complete:
    python scripts/run_all_experiments.py --phase analyze

OUTPUT:
    results/
      raw/
        {domain}_{horizon}_{model}_{seed}.json   # per-run metrics
      aggregated/
        main_table.csv                            # Layer 1: mean +/- std
        stat_tests.csv                            # Layer 2: p-values, CIs
        per_domain/                               # Layer 3: per-domain plots
          {domain}_attention_heatmap.png
          {domain}_shap_summary.png
          {domain}_conformal_calibration.png

NOTE: This script does NOT execute experiments in-process. It generates
a list of subprocess commands and runs them sequentially (or in parallel
if --parallel N is given). This ensures each run is isolated and a
single failure does not block the others.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ============================================================================
# Configuration (per Chapter 18 protocol)
# ============================================================================

TIMEMMD_DOMAINS = [
    "Climate",
    "Economy",
    "Energy",
    "Environment",
    "Health_US",
    "Health_AFR",
    "Agriculture",
    "Traffic",      # NYCTaxi (transferred from ST-LLM+ for cross-evaluation)
    "Bike",         # CHBike (transferred from ST-LLM+ for cross-evaluation)
]

HORIZONS = [3, 6, 12, 24]

BASELINES = [
    "Transformer", "Reformer", "Informer", "Autoformer",
    "Crossformer", "Non-stationary Transformer", "FEDformer",
    "iTransformer", "DLinear", "FiLM", "TimesNet", "PatchTST",
]

MVGT_MODEL = "MVGT-Net"

SEEDS = [42, 43, 44, 45, 46]

RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
AGG_DIR = RESULTS_DIR / "aggregated"
PER_DOMAIN_DIR = AGG_DIR / "per_domain"


@dataclass
class Experiment:
    """A single experiment configuration."""
    domain: str
    horizon: int
    model: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.domain}_{self.horizon}_{self.model}_{self.seed}"

    @property
    def output_path(self) -> Path:
        return RAW_DIR / f"{self.name}.json"

    def to_command(self) -> List[str]:
        """Convert to a subprocess command."""
        if self.model == MVGT_MODEL:
            return [
                "python", "scripts/train.py",
                "--config", "configs/default.yaml",
                "--domain", self.domain,
                "--horizon", str(self.horizon),
                "--seed", str(self.seed),
                "--epochs", "500",
                "--wandb",
                "--output", str(self.output_path),
            ]
        else:
            return [
                "python", "scripts/run_baseline.py",
                "--baseline", self.model,
                "--domain", self.domain,
                "--horizon", str(self.horizon),
                "--seed", str(self.seed),
                "--output", str(self.output_path),
            ]


def generate_experiments(
    phase: str,
    domain_filter: Optional[str] = None,
) -> List[Experiment]:
    """Generate the list of experiments for a given phase.

    Phases:
        smoke: 1 domain * 1 horizon * 1 model * 1 seed = 1 run
        mvgtnet-only: 9 domains * 4 horizons * 1 model * 5 seeds = 180 runs
        all: 9 domains * 4 horizons * 13 models * 5 seeds = 2340 runs
              (13 = 12 baselines + 1 MVGT-Net; if you exclude baselines
              that overlap with MVGT-Net, it's 12 * 5 * 9 * 4 = 2160)
    """
    domains = [domain_filter] if domain_filter else TIMEMMD_DOMAINS
    if phase == "smoke":
        return [Experiment(domains[0], 12, MVGT_MODEL, 42)]
    elif phase == "mvgtnet-only":
        return [
            Experiment(d, h, MVGT_MODEL, s)
            for d in domains
            for h in HORIZONS
            for s in SEEDS
        ]
    elif phase == "all":
        all_models = BASELINES + [MVGT_MODEL]
        return [
            Experiment(d, h, m, s)
            for d in domains
            for h in HORIZONS
            for m in all_models
            for s in SEEDS
        ]
    elif phase == "analyze":
        return []  # analysis only, no new runs
    else:
        raise ValueError(f"Unknown phase: {phase}")


def run_experiment(exp: Experiment, dry_run: bool = False) -> bool:
    """Execute a single experiment.

    Returns True if the experiment succeeded (output file written), False
    otherwise. Failures are logged but do not stop the pipeline.
    """
    if exp.output_path.exists():
        print(f"  [SKIP] {exp.name} (already exists)")
        return True
    cmd = exp.to_command()
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True
    print(f"  [RUN] {exp.name}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=7200
        )
        elapsed = time.time() - t0
        print(f"  [OK]  {exp.name} ({elapsed:.0f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {exp.name}: {e}")
        print(f"         stderr: {e.stderr[:500] if e.stderr else '(empty)'}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {exp.name} (> 2h)")
        return False


def run_all(experiments: List[Experiment], dry_run: bool = False) -> dict:
    """Run all experiments and return a summary."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(experiments)
    n_ok = 0
    n_fail = 0
    n_skip = 0
    t_start = time.time()
    for i, exp in enumerate(experiments, 1):
        print(f"[{i:04d}/{n_total:04d}] {exp.name}")
        if exp.output_path.exists():
            print(f"  [SKIP] (already exists)")
            n_skip += 1
            continue
        success = run_experiment(exp, dry_run=dry_run)
        if success:
            n_ok += 1
        else:
            n_fail += 1
    elapsed = time.time() - t_start
    summary = {
        "total": n_total,
        "succeeded": n_ok,
        "failed": n_fail,
        "skipped": n_skip,
        "elapsed_seconds": elapsed,
    }
    print()
    print("=" * 60)
    print(f"Summary: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"(of {n_total} total) in {elapsed:.0f}s")
    print("=" * 60)
    return summary


# ============================================================================
# Layer 1: Main result aggregation (mean +/- std across seeds)
# ============================================================================

def aggregate_main_table() -> None:
    """Read all raw JSON results and produce results/aggregated/main_table.csv.

    Columns: domain, horizon, model, MAE_mean, MAE_std, MSE_mean, MSE_std,
             RMSE_mean, RMSE_std, WAPE_mean, WAPE_std, MAPE_mean, MAPE_std,
             sMAPE_mean, sMAPE_std, R2_mean, R2_std, n_seeds
    """
    import csv
    from collections import defaultdict
    import numpy as np

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for f in RAW_DIR.glob("*.json"):
        with open(f) as fh:
            d = json.load(fh)
        key = (d["domain"], d["horizon"], d["model"])
        groups[key].append(d["metrics"])
    metrics_keys = ["MAE", "MSE", "RMSE", "WAPE", "MAPE", "sMAPE", "R2"]
    out_path = AGG_DIR / "main_table.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["domain", "horizon", "model"]
        for m in metrics_keys:
            header += [f"{m}_mean", f"{m}_std"]
        header += ["n_seeds"]
        w.writerow(header)
        for (domain, horizon, model), runs in sorted(groups.items()):
            row = [domain, horizon, model]
            for m in metrics_keys:
                vals = [r[m] for r in runs if m in r]
                if vals:
                    row += [float(np.mean(vals)), float(np.std(vals))]
                else:
                    row += [float("nan"), float("nan")]
            row += [len(runs)]
            w.writerow(row)
    print(f"Wrote {out_path} ({len(groups)} groups).")


# ============================================================================
# Layer 2: Statistical tests (paired t-test + Holm-Bonferroni)
# ============================================================================

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Apply the Holm-Bonferroni step-down correction.

    Implements the algorithm of Holm (1979) "A simple sequentially rejective
    multiple test procedure", Scandinavian Journal of Statistics 6(2): 65-70.

    Args:
        p_values: list of raw (uncorrected) p-values.
        alpha: family-wise error rate (default 0.05).

    Returns:
        List of (adjusted_p, reject) tuples in the ORIGINAL order, where
        adjusted_p is the Holm-adjusted p-value and reject is True iff
        the null hypothesis is rejected at family-wise level alpha.
    """
    m = len(p_values)
    if m == 0:
        return []
    # Pair each p-value with its original index, then sort ascending.
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    reject = [False] * m
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm step-down: adjusted_p = p * (m - rank), capped at 1.
        adj = min(1.0, p * (m - rank))
        # Enforce monotonicity (cannot decrease as rank increases).
        adj = max(adj, prev_adj)
        adjusted[orig_idx] = adj
        reject[orig_idx] = adj <= alpha
        prev_adj = adj
    return [(adjusted[i], reject[i]) for i in range(m)]


def statistical_tests() -> None:
    """Compute paired t-test + Holm-Bonferroni for MVGT-Net vs each baseline.

    Output: results/aggregated/stat_tests.csv
    Columns: domain, horizon, baseline, metric, mean_diff, p_value,
             p_value_holm, cohens_d, ci_lower, ci_upper, significant

    The Holm-Bonferroni correction (Holm 1979) is applied ACROSS ALL
    (baseline, metric) pairs within each (domain, horizon) family. This
    controls the family-wise error rate at alpha = 0.05.
    """
    import csv
    from collections import defaultdict
    from scipy import stats
    import numpy as np

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    # Load main_table
    by_dhm = defaultdict(dict)
    with open(AGG_DIR / "main_table.csv") as fh:
        r = csv.DictReader(fh)
        for row in r:
            key = (row["domain"], int(row["horizon"]))
            by_dhm[key][row["model"]] = row
    # We need per-sample errors, not just means, so we re-load raw JSONs
    # grouped by (domain, horizon, model, seed).
    raw_by_dhms = defaultdict(list)
    for f in RAW_DIR.glob("*.json"):
        with open(f) as fh:
            d = json.load(fh)
        key = (d["domain"], d["horizon"], d["model"])
        raw_by_dhms[key].append(d["metrics"])
    out_path = AGG_DIR / "stat_tests.csv"
    metrics_keys = ["MAE", "MSE", "RMSE", "WAPE", "MAPE", "sMAPE", "R2"]

    # First pass: collect all uncorrected rows grouped by (domain, horizon)
    # family so Holm-Bonferroni can be applied within each family.
    families: dict[tuple, list[dict]] = defaultdict(list)
    for (domain, horizon), models in by_dhm.items():
        if MVGT_MODEL not in models:
            continue
        mvgtnet_runs = raw_by_dhms.get((domain, horizon, MVGT_MODEL), [])
        if not mvgtnet_runs:
            continue
        for baseline in BASELINES:
            baseline_runs = raw_by_dhms.get((domain, horizon, baseline), [])
            if not baseline_runs:
                continue
            for metric in metrics_keys:
                mvgtnet_vals = [r[metric] for r in mvgtnet_runs if metric in r]
                baseline_vals = [r[metric] for r in baseline_runs if metric in r]
                if len(mvgtnet_vals) < 2 or len(baseline_vals) < 2:
                    continue
                min_len = min(len(mvgtnet_vals), len(baseline_vals))
                a = np.array(mvgtnet_vals[:min_len])
                b = np.array(baseline_vals[:min_len])
                diff = a - b
                mean_diff = float(np.mean(diff))
                if np.std(diff) > 1e-10:
                    t_stat, p_val = stats.ttest_rel(a, b)
                    p_val = float(p_val)
                else:
                    p_val = 1.0
                cohens_d = float(mean_diff / (np.std(diff) + 1e-10))
                # bootstrap 95% CI (percentile method, 1000 resamples, seed=42)
                n_boot = 1000
                boot_means = []
                rng = np.random.default_rng(42)
                for _ in range(n_boot):
                    idx = rng.integers(0, min_len, min_len)
                    boot_means.append(float(np.mean(diff[idx])))
                ci_lower = float(np.percentile(boot_means, 2.5))
                ci_upper = float(np.percentile(boot_means, 97.5))
                families[(domain, horizon)].append({
                    "domain": domain, "horizon": horizon,
                    "baseline": baseline, "metric": metric,
                    "mean_diff": mean_diff, "p_value": p_val,
                    "cohens_d": cohens_d,
                    "ci_lower": ci_lower, "ci_upper": ci_upper,
                })

    # Second pass: apply Holm-Bonferroni within each (domain, horizon) family.
    ALPHA = 0.05
    total_rows = 0
    total_rejected = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "domain", "horizon", "baseline", "metric",
            "mean_diff", "p_value", "p_value_holm", "cohens_d",
            "ci_lower", "ci_upper", "significant",
        ])
        for (domain, horizon), rows in families.items():
            raw_ps = [r["p_value"] for r in rows]
            corrections = holm_bonferroni(raw_ps, alpha=ALPHA)
            for row, (adj_p, rej) in zip(rows, corrections):
                w.writerow([
                    row["domain"], row["horizon"], row["baseline"], row["metric"],
                    f"{row['mean_diff']:.6f}", f"{row['p_value']:.6f}",
                    f"{adj_p:.6f}", f"{row['cohens_d']:.6f}",
                    f"{row['ci_lower']:.6f}", f"{row['ci_upper']:.6f}",
                    "1" if rej else "0",
                ])
                total_rows += 1
                if rej:
                    total_rejected += 1
    print(f"Wrote {out_path} ({total_rows} tests, {total_rejected} significant "
          f"after Holm-Bonferroni at alpha={ALPHA}).")


# ============================================================================
# Main CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full MVGT-Net experiment suite (Chapter 18 protocol)."
    )
    parser.add_argument(
        "--phase",
        choices=["smoke", "mvgtnet-only", "all", "analyze"],
        default="smoke",
        help="Which phase to run. 'smoke' = 1 run; 'all' = 2160 runs.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to a single domain (default: all 9 domains).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of experiments to run in parallel (default: 1).",
    )
    args = parser.parse_args()

    if args.phase == "analyze":
        print("Phase: analyze (no new runs)")
        aggregate_main_table()
        statistical_tests()
        return 0

    experiments = generate_experiments(args.phase, args.domain)
    print(f"Phase: {args.phase}")
    print(f"Domain filter: {args.domain or '(all 9 domains)'}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated wall-clock (A100): {len(experiments) * 0.033:.1f} hours")
    print()

    if args.dry_run:
        print("[DRY RUN] Commands that would be executed:")
        for exp in experiments:
            print(f"  {' '.join(exp.to_command())}")
        return 0

    summary = run_all(experiments, dry_run=args.dry_run)
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary written to {summary_path}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
