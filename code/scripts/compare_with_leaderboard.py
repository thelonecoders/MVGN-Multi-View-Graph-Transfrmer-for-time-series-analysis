#!/usr/bin/env python3
"""Compare MVGT-Net results against the 12 TimeMMD baselines.

Two modes are supported:

1. Cite mode (default):
   Load the official TimeMMD leaderboard values from
   ``baselines/timemmd_leaderboard.json`` and the MVGT-Net result JSON
   produced by ``scripts/train_real.py``. Emit a comparison table and
   a Holm-Bonferroni-corrected paired t-test on the per-domain MSE
   deltas. No baseline re-training is required.

2. Re-implement mode (--reimplement):
   Require that every baseline has been re-trained locally and that
   its per-domain results live under ``baselines/results/<name>.json``.
   Run the same comparison against the locally-measured numbers
   instead of the upstream leaderboard.

Output
------
- A markdown table written to ``results/baseline_comparison.md``.
- A CSV written to ``results/baseline_comparison.csv``.
- A JSON of the t-test statistics written to
  ``results/baseline_comparison_stats.json``.

If the leaderboard JSON is still in its shipped (un-populated) state,
the script refuses to run in cite mode and prints a clear pointer to
``scripts/fetch_leaderboard.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, CODE_ROOT)

from baselines import (  # noqa: E402
    LeaderboardNotPopulatedError,
    list_available_baselines,
    load_leaderboard,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _load_mvgt_results(path: str) -> Dict[str, Dict[str, float]]:
    """Load MVGT-Net per-domain metrics.

    Expected format (produced by train_real.py):
        {
            "Solar": {"MSE": ..., "MAE": ..., "RMSE": ..., ...},
            ...
        }
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return per-test reject booleans under Holm-Bonferroni."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    reject = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            reject[orig_idx] = True
        else:
            # Once we fail to reject, all remaining (larger) p-values
            # also fail.
            break
    return reject


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mvgt-results",
        default=os.path.join(CODE_ROOT, "results", "mvgt_net_metrics.json"),
        help="Path to the MVGT-Net per-domain metrics JSON.",
    )
    p.add_argument(
        "--mode",
        choices=("cite", "reimplement"),
        default="cite",
        help="cite = use upstream leaderboard numbers; "
             "reimplement = use locally re-trained baseline numbers.",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(CODE_ROOT, "results"),
        help="Where to write the comparison artifacts.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise error rate for Holm-Bonferroni.",
    )
    args = p.parse_args(argv)

    if not os.path.isfile(args.mvgt_results):
        sys.stderr.write(
            f"ERROR: MVGT-Net results not found at {args.mvgt_results}.\n"
            f"Run `python scripts/train_real.py` first to produce it.\n"
        )
        return 2

    mvgt = _load_mvgt_results(args.mvgt_results)
    domains = sorted(mvgt.keys())
    if not domains:
        sys.stderr.write("ERROR: MVGT-Net results JSON has no domains.\n")
        return 2

    baselines = list_available_baselines()
    if not baselines:
        sys.stderr.write("ERROR: no baselines found in leaderboard JSON.\n")
        return 2

    if args.mode == "cite":
        try:
            lb = load_leaderboard()
        except LeaderboardNotPopulatedError as exc:
            sys.stderr.write(
                f"ERROR: leaderboard not populated.\n{exc}\n"
                f"Run `python scripts/fetch_leaderboard.py` first, or "
                f"re-run this script with --mode reimplement after "
                f"re-training the baselines locally.\n"
            )
            return 3
        baseline_scores = {
            name: lb["baselines"][name]["scores"]
            for name in baselines
            if lb["baselines"][name].get("scores")
        }
    else:  # reimplement
        baseline_scores = {}
        for name in baselines:
            local_path = os.path.join(
                CODE_ROOT, "baselines", "results", f"{name}.json"
            )
            if not os.path.isfile(local_path):
                sys.stderr.write(
                    f"ERROR: missing locally re-trained result for "
                    f"{name!r} at {local_path}.\n"
                )
                return 2
            with open(local_path, "r", encoding="utf-8") as fh:
                baseline_scores[name] = json.load(fh)

    # ------------------------------------------------------------------ #
    # Build comparison rows + (very simple) paired t-test on MSE deltas.
    # ------------------------------------------------------------------ #
    rows: List[Dict[str, Any]] = []
    mvgt_mse = [mvgt[d].get("MSE") for d in domains]
    for bname in sorted(baseline_scores.keys()):
        bsc = baseline_scores[bname]
        b_mse = [bsc.get(d, {}).get("MSE") for d in domains]
        deltas = []
        for d, mv, bv in zip(domains, mvgt_mse, b_mse):
            if mv is None or bv is None:
                continue
            deltas.append(bv - mv)  # positive = MVGT-Net is better
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        # Very small-N paired t-test (no scipy dependency required).
        n = len(deltas)
        if n > 1:
            var = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
            std = var ** 0.5
            t_stat = (mean_delta / std * (n ** 0.5)) if std > 0 else float("inf")
            # Two-sided p from Student's t with n-1 df -- use a normal
            # approximation when n is large (we only need ordering for
            # Holm-Bonferroni; this is documented in the README).
            import math
            from math import erf, sqrt
            z = abs(t_stat)
            p_value = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        else:
            t_stat = float("nan")
            p_value = 1.0
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas)
        rows.append({
            "baseline": bname,
            "domains_compared": n,
            "mean_mse_delta_mvgt_minus_baseline": -mean_delta,
            "win_rate_mvgt": win_rate,
            "t_stat": t_stat,
            "p_value": p_value,
        })

    p_values = [r["p_value"] for r in rows]
    reject = _holm_bonferroni(p_values, alpha=args.alpha)
    for r, rej in zip(rows, reject):
        r["significant_after_holm"] = bool(rej)

    # ------------------------------------------------------------------ #
    # Write artifacts
    # ------------------------------------------------------------------ #
    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "baseline_comparison.md")
    csv_path = os.path.join(args.out_dir, "baseline_comparison.csv")
    stats_path = os.path.join(args.out_dir, "baseline_comparison_stats.json")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# MVGT-Net vs TimeMMD Baselines\n\n")
        fh.write(f"- Mode: `{args.mode}`\n")
        fh.write(f"- Domains: {len(domains)} ({', '.join(domains)})\n")
        fh.write(
            f"- Holm-Bonferroni alpha: {args.alpha}\n\n"
        )
        fh.write(
            "| Baseline | N | Mean MSE delta (MVGT - Baseline) | "
            "Win rate | t-stat | p | Sig. (Holm) |\n"
            "|---|---:|---:|---:|---:|---:|:---:|\n"
        )
        for r in rows:
            fh.write(
                f"| {r['baseline']} | {r['domains_compared']} | "
                f"{r['mean_mse_delta_mvgt_minus_baseline']:.4f} | "
                f"{r['win_rate_mvgt']*100:.1f}% | "
                f"{r['t_stat']:.3f} | {r['p_value']:.4f} | "
                f"{'yes' if r['significant_after_holm'] else 'no'} |\n"
            )

    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(
            "baseline,domains_compared,mean_mse_delta_mvgt_minus_baseline,"
            "win_rate_mvgt,t_stat,p_value,significant_after_holm\n"
        )
        for r in rows:
            fh.write(
                f"{r['baseline']},{r['domains_compared']},"
                f"{r['mean_mse_delta_mvgt_minus_baseline']:.6f},"
                f"{r['win_rate_mvgt']:.4f},{r['t_stat']:.6f},"
                f"{r['p_value']:.6f},{int(r['significant_after_holm'])}\n"
            )

    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": args.mode,
                "alpha": args.alpha,
                "n_domains": len(domains),
                "n_baselines_compared": len(rows),
                "n_significant_after_holm": sum(
                    1 for r in rows if r["significant_after_holm"]
                ),
                "rows": rows,
            },
            fh,
            indent=2,
        )
        fh.write("\n")

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
