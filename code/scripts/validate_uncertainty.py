#!/usr/bin/env python3
"""
validate_uncertainty.py — Verify that the conformal prediction intervals
(Formula E) achieve the nominal coverage rate on real test data.

Trains MVGT-Net, then computes conformal prediction intervals at multiple
nominal coverage levels (80%, 90%, 95%, 99%), and verifies that the
empirical coverage matches the nominal coverage.

Also reports:
  - Average interval width (tighter is better, given coverage is met)
  - Coverage gap (empirical - nominal; should be ≈ 0)
  - Per-domain coverage table

Usage:
  python scripts/validate_uncertainty.py --domain Climate_AQI --epochs 50
  python scripts/validate_uncertainty.py --domain Economy_Trade --epochs 100 \\
      --alpha-levels 0.20 0.10 0.05 0.01

Output:
  uncertainty_results/<domain>/coverage_table.csv
  uncertainty_results/<domain>/coverage_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mvgt_net import MVGTNet, MultiTaskLoss, DOMAIN_REGISTRY  # noqa: E402
from mvgt_net.data import get_dataloaders  # noqa: E402
from mvgt_net.metrics import all_metrics  # noqa: E402
from mvgt_net.uncertainty import ConformalPredictor  # noqa: E402


def compute_conformal_intervals(
    model: MVGTNet,
    calibration_loader,
    test_loader,
    device: torch.device,
    alpha_levels: List[float],
) -> Dict[str, Any]:
    """Compute conformal intervals at multiple alpha levels."""
    # Step 1: Collect calibration residuals
    model.eval()
    cal_preds, cal_targets = [], []
    with torch.no_grad():
        for batch in calibration_loader:
            x = batch["batch_x"].to(device).float()
            y = batch["batch_y"].to(device).float()
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            min_t = min(pred.shape[1], y.shape[1])
            cal_preds.append(pred[:, :min_t].cpu().numpy())
            cal_targets.append(y[:, :min_t].cpu().numpy())
    cal_preds = np.concatenate(cal_preds, axis=0)
    cal_targets = np.concatenate(cal_targets, axis=0)
    cal_residuals = np.abs(cal_preds - cal_targets).flatten()

    # Step 2: Collect test predictions + targets
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["batch_x"].to(device).float()
            y = batch["batch_y"].to(device).float()
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            min_t = min(pred.shape[1], y.shape[1])
            test_preds.append(pred[:, :min_t].cpu().numpy())
            test_targets.append(y[:, :min_t].cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    # Step 3: For each alpha, compute quantile + interval + coverage
    results = []
    n_cal = len(cal_residuals)
    for alpha in alpha_levels:
        nominal_coverage = 1.0 - alpha
        # Quantile of calibration residuals
        # Use the (1-alpha) * (n+1) / n quantile (Angelopoulos & Bates 2023)
        q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)
        q_hat = float(np.quantile(cal_residuals, q_level))

        # Compute intervals on test set
        lower = test_preds - q_hat
        upper = test_preds + q_hat
        # Coverage: fraction of targets within [lower, upper]
        covered = np.sum((test_targets >= lower) & (test_targets <= upper))
        total = test_targets.size
        empirical_coverage = covered / total
        coverage_gap = empirical_coverage - nominal_coverage

        # Average interval width
        avg_width = float(np.mean(upper - lower))

        results.append({
            "alpha": alpha,
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": empirical_coverage,
            "coverage_gap": coverage_gap,
            "quantile": q_hat,
            "avg_interval_width": avg_width,
            "n_calibration_samples": n_cal,
            "n_test_samples": total,
            "covered_count": int(covered),
            "within_tolerance": abs(coverage_gap) < 0.05,
        })

    return {
        "per_alpha": results,
        "test_metrics": {k: float(v) for k, v in all_metrics(
            torch.from_numpy(test_preds), torch.from_numpy(test_targets)
        ).items()},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="Economy_Trade",
                   choices=list(DOMAIN_REGISTRY.keys()))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha-levels", nargs="+", type=float,
                   default=[0.20, 0.10, 0.05, 0.01])
    p.add_argument("--output-dir", type=Path,
                   default=CODE_ROOT / "uncertainty_results")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloaders (use validation as calibration set)
    print(f"Loading data for domain: {args.domain}", file=sys.stderr)
    loaders = get_dataloaders(
        domain=args.domain,
        data_root=str(CODE_ROOT / "data" / "TimeMMD"),
        batch_size=32,
        num_workers=0,
    )

    # Build + train model
    print(f"Building + training model for {args.epochs} epochs", file=sys.stderr)
    cfg = DOMAIN_REGISTRY[args.domain]
    model = MVGTNet(
        num_features=1, hidden_dim=64, num_heads=4, num_layers=2,
        lookback=cfg["lookback"], horizon=cfg["horizon"],
        dropout=0.1, lora_rank=8, lora_alpha=16,
    ).to(device)
    criterion = MultiTaskLoss()
    try:
        from ranger21 import Ranger21
        optimizer = Ranger21(model.parameters(), lr=1e-3)
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loaders["train"]:
            x = batch["batch_x"].to(device).float()
            y = batch["batch_y"].to(device).float()
            optimizer.zero_grad()
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            min_t = min(pred.shape[1], y.shape[1])
            pred = pred[:, :min_t]
            y = y[:, :min_t]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch+1}/{args.epochs}: "
                  f"loss={epoch_loss/max(1,n_batches):.6f}", file=sys.stderr)
    training_time = time.time() - start_time

    # Compute conformal intervals
    print(f"\nComputing conformal intervals at alpha levels: {args.alpha_levels}",
          file=sys.stderr)
    result = compute_conformal_intervals(
        model, loaders["validation"], loaders["test"], device, args.alpha_levels,
    )
    result["domain"] = args.domain
    result["epochs"] = args.epochs
    result["training_time_s"] = training_time

    # Write CSV
    csv_path = out_dir / "coverage_table.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("alpha,nominal_coverage,empirical_coverage,coverage_gap,"
                "quantile,avg_interval_width,n_calibration,n_test,covered,"
                "within_tolerance\n")
        for r in result["per_alpha"]:
            f.write(f"{r['alpha']:.4f},{r['nominal_coverage']:.4f},"
                    f"{r['empirical_coverage']:.4f},{r['coverage_gap']:.4f},"
                    f"{r['quantile']:.6f},{r['avg_interval_width']:.6f},"
                    f"{r['n_calibration_samples']},{r['n_test_samples']},"
                    f"{r['covered_count']},{r['within_tolerance']}\n")

    # Write JSON summary
    json_path = out_dir / "coverage_summary.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults:", file=sys.stderr)
    print(f"  Training time: {training_time:.1f}s", file=sys.stderr)
    for r in result["per_alpha"]:
        status = "✓" if r["within_tolerance"] else "✗"
        print(f"  alpha={r['alpha']:.2f}: nominal={r['nominal_coverage']:.3f} "
              f"empirical={r['empirical_coverage']:.3f} "
              f"gap={r['coverage_gap']:+.4f} {status}", file=sys.stderr)
    print(f"\n  CSV:  {csv_path}", file=sys.stderr)
    print(f"  JSON: {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
