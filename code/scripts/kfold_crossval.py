#!/usr/bin/env python3
"""
kfold_crossval.py — K-fold cross-validation for MVGT-Net.

Splits the training data into K folds, trains K models (each holding out one
fold for validation), and reports the mean + std of test metrics across
folds. This strengthens statistical claims in thesis Chapter 10.

Usage:
  python scripts/kfold_crossval.py --domain Climate_AQI --k 5 --epochs 50
  python scripts/kfold_crossval.py --domain Economy_Trade --k 10 --epochs 100 \\
      --output-dir kfold_results/

Output:
  kfold_results/<domain>/kfold_table.csv
  kfold_results/<domain>/kfold_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mvgt_net import MVGTNet, MultiTaskLoss, DOMAIN_REGISTRY  # noqa: E402
from mvgt_net.data import TimeMMDDataset, fit_normalization, collate_fn  # noqa: E402
from mvgt_net.metrics import all_metrics  # noqa: E402
from torch.utils.data import DataLoader


def make_folds(n_samples: int, k: int, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Return K (train_indices, val_indices) tuples."""
    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))
    return folds


def train_one_fold(
    domain: str,
    train_idx: List[int],
    val_idx: List[int],
    epochs: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Train on K-1 folds and evaluate on the held-out fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = DOMAIN_REGISTRY[domain]
    train_path = CODE_ROOT / "data" / "TimeMMD" / domain / "train.jsonl"

    # Load full train dataset, then subset by indices
    full_dataset = TimeMMDDataset(
        jsonl_path=str(train_path),
        domain=domain,
        lookback=cfg["lookback"],
        horizon=cfg["horizon"],
    )
    norm_stats = fit_normalization(train_path, domain)
    full_dataset.norm_stats = norm_stats

    # Subset
    from torch.utils.data import Subset
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_subset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Build model
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

    history: List[Dict[str, float]] = []
    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
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
            train_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["batch_x"].to(device).float()
                y = batch["batch_y"].to(device).float()
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                min_t = min(pred.shape[1], y.shape[1])
                preds.append(pred[:, :min_t].cpu().numpy())
                targets.append(y[:, :min_t].cpu().numpy())
        if preds:
            vp = np.concatenate(preds, axis=0)
            vt = np.concatenate(targets, axis=0)
            val_metrics = all_metrics(
                torch.from_numpy(vp), torch.from_numpy(vt)
            )
            val_mae = val_metrics["MAE"]
        else:
            val_mae = float("nan")

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, n_batches),
            "val_MAE": val_mae,
        })
        print(f"  [fold] epoch {epoch+1}/{epochs}: "
              f"loss={train_loss/max(1,n_batches):.6f} val_MAE={val_mae:.6f}",
              file=sys.stderr)

    # Final val metrics
    final_metrics = {}
    if preds:
        final_metrics = {k: float(v) for k, v in val_metrics.items()}

    return {
        "best_val_MAE": best_val_mae,
        "final_val_metrics": final_metrics,
        "history": history,
        "training_time_s": time.time() - start_time,
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="Economy_Trade",
                   choices=list(DOMAIN_REGISTRY.keys()))
    p.add_argument("--k", type=int, default=5, help="Number of folds.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=CODE_ROOT / "kfold_results")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full train to get sample count
    train_path = CODE_ROOT / "data" / "TimeMMD" / args.domain / "train.jsonl"
    with train_path.open("r", encoding="utf-8") as f:
        n_samples = sum(1 for line in f if line.strip())
    print(f"Domain: {args.domain}, train samples: {n_samples}, K: {args.k}",
          file=sys.stderr)

    if n_samples < args.k:
        print(f"ERROR: K={args.k} > n_samples={n_samples}", file=sys.stderr)
        return 1

    folds = make_folds(n_samples, args.k, args.seed)

    fold_results: List[Dict[str, Any]] = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_i+1}/{args.k} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ===",
              file=sys.stderr)
        result = train_one_fold(
            args.domain, train_idx, val_idx, args.epochs, device,
            seed=args.seed + fold_i,
        )
        result["fold"] = fold_i + 1
        fold_results.append(result)

    # Aggregate
    final_maes = [r["final_val_metrics"].get("MAE", float("nan"))
                  for r in fold_results]
    final_rmses = [r["final_val_metrics"].get("RMSE", float("nan"))
                   for r in fold_results]

    # Write CSV
    csv_path = out_dir / "kfold_table.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("fold,n_train,n_val,best_val_MAE,final_val_MAE,final_val_RMSE,"
                "training_time_s\n")
        for r in fold_results:
            f.write(f"{r['fold']},{r['n_train_samples']},{r['n_val_samples']},"
                    f"{r['best_val_MAE']:.6f},"
                    f"{r['final_val_metrics'].get('MAE', float('nan')):.6f},"
                    f"{r['final_val_metrics'].get('RMSE', float('nan')):.6f},"
                    f"{r['training_time_s']:.1f}\n")

    # Write JSON summary
    summary = {
        "domain": args.domain,
        "k": args.k,
        "epochs": args.epochs,
        "device": args.device,
        "n_train_total": n_samples,
        "mean_final_MAE": float(np.nanmean(final_maes)),
        "std_final_MAE": float(np.nanstd(final_maes, ddof=1)) if len(final_maes) > 1 else 0.0,
        "mean_final_RMSE": float(np.nanmean(final_rmses)),
        "std_final_RMSE": float(np.nanstd(final_rmses, ddof=1)) if len(final_rmses) > 1 else 0.0,
        "fold_results": fold_results,
    }
    json_path = out_dir / "kfold_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults:", file=sys.stderr)
    print(f"  Mean MAE:  {summary['mean_final_MAE']:.6f} "
          f"± {summary['std_final_MAE']:.6f}", file=sys.stderr)
    print(f"  Mean RMSE: {summary['mean_final_RMSE']:.6f} "
          f"± {summary['std_final_RMSE']:.6f}", file=sys.stderr)
    print(f"  CSV:       {csv_path}", file=sys.stderr)
    print(f"  JSON:      {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
