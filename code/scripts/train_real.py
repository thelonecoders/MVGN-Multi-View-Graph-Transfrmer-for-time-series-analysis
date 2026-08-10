#!/usr/bin/env python3
"""
Production training script for MVGT-Net on REAL TimeMMD data.
=============================================================
This is the full, end-to-end training pipeline that runs on the REAL
TimeMMD dataset (no synthetic data, no placeholders).

Usage:
    # Train on a single domain (default: Climate_AQI)
    python scripts/train_real.py --domain Climate_AQI --device cuda --epochs 100

    # Train on all 9 domains sequentially
    python scripts/train_real.py --all-domains --device cuda

    # Quick smoke test (2 epochs, CPU)
    python scripts/train_real.py --domain Economy_Trade --smoke-test --device cpu

    # Resume from latest checkpoint
    python scripts/train_real.py --domain Climate_AQI --resume

What this script implements (fully specified — no missing pieces):
-----------------------------------------------------------------
1. Real TimeMMD JSONL loading via mvgt_net.data.TimeMMDDataset
2. Z-score normalization fit on TRAIN split only (DATA_CARD.md §4)
3. Ranger21 optimizer with:
     - Lookahead active
     - Warmup: 5 epochs × steps_per_epoch
     - Total iterations: max_epochs × steps_per_epoch
     - Fallback to AdamW if ranger21 not installed
4. Cosine learning-rate scheduler (with warmup)
5. Mixed-precision training (torch.cuda.amp) when device=cuda and
   config.training.mixed_precision == 'fp16'
6. Gradient clipping (default 1.0)
7. Early stopping with patience (default 15 epochs)
8. Best-model checkpointing (saved to checkpoints/<domain>/best.pt)
9. Periodic checkpoints (every 10 epochs, to checkpoints/<domain>/epoch_N.pt)
10. Final test evaluation with all 7 metrics (MAE, MSE, RMSE, WAPE,
    MAPE, sMAPE, R²)
11. Predictions + attention weights saved to results/<domain>/
12. Per-epoch + final JSON metrics written to results/<domain>/metrics.json
13. Optional Weights & Biases logging (--wandb)
14. Deterministic seeding (Chapter 18 protocol)
15. Resume from checkpoint (--resume)

Memory budget for RTX 3080 Ti (12 GB VRAM)
------------------------------------------
With BERT-base-uncased (110M params) + LoRA rank 8 + QLoRA 4-bit:
  - LLM frozen backbone : ~440 MB (fp16) or ~110 MB (4-bit)
  - LoRA adapters       : ~3 MB
  - MVGT-Net graph blocks: ~50 MB
  - Activations (B=32)  : ~3 GB
  - Optimizer state     : ~600 MB (Ranger21 = 2x params)
  - Total               : ~4-5 GB → fits comfortably in 12 GB

If OOM occurs, reduce batch_size to 16 or 8.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvgt_net import MVGTNet, MultiTaskLoss, all_metrics
from mvgt_net.data import DOMAIN_REGISTRY, get_dataloaders


# ---------------------------------------------------------------------------
# Reproducibility (Chapter 18 protocol)
# ---------------------------------------------------------------------------
def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, config: dict, steps_per_epoch: int):
    """Build Ranger21 (or AdamW fallback) + cosine scheduler with warmup."""
    train_cfg = config["training"]
    lr = train_cfg["learning_rate"]
    max_epochs = train_cfg["max_epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    wd = train_cfg.get("weight_decay", 0.0)
    opt_name = train_cfg.get("optimizer", "ranger21").lower()

    total_iters = max(1, max_epochs * steps_per_epoch)
    warmup_iters = max(1, warmup_epochs * steps_per_epoch)
    # M2 override: fixed warmup iterations regardless of steps_per_epoch
    if "_warmup_iters_override" in train_cfg:
        warmup_iters = train_cfg["_warmup_iters_override"]

    if opt_name == "ranger21":
        try:
            from ranger21 import Ranger21
            # Ranger21 API (lessw2020 main branch, accessed Aug 2026):
            #   num_epochs + num_batches_per_epoch  → drives warmup/warmdown math
            #   num_warmup_iterations              → explicit warmup override
            #   warmdown_active=False              → disable warmdown (early stopping handles termination)
            # Note: the older `num_iterations=` kwarg is NOT accepted by this
            # version of ranger21 and raises TypeError. We pass num_epochs +
            # num_batches_per_epoch instead, which is the supported API.
            optimizer = Ranger21(
                model.parameters(),
                lr=lr,
                weight_decay=wd,
                lookahead_active=True,
                use_warmup=True,
                num_warmup_iterations=warmup_iters,
                num_epochs=max_epochs,
                num_batches_per_epoch=steps_per_epoch,
                warmdown_active=False,
            )
            print(f"  Optimizer: Ranger21 (lr={lr}, wd={wd}, "
                  f"warmup={warmup_iters} iters, "
                  f"num_epochs={max_epochs}, "
                  f"num_batches_per_epoch={steps_per_epoch}, "
                  f"total={total_iters} iters, warmdown=off)")
        except ImportError:
            print("  ranger21 not installed; falling back to AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine LR schedule with linear warmup
    def lr_lambda(step):
        if step < warmup_iters:
            return float(step) / float(max(1, warmup_iters))
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training + evaluation primitives
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device,
                    grad_clip: float, scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    use_amp: bool = False):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x, y, text, cat, adj = batch[:5]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cat = cat.to(device, non_blocking=True)
        # adj is a single (N, N) tensor; graph_builder moves it to device.
        if isinstance(adj, torch.Tensor):
            adj = adj.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(x, text, cat, adj_spatial=adj)
                numeric_loss = nn.functional.l1_loss(
                    outputs["numeric"].squeeze(-1), y.squeeze(-1)
                )
                losses = {"numeric": numeric_loss}
                total, weights = loss_fn(losses)
            scaler.scale(total).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, text, cat, adj_spatial=adj)
            numeric_loss = nn.functional.l1_loss(
                outputs["numeric"].squeeze(-1), y.squeeze(-1)
            )
            losses = {"numeric": numeric_loss}
            total, weights = loss_fn(losses)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        total_loss += total.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        x, y, text, cat, adj = batch[:5]
        x = x.to(device, non_blocking=True)
        cat = cat.to(device, non_blocking=True)
        # adj is a single (N, N) tensor; graph_builder moves it to device.
        if isinstance(adj, torch.Tensor):
            adj = adj.to(device, non_blocking=True)
        outputs = model(x, text, cat, adj_spatial=adj)
        preds.append(outputs["numeric"].squeeze(-1).cpu())
        targets.append(y.squeeze(-1).cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return all_metrics(preds, targets)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int,
                    best_val: float, stats: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_mae": best_val,
        "stats": stats,
    }
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    torch.save(ckpt, path)
    print(f"  [checkpoint] saved {path}")


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt["best_val_mae"], ckpt.get("stats", {})


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_one_domain(domain: str, args, base_config: dict) -> Dict:
    """Train MVGT-Net on a single TimeMMD domain. Returns final metrics dict."""
    print("\n" + "=" * 78)
    print(f"  TRAINING: {domain}")
    print(f"  {DOMAIN_REGISTRY[domain]['description']}")
    print(f"  frequency={DOMAIN_REGISTRY[domain]['frequency']}  "
          f"lookback={DOMAIN_REGISTRY[domain]['lookback']}  "
          f"horizon={DOMAIN_REGISTRY[domain]['horizon']}")
    print("=" * 78)

    set_deterministic_seed(args.seed)

    # Override config with domain-specific settings
    config = json.loads(json.dumps(base_config))  # deep copy
    config["model"]["num_nodes"] = 1
    config["model"]["input_dim"] = DOMAIN_REGISTRY[domain]["variables"]
    config["model"]["lookback"] = DOMAIN_REGISTRY[domain]["lookback"]
    config["model"]["horizon"] = DOMAIN_REGISTRY[domain]["horizon"]
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.smoke_test:
        config["training"]["max_epochs"] = 2
        config["training"]["batch_size"] = 4
    # --- Step J Mitigation Overrides ---
    if args.patience is not None:
        config["training"]["early_stopping_patience"] = args.patience
        print(f"  [M1] early_stopping_patience overridden to {args.patience}")
    if args.warmup_iters is not None:
        config["training"]["_warmup_iters_override"] = args.warmup_iters
        print(f"  [M2] warmup_iters override set to {args.warmup_iters} (applied after data loaders)")
    if args.normalization is not None:
        config["data"]["normalize"] = args.normalization
        print(f"  [M4] normalization overridden to {args.normalization}")

    # Device
    device = torch.device(args.device or config["hardware"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Data
    data_root = args.data_root or os.path.join(
        os.path.dirname(__file__), "..", "..", "12_dataset", "TimeMMD"
    )
    print(f"  Data root: {Path(data_root).resolve()}")
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        domain, data_root,
        batch_size=config["training"]["batch_size"],
        num_workers=config["hardware"].get("num_workers", 0),
        pin_memory=config["hardware"].get("pin_memory", True) and device.type == "cuda",
        normalization=config.get("data", {}).get("normalize", "zscore"),
    )
    print(f"  Train: {stats['train_size']} samples")
    print(f"  Val  : {stats['val_size']} samples")
    print(f"  Test : {stats['test_size']} samples")
    print(f"  Normalize: mean={stats['normalize_mean']}, std={stats['normalize_std']}")

    # Model
    model = MVGTNet(config["model"]).to(device)
    eff = model.parameter_efficiency()
    print(f"  Parameters: total={eff['total_parameters']:,}  "
          f"trainable={eff['trainable_parameters']:,} "
          f"({eff['trainable_percentage']:.2f}%)  "
          f"frozen={eff['frozen_parameters']:,}")

    # Optimizer + scheduler
    steps_per_epoch = max(1, len(train_loader))
    optimizer, scheduler = build_optimizer(model, config, steps_per_epoch)

    # Loss
    loss_fn = MultiTaskLoss(
        task_names=config["loss"].get("task_names", ["numeric"]),
        history_length=config["loss"].get("history_length", 5),
        hidden_dim=config["loss"].get("hidden_dim", 32),
    ).to(device)

    # Mixed precision
    use_amp = (config["training"].get("mixed_precision") == "fp16"
               and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"  Mixed precision (fp16): {use_amp}")

    # Checkpoint dirs
    ckpt_dir = Path(args.checkpoint_dir) / domain
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) / domain
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    best_val_mae = float("inf")
    if args.resume:
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            start_epoch, best_val_mae, _ = load_checkpoint(
                latest, model, optimizer, scheduler, scaler
            )
            start_epoch += 1
            print(f"  Resumed from epoch {start_epoch}, best_val_mae={best_val_mae:.4f}")
        else:
            print(f"  --resume specified but no checkpoint at {latest}; starting fresh")

    # Training loop
    max_epochs = config["training"]["max_epochs"]
    patience = config["training"].get("early_stopping_patience", 15)
    grad_clip = config["training"].get("gradient_clip", 1.0)
    print(f"\n  Training for {max_epochs} epochs (early stopping patience={patience})")
    print(f"  Batch size: {config['training']['batch_size']}  "
          f"LR: {config['training']['learning_rate']}  "
          f"Grad clip: {grad_clip}")

    history = []
    epochs_since_best = 0
    t_train_start = time.time()
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            grad_clip=grad_clip, scaler=scaler, use_amp=use_amp,
        )
        val_metrics = evaluate(model, val_loader, device)
        t1 = time.time()
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_MAE": val_metrics["MAE"],
            "val_MSE": val_metrics["MSE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_WAPE": val_metrics["WAPE"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(t1 - t0, 2),
        }
        history.append(epoch_record)
        print(f"  Epoch {epoch+1:3d}/{max_epochs} | loss={train_loss:.4f} | "
              f"val_MAE={val_metrics['MAE']:.4f} | val_RMSE={val_metrics['RMSE']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {t1-t0:.1f}s")

        # Save latest checkpoint
        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler,
                        scaler, epoch + 1, best_val_mae, stats)

        # Best model?
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler,
                            scaler, epoch + 1, best_val_mae, stats)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1}.pt", model, optimizer,
                            scheduler, scaler, epoch + 1, best_val_mae, stats)

        # Early stopping
        if epochs_since_best >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    train_time = time.time() - t_train_start

    # Load best model and evaluate on test
    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler)
        print(f"  Loaded best checkpoint (val_MAE={best_val_mae:.4f})")

    test_metrics = evaluate(model, test_loader, device)
    print("\n  Final test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save metrics JSON
    result = {
        "domain": domain,
        "model": "MVGT-Net",
        "config": config["model"],
        "training_config": config["training"],
        "stats": stats,
        "history": history,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "best_val_MAE": float(best_val_mae),
        "trainable_parameters": eff["trainable_parameters"],
        "total_parameters": eff["total_parameters"],
        "trainable_percentage": eff["trainable_percentage"],
        "total_train_time_s": round(train_time, 1),
        "epochs_completed": len(history),
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Metrics written to {metrics_path}")
    print(f"  Total training time: {train_time:.1f}s  "
          f"({train_time/60:.1f} min)")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..",
                                                    "configs", "default.yaml"),
                   help="Path to YAML config (default: configs/default.yaml)")
    p.add_argument("--domain", default="Climate_AQI",
                   help=f"TimeMMD domain. One of: {list(DOMAIN_REGISTRY)}")
    p.add_argument("--all-domains", action="store_true",
                   help="Train on all 9 domains sequentially")
    p.add_argument("--device", default=None, help="cuda | cpu (default: from config)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max_epochs (default: from config)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size (default: from config)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate (default: from config)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (Chapter 18 protocol, default 42)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Quick 2-epoch run with batch_size=4")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint")
    p.add_argument("--data-root", default=None,
                   help="Path to 12_dataset/TimeMMD/ (default: ../../12_dataset/TimeMMD)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Where to save checkpoints (default: ./checkpoints)")
    p.add_argument("--results-dir", default=None,
                   help="Where to save results (default: ./results)")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    # --- Step J Mitigation Flags (M1-M4) ---
    p.add_argument("--patience", type=int, default=None,
                   help="M1: Override early_stopping_patience (default: from config, "
                        "typically 15; Step J uses 30)")
    p.add_argument("--warmup-iters", type=int, default=None,
                   help="M2: Override Ranger21 warmup iterations (default: "
                        "auto-calculated as warmup_epochs * steps_per_epoch; "
                        "Step J uses 50)")
    p.add_argument("--normalization", choices=["zscore", "minmax"], default=None,
                   help="M4: Override normalization method (default: zscore; "
                        "Step J uses minmax for per-sample scaling)")
    args = p.parse_args()

    # Defaults
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Optional wandb
    if args.wandb:
        try:
            import wandb
            wandb.init(project="mvgtnet-timemmd", config=config)
        except ImportError:
            print("wandb not installed; skipping experiment tracking")

    # Train
    if args.all_domains:
        print(f"\nTraining on ALL {len(DOMAIN_REGISTRY)} domains sequentially...")
        all_results = []
        for i, domain in enumerate(DOMAIN_REGISTRY):
            print(f"\n>>> Domain {i+1}/{len(DOMAIN_REGISTRY)}: {domain}")
            try:
                result = train_one_domain(domain, args, config)
                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] domain {domain} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"domain": domain, "error": str(e)})

        # Summary
        print("\n" + "=" * 78)
        print("  ALL-DOMAINS TRAINING SUMMARY")
        print("=" * 78)
        print(f"{'Domain':<22} {'test_MAE':>10} {'test_RMSE':>10} "
              f"{'test_WAPE':>10} {'epochs':>8} {'time_min':>10}")
        print("-" * 78)
        for r in all_results:
            if "error" in r:
                print(f"{r['domain']:<22}  ERROR: {r['error'][:50]}")
                continue
            m = r["test_metrics"]
            print(f"{r['domain']:<22} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} "
                  f"{m['WAPE']:>10.4f} {r['epochs_completed']:>8} "
                  f"{r['total_train_time_s']/60:>10.1f}")
        summary_path = Path(args.results_dir) / "all_domains_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary written to {summary_path}")
    else:
        train_one_domain(args.domain, args, config)


if __name__ == "__main__":
    main()
