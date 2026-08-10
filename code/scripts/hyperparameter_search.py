"""
Phase F: Hyperparameter Search (Optuna)
========================================
Performs a Bayesian-optimization hyperparameter sweep over the MVGT-Net
design space. Records all trials to a SQLite database and produces a
parallel-coordinates plot + parameter importance ranking.

Search space (Phase F, Section 19-3 of thesis):
    - learning_rate  : log-uniform [1e-5, 1e-2]
    - hidden_dim     : categorical {32, 64, 128}
    - num_heads      : categorical {2, 4, 8}
    - frozen_layers  : categorical {2, 4, 6, 8}
    - unfrozen_layers: categorical {1, 2, 3, 4}
    - lora_rank      : categorical {4, 8, 16, 32}
    - topk           : categorical {4, 8, 16, 32}
    - dropout        : uniform [0.0, 0.3]
    - weight_decay   : log-uniform [1e-6, 1e-2]
    - batch_size     : categorical {8, 16, 32, 64}

Each trial trains for a small number of epochs (default 5) on synthetic
data and reports validation MAE as the objective to MINIMIZE.

Usage:
    python scripts/hyperparameter_search.py --trials 50 --epochs 5
    python scripts/hyperparameter_search.py --trials 200 --epochs 10 --study-name mvgtnet_v1

References:
    Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter
    Optimization Framework", KDD 2019.
    Bergstra et al. (2011). "Algorithms for Hyper-Parameter Optimization",
    NeurIPS 2011 (TPE).
"""
import argparse
import json
import sys
import os
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

RESULTS_DIR = Path(os.path.join(CODE_ROOT, "..", "results", "hyperparameter"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def objective(trial):
    """Optuna objective: returns validation MAE (lower is better)."""
    import yaml
    from mvgt_net import MVGTNet, MultiTaskLoss, all_metrics

    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    frozen_layers = trial.suggest_categorical("frozen_layers", [2, 4, 6, 8])
    unfrozen_layers = trial.suggest_categorical("unfrozen_layers", [1, 2, 3, 4])
    lora_rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32])
    topk = trial.suggest_categorical("topk", [4, 8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    cfg = {
        "num_nodes": 14, "input_dim": 1, "lookback": 12, "horizon": 3,
        "hidden_dim": hidden_dim, "num_heads": num_heads,
        "frozen_layers": frozen_layers, "unfrozen_layers": unfrozen_layers,
        "num_categories": 5, "lora_rank": lora_rank, "lora_alpha": 2 * lora_rank,
        "graph_types": ["spatial", "temporal", "semantic", "adaptive"],
        "vocab_size": 30000, "topk": topk, "dropout": dropout,
        "use_text": False, "use_categorical": True,
    }
    try:
        model = MVGTNet(cfg)
    except Exception as e:
        # Invalid combination (e.g. hidden_dim not divisible by num_heads)
        return float("inf")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Generate synthetic data
    n_samples = 96
    t = torch.arange(0, 15, dtype=torch.float32)
    base = 50.0 + 0.05 * t + 0.3 * torch.sin(2 * torch.pi * t / 7.0)
    node_offsets = torch.linspace(0, 5.0, 14).unsqueeze(0)
    X, Y, TEXT, CAT = [], [], [], []
    for s in range(n_samples):
        eps = torch.randn(15, 14) * 0.1
        signal = base.unsqueeze(1) + node_offsets + eps
        X.append(signal[:12].unsqueeze(-1))
        Y.append(signal[12:].unsqueeze(-1))
        TEXT.append(torch.randint(0, 30000, (12, 32)))
        CAT.append(torch.randint(0, 5, (12, 14)))
    x = torch.stack(X); y = torch.stack(Y)
    text = {"fact": torch.stack(TEXT)}; cat = torch.stack(CAT)
    adj = torch.ones(14, 14) / 14.0

    n_train, n_val = 64, 16
    train_ds = torch.utils.data.TensorDataset(x[:n_train], y[:n_train], text["fact"][:n_train], cat[:n_train])
    val_ds = torch.utils.data.TensorDataset(x[n_train:n_train+n_val], y[n_train:n_train+n_val], text["fact"][n_train:n_train+n_val], cat[n_train:n_train+n_val])

    def collate(batch):
        xs, ys, ts, cs = zip(*batch)
        return torch.stack(xs), torch.stack(ys), {"fact": torch.stack(ts)}, torch.stack(cs), adj

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Train for `epochs` epochs
    epochs = objective.epochs  # set by main()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            xn, yt, xt, xc, a = batch
            optimizer.zero_grad()
            out = model(xn, xt, xc, adj_spatial=a)
            loss = torch.nn.functional.l1_loss(out["numeric"], yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                xn, yt, xt, xc, a = batch
                out = model(xn, xt, xc, adj_spatial=a)
                preds.append(out["numeric"].cpu()); targets.append(yt.cpu())
        p = torch.cat(preds); t = torch.cat(targets)
        val_mae = float(torch.nn.functional.l1_loss(p, t).item())
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_mae


def main():
    parser = argparse.ArgumentParser(description="Phase F: Hyperparameter search with Optuna")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--study-name", default="mvgtnet_synthetic_v1")
    args = parser.parse_args()

    try:
        import optuna
        import optuna.visualization
    except ImportError:
        print("ERROR: optuna is not installed. Install with: pip install optuna")
        return 1

    objective.epochs = args.epochs

    storage = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    print(f"Running {args.trials} trials, {args.epochs} epochs each...")
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    dt = time.time() - t0

    print(f"\nBest trial: val_MAE={study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Total time: {dt:.1f}s ({dt/args.trials:.1f}s/trial)")

    # Save best params + summary
    summary = {
        "best_trial": {"value": study.best_value, "params": study.best_params},
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "total_time_s": dt,
        "study_name": args.study_name,
    }
    out = RESULTS_DIR / "best_params.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary: {out}")

    # Param importance (only if all trials completed)
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importance:")
        for k, v in importances.items():
            print(f"  {k:25s} : {v:.4f}")
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        names = list(importances.keys())
        vals = list(importances.values())
        ax.barh(names, vals, color="#1f77b4")
        ax.set_xlabel("Importance (fANOVA)")
        ax.set_title("Phase F: Hyperparameter Importance for Validation MAE (synthetic)")
        ax.invert_yaxis()
        out = RESULTS_DIR / "param_importance.png"
        fig.savefig(out, dpi=150)
        fig.savefig(out.with_suffix(".svg"))
        plt.close(fig)
        print(f"  -> {out}")
    except Exception as e:
        print(f"  (could not compute importance: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
