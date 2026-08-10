#!/usr/bin/env python3
"""experiment_tracker.py — Lightweight experiment tracking integration.

Provides a single ExperimentTracker class that writes structured run logs
to disk in JSON format, with optional TensorBoard and Weights & Biases (W&B)
integration when those packages are available.

Usage in training scripts:

    from mvgt_net.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(
        experiment_name="mvgt_net_economy_trade",
        run_name="lora_r8_bs32_lr1e-3",
        config=cfg,                       # any dict-like config object
        output_dir="logs/experiments",
        backend="auto",                   # "auto" | "tensorboard" | "wandb" | "json"
    )
    tracker.log_params({
        "lora_rank": 8, "lora_alpha": 16, "batch_size": 32,
        "lr": 1e-3, "max_epochs": 100, "warmup": 5,
    })
    for epoch in range(max_epochs):
        # ... training loop ...
        tracker.log_metrics({"train/loss": loss, "val/mae": mae}, step=epoch)
    tracker.finish()

Outputs:
    logs/experiments/<experiment_name>/<run_name>_<timestamp>/
        experiment.json   — full params + metrics history
        metrics.csv       — flat CSV of all logged metrics
        (tensorboard/)    — TensorBoard event files if backend includes tb
        (wandb run)       — W&B online run if backend includes wandb
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Multi-backend experiment tracker (JSON / TensorBoard / W&B)."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        config: dict | None = None,
        output_dir: str | Path = "logs/experiments",
        backend: str = "auto",
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.config = config or {}
        self.backend = backend

        # Create run directory
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(output_dir) / experiment_name / f"{run_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.params: dict[str, Any] = {}
        self.metrics_history: list[dict[str, Any]] = []
        self.start_time = time.time()

        # Backends
        self._tb = None
        self._wandb = None
        self._init_backends()

        # Write initial experiment.json
        self._write_experiment_json()

    def _init_backends(self) -> None:
        backends = [self.backend] if self.backend != "auto" else ["tensorboard", "wandb", "json"]
        for b in backends:
            if b == "tensorboard":
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    tb_dir = self.run_dir / "tensorboard"
                    tb_dir.mkdir(exist_ok=True)
                    self._tb = SummaryWriter(log_dir=str(tb_dir))
                    print(f"[ExperimentTracker] TensorBoard logging to {tb_dir}")
                except ImportError:
                    if self.backend == "tensorboard":
                        print("[ExperimentTracker] WARNING: tensorboard not available (pip install tensorboard)", file=sys.stderr)
            elif b == "wandb":
                try:
                    import wandb
                    self._wandb = wandb.init(
                        project=self.experiment_name,
                        name=self.run_name,
                        config=self.config,
                        dir=str(self.run_dir),
                    )
                    print(f"[ExperimentTracker] W&B run: {self._wandb.url}")
                except ImportError:
                    if self.backend == "wandb":
                        print("[ExperimentTracker] WARNING: wandb not available (pip install wandb)", file=sys.stderr)
                except Exception as e:
                    print(f"[ExperimentTracker] WARNING: W&B init failed: {e}", file=sys.stderr)

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        self._write_experiment_json()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        entry = {"step": step if step is not None else len(self.metrics_history), "timestamp": time.time()}
        entry.update(metrics)
        self.metrics_history.append(entry)

        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, entry["step"])

        if self._wandb is not None:
            self._wandb.log(metrics, step=entry["step"])

        # Persist to CSV incrementally
        self._append_metrics_csv(entry)
        self._write_experiment_json()

    def _write_experiment_json(self) -> None:
        data = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "config": self.config,
            "params": self.params,
            "start_time": self.start_time,
            "metrics_count": len(self.metrics_history),
            "last_metrics": self.metrics_history[-1] if self.metrics_history else None,
            "run_dir": str(self.run_dir),
        }
        with open(self.run_dir / "experiment.json", "w") as fh:
            json.dump(data, fh, indent=2)

    def _append_metrics_csv(self, entry: dict[str, Any]) -> None:
        csv_path = self.run_dir / "metrics.csv"
        file_exists = csv_path.exists()
        # Collect all keys across history for consistent columns
        all_keys: list[str] = []
        seen = set()
        for m in self.metrics_history + [entry]:
            for k in m:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: entry.get(k, "") for k in all_keys})

    def finish(self, status: str = "completed") -> None:
        elapsed = time.time() - self.start_time
        summary = {
            "status": status,
            "elapsed_seconds": elapsed,
            "final_metrics": self.metrics_history[-1] if self.metrics_history else None,
            "total_steps": len(self.metrics_history),
        }
        with open(self.run_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
        print(f"[ExperimentTracker] Run '{self.run_name}' {status} in {elapsed:.1f}s. Artifacts: {self.run_dir}")

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish(status="failed" if exc_type else "completed")


if __name__ == "__main__":
    # Smoke test
    with ExperimentTracker("smoke_test", "demo_run", backend="json") as t:
        t.log_params({"lr": 1e-3, "batch_size": 32})
        for step in range(5):
            t.log_metrics({"train/loss": 1.0 / (step + 1), "val/mae": 0.5 + step * 0.01}, step=step)
    print("[OK] ExperimentTracker smoke test passed.")
