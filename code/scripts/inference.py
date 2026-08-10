#!/usr/bin/env python3
"""
inference.py — Load a trained MVGT-Net checkpoint and run inference.

Supports three modes:
  1. Single-record inference from a JSONL line
  2. Batch inference on a full JSONL file
  3. Random-record inference (loads a random record from the test split)

Usage:
  # Single record
  python scripts/inference.py \\
      --checkpoint checkpoints/Economy_Trade/best.pt \\
      --domain Economy_Trade \\
      --input data/TimeMMD/Economy_Trade/test.jsonl \\
      --record-index 0

  # Batch (full file)
  python scripts/inference.py \\
      --checkpoint checkpoints/Economy_Trade/best.pt \\
      --domain Economy_Trade \\
      --input data/TimeMMD/Economy_Trade/test.jsonl \\
      --batch

  # Random record
  python scripts/inference.py \\
      --checkpoint checkpoints/Climate_AQI/best.pt \\
      --domain Climate_AQI \\
      --input data/TimeMMD/Climate_AQI/test.jsonl \\
      --random

  # Save predictions to JSON
  python scripts/inference.py \\
      --checkpoint checkpoints/Economy_Trade/best.pt \\
      --domain Economy_Trade \\
      --input data/TimeMMD/Economy_Trade/test.jsonl \\
      --batch --output predictions.json

Output:
  Prints predictions (and ground truth if available) to stdout.
  If --output is given, writes a JSON file with per-record predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add the parent directory to sys.path so we can import mvgt_net
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mvgt_net import MVGTNet, DOMAIN_REGISTRY  # noqa: E402
from mvgt_net.data import TimeMMDDataset, fit_normalization  # noqa: E402
from mvgt_net.metrics import all_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MVGT-Net inference: load a checkpoint and predict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the trained checkpoint (.pt file).",
    )
    p.add_argument(
        "--domain", required=True, choices=list(DOMAIN_REGISTRY.keys()),
        help="TimeMMD domain name.",
    )
    p.add_argument(
        "--input", required=True, type=Path,
        help="Path to the input JSONL file (test or validation split).",
    )
    p.add_argument(
        "--record-index", type=int, default=None,
        help="Index of the record to predict (0-based). Mutually exclusive "
             "with --batch and --random.",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Run inference on every record in --input.",
    )
    p.add_argument(
        "--random", action="store_true",
        help="Run inference on a randomly-selected record from --input.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="If given, write predictions to this JSON file.",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --random mode.",
    )
    return p.parse_args()


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Dict[str, Any]:
    """Load a checkpoint and return its contents."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain "
            f"'model_state_dict'. Available keys: {list(ckpt.keys())}"
        )
    return ckpt


def build_model(domain: str, checkpoint: Dict[str, Any], device: torch.device) -> MVGTNet:
    """Reconstruct the MVGTNet model from the checkpoint config."""
    cfg = checkpoint.get("config", {})
    model_kwargs = dict(
        num_features=cfg.get("num_features", 1),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 2),
        lookback=DOMAIN_REGISTRY[domain]["lookback"],
        horizon=DOMAIN_REGISTRY[domain]["horizon"],
        dropout=cfg.get("dropout", 0.1),
        lora_rank=cfg.get("lora_rank", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        use_text=cfg.get("use_text", True),
        use_graph=cfg.get("use_graph", True),
    )
    model = MVGTNet(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_normalization(
    domain: str, checkpoint: Dict[str, Any], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Load the train-fit normalization statistics from the checkpoint."""
    norm = checkpoint.get("norm_stats", None)
    if norm is None:
        # Fall back to fitting on the train split (less ideal but works).
        train_path = Path(f"data/TimeMMD/{domain}/train.jsonl")
        if not train_path.exists():
            raise FileNotFoundError(
                f"No norm_stats in checkpoint and no train file at {train_path}"
            )
        norm = fit_normalization(train_path, domain)
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in norm.items()
    }


def predict_record(
    model: MVGTNet,
    record: Dict[str, Any],
    norm_stats: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Run inference on a single record and return predictions + metrics."""
    # Build input tensors from the record.
    x = torch.as_tensor(record["batch_x"], dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    elif x.ndim == 2:
        x = x.unsqueeze(0)  # (1, T, F)
    # Normalize
    mean = norm_stats.get("mean", torch.zeros(1, device=device))
    std = norm_stats.get("std", torch.ones(1, device=device)) + 1e-8
    x = (x - mean) / std

    # Forward pass
    with torch.no_grad():
        pred = model(x)
    if isinstance(pred, tuple):
        pred = pred[0]
    # Denormalize
    pred = pred * std + mean

    out: Dict[str, Any] = {
        "prediction_shape": list(pred.shape),
        "prediction_summary": {
            "mean": float(pred.mean().item()),
            "std": float(pred.std().item()),
            "min": float(pred.min().item()),
            "max": float(pred.max().item()),
        },
    }

    # If ground truth is present, compute metrics.
    if "batch_y" in record:
        y = torch.as_tensor(record["batch_y"], dtype=torch.float32, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(0).unsqueeze(-1)
        elif y.ndim == 2:
            y = y.unsqueeze(0)
        # Match shapes
        try:
            metrics = all_metrics(pred, y)
            out["ground_truth_summary"] = {
                "mean": float(y.mean().item()),
                "std": float(y.std().item()),
                "min": float(y.min().item()),
                "max": float(y.max().item()),
            }
            out["metrics"] = {k: float(v) for k, v in metrics.items()}
        except Exception as e:
            out["metrics_error"] = str(e)

    return out


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    # Modes are mutually exclusive.
    modes = [args.record_index is not None, args.batch, args.random]
    if sum(bool(m) for m in modes) != 1:
        print(
            "ERROR: exactly one of --record-index, --batch, --random must be set.",
            file=sys.stderr,
        )
        return 2

    # Load checkpoint
    print(f"[1/4] Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    ckpt = load_checkpoint(args.checkpoint, device)

    # Build model
    print(f"[2/4] Building model for domain: {args.domain}", file=sys.stderr)
    model = build_model(args.domain, ckpt, device)
    print(
        f"      Model params: {sum(p.numel() for p in model.parameters()):,}",
        file=sys.stderr,
    )

    # Load normalization
    print("[3/4] Loading normalization statistics", file=sys.stderr)
    norm_stats = load_normalization(args.domain, ckpt, device)

    # Load input records
    print(f"[4/4] Loading input records: {args.input}", file=sys.stderr)
    records: List[Dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"      Loaded {len(records)} records", file=sys.stderr)

    if args.random:
        rng = random.Random(args.seed)
        idx = rng.randrange(len(records))
        records = [records[idx]]
        print(f"      Selected random record index: {idx}", file=sys.stderr)
    elif args.record_index is not None:
        if args.record_index >= len(records):
            print(
                f"ERROR: --record-index {args.record_index} out of range "
                f"(file has {len(records)} records).",
                file=sys.stderr,
            )
            return 2
        records = [records[args.record_index]]

    # Run inference
    results: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        try:
            result = predict_record(model, rec, norm_stats, device)
            result["record_index"] = i if args.batch else (
                args.record_index if args.record_index is not None else "random"
            )
            results.append(result)
        except Exception as e:
            results.append({"record_index": i, "error": str(e)})

    # Output
    output_data = {
        "checkpoint": str(args.checkpoint),
        "domain": args.domain,
        "input_file": str(args.input),
        "device": str(device),
        "num_records_processed": len(results),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nPredictions written to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
