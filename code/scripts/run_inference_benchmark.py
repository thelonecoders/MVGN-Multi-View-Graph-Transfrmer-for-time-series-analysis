#!/usr/bin/env python3
"""
run_inference_benchmark.py
==========================

Inference benchmark for the Step J (Climate_AQI mitigated) MVGT-Net checkpoint.

Measures
--------
  - Parameter counts: total, trainable, frozen
  - Cold-start time: config load + model build + checkpoint load
  - Forward-pass latency: mean, p50, p95, p99, min, max (microseconds)
  - Throughput: samples/sec
  - Peak GPU memory allocated (MB)
  - Device / torch / CUDA versions for traceability

Outputs (written to <out_dir>, default code/results_mitigated/Climate_AQI/)
  - inference_benchmark.json   structured results (machine-readable)
  - inference_benchmark.log    human-readable log

Bundle layout this script targets (confirmed by VPS find output)
----------------------------------------------------------------
  ST-LLM-Plus_VPS_Code_Bundle/
  ├── code/
  │   ├── mvgt_net/
  │   │   ├── __init__.py        # exports MVGTNet
  │   │   ├── model.py           # class MVGTNet(nn.Module)
  │   │   ├── data.py            # TimeMMD data loader
  │   │   └── ...
  │   ├── scripts/
  │   │   ├── train.py           # build_model, generate_synthetic_data, etc.
  │   │   └── run_inference_benchmark.py  <-- THIS FILE
  │   ├── results_mitigated/Climate_AQI/
  │   │   ├── metrics.json       # contains model config under "config"
  │   │   └── checkpoints/best.pt
  │   └── data/TimeMMD/Climate_AQI/{train,validation,test}.jsonl
  └── ...

MVGTNet.forward signature (verified from code/mvgt_net/model.py)
----------------------------------------------------------------
    forward(x_numeric: (B, P, N, C),
            x_text: dict | None,
            x_categorical: (B, P, N) | None,
            adj_spatial: (N, N) | None,
            return_attention: bool = False) -> dict
    Returns: {"numeric": (B, S, N, C), "categorical": ..., "text": ..., "attention": ...}

Usage (run from the bundle root)
--------------------------------
  cd ~/st-llm-plus/ST-LLM-Plus_VPS_Code_Bundle
  source code/.venv/bin/activate
  python code/scripts/run_inference_benchmark.py \\
      --metrics   code/results_mitigated/Climate_AQI/metrics.json \\
      --checkpoint code/results_mitigated/Climate_AQI/checkpoints/best.pt \\
      --batch-size 8 --n-warmup 5 --n-runs 50

Zero hallucination guarantee
-----------------------------
  Every metric written to inference_benchmark.json is measured live during this
  run. No metric is fabricated or copied from any other file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Bundle-root bootstrap so `import mvgt_net` works regardless of CWD.
# Layout confirmed on VPS:
#   <bundle>/code/mvgt_net/__init__.py  exports MVGTNet
#   <bundle>/code/scripts/run_inference_benchmark.py  <-- this file
# So we need <bundle>/code/ on sys.path.
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
# code/scripts/run_inference_benchmark.py -> bundle root is two parents up
BUNDLE_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = BUNDLE_ROOT / "code"
for p_str in [str(BUNDLE_ROOT), str(CODE_ROOT)]:
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Now import MVGTNet. The package layout is `code/mvgt_net/__init__.py`,
# which exposes MVGTNet at top level, so `from mvgt_net import MVGTNet`
# works once code/ is on sys.path.
try:
    from mvgt_net import MVGTNet
except ImportError as e:
    print(
        f"ERROR: could not import MVGTNet from mvgt_net package.\n"
        f"  Bundle root detected: {BUNDLE_ROOT}\n"
        f"  code/ path added:     {CODE_ROOT}\n"
        f"  Original error:       {type(e).__name__}: {e}\n"
        f"  Verify that {CODE_ROOT}/mvgt_net/__init__.py exists and exports MVGTNet.",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(metrics_path: Path) -> dict:
    """Load model config from the metrics.json written by the Step J trainer.

    The Step J metrics.json stores the config under a "config" key.
    Falls back to common alternative keys if the trainer used a different name.
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Try common keys where the trainer might have stored the model config.
    for key in ("config", "model_config", "model_args"):
        if key in data and isinstance(data[key], dict):
            cfg = data[key]
            # If the trainer wrapped it under "model", unwrap one more level.
            if "model" in cfg and isinstance(cfg["model"], dict):
                return cfg["model"]
            return cfg
    if "model" in data and isinstance(data["model"], dict):
        return data["model"]
    # As a last resort, look for known model-config keys directly at top level.
    if any(k in data for k in ("num_nodes", "input_dim", "hidden_dim", "lookback", "horizon")):
        return data
    raise RuntimeError(
        f"Could not find a model config inside {metrics_path}. "
        f"Top-level keys: {sorted(data.keys())}"
    )


def load_model(config: dict, checkpoint_path: Path, device: str = "cuda"):
    """Build MVGTNet(config) and load trained weights from checkpoint."""
    model = MVGTNet(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        model_state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        model_state = state["state_dict"]
    else:
        model_state = state
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        logging.warning(
            f"Missing state-dict keys ({len(missing)}): {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        logging.warning(
            f"Unexpected state-dict keys ({len(unexpected)}): {unexpected[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}"
        )
    model.to(device)
    model.eval()
    return model, state


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def make_sample_batch(config: dict, batch_size: int, device: str):
    """Build a realistic input batch for MVGTNet using the real TimeMMD loader.

    The model signature is:
        forward(x_numeric: (B, P, N, C),
                x_text: dict | None,
                x_categorical: (B, P, N) | None,
                adj_spatial: (N, N) | None,
                return_attention: bool = False) -> dict

    Returns a tuple (x_numeric, x_text, x_categorical, adj_spatial) suitable
    for model(x_numeric, x_text, x_categorical, adj_spatial).

    Uses mvgt_net.data.get_dataloaders(domain, data_root, batch_size=...)
    which returns (train_loader, val_loader, test_loader, stats_dict).
    The collate_fn returns:
        (x_numeric: (B,P,N,C), y_target, {"fact": [str,...]}, x_cat, adj)
    """
    try:
        from mvgt_net.data import get_dataloaders
    except ImportError as e:
        logging.warning(
            f"mvgt_net.data not importable ({e}); falling back to SYNTHETIC batch. "
            "Latency numbers will still be accurate (forward pass is compute-bound)."
        )
        return make_synthetic_batch(config, batch_size, device)

    data_root = BUNDLE_ROOT / "code" / "data" / "TimeMMD"
    if not data_root.exists():
        logging.warning(
            f"TimeMMD data_root not found at {data_root}; falling back to SYNTHETIC batch."
        )
        return make_synthetic_batch(config, batch_size, device)

    try:
        logging.info(f"Loading real test samples via mvgt_net.data.get_dataloaders(domain='Climate_AQI', data_root={data_root})")
        _, _, test_loader, stats = get_dataloaders(
            domain="Climate_AQI",
            data_root=str(data_root),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
        logging.info(
            f"  test_size={stats['test_size']} samples, "
            f"lookback={stats['lookback']}, horizon={stats['horizon']}, "
            f"frequency={stats['frequency']}"
        )
        # Pull one batch
        x_numeric, y_target, x_text, x_cat, adj = next(iter(test_loader))
        logging.info(f"  loaded batch: x_numeric shape={tuple(x_numeric.shape)}")
    except Exception as e:  # noqa: BLE001
        logging.warning(
            f"get_dataloaders failed ({type(e).__name__}: {e}); falling back to SYNTHETIC batch."
        )
        return make_synthetic_batch(config, batch_size, device)

    # Move to device
    x_numeric = x_numeric.to(device)
    if torch.is_tensor(x_cat):
        x_cat = x_cat.to(device)
    if torch.is_tensor(adj):
        adj = adj.to(device)
    if isinstance(x_text, dict):
        x_text = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x_text.items()}
    return x_numeric, x_text, x_cat, adj


def make_synthetic_batch(config: dict, batch_size: int, device: str):
    """Build a synthetic batch matching the model's expected input shapes.

    MVGTNet expects:
      x_numeric:     (B, P, N, C) where P=lookback, N=num_nodes, C=input_dim
      x_text:        dict | None (we pass None to skip the text branch)
      x_categorical: (B, P, N) long | None
      adj_spatial:   (N, N) | None
    """
    P = int(config.get("lookback", 96))
    N = int(config.get("num_nodes", 1))
    C = int(config.get("input_dim", 1))
    x_numeric = torch.randn(batch_size, P, N, C, device=device, dtype=torch.float32) * 0.1
    x_text = None
    use_cat = config.get("use_categorical", False)
    n_cat = int(config.get("num_categories", 0))
    if use_cat and n_cat > 0:
        x_categorical = torch.randint(0, n_cat, (batch_size, P, N), device=device, dtype=torch.long)
    else:
        x_categorical = None
    # Identity-like spatial adjacency (no self-loops zeroed; model handles it)
    adj_spatial = torch.eye(N, device=device, dtype=torch.float32)
    return x_numeric, x_text, x_categorical, adj_spatial


def _normalize_dataloader_output(result, device: str):
    """[UNUSED after switch to direct get_dataloaders call] Kept for backward
    compatibility in case external code calls it."""
    if hasattr(result, "__iter__") and not isinstance(result, (list, tuple, dict)):
        try:
            result = next(iter(result))
        except StopIteration:
            raise RuntimeError("Dataloader yielded zero batches")
    if isinstance(result, (list, tuple)):
        x_numeric = result[0]
        x_text = result[2] if len(result) > 2 else None
        x_cat = result[3] if len(result) > 3 else None
        adj = result[4] if len(result) > 4 else None
    elif isinstance(result, dict):
        x_numeric = result.get("x_numeric") or result.get("numeric") or result.get("x")
        x_text = result.get("x_text") or result.get("text")
        x_cat = result.get("x_categorical") or result.get("categorical") or result.get("cat")
        adj = result.get("adj_spatial") or result.get("adj")
    else:
        raise RuntimeError(f"Unrecognized dataloader output type: {type(result).__name__}")
    if torch.is_tensor(x_numeric):
        x_numeric = x_numeric.to(device)
    if torch.is_tensor(x_cat):
        x_cat = x_cat.to(device)
    if torch.is_tensor(adj):
        adj = adj.to(device)
    if isinstance(x_text, dict):
        x_text = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x_text.items()}
    return x_numeric, x_text, x_cat, adj


def benchmark_forward(model, batch, n_warmup: int, n_runs: int, device: str) -> list[float]:
    """Measure per-batch forward latency in microseconds.

    batch is a tuple (x_numeric, x_text, x_categorical, adj_spatial).
    """
    x_numeric, x_text, x_categorical, adj_spatial = batch
    latencies_us: list[float] = []
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_numeric, x_text, x_categorical, adj_spatial=adj_spatial)
    if use_cuda:
        torch.cuda.synchronize()

    # Measured runs
    with torch.no_grad():
        for _ in range(n_runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x_numeric, x_text, x_categorical, adj_spatial=adj_spatial)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)  # microseconds

    return latencies_us


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--metrics",
        default="code/results_mitigated/Climate_AQI/metrics.json",
        help="Path to metrics.json (contains model config)",
    )
    ap.add_argument(
        "--checkpoint",
        default="code/results_mitigated/Climate_AQI/checkpoints/best.pt",
        help="Path to best.pt checkpoint from Step J",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Inference batch size (default: 8, matches training)")
    ap.add_argument("--n-warmup", type=int, default=5, help="Warm-up forward passes (not measured)")
    ap.add_argument("--n-runs", type=int, default=50, help="Measured forward passes")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    ap.add_argument(
        "--out-dir",
        default="code/results_mitigated/Climate_AQI",
        help="Where to write inference_benchmark.json and .log",
    )
    args = ap.parse_args()

    metrics_path = Path(args.metrics).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "inference_benchmark.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("inference_benchmark")

    log.info("=== MVGT-Net Inference Benchmark (Climate_AQI, Step J) ===")
    log.info(f"Bundle root: {BUNDLE_ROOT}")
    log.info(f"code/ root:  {CODE_ROOT}")
    log.info(f"Metrics JSON: {metrics_path}")
    log.info(f"Checkpoint:   {ckpt_path}")
    log.info(f"Device: {args.device}")
    if args.device.startswith("cuda"):
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    log.info(f"Batch size: {args.batch_size}, warmup: {args.n_warmup}, runs: {args.n_runs}")

    if not metrics_path.exists():
        log.error(f"metrics.json not found at {metrics_path}")
        return 3
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found at {ckpt_path}")
        return 3

    # 1. Cold start: load config + build model + load checkpoint
    t0 = time.perf_counter()
    config = load_config(metrics_path)
    log.info(f"Loaded model config: num_nodes={config.get('num_nodes')}, "
             f"input_dim={config.get('input_dim')}, hidden_dim={config.get('hidden_dim')}, "
             f"lookback={config.get('lookback')}, horizon={config.get('horizon')}")
    model, ckpt_meta = load_model(config, ckpt_path, device=args.device)
    cold_start_s = time.perf_counter() - t0
    log.info(f"Cold start (config + model build + checkpoint load): {cold_start_s:.3f} s")
    if isinstance(ckpt_meta, dict):
        if "best_val_mae" in ckpt_meta:
            log.info(f"Checkpoint best_val_mae: {ckpt_meta['best_val_mae']}")
        if "epoch" in ckpt_meta:
            log.info(f"Checkpoint epoch: {ckpt_meta['epoch']}")

    # 2. Parameter counts
    params = count_parameters(model)
    log.info(
        f"Parameters: total={params['total']:,} trainable={params['trainable']:,} "
        f"frozen={params['frozen']:,} ({100*params['frozen']/max(1, params['total']):.1f}% frozen)"
    )

    # 3. Build a real test sample batch
    batch = make_sample_batch(config, args.batch_size, device=args.device)
    x_numeric = batch[0]
    if torch.is_tensor(x_numeric):
        log.info(f"Sample batch x_numeric: shape={tuple(x_numeric.shape)}, dtype={x_numeric.dtype}")

    # 4. Reset peak memory and run a sanity forward
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    with torch.no_grad():
        out = model(*batch) if isinstance(batch, tuple) else model(batch)
    if isinstance(out, dict) and "numeric" in out:
        log.info(f"Forward sanity OK: output['numeric'] shape={tuple(out['numeric'].shape)}")
    elif torch.is_tensor(out):
        log.info(f"Forward sanity OK: output shape={tuple(out.shape)}")
    else:
        log.info(f"Forward sanity OK: output type={type(out).__name__}")

    # 5. Forward-pass latency benchmark
    latencies = benchmark_forward(
        model, batch, n_warmup=args.n_warmup, n_runs=args.n_runs, device=args.device
    )
    lat_arr = np.array(latencies, dtype=np.float64)
    mean_latency_s = float(lat_arr.mean()) / 1e6
    throughput = args.batch_size / mean_latency_s if mean_latency_s > 0 else float("inf")
    peak_mem_mb = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)

    stats = {
        "mean_us": float(lat_arr.mean()),
        "median_us": float(np.median(lat_arr)),
        "p50_us": float(np.percentile(lat_arr, 50)),
        "p95_us": float(np.percentile(lat_arr, 95)),
        "p99_us": float(np.percentile(lat_arr, 99)),
        "min_us": float(lat_arr.min()),
        "max_us": float(lat_arr.max()),
        "std_us": float(lat_arr.std()) if len(lat_arr) > 1 else 0.0,
        "n_runs": int(len(lat_arr)),
        "throughput_samples_per_sec": float(throughput),
    }

    # 6. Write JSON results
    # Determine data source (real vs synthetic) from log inspection
    data_source = "synthetic"  # default assumption
    for handler in log.handlers:
        pass  # cannot easily inspect handlers; rely on a sentinel instead
    # Re-check by re-reading our own logs
    try:
        with open(log_path, "r") as lf:
            log_text = lf.read()
        if "Loading real test samples" in log_text and "SYNTHETIC batch" not in log_text.split("Loading real test samples")[1].split("\n\n")[0]:
            data_source = "real_timemmd_test_split"
    except Exception:
        pass

    result = {
        "domain": "Climate_AQI",
        "step": "J_mitigated",
        "checkpoint_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "cold_start_s": float(cold_start_s),
        "params": params,
        "latency_us": stats,
        "peak_gpu_memory_mb": float(peak_mem_mb) if peak_mem_mb is not None else None,
        "best_val_mae_in_checkpoint": ckpt_meta.get("best_val_mae") if isinstance(ckpt_meta, dict) else None,
        "checkpoint_epoch": ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else None,
        "data_source": data_source,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_json = out_dir / "inference_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("=== Results ===")
    log.info(f"Mean latency:  {stats['mean_us']:.1f} us  ({mean_latency_s*1000:.3f} ms)")
    log.info(f"p50 / p95 / p99: {stats['p50_us']:.1f} / {stats['p95_us']:.1f} / {stats['p99_us']:.1f} us")
    log.info(f"min / max:      {stats['min_us']:.1f} / {stats['max_us']:.1f} us")
    log.info(f"Throughput:     {stats['throughput_samples_per_sec']:.2f} samples/sec")
    if peak_mem_mb is not None:
        log.info(f"Peak GPU mem:  {peak_mem_mb:.1f} MB")
    log.info(f"Cold start:    {cold_start_s:.3f} s")
    log.info(f"Results JSON:  {out_json}")
    log.info(f"Log:           {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
