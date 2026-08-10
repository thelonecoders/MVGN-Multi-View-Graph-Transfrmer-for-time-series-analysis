#!/usr/bin/env python3
"""
run_integrated_gradients.py
===========================

Integrated Gradients (IG) interpretability pass for the Step J
(Climate_AQI mitigated) MVGT-Net checkpoint.

This is the SECOND interpretability method for Climate_AQI, complementing
the SHAP pass already produced under code/results_mitigated/Climate_AQI/shap/.

Method
------
  - Captum.attr.IntegratedGradients
  - Wrapper reduces model output['numeric'] to a single scalar per sample
    (mean over horizon, nodes, and feature dims)
  - Baseline: all-zeros x_numeric with adj_spatial = identity
  - Attribution shape == x_numeric shape: (B, P, N, C)
  - Aggregation:
        per_feature_importance = mean(|attribution|, axis=(B, P, N)) -> (C,)
        per_node_importance     = mean(|attribution|, axis=(B, P, C)) -> (N,)
        per_timestep_importance = mean(|attribution|, axis=(B, N, C)) -> (P,)

Outputs (written to <out_dir>, default
code/results_mitigated/Climate_AQI/integrated_gradients/)
  - ig_attributions.npy           (n_samples, P, N, C)
  - ig_feature_importance.json    aggregated importances + metadata
  - ig_attribution_bar.png         bar chart of per-feature importance
  - ig_attribution_heatmap.png    heatmap of (P, C) mean |attribution|
  - ig_run.log                    human-readable log

Bundle layout (confirmed by VPS find output)
---------------------------------------------
  ST-LLM-Plus_VPS_Code_Bundle/
  ├── code/
  │   ├── mvgt_net/__init__.py            # exports MVGTNet
  │   ├── mvgt_net/model.py               # class MVGTNet(nn.Module)
  │   ├── scripts/run_integrated_gradients.py  <-- THIS FILE
  │   ├── results_mitigated/Climate_AQI/
  │   │   ├── metrics.json
  │   │   └── checkpoints/best.pt
  │   └── data/TimeMMD/Climate_AQI/{train,validation,test}.jsonl

MVGTNet.forward signature (verified from code/mvgt_net/model.py)
----------------------------------------------------------------
    forward(x_numeric: (B, P, N, C),
            x_text: dict | None,
            x_categorical: (B, P, N) | None,
            adj_spatial: (N, N) | None,
            return_attention: bool = False) -> dict
    Returns: {"numeric": (B, S, N, C), ...}

Requirements
------------
  - Same .venv used for Step J/K (PyTorch 2.5.1+cu121 verified)
  - `pip install captum matplotlib` (if not already installed)

Usage (run from the bundle root)
--------------------------------
  cd ~/st-llm-plus/ST-LLM-Plus_VPS_Code_Bundle
  source code/.venv/bin/activate
  pip install captum matplotlib
  python code/scripts/run_integrated_gradients.py \\
      --metrics   code/results_mitigated/Climate_AQI/metrics.json \\
      --checkpoint code/results_mitigated/Climate_AQI/checkpoints/best.pt \\
      --n-samples 10 --n-steps 64

Zero hallucination guarantee
----------------------------
  - All attribution values are computed live by captum.
  - No number in the JSON is copied from any other file.
  - If the script fails, no partial JSON is written.
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

# ---------------------------------------------------------------------------
# Optional deps -- fail with a clear message if captum is missing.
# ---------------------------------------------------------------------------
try:
    from captum.attr import IntegratedGradients
except ImportError as e:
    print(
        "ERROR: captum is not installed.\n"
        "  Fix:  pip install captum\n"
        f"  Original error: {e}",
        file=sys.stderr,
    )
    sys.exit(2)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe for headless VPS
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # Register Noto Sans SC if present (for any non-Latin labels we add later)
    for font_path in [
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print(
        "WARNING: matplotlib is not installed; .png files will be skipped.\n"
        "  Fix:  pip install matplotlib",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Bundle-root bootstrap -- same as run_inference_benchmark.py
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
BUNDLE_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = BUNDLE_ROOT / "code"
for p_str in [str(BUNDLE_ROOT), str(CODE_ROOT)]:
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

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
# Helpers (config + model loading -- identical to run_inference_benchmark.py)
# ---------------------------------------------------------------------------
def load_config(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("config", "model_config", "model_args"):
        if key in data and isinstance(data[key], dict):
            cfg = data[key]
            if "model" in cfg and isinstance(cfg["model"], dict):
                return cfg["model"]
            return cfg
    if "model" in data and isinstance(data["model"], dict):
        return data["model"]
    if any(k in data for k in ("num_nodes", "input_dim", "hidden_dim", "lookback", "horizon")):
        return data
    raise RuntimeError(
        f"Could not find a model config inside {metrics_path}. "
        f"Top-level keys: {sorted(data.keys())}"
    )


def load_model(config: dict, checkpoint_path: Path, device: str = "cuda"):
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
        logging.warning(f"Missing state-dict keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        logging.warning(f"Unexpected state-dict keys ({len(unexpected)}): {unexpected[:5]}")
    model.to(device)
    model.eval()
    return model, state


def load_real_samples(config: dict, n_samples: int, device: str):
    """Load N real test samples as a single x_numeric tensor.

    Uses mvgt_net.data.get_dataloaders(domain, data_root, batch_size=...)
    which returns (train_loader, val_loader, test_loader, stats_dict).
    The collate_fn returns:
        (x_numeric: (B,P,N,C), y_target, {"fact": [str,...]}, x_cat, adj)

    Returns: x_numeric tensor of shape (n_samples, P, N, C).
    """
    try:
        from mvgt_net.data import get_dataloaders
    except ImportError as e:
        logging.warning(
            f"mvgt_net.data not importable ({e}); using SYNTHETIC samples."
        )
        return _synthetic_fallback(config, n_samples, device)

    data_root = BUNDLE_ROOT / "code" / "data" / "TimeMMD"
    if not data_root.exists():
        logging.warning(
            f"TimeMMD data_root not found at {data_root}; using SYNTHETIC samples."
        )
        return _synthetic_fallback(config, n_samples, device)

    try:
        logging.info(
            f"Loading real test samples via mvgt_net.data.get_dataloaders("
            f"domain='Climate_AQI', data_root={data_root})"
        )
        _, _, test_loader, stats = get_dataloaders(
            domain="Climate_AQI",
            data_root=str(data_root),
            batch_size=n_samples,
            num_workers=0,
            pin_memory=False,
        )
        logging.info(
            f"  test_size={stats['test_size']} samples, "
            f"lookback={stats['lookback']}, horizon={stats['horizon']}, "
            f"frequency={stats['frequency']}"
        )
        # Pull one batch -- the collate output is:
        # (x_numeric, y_target, {"fact": [...]}, x_cat, adj)
        x_numeric, _, _, _, _ = next(iter(test_loader))
        logging.info(f"  loaded batch: x_numeric shape={tuple(x_numeric.shape)}")
        return x_numeric.to(device)
    except Exception as e:  # noqa: BLE001
        logging.warning(
            f"get_dataloaders failed ({type(e).__name__}: {e}); using SYNTHETIC samples."
        )
        return _synthetic_fallback(config, n_samples, device)


def _synthetic_fallback(config: dict, n_samples: int, device: str) -> torch.Tensor:
    """Last-resort synthetic data (NOT for production interpretation)."""
    P = int(config.get("lookback", 96))
    N = int(config.get("num_nodes", 1))
    C = int(config.get("input_dim", 1))
    return torch.randn(n_samples, P, N, C, device=device, dtype=torch.float32) * 0.1


def _extract_x_numeric(result, n_samples: int, device: str) -> torch.Tensor:
    """[DEPRECATED -- kept for backward compat] Pull x_numeric out of a
    dataloader output and move to device."""
    if hasattr(result, "__iter__") and not isinstance(result, (list, tuple, dict)):
        try:
            result = next(iter(result))
        except StopIteration:
            raise RuntimeError("Dataloader yielded zero batches")
    if isinstance(result, (list, tuple)):
        x_numeric = result[0]
    elif isinstance(result, dict):
        x_numeric = result.get("x_numeric") or result.get("numeric") or result.get("x")
    else:
        raise RuntimeError(f"Unrecognized dataloader output type: {type(result).__name__}")
    if not torch.is_tensor(x_numeric):
        raise RuntimeError(f"x_numeric is not a tensor (got {type(x_numeric).__name__})")
    return x_numeric.to(device)


# ---------------------------------------------------------------------------
# Model wrapper for Captum
# ---------------------------------------------------------------------------
class MVGTNetWrapper(torch.nn.Module):
    """Wraps MVGTNet so Captum sees: forward(x_numeric) -> scalar per sample.

    Captum requires a single-tensor input. The wrapper holds x_text,
    x_categorical, and adj_spatial fixed at their baseline values (None /
    zero / identity) so the attribution is computed against x_numeric only.
    """

    def __init__(self, model, adj_spatial=None, x_categorical=None):
        super().__init__()
        self.model = model
        self.adj_spatial = adj_spatial
        self.x_categorical = x_categorical

    def forward(self, x_numeric: torch.Tensor) -> torch.Tensor:
        out = self.model(
            x_numeric,
            None,                # x_text=None to skip text branch
            self.x_categorical,  # baseline (None or zeros)
            adj_spatial=self.adj_spatial,
        )
        if isinstance(out, dict):
            out = out.get("numeric", next(iter(out.values())))
        # out shape: (B, S, N, C) or (B, S, N*C) etc. Reduce to (B,) scalar.
        if out.dim() == 1:
            return out
        return out.mean(dim=tuple(range(1, out.dim())))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_feature_importance(per_feature: np.ndarray, out_path: Path, top_k: int = 20):
    n = len(per_feature)
    k = min(top_k, n)
    order = np.argsort(per_feature)[::-1][:k]
    fig, ax = plt.subplots(figsize=(8, max(4, k * 0.35)), constrained_layout=True)
    ax.barh(range(k), per_feature[order][::-1], color="#3b82f6")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"f{i}" for i in order][::-1])
    ax.set_xlabel("mean |Integrated Gradients attribution|")
    ax.set_title(f"Top-{k} feature importance (Integrated Gradients)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attribution_heatmap(mean_abs_attr_2d: np.ndarray, out_path: Path,
                              x_label: str, y_label: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(mean_abs_attr_2d, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean |attribution|")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    ap.add_argument("--n-samples", type=int, default=10, help="Number of test samples to attribute (default: 10)")
    ap.add_argument("--n-steps", type=int, default=64, help="IG integration steps (default: 64)")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    ap.add_argument(
        "--out-dir",
        default="code/results_mitigated/Climate_AQI/integrated_gradients",
        help="Where to write IG outputs",
    )
    args = ap.parse_args()

    metrics_path = Path(args.metrics).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ig_run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("integrated_gradients")

    log.info("=== Integrated Gradients (Climate_AQI, Step J) ===")
    log.info(f"Bundle root: {BUNDLE_ROOT}")
    log.info(f"Metrics JSON: {metrics_path}")
    log.info(f"Checkpoint:   {ckpt_path}")
    log.info(f"Device: {args.device}")
    if args.device.startswith("cuda"):
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"n_samples: {args.n_samples}, n_steps: {args.n_steps}")

    if not metrics_path.exists():
        log.error(f"metrics.json not found at {metrics_path}")
        return 3
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found at {ckpt_path}")
        return 3

    # 1. Load model
    config = load_config(metrics_path)
    log.info(
        f"Config: num_nodes={config.get('num_nodes')}, input_dim={config.get('input_dim')}, "
        f"hidden_dim={config.get('hidden_dim')}, lookback={config.get('lookback')}, "
        f"horizon={config.get('horizon')}"
    )
    model, ckpt_meta = load_model(config, ckpt_path, device=args.device)
    log.info("Model loaded.")
    if isinstance(ckpt_meta, dict) and "best_val_mae" in ckpt_meta:
        log.info(f"Checkpoint best_val_mae: {ckpt_meta['best_val_mae']}")

    # 2. Load real test samples as a single x_numeric tensor
    x_numeric = load_real_samples(config, args.n_samples, device=args.device)
    if x_numeric.dim() == 3:
        # (B, P, C) -> add node dim
        x_numeric = x_numeric.unsqueeze(2)
    elif x_numeric.dim() == 2:
        # (P, C) -> add batch and node dims
        x_numeric = x_numeric.unsqueeze(0).unsqueeze(2)
    log.info(f"x_numeric tensor shape: {tuple(x_numeric.shape)}, dtype: {x_numeric.dtype}")

    n_samples, P, N, C = x_numeric.shape
    # Baseline pieces for x_text / x_categorical / adj_spatial
    adj_spatial = torch.eye(N, device=args.device, dtype=torch.float32)
    x_categorical = None
    if config.get("use_categorical", False) and int(config.get("num_categories", 0)) > 0:
        x_categorical = torch.zeros(n_samples, P, N, device=args.device, dtype=torch.long)

    # 3. Build wrapper + IG
    wrapper = MVGTNetWrapper(model, adj_spatial=adj_spatial, x_categorical=x_categorical).to(args.device).eval()
    ig = IntegratedGradients(wrapper)

    # Baseline: all-zeros x_numeric
    baseline = torch.zeros_like(x_numeric)

    # 4. Compute attributions
    # NOTE on the target= argument:
    #   Captum's `target` parameter selects a column from the model's output
    #   (e.g. target=5 means "attribute w.r.t. the 5-th class probability").
    #   Our wrapper (MVGTNetWrapper) already reduces the output to a single
    #   scalar per sample (shape (B,)), so there is no column to select and
    #   we must OMIT `target`. Passing target=0 here raises:
    #     AssertionError: Cannot choose target column with output shape (N,).
    #   See captum/_utils/common.py:_verify_select_column for the assertion.
    log.info(f"Computing Integrated Gradients (n_steps={args.n_steps})...")
    t0 = time.perf_counter()
    attributions = torch.zeros_like(x_numeric)
    for i in range(n_samples):
        x_i = x_numeric[i : i + 1].clone().detach().requires_grad_(True)
        attr_i = ig.attribute(x_i, baselines=torch.zeros_like(x_i), n_steps=args.n_steps)
        attributions[i] = attr_i.detach()
    elapsed = time.perf_counter() - t0
    log.info(f"IG computed for {n_samples} samples in {elapsed:.1f} s")

    # 5. Aggregate
    attr_np = attributions.detach().cpu().numpy()  # (n, P, N, C)
    abs_attr = np.abs(attr_np)
    # Per-feature: average over batch, time, node -> (C,)
    per_feature = abs_attr.mean(axis=(0, 1, 2))
    # Per-node: average over batch, time, feature -> (N,)
    per_node = abs_attr.mean(axis=(0, 1, 3))
    # Per-timestep: average over batch, node, feature -> (P,)
    per_timestep = abs_attr.mean(axis=(0, 2, 3))
    # 2D heatmap: (P, C)
    mean_abs_attr_PC = abs_attr.mean(axis=(0, 2))

    # 6. Save outputs
    np.save(out_dir / "ig_attributions.npy", attr_np)

    # Determine data source (real vs synthetic) from log inspection
    data_source = "synthetic"
    try:
        with open(log_path, "r") as lf:
            log_text = lf.read()
        if "Loading real test samples" in log_text and "SYNTHETIC samples" not in log_text.split("Loading real test samples")[1].split("\n\n")[0]:
            data_source = "real_timemmd_test_split"
    except Exception:
        pass

    feature_importance = {
        "method": "IntegratedGradients",
        "library": "captum",
        "n_samples": int(n_samples),
        "input_shape": [int(P), int(N), int(C)],
        "n_steps": int(args.n_steps),
        "elapsed_seconds": float(elapsed),
        "per_feature_importance": [
            {"feature_index": int(i), "mean_abs_attribution": float(per_feature[i])}
            for i in range(len(per_feature))
        ],
        "top_5_features": [
            {"feature_index": int(i), "mean_abs_attribution": float(per_feature[i])}
            for i in np.argsort(per_feature)[::-1][:5].tolist()
        ],
        "per_node_importance": [
            {"node_index": int(i), "mean_abs_attribution": float(per_node[i])}
            for i in range(len(per_node))
        ],
        "per_timestep_importance_peak_idx": int(np.argmax(per_timestep)),
        "per_timestep_importance_peak_value": float(per_timestep.max()),
        "baseline_strategy": "zeros",
        "data_source": data_source,
        "checkpoint_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
        "device": args.device,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "ig_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2, ensure_ascii=False)

    # 7. Plots
    if HAS_MPL:
        plot_feature_importance(per_feature, out_dir / "ig_attribution_bar.png")
        plot_attribution_heatmap(
            mean_abs_attr_PC, out_dir / "ig_attribution_heatmap.png",
            x_label="Lookback timestep (P)",
            y_label="Feature index (C)",
            title="Mean |Integrated Gradients attribution| (averaged over batch and node)",
        )
        log.info("Plots saved: ig_attribution_bar.png, ig_attribution_heatmap.png")
    else:
        log.warning("matplotlib not available; skipping PNG outputs.")

    # 8. Summary
    log.info("=== Integrated Gradients Summary ===")
    log.info(f"Attributions shape: {attr_np.shape}")
    log.info(f"Top-5 features by |IG|:")
    for fi in feature_importance["top_5_features"]:
        log.info(f"  feature {fi['feature_index']}: {fi['mean_abs_attribution']:.6f}")
    log.info(
        f"Peak timestep: t={feature_importance['per_timestep_importance_peak_idx']} "
        f"(value={feature_importance['per_timestep_importance_peak_value']:.6f})"
    )
    log.info(f"Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
