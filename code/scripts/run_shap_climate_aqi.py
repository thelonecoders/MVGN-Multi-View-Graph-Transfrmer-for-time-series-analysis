#!/usr/bin/env python3
"""Standalone SHAP analysis for Climate_AQI MVGT-Net checkpoint.

Bypasses the broken run_shap_analysis.py — uses get_dataloaders (the same
data path train_real.py uses) to guarantee normalization matches training.
Explains the x_numeric time-series input; text/categorical/adjacency are
held at background values (standard SHAP practice for multi-input models).
"""
import sys, os, json, argparse
import numpy as np
import torch
from pathlib import Path

# Ensure mvgt_net is importable
sys.path.insert(0, os.getcwd())

def infer_data_root(jsonl_path):
    p = Path(jsonl_path)
    split = p.stem
    domain = p.parent.name
    data_root = str(p.parent.parent)
    return data_root, domain, split

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--background-data", required=True, help="Path to train.jsonl")
    parser.add_argument("--test-data", required=True, help="Path to test.jsonl")
    parser.add_argument("--n-background", type=int, default=100)
    parser.add_argument("--n-explain", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load checkpoint + config
    print(f"[1/7] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    metrics_candidates = [
        os.path.join(ckpt_dir, "metrics.json"),
        os.path.normpath(os.path.join(ckpt_dir, "..", "results_mitigated", args.domain, "metrics.json")),
        os.path.normpath(os.path.join(ckpt_dir, "..", "..", "results_mitigated", args.domain, "metrics.json")),
        os.path.normpath(os.path.join(ckpt_dir, "..", "results", args.domain, "metrics.json")),
    ]
    metrics_path = next((c for c in metrics_candidates if os.path.isfile(c)), None)
    if metrics_path is None:
        print("ERROR: Could not find metrics.json. Checked:")
        for c in metrics_candidates:
            print(f"  {c}")
        return 1

    with open(metrics_path) as f:
        metrics = json.load(f)
    config = metrics["config"]
    print(f"  Config from: {metrics_path}")
    print(f"  Epoch: {ckpt.get('epoch')}, val_MAE: {ckpt.get('best_val_mae')}")

    # 2. Build model
    print(f"[2/7] Building MVGTNet")
    from mvgt_net.model import MVGTNet
    model = MVGTNet(config).to(device)
    state_key = "model_state" if "model_state" in ckpt else ("model_state_dict" if "model_state_dict" in ckpt else "state_dict")
    model.load_state_dict(ckpt[state_key])
    model.eval()
    print(f"  State loaded from key: '{state_key}'")

    # 3. Load data via get_dataloaders (same path as training)
    print(f"[3/7] Loading data via get_dataloaders (minmax normalization)")
    data_root, _, _ = infer_data_root(args.background_data)
    print(f"  data_root: {data_root}")

    from mvgt_net.data import get_dataloaders
    batch_size = max(args.n_background, args.n_explain)
    train_loader, val_loader, test_loader, data_stats = get_dataloaders(
        domain=args.domain,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        normalization="minmax",
    )

    bg_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    # collate_fn returns: (x_numeric, y_target, x_text_dict, x_cat, adj)
    bg_x, bg_y, bg_text, bg_cat, bg_adj = bg_batch
    test_x, test_y, test_text, test_cat, test_adj = test_batch

    n_bg = min(args.n_background, bg_x.shape[0])
    n_ex = min(args.n_explain, test_x.shape[0])
    bg_x = bg_x[:n_bg].to(device)
    bg_cat = bg_cat[:n_bg].to(device)
    bg_adj = bg_adj.to(device)
    test_x = test_x[:n_ex].to(device)

    print(f"  Background: {tuple(bg_x.shape)}")
    print(f"  Test:      {tuple(test_x.shape)}")

    # 4. Model wrapper — explain x_numeric, fix text/cat/adj at background values
    print(f"[4/7] Creating model wrapper")

    class ShapWrapper(torch.nn.Module):
        def __init__(self, model, text, cat, adj, n_bg):
            super().__init__()
            self.model = model
            self.text = text
            self.cat = cat
            self.adj = adj
            self.n_bg = n_bg
            self._pred_key = None

        def _get_pred(self, out):
            """Extract prediction tensor from model output (dict/list/tensor)."""
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, dict):
                if self._pred_key is None:
                    # Try common keys first
                    for k in ['pred', 'prediction', 'logits', 'output', 'y_pred',
                              'forecast', 'y_hat', 'out', 'forecast_pred', 'logit']:
                        if k in out:
                            self._pred_key = k
                            break
                    # Fallback: first tensor value
                    if self._pred_key is None:
                        for k, v in out.items():
                            if isinstance(v, torch.Tensor):
                                self._pred_key = k
                                break
                return out[self._pred_key]
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

        def forward(self, x_numeric):
            B = x_numeric.shape[0]
            text_fact = self.text["fact"]
            # Cycle text + cat to match whatever batch size SHAP requests
            cycled_text = [text_fact[i % len(text_fact)] for i in range(B)]
            idx = torch.arange(B, device=self.cat.device) % self.n_bg
            cycled_cat = self.cat[idx]
            out = self.model(x_numeric, {"fact": cycled_text}, cycled_cat, self.adj)
            return self._get_pred(out)

    wrapper = ShapWrapper(model, bg_text, bg_cat, bg_adj, n_bg).to(device).eval()

    # Test the wrapper to confirm it returns a tensor and find pred_key
    with torch.no_grad():
        test_out = wrapper(bg_x[:2])
        if not isinstance(test_out, torch.Tensor):
            print(f"  ERROR: wrapper output is {type(test_out).__name__}, not tensor")
            print(f"  Output: {test_out}")
            return 1
        print(f"  Wrapper OK - output shape: {tuple(test_out.shape)}, pred_key: {wrapper._pred_key}")

    # 5. Build SHAP explainer - PermutationExplainer is most robust for dict-output models
    print(f"[5/7] Building SHAP explainer")
    import shap

    def forward_numpy(x_flat_np):
        """Numpy in -> numpy out. Used by Permutation/Kernel explainer."""
        B = x_flat_np.shape[0]
        L = bg_x.shape[1]
        x = torch.tensor(x_flat_np, dtype=torch.float32, device=device).reshape(B, L, 1, 1)
        with torch.no_grad():
            out = wrapper(x)
        return out.detach().cpu().numpy().reshape(B, -1)

    bg_flat = bg_x.cpu().numpy().reshape(n_bg, -1)
    test_flat = test_x.cpu().numpy().reshape(n_ex, -1)

    explainer_type = "permutation"
    sv = None
    try:
        explainer = shap.PermutationExplainer(forward_numpy, masker=bg_flat)
        print("  Using shap.PermutationExplainer (max_evals=25, ~8x faster)")
        sv = explainer.shap_values(test_flat, max_evals=25)
    except Exception as e:
        print(f"  PermutationExplainer failed: {e}")
        try:
            explainer = shap.ExactExplainer(forward_numpy, masker=bg_flat)
            explainer_type = "exact"
            print("  Using shap.ExactExplainer")
            sv = explainer.shap_values(test_flat)
        except Exception as e2:
            print(f"  ExactExplainer failed: {e2}")
            explainer_type = "kernel"
            explainer = shap.KernelExplainer(forward_numpy, bg_flat)
            print("  Fallback: Using shap.KernelExplainer (slower)")
            sv = explainer.shap_values(test_flat, nsamples=200)

    # Normalize sv to numpy array
    if isinstance(sv, list):
        sv = np.array(sv)
        if sv.ndim > 3:
            sv = sv.sum(axis=0)
    if hasattr(sv, 'values'):
        sv = sv.values
    sv = np.asarray(sv)
    print(f"  SHAP values shape: {sv.shape}")

    npy_path = os.path.join(args.output_dir, f"{args.domain}_shap_values.npy")
    np.save(npy_path, sv)
    print(f"  Saved: {npy_path}")

    # 7. Feature importance + plots
    print(f"[7/7] Feature importance + plots")

    # Flatten to [n_explain, lookback]
    if sv.ndim == 4:
        sv_flat = sv.squeeze(-1).squeeze(-1)
    elif sv.ndim == 3:
        sv_flat = sv.squeeze(-1)
    elif sv.ndim == 2:
        sv_flat = sv
    else:
        sv_flat = sv.reshape(sv.shape[0], -1)

    mean_abs_shap = np.abs(sv_flat).mean(axis=0)
    lookback = bg_x.shape[1]
    n_features = len(mean_abs_shap)

    if n_features == lookback:
        feature_names = [f"t-{lookback-i}" for i in range(lookback)]
    else:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    top_idx = np.argsort(mean_abs_shap)[::-1]

    importance = {
        "domain": args.domain,
        "n_background": n_bg,
        "n_explain": n_ex,
        "shap_values_shape": list(sv.shape),
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap.tolist(),
        "top_10_features": [
            {"rank": r+1, "feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
            for r, i in enumerate(top_idx[:10])
        ],
        "total_shap_magnitude": float(np.abs(sv_flat).sum()),
        "mean_shap_magnitude": float(np.abs(sv_flat).mean()),
    }

    json_path = os.path.join(args.output_dir, f"{args.domain}_feature_importance.json")
    with open(json_path, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"  Saved: {json_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Bar chart
        top_n = min(20, n_features)
        top_i = top_idx[:top_n]
        top_names = [feature_names[i] for i in top_i]
        top_vals = [mean_abs_shap[i] for i in top_i]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), top_vals[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Features by SHAP Importance - {args.domain}")
        fig.tight_layout()
        bar_path = os.path.join(args.output_dir, f"{args.domain}_shap_bar.png")
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {bar_path}")

        # Beeswarm summary
        try:
            test_x_np = test_x.cpu().numpy()
            if test_x_np.ndim == 4:
                test_x_flat = test_x_np.squeeze(-1).squeeze(-1)
            else:
                test_x_flat = test_x_np.reshape(n_ex, -1)
            shap.summary_plot(sv_flat, test_x_flat, feature_names=feature_names, show=False)
            summary_path = os.path.join(args.output_dir, f"{args.domain}_shap_summary.png")
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {summary_path}")
        except Exception as e:
            print(f"  WARN: summary plot failed: {e}")
    except Exception as e:
        print(f"  WARN: plotting failed: {e}")

    # Explanation report
    report = {
        "domain": args.domain,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_mae": ckpt.get("best_val_mae"),
        "test_r2": metrics.get("test_metrics", {}).get("R2"),
        "test_mae": metrics.get("test_metrics", {}).get("MAE"),
        "n_background": n_bg,
        "n_explain": n_ex,
        "shap_values_shape": list(sv.shape),
        "top_5_features": importance["top_10_features"][:5],
        "total_shap_magnitude": importance["total_shap_magnitude"],
        "mean_shap_magnitude": importance["mean_shap_magnitude"],
        "note": "SHAP values computed on x_numeric (time series input). Text/categorical/adjacency held at background values.",
    }
    report_path = os.path.join(args.output_dir, f"{args.domain}_explanation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SHAP analysis complete!")
    print(f"{'='*60}")
    print(f"Outputs in: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
