#!/usr/bin/env python3
"""SHAP v3 - aggregates 96 time steps into 12 weekly features for speed."""
import sys, os, json, argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.getcwd())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--background-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--n-background", type=int, default=25)
    parser.add_argument("--n-explain", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-features", type=int, default=12,
                        help="Number of aggregated features (12=weekly, 4=monthly)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load checkpoint + config
    print(f"[1/7] Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    metrics_candidates = [
        os.path.join(ckpt_dir, "metrics.json"),
        os.path.normpath(os.path.join(ckpt_dir, "..", "results_mitigated", args.domain, "metrics.json")),
        os.path.normpath(os.path.join(ckpt_dir, "..", "..", "results_mitigated", args.domain, "metrics.json")),
    ]
    metrics_path = next((c for c in metrics_candidates if os.path.isfile(c)), None)
    if metrics_path is None:
        print("ERROR: Could not find metrics.json")
        return 1

    with open(metrics_path) as f:
        metrics = json.load(f)
    config = metrics["config"]
    print(f"  Config from: {metrics_path}", flush=True)
    print(f"  Epoch: {ckpt.get('epoch')}, val_MAE: {ckpt.get('best_val_mae')}", flush=True)

    # 2. Build model
    print(f"[2/7] Building MVGTNet", flush=True)
    from mvgt_net.model import MVGTNet
    model = MVGTNet(config).to(device)
    state_key = "model_state" if "model_state" in ckpt else ("model_state_dict" if "model_state_dict" in ckpt else "state_dict")
    model.load_state_dict(ckpt[state_key])
    model.eval()
    print(f"  State loaded from key: '{state_key}'", flush=True)

    # 3. Load data
    print(f"[3/7] Loading data via get_dataloaders", flush=True)
    p = Path(args.background_data)
    data_root = str(p.parent.parent)
    print(f"  data_root: {data_root}", flush=True)

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

    bg_x, bg_y, bg_text, bg_cat, bg_adj = bg_batch
    test_x, test_y, test_text, test_cat, test_adj = test_batch

    n_bg = min(args.n_background, bg_x.shape[0])
    n_ex = min(args.n_explain, test_x.shape[0])
    bg_x = bg_x[:n_bg].to(device)
    bg_cat = bg_cat[:n_bg].to(device)
    bg_adj = bg_adj.to(device)
    test_x = test_x[:n_ex].to(device)

    lookback = bg_x.shape[1]
    n_features = args.n_features
    # Aggregate lookback (96) into n_features (12) chunks
    chunk_size = lookback // n_features
    print(f"  Lookback={lookback}, aggregating into {n_features} features (chunk_size={chunk_size})", flush=True)
    print(f"  Background: {tuple(bg_x.shape)}", flush=True)
    print(f"  Test:      {tuple(test_x.shape)}", flush=True)

    # 4. Wrapper - aggregate input, expand back, run model, extract output
    print(f"[4/7] Creating model wrapper", flush=True)

    class ShapWrapperAgg(torch.nn.Module):
        """Takes (B, n_features) input, expands to (B, lookback, 1, 1), runs model."""
        def __init__(self, model, text, cat, adj, n_bg, lookback, n_features, chunk_size):
            super().__init__()
            self.model = model
            self.text = text
            self.cat = cat
            self.adj = adj
            self.n_bg = n_bg
            self.lookback = lookback
            self.n_features = n_features
            self.chunk_size = chunk_size

        def forward(self, x_agg):
            # x_agg: (B, n_features) - expand each feature to chunk_size timesteps
            B = x_agg.shape[0]
            # Repeat each value chunk_size times along time dim
            x_expanded = x_agg.unsqueeze(2).repeat(1, 1, self.chunk_size)  # (B, n_features, chunk_size)
            x_expanded = x_expanded.reshape(B, self.lookback, 1, 1)

            text_fact = self.text["fact"]
            cycled_text = [text_fact[i % len(text_fact)] for i in range(B)]
            idx = torch.arange(B, device=self.cat.device) % self.n_bg
            cycled_cat = self.cat[idx]
            out = self.model(x_expanded, {"fact": cycled_text}, cycled_cat, self.adj)
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, dict):
                if "numeric" in out:
                    return out["numeric"]
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        return v
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

    wrapper = ShapWrapperAgg(model, bg_text, bg_cat, bg_adj, n_bg, lookback, n_features, chunk_size).to(device).eval()

    with torch.no_grad():
        test_out = wrapper(torch.randn(2, n_features, device=device))
        if not isinstance(test_out, torch.Tensor):
            print(f"  ERROR: wrapper output is {type(test_out).__name__}")
            return 1
        print(f"  Wrapper OK - output shape: {tuple(test_out.shape)}", flush=True)

    # 5. Build SHAP explainer
    print(f"[5/7] Building SHAP explainer", flush=True)
    import shap

    def forward_numpy(x_agg_np):
        x = torch.tensor(x_agg_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = wrapper(x)
        return out.detach().cpu().numpy().reshape(x_agg_np.shape[0], -1)

    # Aggregate background + test data into n_features features
    def aggregate(x_4d):
        # x_4d: (B, lookback, 1, 1) -> (B, n_features)
        flat = x_4d.squeeze(-1).squeeze(-1).cpu().numpy()  # (B, lookback)
        return flat.reshape(flat.shape[0], n_features, chunk_size).mean(axis=2)

    bg_agg = aggregate(bg_x)
    test_agg = aggregate(test_x)
    print(f"  Aggregated shapes: bg={bg_agg.shape}, test={test_agg.shape}", flush=True)

    sv = None
    explainer_type = "permutation"

    # shap 0.52: 2*12+1 = 25, so max_evals=25 is now VALID
    try:
        explainer = shap.PermutationExplainer(forward_numpy, masker=bg_agg)
        print(f"  PermutationExplainer (max_evals=25, valid for {n_features} features)", flush=True)
        result = explainer(test_agg, max_evals=25)
        sv = result.values if hasattr(result, 'values') else np.asarray(result)
        print(f"  ✓ Got values via __call__()", flush=True)
    except Exception as e:
        print(f"  PermutationExplainer failed: {e}", flush=True)
        try:
            explainer = shap.KernelExplainer(forward_numpy, bg_agg)
            explainer_type = "kernel"
            print(f"  Fallback: KernelExplainer (nsamples=50)", flush=True)
            sv = explainer.shap_values(test_agg, nsamples=50)
            if isinstance(sv, list):
                sv = np.array(sv)
                if sv.ndim > 3:
                    sv = sv.sum(axis=0)
        except Exception as e2:
            print(f"  KernelExplainer failed: {e2}", flush=True)
            return 1

    if hasattr(sv, 'values') and not isinstance(sv, np.ndarray):
        sv = sv.values
    sv = np.asarray(sv)
    print(f"  SHAP values shape: {sv.shape}", flush=True)

    # 6. Save SHAP values
    print(f"[6/7] Saving SHAP values", flush=True)
    npy_path = os.path.join(args.output_dir, f"{args.domain}_shap_values.npy")
    np.save(npy_path, sv)
    print(f"  Saved: {npy_path}", flush=True)

    # 7. Feature importance + plots
    print(f"[7/7] Feature importance + plots", flush=True)

    if sv.ndim > 2:
        sv_flat = sv.reshape(sv.shape[0], -1)
    else:
        sv_flat = sv

    mean_abs_shap = np.abs(sv_flat).mean(axis=0)
    n_actual = len(mean_abs_shap)

    if n_actual == n_features:
        # Weekly feature names: t-0..7, t-8..15, etc.
        feature_names = []
        for i in range(n_features):
            start = i * chunk_size
            end = (i + 1) * chunk_size - 1
            feature_names.append(f"t-{start}..{end} (lag {start}-{end}d)")
    else:
        feature_names = [f"feature_{i}" for i in range(n_actual)]

    top_idx = np.argsort(mean_abs_shap)[::-1]

    importance = {
        "domain": args.domain,
        "explainer_type": explainer_type,
        "n_features_aggregated": n_features,
        "chunk_size": chunk_size,
        "lookback": lookback,
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
        "interpretation_note": "Each feature represents an aggregated chunk of historical time steps. For daily data with lookback=96, 12 features = weekly aggregations (8 days each).",
    }

    json_path = os.path.join(args.output_dir, f"{args.domain}_feature_importance.json")
    with open(json_path, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"  Saved: {json_path}", flush=True)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_n = min(12, n_actual)
        top_i = top_idx[:top_n]
        top_names = [feature_names[i] for i in top_i]
        top_vals = [mean_abs_shap[i] for i in top_i]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(top_n), top_vals[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Features by SHAP Importance - {args.domain}\n(weekly aggregations of 8-day chunks)")
        fig.tight_layout()
        bar_path = os.path.join(args.output_dir, f"{args.domain}_shap_bar.png")
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {bar_path}", flush=True)

        try:
            shap.summary_plot(sv_flat, test_agg, feature_names=feature_names, show=False)
            summary_path = os.path.join(args.output_dir, f"{args.domain}_shap_summary.png")
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {summary_path}", flush=True)
        except Exception as e:
            print(f"  WARN: summary plot failed: {e}", flush=True)
    except Exception as e:
        print(f"  WARN: plotting failed: {e}", flush=True)

    # Report
    report = {
        "domain": args.domain,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_mae": ckpt.get("best_val_mae"),
        "test_r2": metrics.get("test_metrics", {}).get("R2"),
        "test_mae": metrics.get("test_metrics", {}).get("MAE"),
        "explainer_type": explainer_type,
        "n_features": n_features,
        "n_background": n_bg,
        "n_explain": n_ex,
        "top_5_features": importance["top_10_features"][:5],
        "interpretation_note": importance["interpretation_note"],
    }
    report_path = os.path.join(args.output_dir, f"{args.domain}_explanation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"SHAP analysis complete!", flush=True)
    print(f"{'='*60}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
