#!/usr/bin/env python3
"""Run the Q5 SHAP interpretability analysis on a trained MVGT-Net.

This script implements the full Q5 protocol described in thesis
Chapter 5 (Research Question 5):

1. Load a trained MVGT-Net checkpoint.
2. Build a SHAP DeepExplainer with a background sample from the
   training set.
3. Compute SHAP values for ``n_explain`` test-set samples per domain.
4. Compute the attention-adjacency alignment (Pearson correlation
   between the learned attention map and the domain's adjacency
   matrix).
5. Compute the SHAP stability (Kendall tau between fold-pair
   rankings) -- requires --fold-pair to be set.
6. Generate the SHAP summary plot (beeswarm) for each domain.
7. Write all artifacts to ``results/shap/``.

Usage
-----
    python3 scripts/run_shap_analysis.py \\
        --checkpoint checkpoints/best_Solar.pt \\
        --domain Solar \\
        --background-data data/Solar/train.jsonl \\
        --test-data data/Solar/test.jsonl

    # With fold-pair stability:
    python3 scripts/run_shap_analysis.py \\
        --checkpoint checkpoints/best_Solar_fold0.pt \\
        --checkpoint-pair checkpoints/best_Solar_fold1.pt \\
        --domain Solar \\
        --fold-pair 0,1

Outputs (per domain)
--------------------
- ``results/shap/<domain>_shap_values.npy``  -- raw SHAP values
- ``results/shap/<domain>_feature_importance.json``  -- mean(|SHAP|) per feature
- ``results/shap/<domain>_attention_alignment.json``  -- Pearson + Spearman
- ``results/shap/<domain>_shap_summary.png``  -- beeswarm plot
- ``results/shap/<domain>_stability.json``  -- Kendall tau (if --fold-pair)

If ``shap`` is not installed, the script prints install instructions
and exits with code 2 (matching ``fetch_leaderboard.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, CODE_ROOT)


def _load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Load a MVGT-Net checkpoint into a model instance.

    Returns
    -------
    (model, config) : tuple
        ``model`` is the loaded nn.Module in eval mode on ``device``.
        ``config`` is the training config saved alongside the
        checkpoint.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyTorch is not installed. Install with `pip install -r requirements.txt`.\n"
        )
        raise SystemExit(2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    # Lazy import to avoid a hard torch dependency at module import time.
    # If checkpoint has no config, try to load from sibling metrics.json
    if not config:
        import os as _os
        import json as _json
        ckpt_dir = _os.path.dirname(_os.path.abspath(checkpoint_path))
        domain = _os.path.basename(ckpt_dir)
        candidates = [
            _os.path.join(ckpt_dir, "metrics.json"),
            _os.path.join(ckpt_dir, "..", "results_mitigated", domain, "metrics.json"),
            _os.path.join(ckpt_dir, "..", "results", domain, "metrics.json"),
            _os.path.join(ckpt_dir, "..", "..", "results_mitigated", domain, "metrics.json"),
            _os.path.join(ckpt_dir, "..", "..", "results", domain, "metrics.json"),
        ]
        for candidate in candidates:
            candidate = _os.path.normpath(candidate)
            if _os.path.isfile(candidate):
                with open(candidate) as f:
                    metrics = _json.load(f)
                if "config" in metrics:
                    config = metrics["config"]
                    print(f"  [shap] Loaded model config from {candidate}")
                    break
    # Last resort: reconstruct from stats + defaults
    if not config:
        stats = ckpt.get("stats", {}) if isinstance(ckpt, dict) else {}
        config = {
            "num_nodes": stats.get("num_nodes", 1),
            "input_dim": stats.get("variables", 1),
            "hidden_dim": 64,
            "lookback": stats.get("lookback", 96),
            "horizon": stats.get("horizon", 96),
            "use_text": True,
            "use_categorical": False,
            "num_categories": 0,
            "text_model": "bert-base-uncased",
            "graph_types": ["spatial", "temporal", "semantic", "adaptive"],
            "topk": 8,
            "num_heads": 4,
            "dropout": 0.1,
        }
        print(f"  [shap] Reconstructed config from stats + defaults")
    # Lazy import to avoid a hard torch dependency at module import time.
    from mvgt_net.model import MVGTNet  # type: ignore
    model = MVGTNet(config)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, config


def _load_samples(
    data_path: str,
    n_samples: int,
    device: str = "cpu",
) -> Tuple[Any, Any, Any, Any, Any]:
    """Load ``n_samples`` samples from a JSONL file as torch tensors.

    Returns
    -------
    (x_numeric, x_text, adj, tod_idx, dow_idx) : tuple of torch.Tensor
    """
    try:
        import torch  # type: ignore
    except ImportError:
        raise SystemExit(2)
    from mvgt_net.data import load_samples  # type: ignore
    return load_samples(data_path, n_samples, device=device)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True,
                   help="Path to the trained MVGT-Net checkpoint.")
    p.add_argument("--checkpoint-pair", default=None,
                   help="Second checkpoint (for fold-pair stability).")
    p.add_argument("--domain", required=True,
                   help="TimeMMD domain name (e.g., Solar).")
    p.add_argument("--background-data", required=True,
                   help="JSONL with background samples (training set).")
    p.add_argument("--test-data", required=True,
                   help="JSONL with test samples.")
    p.add_argument("--n-background", type=int, default=64,
                   help="Background sample size for SHAP expectation.")
    p.add_argument("--n-explain", type=int, default=128,
                   help="Number of test samples to explain.")
    p.add_argument("--output-dir", default=os.path.join(CODE_ROOT, "results", "shap"),
                   help="Output directory.")
    p.add_argument("--fold-pair", default=None,
                   help="Comma-separated fold indices, e.g. '0,1'.")
    p.add_argument("--device", default="cpu",
                   help="Device for the model ('cpu' or 'cuda'). "
                        "SHAP values are always computed on CPU.")
    args = p.parse_args(argv)

    # Lazy import so the script can print a friendly error if shap
    # is not installed.
    try:
        from mvgt_net.shap_explainer import (
            ShapConfig,
            ShapExplainer,
            ShapNotInstalledError,
            compute_attention_adjacency_alignment,
            compute_shap_stability,
        )
    except ShapNotInstalledError as exc:
        sys.stderr.write(str(exc))
        return 2

    # ------------------------------------------------------------------ #
    # Load model + samples
    # ------------------------------------------------------------------ #
    print(f"[1/6] Loading checkpoint: {args.checkpoint}")
    model, config = _load_model(args.checkpoint, device=args.device)

    print(f"[2/6] Loading background samples from {args.background_data}")
    bg = _load_samples(args.background_data, args.n_background, device=args.device)

    print(f"[3/6] Loading test samples from {args.test_data}")
    test = _load_samples(args.test_data, args.n_explain, device=args.device)

    # ------------------------------------------------------------------ #
    # Run SHAP
    # ------------------------------------------------------------------ #
    os.makedirs(args.output_dir, exist_ok=True)
    explainer = ShapExplainer(
        model, background=bg,
        config=ShapConfig(
            n_background=args.n_background,
            n_explain=args.n_explain,
            output_dir=args.output_dir,
            device=args.device,
        ),
    )

    print("[4/6] Computing SHAP values")
    shap_values = explainer.explain(test)
    npy_path = os.path.join(args.output_dir, f"{args.domain}_shap_values.npy")
    import numpy as np
    np.save(npy_path, shap_values)
    print(f"      wrote {npy_path}")

    print("[5/6] Computing feature importance + summary plot")
    importance = explainer.feature_importance(test)
    json_path = os.path.join(
        args.output_dir, f"{args.domain}_feature_importance.json"
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(importance, fh, indent=2)
        fh.write("\n")
    print(f"      wrote {json_path}")

    png_path = os.path.join(
        args.output_dir, f"{args.domain}_shap_summary.png"
    )
    explainer.summary_plot(test, out_path=png_path)
    print(f"      wrote {png_path}")

    # ------------------------------------------------------------------ #
    # Attention-adjacency alignment
    # ------------------------------------------------------------------ #
    print("[6/6] Attention-adjacency alignment")
    try:
        with torch.no_grad():  # type: ignore
            attention = model.last_attention_map(*test)  # type: ignore
        attention = attention.cpu().numpy()
    except Exception as exc:
        sys.stderr.write(
            f"WARN: could not extract attention map from model: {exc}\n"
        )
        attention = None

    alignment_path = os.path.join(
        args.output_dir, f"{args.domain}_attention_alignment.json"
    )
    if attention is not None:
        # The adjacency is the third element of the test tuple.
        adj = test[2].cpu().numpy() if hasattr(test[2], "cpu") else np.asarray(test[2])
        pearson = compute_attention_adjacency_alignment(
            attention.mean(axis=0), adj, method="pearson"
        )
        try:
            spearman = compute_attention_adjacency_alignment(
                attention.mean(axis=0), adj, method="spearman"
            )
        except Exception:
            spearman = None
        with open(alignment_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "domain": args.domain,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n_samples": attention.shape[0],
                    "q5_threshold_pearson": 0.5,
                    "q5_pass": bool(pearson >= 0.5),
                },
                fh, indent=2,
            )
            fh.write("\n")
        print(f"      wrote {alignment_path} (Pearson={pearson:.3f})")
    else:
        with open(alignment_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"domain": args.domain, "pearson": None, "q5_pass": None,
                 "note": "Attention map not available for this model."},
                fh, indent=2,
            )
            fh.write("\n")

    # ------------------------------------------------------------------ #
    # Optional: fold-pair stability
    # ------------------------------------------------------------------ #
    if args.checkpoint_pair and args.fold_pair:
        print(f"      Loading pair checkpoint: {args.checkpoint_pair}")
        model2, _ = _load_model(args.checkpoint_pair, device=args.device)
        explainer2 = ShapExplainer(
            model2, background=bg,
            config=ShapConfig(
                n_background=args.n_background,
                n_explain=args.n_explain,
                output_dir=args.output_dir,
                device=args.device,
            ),
        )
        importance2 = explainer2.feature_importance(test)
        try:
            tau = compute_shap_stability(importance, importance2, method="kendall")
        except Exception as exc:
            sys.stderr.write(f"WARN: Kendall tau failed: {exc}\n")
            tau = None
        stability_path = os.path.join(
            args.output_dir, f"{args.domain}_stability.json"
        )
        with open(stability_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "domain": args.domain,
                    "fold_pair": args.fold_pair,
                    "kendall_tau": tau,
                    "q5_threshold_tau": 0.7,
                    "q5_pass": bool(tau is not None and tau >= 0.7),
                },
                fh, indent=2,
            )
            fh.write("\n")
        print(f"      wrote {stability_path} (tau={tau})")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
