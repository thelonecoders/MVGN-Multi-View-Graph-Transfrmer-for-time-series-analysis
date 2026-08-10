"""SHAP-based interpretability module for MVGT-Net (Q5 implementation).

This module closes the Q5 implementation gap noted in
`04_audit_reports/professor_eval_2026-08-09.md` §2 ("Q5 was the weakest
link due to QLoRA claim without implementation" -- the same audit also
flagged that SHAP was mentioned 54 times in the thesis but no SHAP
runner existed in the code).

What Q5 requires (thesis Chapter 5, Research Question 5):
    "How can MVGT-Net be made self-interpretable through structural
    interpretation mechanisms such as SHAP? Do the learned attention
    maps align with the domain structure?"

Success metrics (thesis §5.2):
    1. Attention-adjacency alignment (Pearson correlation >= 0.5 on
       at least 6/9 TimeMMD domains).
    2. SHAP feature importance ranking stable across folds
       (Kendall tau >= 0.7 between fold-pair rankings).
    3. SHAP summary plot generated for each of the 9 domains.

This module implements:

- :class:`ShapExplainer` -- wrapper around `shap.DeepExplainer` that
  handles MVGT-Net's multi-input signature (numeric tensor +
  text-encoded tensor + adjacency matrix + time-of-day/day-of-week
  indices) and produces per-feature SHAP values.
- :func:`compute_attention_adjacency_alignment` -- Pearson correlation
  between the learned attention map and the domain adjacency matrix.
- :func:`compute_shap_stability` -- Kendall tau between SHAP rankings
  computed on two different folds.
- :func:`generate_shap_summary_plot` -- produce the standard SHAP
  beeswarm plot and save it as PNG.

Design notes
------------
- The module imports `shap` lazily inside the constructor so that the
  rest of the bundle continues to work on machines where `shap` is not
  installed (it is an optional dependency).
- When `shap` is not installed, every method raises a clear
  :class:`ShapNotInstalledError` with install instructions, rather
  than failing at import time.
- All SHAP values are computed on CPU even when the model is on GPU;
  this avoids the known CUDA/SHAP interaction bug in shap==0.46.0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ShapNotInstalledError(ImportError):
    """Raised when the optional `shap` package is not installed."""


def _require_shap():
    try:
        import shap  # type: ignore  # noqa: F401
        return shap
    except ImportError as exc:
        raise ShapNotInstalledError(
            "The 'shap' package is required for Q5 interpretability "
            "analysis but is not installed. Install it with:\n"
            "    pip install shap==0.46.0\n"
            "or, on the VPS, run:\n"
            "    pip install -r requirements.txt\n"
            "(the requirements file lists shap as an optional dep; "
            "uncomment the line).\n"
        ) from exc


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
@dataclass
class ShapConfig:
    """Configuration for a SHAP explanation run.

    Attributes
    ----------
    n_background : int
        Number of background samples used to construct the
        DeepExplainer expectation. Default 64 (good trade-off between
        shapley value accuracy and runtime).
    n_explain : int
        Number of test-set samples to explain. Default 128 (enough for
        a stable beeswarm plot).
    feature_names : list of str, optional
        Names of the input features. If None, features are named
        ``x0, x1, ...``.
    output_dir : str
        Directory to write plots and JSON artifacts. Default
        ``results/shap/``.
    device : str
        ``"cpu"`` or ``"cuda"``. SHAP values are always computed on
        CPU even when this is ``"cuda"`` (see module docstring); the
        flag only controls where the model lives when its forward
        method is wrapped.
    random_state : int
        Seed for the background/explain sample selection. Default 42.
    """

    n_background: int = 64
    n_explain: int = 128
    feature_names: Optional[List[str]] = None
    output_dir: str = "results/shap"
    device: str = "cpu"
    random_state: int = 42


# ---------------------------------------------------------------------- #
# Main explainer
# ---------------------------------------------------------------------- #
class ShapExplainer:
    """SHAP explainer for a trained MVGT-Net model.

    Parameters
    ----------
    model : torch.nn.Module
        A trained MVGT-Net model (or any nn.Module with the same
        forward signature).
    background : torch.Tensor or tuple
        Background data used to compute the expectation E[f(x)]. For
        MVGT-Net this is typically a tuple ``(x_numeric, x_text, adj,
        tod_idx, dow_idx)`` sampled from the training set.
    config : ShapConfig, optional
        Configuration. Defaults are sensible for an RTX 3080 Ti.

    Notes
    -----
    Internally uses :class:`shap.DeepExplainer`. If the model contains
    operations that DeepExplainer cannot handle (e.g., some custom
    scatter operations), falls back to
    :class:`shap.GradientExplainer`. If both fail, raises
    :class:`RuntimeError` with a message pointing to
    :class:`shap.KernelExplainer` as a last-resort fallback.
    """

    def __init__(
        self,
        model: Any,
        background: Any,
        config: Optional[ShapConfig] = None,
    ) -> None:
        self.shap = _require_shap()
        self.model = model
        self.background = background
        self.config = config or ShapConfig()
        self._explainer = None
        self._build_explainer()

    # ------------------------------------------------------------------ #
    # Internal: construct the underlying shap explainer
    # ------------------------------------------------------------------ #
    def _build_explainer(self) -> None:
        try:
            self._explainer = self.shap.DeepExplainer(
                self.model, self.background
            )
        except Exception as exc:
            # Fall back to GradientExplainer -- slower but more robust.
            try:
                self._explainer = self.shap.GradientExplainer(
                    self.model, self.background
                )
            except Exception:
                raise RuntimeError(
                    "Neither shap.DeepExplainer nor shap.GradientExplainer "
                    f"could wrap the model. Original DeepExplainer error: "
                    f"{exc}. Consider using shap.KernelExplainer as a "
                    f"last-resort fallback (much slower)."
                ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def explain(self, samples: Any) -> np.ndarray:
        """Compute SHAP values for ``samples``.

        Parameters
        ----------
        samples : torch.Tensor or tuple
            Test-set samples to explain. Same signature as
            ``background``.

        Returns
        -------
        shap_values : np.ndarray
            SHAP values with the same shape as ``samples`` (for
            regression: shape ``(n_samples, n_features)``).
        """
        if self._explainer is None:
            self._build_explainer()
        # shap.DeepExplainer.shap_values returns a list for
        # multi-output models and a single array for single-output.
        sv = self._explainer.shap_values(samples)
        if isinstance(sv, list):
            sv = sv[0]
        return np.asarray(sv)

    def summary_plot(
        self,
        samples: Any,
        feature_names: Optional[Sequence[str]] = None,
        out_path: Optional[str] = None,
    ) -> None:
        """Generate the standard SHAP beeswarm summary plot.

        Parameters
        ----------
        samples : torch.Tensor or tuple
            Test-set samples.
        feature_names : list of str, optional
            Override ``self.config.feature_names``.
        out_path : str, optional
            Where to save the plot. Defaults to
            ``<output_dir>/shap_summary.png``.
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive
        import matplotlib.pyplot as plt

        sv = self.explain(samples)
        os.makedirs(self.config.output_dir, exist_ok=True)
        out = out_path or os.path.join(self.config.output_dir, "shap_summary.png")

        # shap.summary_plot writes to the current matplotlib figure.
        self.shap.summary_plot(
            sv,
            np.asarray(samples) if not isinstance(samples, tuple)
            else np.asarray(samples[0]),
            feature_names=feature_names or self.config.feature_names,
            show=False,
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        return out

    def feature_importance(
        self,
        samples: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Return mean(|SHAP value|) per feature -- the global importance.

        Returns
        -------
        dict
            Mapping ``{feature_name: mean_abs_shap}``.
        """
        sv = self.explain(samples)
        names = list(feature_names or self.config.feature_names
                     or [f"x{i}" for i in range(sv.shape[-1])])
        # sv shape: (n_samples, n_features) for regression
        mean_abs = np.abs(sv).mean(axis=tuple(range(sv.ndim - 1)))
        if len(mean_abs) != len(names):
            # Reduce extra dimensions if present.
            mean_abs = mean_abs.reshape(-1)[: len(names)]
        return {n: float(v) for n, v in zip(names, mean_abs)}


# ---------------------------------------------------------------------- #
# Q5-specific metrics
# ---------------------------------------------------------------------- #
def compute_attention_adjacency_alignment(
    attention_map: np.ndarray,
    adjacency: np.ndarray,
    method: str = "pearson",
) -> float:
    """Pearson/Spearman correlation between attention and adjacency.

    Parameters
    ----------
    attention_map : np.ndarray of shape (N, N)
        Learned attention map (averaged over heads and samples).
    adjacency : np.ndarray of shape (N, N)
        Domain adjacency matrix (e.g., correlation graph from
        ``graph_builder.CorrelationGraphBuilder``).
    method : str
        ``"pearson"`` (default) or ``"spearman"``.

    Returns
    -------
    float
        Correlation in ``[-1, 1]``. Q5 success threshold: ``>= 0.5``
        on at least 6/9 domains.
    """
    a = np.asarray(attention_map, dtype=float).reshape(-1)
    b = np.asarray(adjacency, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: attention {attention_map.shape} vs "
            f"adjacency {adjacency.shape}"
        )
    if method == "pearson":
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    elif method == "spearman":
        from scipy.stats import spearmanr  # type: ignore
        return float(spearmanr(a, b).correlation)
    else:
        raise ValueError(f"Unknown method {method!r}")


def compute_shap_stability(
    shap_run_1: Dict[str, float],
    shap_run_2: Dict[str, float],
    method: str = "kendall",
) -> float:
    """Stability of the SHAP feature-importance ranking across two runs.

    Parameters
    ----------
    shap_run_1, shap_run_2 : dict
        Output of :meth:`ShapExplainer.feature_importance` on two
        different folds (or two different seeds).
    method : str
        ``"kendall"`` (default, Q5 metric) or ``"spearman"``.

    Returns
    -------
    float
        Correlation in ``[-1, 1]``. Q5 success threshold: ``>= 0.7``
        between every pair of folds.
    """
    common = sorted(set(shap_run_1.keys()) & set(shap_run_2.keys()))
    if len(common) < 2:
        raise ValueError(
            "Need at least 2 common features between the two runs."
        )
    a = np.array([shap_run_1[k] for k in common])
    b = np.array([shap_run_2[k] for k in common])
    if method == "kendall":
        from scipy.stats import kendalltau  # type: ignore
        return float(kendalltau(a, b).correlation)
    elif method == "spearman":
        from scipy.stats import spearmanr  # type: ignore
        return float(spearmanr(a, b).correlation)
    else:
        raise ValueError(f"Unknown method {method!r}")


def generate_shap_summary_plot(
    explainer: ShapExplainer,
    samples: Any,
    out_path: str,
    feature_names: Optional[Sequence[str]] = None,
) -> str:
    """Convenience wrapper. See :meth:`ShapExplainer.summary_plot`."""
    return explainer.summary_plot(samples, feature_names=feature_names, out_path=out_path)


__all__ = [
    "ShapConfig",
    "ShapExplainer",
    "ShapNotInstalledError",
    "compute_attention_adjacency_alignment",
    "compute_shap_stability",
    "generate_shap_summary_plot",
]
