"""
MVGT-Net: Causal Sensitivity Analysis (Section 6-19)
=====================================================

Implements the leave-one-view-out sensitivity probe described in
Section 6-19 of the thesis. This is an OBSERVATIONAL approximation
to a counterfactual; it does not recover the true causal effect
without additional identifying assumptions (Pearl 2009).

Theory:
    Sensitivity(v, i) = || f(x; A^multi) - f(x; A^multi with A^v = I) ||_2

    A large sensitivity score indicates that the model relies on view v
    for this prediction; a small score indicates that view v is redundant
    given the other views.

Limitations (documented in Section 6-19):
    - The probe assumes views are independent; in practice the adaptive
      and semantic views both depend on learnable embeddings.
    - For correlated views, the score underestimates the held-out view's
      true contribution.
    - A full causal analysis requires a structural causal model or
      instrumental variable; this is a diagnostic, not a causal claim.

References:
    Pearl, J. "Causality: Models, Reasoning, and Inference."
        Cambridge University Press, 2nd ed., 2009.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SensitivityConfig:
    """Configuration for the causal sensitivity probe.

    Attributes:
        views_to_probe: list of view names to ablate. Must match the
            view names registered in the model's graph builder.
        normalize: if True, divide the sensitivity score by the norm of
            the original prediction to get a per-node score in [0, 1].
        aggregation: how to aggregate per-node scores across the batch.
            'mean' returns the mean; 'max' returns the max.
    """
    views_to_probe: List[str] = field(
        default_factory=lambda: ["spatial", "temporal", "semantic", "adaptive"]
    )
    normalize: bool = True
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.aggregation not in ("mean", "max"):
            raise ValueError(
                f"aggregation must be 'mean' or 'max', got {self.aggregation}"
            )


class CausalSensitivityProbe(nn.Module):
    """Leave-one-view-out sensitivity probe for MVGT-Net.

    Wraps a trained MVGT-Net model and, for each view v in the config,
    replaces A^v with the identity matrix at inference time, then
    measures the change in the model's prediction. The change is the
    sensitivity score for that view.

    Usage:
        model = MVGTNet(...)
        model.load_state_dict(torch.load("checkpoints/best.pt"))
        probe = CausalSensitivityProbe(model, SensitivityConfig())
        scores = probe.compute_sensitivities(test_batch)
        # scores = {"spatial": 0.23, "temporal": 0.45, ...}

    Note: this probe is post-hoc and requires NO retraining. It performs
    one forward pass per view per batch, so the cost is O(V * 4) forward
    passes for a batch with V nodes.
    """

    def __init__(self, model: nn.Module, config: Optional[SensitivityConfig] = None) -> None:
        super().__init__()
        self.model = model
        self.config = config or SensitivityConfig()

    @torch.no_grad()
    def compute_sensitivities(
        self,
        inputs: torch.Tensor,
        adjacency_views: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute the leave-one-view-out sensitivity score for each view.

        Args:
            inputs: model inputs, shape and type per MVGTNet.forward.
            adjacency_views: dict mapping view name to its adjacency
                matrix. Each value must have shape (N, N) where N is the
                number of nodes. The dict must contain all views in
                self.config.views_to_probe.

        Returns:
            Dict mapping view name to its sensitivity score. Higher score
            means the model relies more on that view.
        """
        self.model.eval()
        # Original prediction with all views active.
        baseline_pred = self.model(inputs, adjacency_views)
        baseline_norm = baseline_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        scores: Dict[str, float] = {}
        for view_name in self.config.views_to_probe:
            if view_name not in adjacency_views:
                raise KeyError(
                    f"View '{view_name}' not found in adjacency_views. "
                    f"Available: {list(adjacency_views.keys())}"
                )
            # Replace this view's adjacency with the identity matrix.
            ablated_views = dict(adjacency_views)
            n = adjacency_views[view_name].shape[-1]
            eye = torch.eye(
                n,
                dtype=adjacency_views[view_name].dtype,
                device=adjacency_views[view_name].device,
            )
            ablated_views[view_name] = eye
            ablated_pred = self.model(inputs, ablated_views)
            delta = (baseline_pred - ablated_pred).norm(dim=-1)
            if self.config.normalize:
                delta = delta / baseline_norm.squeeze(-1)
            if self.config.aggregation == "mean":
                scores[view_name] = float(delta.mean().item())
            else:  # max
                scores[view_name] = float(delta.max().item())
        return scores

    @torch.no_grad()
    def compute_per_node_sensitivities(
        self,
        inputs: torch.Tensor,
        adjacency_views: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-node sensitivity scores (no aggregation across nodes).

        Returns:
            Dict mapping view name to a tensor of shape (N,) containing
            the per-node sensitivity score for that view.
        """
        self.model.eval()
        baseline_pred = self.model(inputs, adjacency_views)
        baseline_norm = baseline_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        per_node: Dict[str, torch.Tensor] = {}
        for view_name in self.config.views_to_probe:
            ablated_views = dict(adjacency_views)
            n = adjacency_views[view_name].shape[-1]
            eye = torch.eye(
                n,
                dtype=adjacency_views[view_name].dtype,
                device=adjacency_views[view_name].device,
            )
            ablated_views[view_name] = eye
            ablated_pred = self.model(inputs, ablated_views)
            delta = (baseline_pred - ablated_pred).norm(dim=-1)
            if self.config.normalize:
                delta = delta / baseline_norm.squeeze(-1)
            per_node[view_name] = delta.detach().cpu()
        return per_node
