"""
Multi-View Graph Builder
========================
Constructs the multi-view adjacency matrix A^multi (Proposed Formula A):

    A^multi = alpha * A^spatial + beta * A^temporal + gamma * A^semantic + delta * A^adaptive

where the four weights are learned via a softmax-constrained parameter vector
so that alpha + beta + gamma + delta = 1 (single softmax is mathematically
sufficient; a second softmax would be redundant).

Each view A^view is constructed as follows:
  - A^spatial:  pre-computed geographic adjacency (passed in by caller)
  - A^temporal: Pearson correlation of each node's time series, thresholded
  - A^semantic: cosine similarity of text embeddings (if available)
  - A^adaptive: learned via a trainable embedding matrix E @ E^T

Top-k sparsification is applied to control complexity:
    E^multi <= k * V (default k=8)

All A^view matrices are row-normalized before combination to ensure stable
spectral properties (the original ST-LLM+ does not row-normalize, but
MVGT-Net adds this for stability under multi-view combination).

Reference:
  - GraphWaveNet (Wu et al., IJCAI 2019): adaptive graph via E @ E^T
  - MTGNN (Wu et al., NeurIPS 2020): adaptive graph with top-k sparsification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize an adjacency matrix: A_norm[i] = A[i] / sum(A[i]).

    Args:
        A: (..., N, N) adjacency matrix
        eps: small constant to avoid division by zero

    Returns:
        Row-normalized adjacency matrix with the same shape.
    """
    row_sum = A.sum(dim=-1, keepdim=True).clamp(min=eps)
    return A / row_sum


def topk_sparsify(A: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k largest entries per row; set the rest to zero.

    Args:
        A: (..., N, N) adjacency matrix
        k: number of edges to keep per node

    Returns:
        Sparsified adjacency matrix (same shape).
    """
    N = A.shape[-1]
    if k >= N:
        return A
    # Find the threshold per row: the k-th largest value
    topk_values, _ = A.topk(k, dim=-1, largest=True, sorted=False)
    # Use the minimum of the top-k as the threshold (per row)
    threshold = topk_values.min(dim=-1, keepdim=True).values
    mask = A >= threshold
    return A * mask.float()


class MultiViewGraphBuilder(nn.Module):
    """Constructs A^multi from four dependency types.

    Args:
        num_nodes:    number of graph nodes N
        graph_types:  list of view names to include (default: all 4)
        topk:         top-k sparsification parameter (default 8)
        adaptive_dim: dimension of the learnable adaptive embedding (default 16)
    """

    def __init__(self, num_nodes: int,
                 graph_types: list = None,
                 topk: int = 8,
                 adaptive_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_types = graph_types or ["spatial", "temporal", "semantic", "adaptive"]
        self.topk = topk

        # Learnable view weights (4 parameters, softmax-constrained)
        # Initialize to uniform: each view starts with weight 0.25
        n_views = len(self.graph_types)
        self.view_logits = nn.Parameter(torch.zeros(n_views))

        # Adaptive graph: learnable node embedding matrix
        if "adaptive" in self.graph_types:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(num_nodes, adaptive_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None

    def get_view_weights(self) -> dict:
        """Return the current view weights as a dict {view_name: weight}."""
        weights = F.softmax(self.view_logits, dim=0)
        return {name: float(w) for name, w in zip(self.graph_types, weights)}

    def _build_temporal_graph(self, x_numeric: torch.Tensor) -> torch.Tensor:
        """Build A^temporal from Pearson correlation of node time series.

        Args:
            x_numeric: (B, P, N, C) numeric time series

        Returns:
            A_temporal: (N, N) averaged Pearson correlation matrix
        """
        # Average over batch and features: (N, P)
        # We compute Pearson correlation between each pair of nodes' time series.
        B, P, N, C = x_numeric.shape
        # Reshape to (N, B*P*C) for correlation computation
        node_series = x_numeric.permute(2, 0, 1, 3).reshape(N, -1)  # (N, B*P*C)
        # Center each node's series
        node_series = node_series - node_series.mean(dim=-1, keepdim=True)
        # Compute correlation = cov / (std_i * std_j)
        # cov = node_series @ node_series^T / (T-1)
        cov = node_series @ node_series.t()  # (N, N)
        std = node_series.std(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        corr = cov / (std * std.t() * (node_series.shape[-1] - 1))
        # Take absolute value (we care about dependency, not direction)
        A_temporal = corr.abs()
        return A_temporal

    def _build_semantic_graph(self, x_text: dict = None,
                              h: torch.Tensor = None) -> torch.Tensor:
        """Build A^semantic from cosine similarity of text or node embeddings.

        Args:
            x_text: optional dict with "fact"/"preds" keys
            h:      (B, N, 3D) node embeddings (fallback if no text)

        Returns:
            A_semantic: (N, N) cosine similarity matrix
        """
        N = self.num_nodes
        if h is not None:
            # Use the numeric embedding (first D dims of h) as a proxy for node semantics
            # Average over batch: (N, D)
            emb = h[:, :, :h.shape[-1] // 3].mean(dim=0)  # (N, D)
            # Cosine similarity
            emb_norm = F.normalize(emb, p=2, dim=-1)
            A_semantic = emb_norm @ emb_norm.t()  # (N, N)
            return A_semantic.clamp(min=0)  # Non-negative
        return torch.eye(N, device=h.device if h is not None else "cpu")

    def _build_adaptive_graph(self, device) -> torch.Tensor:
        """Build A^adaptive = E @ E^T (learnable).

        Returns:
            A_adaptive: (N, N) learned adjacency
        """
        if self.adaptive_embedding is None:
            return torch.eye(self.num_nodes, device=device)
        E = self.adaptive_embedding.to(device)
        A_adaptive = E @ E.t()  # (N, N)
        return A_adaptive.clamp(min=0)

    def forward(self, x_numeric: torch.Tensor,
                h: torch.Tensor = None,
                adj_spatial: torch.Tensor = None,
                x_text: dict = None) -> tuple:
        """
        Args:
            x_numeric:   (B, P, N, C) numeric time series
            h:           (B, N, 3D) node embeddings (for semantic graph)
            adj_spatial: (N, N) pre-computed spatial adjacency (optional)
            x_text:      dict with text inputs (optional)

        Returns:
            adj_multi:      (N, N) combined multi-view adjacency
            adj_components: dict of individual A^view matrices (for interpretability)
            view_weights:   dict of current view weights {name: float}
        """
        B, P, N, C = x_numeric.shape
        device = x_numeric.device
        assert N == self.num_nodes

        adj_components = {}

        # Build each view
        if "spatial" in self.graph_types:
            if adj_spatial is not None:
                A_spatial = adj_spatial.to(device)
            else:
                # Fallback: identity matrix (no spatial info)
                A_spatial = torch.eye(N, device=device)
            adj_components["spatial"] = A_spatial
        else:
            A_spatial = torch.zeros(N, N, device=device)

        if "temporal" in self.graph_types:
            A_temporal = self._build_temporal_graph(x_numeric)
            adj_components["temporal"] = A_temporal
        else:
            A_temporal = torch.zeros(N, N, device=device)

        if "semantic" in self.graph_types:
            A_semantic = self._build_semantic_graph(x_text, h)
            adj_components["semantic"] = A_semantic
        else:
            A_semantic = torch.zeros(N, N, device=device)

        if "adaptive" in self.graph_types:
            A_adaptive = self._build_adaptive_graph(device)
            adj_components["adaptive"] = A_adaptive
        else:
            A_adaptive = torch.zeros(N, N, device=device)

        # Row-normalize each view before combination
        A_spatial_n = row_normalize(A_spatial)
        A_temporal_n = row_normalize(A_temporal)
        A_semantic_n = row_normalize(A_semantic)
        A_adaptive_n = row_normalize(A_adaptive)

        # Combine with softmax-constrained weights
        weights = F.softmax(self.view_logits, dim=0)  # (n_views,)
        view_weight_map = {name: w for name, w in zip(self.graph_types, weights)}

        adj_multi = torch.zeros(N, N, device=device)
        if "spatial" in self.graph_types:
            adj_multi = adj_multi + view_weight_map["spatial"] * A_spatial_n
        if "temporal" in self.graph_types:
            adj_multi = adj_multi + view_weight_map["temporal"] * A_temporal_n
        if "semantic" in self.graph_types:
            adj_multi = adj_multi + view_weight_map["semantic"] * A_semantic_n
        if "adaptive" in self.graph_types:
            adj_multi = adj_multi + view_weight_map["adaptive"] * A_adaptive_n

        # Top-k sparsification
        if self.topk < N:
            adj_multi = topk_sparsify(adj_multi, self.topk)
            # Re-normalize after sparsification
            adj_multi = row_normalize(adj_multi)

        return adj_multi, adj_components, self.get_view_weights()
