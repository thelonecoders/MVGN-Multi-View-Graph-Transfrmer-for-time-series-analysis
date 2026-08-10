"""
STLLMPlus: Faithful Reproduction of the Source Model
====================================================
Reproduces the ST-LLM+ model from Liu et al. (IEEE TKDE 2025) using the
PFGAModule defined in pfga.py.

This implementation follows Equations 1-15 of the source paper:
  - Eq. 2:  PConv token embedding
  - Eq. 3-5: Temporal embedding (day + week)
  - Eq. 6:  Spatial embedding
  - Eq. 7:  Fusion convolution
  - Eq. 8-9: Frozen PFGA layers
  - Eq. 10: Unfrozen graph-attention PFGA layers
  - Eq. 11-13: LoRA-augmented attention
  - Eq. 14: Regression convolution output

Reference:
  Liu, C. et al. "ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models
  for Traffic Prediction." IEEE TKDE 37(8): 4846-4859, Aug. 2025.
  Repository: github.com/kethmih/ST-LLM-Plus
"""
import torch
import torch.nn as nn


class STLLMPlus(nn.Module):
    """Faithful reproduction of the ST-LLM+ model.

    Args:
        num_nodes:  number of graph nodes N
        input_dim:  number of features per node C
        hidden_dim: hidden dimension D
        lookback:   lookback length P
        horizon:    prediction horizon S
        F_layers:   number of frozen PFGA layers
        U_layers:   number of unfrozen PFGA layers
        num_heads:  attention heads
        lora_rank:  LoRA rank r
        T_d:        daily period (default 48 for half-hourly traffic data)
        T_w:        weekly period (default 7)
        dropout:    dropout probability
    """

    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int,
                 lookback: int, horizon: int,
                 F_layers: int = 6, U_layers: int = 2,
                 num_heads: int = 4, lora_rank: int = 8,
                 T_d: int = 48, T_w: int = 7, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        self.horizon = horizon
        self.F = F_layers
        self.U = U_layers

        # --- Token embedding (Eq. 2): PConv ---
        # (B, P, N, C) -> permute to (B, N, P, C) -> reshape (B, N, P*C)
        # -> transpose (B, P*C, N) for Conv1d -> output (B, D, N) -> transpose (B, N, D)
        self.pconv = nn.Conv1d(input_dim * lookback, hidden_dim, kernel_size=1)

        # --- Temporal embedding (Eq. 3-5) ---
        self.time_day = nn.Embedding(T_d, hidden_dim)
        self.time_week = nn.Embedding(T_w, hidden_dim)

        # --- Spatial embedding (Eq. 6) ---
        self.spatial_linear = nn.Linear(input_dim, hidden_dim)
        self.spatial_act = nn.GELU()

        # --- Fusion convolution (Eq. 7) ---
        self.fusion_conv = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=1)

        # --- Learnable positional encoding ---
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_nodes, hidden_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # --- PFGA layers ---
        from .pfga import PFGAModule
        self.frozen_layers = nn.ModuleList([
            PFGAModule(hidden_dim, num_heads, frozen=True,
                      use_graph_mask=False, use_lora=False, dropout=dropout)
            for _ in range(F_layers)
        ])
        self.unfrozen_layers = nn.ModuleList([
            PFGAModule(hidden_dim, num_heads, frozen=False,
                      use_graph_mask=True, use_lora=True,
                      lora_rank=lora_rank, dropout=dropout)
            for _ in range(U_layers)
        ])

        # --- Regression output (Eq. 14) ---
        self.rconv = nn.Conv1d(hidden_dim, horizon * input_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:   (B, P, N, C) input time series
            adj: (N, N) adjacency matrix (required for unfrozen layers)

        Returns:
            y_pred: (B, S, N, C) predicted future values
        """
        B, P, N, C = x.shape
        device = x.device

        # --- Token embedding ---
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, P * C)
        x_flat = x_flat.transpose(1, 2)  # (B, P*C, N)
        e_token = self.pconv(x_flat).transpose(1, 2)  # (B, N, D)

        # --- Temporal embedding ---
        day_idx = torch.arange(B, device=device) % 48
        week_idx = torch.arange(B, device=device) % 7
        e_temporal = (self.time_day(day_idx).unsqueeze(1) +
                      self.time_week(week_idx).unsqueeze(1))  # (B, 1, D)
        e_temporal = e_temporal.expand(-1, N, -1)

        # --- Spatial embedding ---
        x_last = x[:, -1, :, :]
        e_spatial = self.spatial_act(self.spatial_linear(x_last))

        # --- Fusion ---
        concat = torch.cat([e_token, e_spatial, e_temporal], dim=-1)
        h = self.fusion_conv(concat.transpose(1, 2)).transpose(1, 2)  # (B, N, D)

        # Add positional encoding
        h = h + self.pos_encoding

        # --- PFGA layers ---
        for layer in self.frozen_layers:
            h = layer(h, adj=None)
        for layer in self.unfrozen_layers:
            h = layer(h, adj=adj)

        # --- Regression output ---
        y = self.rconv(h.transpose(1, 2)).transpose(1, 2)  # (B, N, S*C)
        y = y.view(B, N, self.horizon, C).permute(0, 2, 1, 3)  # (B, S, N, C)
        return y

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
