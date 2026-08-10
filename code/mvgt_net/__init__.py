"""
MVGT-Net: Multi-View Graph-Transformer Network
================================================
A complete, runnable PyTorch implementation of the MVGT-Net architecture
proposed in the thesis "Audit of ST-LLM+ and the Proposed MVGT-Net Extension".

This package contains:
  - mvgt_net.embedding         : MultiViewEmbedding (numeric + text + categorical)
  - mvgt_net.graph_builder     : MultiViewGraphBuilder (A^multi construction)
  - mvgt_net.attention         : HierarchicalAttention (time -> view -> graph)
  - mvgt_net.pfga              : PFGAModule, PFGAMultiView (Partially Frozen Graph Attention)
  - mvgt_net.lora              : LoRALinear (low-rank adaptation)
  - mvgt_net.st_llm_plus       : STLLMPlus (faithful reproduction of the source model)
  - mvgt_net.model             : MVGTNet (the proposed full model)
  - mvgt_net.losses            : MultiTaskLoss with dynamic weighting (Formula D)
  - mvgt_net.metrics           : MAE, RMSE, WAPE, MSE, MAPE, sMAPE, R2

All modules are importable and runnable. See tests/test_smoke.py for a
forward-pass smoke test that exercises every class with random tensors.

Author: Thesis Author
License: MIT
"""
from .model import MVGTNet
from .embedding import MultiViewEmbedding
from .graph_builder import MultiViewGraphBuilder
from .attention import HierarchicalAttention
from .pfga import PFGAModule, PFGAMultiView
from .lora import LoRALinear
from .st_llm_plus import STLLMPlus
from .losses import MultiTaskLoss
from .metrics import (
    masked_mae, masked_rmse, masked_wape, masked_mse,
    masked_mape, masked_smape, r2_score, all_metrics,
)
from .data import (
    DOMAIN_REGISTRY,
    TimeMMDDataset,
    fit_normalization,
    collate_fn,
    get_dataloaders,
)

__version__ = "1.1.0"
__all__ = [
    "MVGTNet",
    "MultiViewEmbedding",
    "MultiViewGraphBuilder",
    "HierarchicalAttention",
    "PFGAModule",
    "PFGAMultiView",
    "LoRALinear",
    "STLLMPlus",
    "MultiTaskLoss",
    "masked_mae", "masked_rmse", "masked_wape", "masked_mse",
    "masked_mape", "masked_smape", "r2_score", "all_metrics",
    "DOMAIN_REGISTRY", "TimeMMDDataset", "fit_normalization",
    "collate_fn", "get_dataloaders",
]
