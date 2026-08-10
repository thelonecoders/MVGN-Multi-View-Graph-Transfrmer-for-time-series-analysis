"""
MVGTNet: The Proposed Full Model
================================
Multi-View Graph-Transformer Network.

Combines:
  1. MultiViewEmbedding (numeric + text + categorical)
  2. MultiViewGraphBuilder (A^multi = weighted combination of 4 views)
  3. HierarchicalAttention (time -> view -> graph)
  4. PFGAMultiView (F frozen + U unfrozen graph-attention layers, LoRA-augmented)
  5. Multi-task output heads (numeric regression + categorical + text generation)

This is the main entry point for training and inference.

Reference:
  This thesis, Chapters 6-8. Proposed formulas A-D are implemented here.
"""
import torch
import torch.nn as nn

from .embedding import MultiViewEmbedding
from .graph_builder import MultiViewGraphBuilder
from .attention import HierarchicalAttention
from .pfga import PFGAMultiView


class MVGTNet(nn.Module):
    """Multi-View Graph-Transformer Network.

    Args:
        config: dict with keys:
            - num_nodes (int)
            - input_dim (int)
            - hidden_dim (int)
            - lookback (int)
            - horizon (int)
            - use_text (bool, default True)
            - use_categorical (bool, default False)
            - num_categories (int, default 0)
            - text_model (str, default "bert-base-uncased")
            - graph_types (list, default all 4)
            - topk (int, default 8)
            - num_heads (int, default 4)
            - frozen_layers (int, default 6)
            - unfrozen_layers (int, default 2)
            - lora_rank (int, default 8)
            - use_qlora (bool, default False)
            - use_text_gen (bool, default False)
            - dropout (float, default 0.1)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_nodes = config["num_nodes"]
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.lookback = config["lookback"]
        self.horizon = config["horizon"]
        self.use_text = config.get("use_text", True)
        self.use_categorical = config.get("use_categorical", False)
        self.num_categories = config.get("num_categories", 0)

        D = self.hidden_dim
        D3 = 3 * D  # MultiViewEmbedding outputs (B, N, 3D)

        # 1. Multi-View Embedding Layer
        self.embedding = MultiViewEmbedding(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=D,
            lookback=self.lookback,
            use_text=self.use_text,
            use_categorical=self.use_categorical,
            num_categories=self.num_categories,
            text_model_name=config.get("text_model", "bert-base-uncased"),
        )

        # 2. Multi-View Graph Builder
        self.graph_builder = MultiViewGraphBuilder(
            num_nodes=self.num_nodes,
            graph_types=config.get("graph_types",
                                   ["spatial", "temporal", "semantic", "adaptive"]),
            topk=config.get("topk", 8),
        )

        # 3. Hierarchical Attention (time -> view -> graph)
        self.hierarchical_attention = HierarchicalAttention(
            hidden_dim=D3,
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
        )

        # 4. PFGA with Multi-View Graph
        self.pfga = PFGAMultiView(
            hidden_dim=D3,
            num_layers=config.get("frozen_layers", 6) + config.get("unfrozen_layers", 2),
            num_frozen_layers=config.get("frozen_layers", 6),
            num_unfrozen_layers=config.get("unfrozen_layers", 2),
            num_heads=config.get("num_heads", 4),
            lora_rank=config.get("lora_rank", 8),
            use_qlora=config.get("use_qlora", False),
            dropout=config.get("dropout", 0.1),
        )

        # 5. Multi-Task Output Heads
        # 5a. Numeric prediction (regression)
        self.numeric_head = nn.Conv1d(
            in_channels=D3,
            out_channels=self.horizon * self.input_dim,
            kernel_size=1,
        )

        # 5b. Categorical prediction (optional)
        if self.use_categorical and self.num_categories > 0:
            self.categorical_head = nn.Linear(D3, self.num_categories)
        else:
            self.categorical_head = None

        # 5c. Text generation head (optional)
        if config.get("use_text_gen", False):
            try:
                from transformers import T5ForConditionalGeneration, T5Config
                t5_config = T5Config(d_model=D3, d_kv=D3 // 4,
                                     num_layers=2, num_heads=4,
                                     vocab_size=32128)
                self.text_decoder = T5ForConditionalGeneration(t5_config)
            except ImportError:
                import warnings
                warnings.warn(
                    "transformers not installed; text generation head disabled. "
                    "Install with: pip install transformers"
                )
                self.text_decoder = None
        else:
            self.text_decoder = None

    def forward(self, x_numeric: torch.Tensor,
                x_text: dict = None,
                x_categorical: torch.Tensor = None,
                adj_spatial: torch.Tensor = None,
                return_attention: bool = False) -> dict:
        """
        Args:
            x_numeric:     (B, P, N, C) numerical time series
            x_text:        dict with "fact" and/or "preds" keys (B, P, max_len)
            x_categorical: (B, P, N) long tensor of category indices
            adj_spatial:   (N, N) pre-computed spatial adjacency
            return_attention: whether to return attention weights for interpretability

        Returns:
            dict with keys:
                - "numeric":     (B, S, N, C) numeric prediction
                - "categorical": (B, N, num_categories) logits (if use_categorical)
                - "text":        generated text IDs (if use_text_gen and not training)
                - "attention":   dict of attention weights (if return_attention)
                - "adj_multi":   (N, N) multi-view adjacency
                - "view_weights": dict of view weights
        """
        # 1. Multi-View Embedding
        h = self.embedding(x_numeric, x_text, x_categorical)  # (B, N, 3D)

        # 2. Build Multi-View Graph
        adj_multi, adj_components, view_weights = self.graph_builder(
            x_numeric, h, adj_spatial, x_text
        )

        # 3. Hierarchical Attention
        h, attn_weights = self.hierarchical_attention(h, adj_multi)

        # 4. PFGA with Multi-View Graph
        h = self.pfga(h, adj_multi)

        # 5. Multi-Task Outputs
        outputs = {}

        # 5a. Numeric prediction
        numeric_out = self.numeric_head(h.transpose(1, 2)).transpose(1, 2)
        numeric_out = numeric_out.view(
            numeric_out.size(0), self.num_nodes,
            self.horizon, self.input_dim,
        ).permute(0, 2, 1, 3)  # (B, S, N, C)
        outputs["numeric"] = numeric_out

        # 5b. Categorical prediction
        if self.categorical_head is not None:
            outputs["categorical"] = self.categorical_head(h)

        # 5c. Text generation (eval mode only)
        if self.text_decoder is not None and not self.training:
            # Use h as the encoder hidden state for T5 generation
            outputs["text"] = self.text_decoder.generate(
                encoder_outputs=type("", (), {
                    "last_hidden_state": h,
                    "encoder_attentions": None,
                    "encoder_hidden_states": None,
                })(),
                max_length=50,
                num_beams=4,
            )

        if return_attention:
            outputs["attention"] = attn_weights
            outputs["adj_components"] = adj_components
            outputs["adj_multi"] = adj_multi
            outputs["view_weights"] = view_weights

        return outputs

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def parameter_efficiency(self) -> dict:
        """Return parameter efficiency metrics (matches ST-LLM+ Table II)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.num_trainable_parameters()
        frozen = self.num_frozen_parameters()
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percentage": 100.0 * trainable / total if total > 0 else 0,
            "frozen_percentage": 100.0 * frozen / total if total > 0 else 0,
        }
