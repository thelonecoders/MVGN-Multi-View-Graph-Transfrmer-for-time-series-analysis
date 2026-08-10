"""
LoRA: Low-Rank Adaptation for Linear Layers
===========================================
Faithful reproduction of the LoRA module used in ST-LLM+ (Equations 11-13).

Instead of updating the full weight matrix W in a linear layer, LoRA
introduces two low-rank matrices L (shape: in_features x r) and M
(shape: r x out_features) such that the effective weight becomes:

    W' = W + L @ M

where r << min(in_features, out_features). This reduces trainable
parameters from (in_features * out_features) to r * (in_features + out_features).

Reference:
  Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models."
  ICLR 2022. (Cited as ref [68] in this thesis.)
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA low-rank adaptation.

    Args:
        in_features:  input dimension
        out_features: output dimension
        r:            LoRA rank (default 8; use 16 for higher capacity)
        bias:         whether to include bias in the base linear layer
        dropout:      dropout probability applied to LoRA input (default 0.0)

    Forward:
        x: (..., in_features) -> (..., out_features)

    Trainable parameters:
        - Base layer weight W is FROZEN (requires_grad=False)
        - Base layer bias b is FROZEN (if present)
        - LoRA matrices L (in_features x r) and M (r x out_features) are TRAINABLE
        - Scaling factor alpha/r is applied to the LoRA output

    Formula:
        W' = W + (alpha / r) * L @ M
        y  = x @ W'^T + b
    """

    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, bias: bool = True, dropout: float = 0.0,
                 alpha: int = 16):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        assert r <= min(in_features, out_features), \
            f"LoRA rank r={r} must be <= min(in={in_features}, out={out_features})"

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # Base linear layer (frozen)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA matrices (trainable). Initialize L with kaiming, M with zeros
        # so the initial LoRA output is zero (model starts identical to base).
        self.lora_L = nn.Parameter(torch.empty(in_features, r))
        self.lora_M = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_L, a=5 ** 0.5)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base (frozen) forward pass
        base_out = self.base(x)
        # LoRA delta: x @ L @ M * scaling
        lora_in = self.lora_dropout(x)
        lora_out = (lora_in @ self.lora_L) @ self.lora_M * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, scaling={self.scaling:.4f}")

    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters (LoRA only)."""
        return self.lora_L.numel() + self.lora_M.numel()

    def merge(self) -> nn.Linear:
        """Merge LoRA delta into the base weight and return a plain nn.Linear.

        Useful for inference efficiency.
        """
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight +
                                (self.lora_L @ self.lora_M).T * self.scaling)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
