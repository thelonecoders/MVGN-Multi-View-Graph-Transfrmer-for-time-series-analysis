"""Factory function returning a stub instance of any of the 12 baselines.

Usage
-----
>>> from baselines import build_baseline
>>> m = build_baseline("iTransformer")
>>> m
<iTransformer name='iTransformer' lookback=96 horizon=96>
>>> m.forward(x)        # raises NotImplementedError -- see base_baseline.py
"""

from __future__ import annotations

from typing import Dict

from .base_baseline import BaseBaseline, BaselineConfig
from .transformer import TransformerBaseline
from .reformer import ReformerBaseline
from .informer import InformerBaseline
from .autoformer import AutoformerBaseline
from .crossformer import CrossformerBaseline
from .nonstationary_transformer import NonstationaryTransformerBaseline
from .fedformer import FEDformerBaseline
from .itransformer import iTransformerBaseline
from .dlinear import DLinearBaseline
from .film import FiLMBaseline
from .timesnet import TimesNetBaseline
from .patchtst import PatchTSTBaseline


_REGISTRY: Dict[str, type] = {
    "Transformer": TransformerBaseline,
    "Reformer": ReformerBaseline,
    "Informer": InformerBaseline,
    "Autoformer": AutoformerBaseline,
    "Crossformer": CrossformerBaseline,
    "Non-stationary Transformer": NonstationaryTransformerBaseline,
    "FEDformer": FEDformerBaseline,
    "iTransformer": iTransformerBaseline,
    "DLinear": DLinearBaseline,
    "FiLM": FiLMBaseline,
    "TimesNet": TimesNetBaseline,
    "PatchTST": PatchTSTBaseline,
}


def build_baseline(name: str, config: BaselineConfig | None = None) -> BaseBaseline:
    """Instantiate the named baseline.

    Parameters
    ----------
    name : str
        Canonical baseline name. Must be one of the 12 keys in the
        registry (see :func:`list_available_baselines`).
    config : BaselineConfig, optional
        Hyper-parameter override. If None, defaults from the subclass
        are used.

    Returns
    -------
    BaseBaseline
        A stub instance. ``forward()`` and ``train_step()`` raise
        :class:`NotImplementedError` until the subclass body is
        replaced with the original authors' code.

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown baseline {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[name]
    if config is None:
        config = BaselineConfig(name=name)
    return cls(config=config)
