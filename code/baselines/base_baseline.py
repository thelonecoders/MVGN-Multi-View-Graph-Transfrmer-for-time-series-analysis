"""Abstract base class for all TimeMMD baselines.

Each baseline follows the same interface so that the experiment runner
(`scripts/run_all_experiments.py`) can sweep them uniformly. Concrete
subclasses live in `transformer.py`, `informer.py`, etc.

Design notes
------------
- The base class intentionally does **not** require PyTorch at import
  time. This lets the cite-mode path (which only loads the leaderboard
  JSON) work on machines without a GPU or PyTorch.
- The `forward()` and `train_step()` methods raise
  :class:`NotImplementedError` in the base class. The 12 stub subclasses
  shipped in this package inherit that default behavior. To run a
  head-to-head re-training, override these two methods in the subclass
  with the original authors' code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BaselineConfig:
    """Hyper-parameter container shared by every baseline.

    Attributes
    ----------
    name : str
        Canonical baseline name (must match a key in
        ``timemmd_leaderboard.json``).
    lookback : int
        Input window length P (default 96, TimeMMD convention).
    horizon : int
        Forecast horizon S (default 96 for long-horizon, 12 for
        traffic-style short horizon).
    learning_rate : float
        Adam learning rate. The TimeMMD leaderboard reports 1e-3 for
        most baselines, 1e-4 for the larger Transformer variants.
    batch_size : int
        Mini-batch size. TimeMMD uses 32 throughout.
    epochs : int
        Maximum training epochs (early stopping with patience 5).
    extras : dict
        Model-specific hyper-parameters (kernel size, d_model, n_heads,
        etc.) go here. Each subclass documents its own ``extras`` schema.
    """

    name: str
    lookback: int = 96
    horizon: int = 96
    learning_rate: float = 1.0e-3
    batch_size: int = 32
    epochs: int = 50
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseBaseline:
    """Abstract base class implementing the uniform baseline interface."""

    #: Canonical name -- subclasses MUST override.
    NAME: str = "base"

    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        self.config: BaselineConfig = config or BaselineConfig(name=self.NAME)

    # ------------------------------------------------------------------ #
    # Required overrides
    # ------------------------------------------------------------------ #
    def forward(self, x):  # noqa: D401 -- see class docstring
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (B, P, N, C)
            Historical multivariate time series.

        Raises
        ------
        NotImplementedError
            In the base class and in stub subclasses. Override in a
            concrete subclass to enable training/inference.
        """
        raise NotImplementedError(
            f"{self.NAME}.forward() is a stub. Override it with the "
            "original authors' implementation to enable re-training."
        )

    def train_step(self, batch, optimizer):
        """Single gradient step.

        Parameters
        ----------
        batch : tuple
            ``(x, y)`` where ``x`` is the input window and ``y`` is the
            target window of shape ``(B, S, N, C)``.
        optimizer : torch.optim.Optimizer
            Pre-constructed optimizer (Adam by default in TimeMMD).

        Raises
        ------
        NotImplementedError
            In the base class and in stub subclasses.
        """
        raise NotImplementedError(
            f"{self.NAME}.train_step() is a stub. Override it to enable "
            "training."
        )

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.NAME!r} "
            f"lookback={self.config.lookback} horizon={self.config.horizon}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the baseline config for reproducibility logging."""
        return {
            "name": self.NAME,
            "class": self.__class__.__name__,
            "config": {
                "lookback": self.config.lookback,
                "horizon": self.config.horizon,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "extras": dict(self.config.extras),
            },
        }
