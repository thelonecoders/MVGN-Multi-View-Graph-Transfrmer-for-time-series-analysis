"""Stub class for the Reformer baseline.

Family
------
Transformer-based

Reference
---------
Kitaev et al., 2020, Reformer: The Efficient Transformer
Upstream code: https://github.com/thuml/Time-Series-Library

This file is a **stub**. The class imports cleanly and exposes the
uniform interface defined by :class:`BaseBaseline`, but
:meth:`forward` and :meth:`train_step` raise
:class:`NotImplementedError`. To enable head-to-head re-training on the
VPS:

1. Clone the upstream repository listed above (verify the licence
   first -- most are MIT or Apache-2.0).
2. Copy the model class body into :meth:`forward` and the training loop
   body into :meth:`train_step`, retaining the :class:`BaseBaseline`
   wrapper so the experiment runner can still dispatch via
   :func:`build_baseline`.
3. Re-run ``python scripts/run_all_experiments.py --baselines Reformer``.

Until step 2 is complete, use the **cite mode** path: load the official
leaderboard numbers via :func:`baselines.load_leaderboard` and report
those as the Reformer reference values.
"""

from __future__ import annotations

from .base_baseline import BaseBaseline, BaselineConfig


class ReformerBaseline(BaseBaseline):
    """Stub for the Reformer baseline. See module docstring."""

    NAME = "Reformer"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config or BaselineConfig(name=self.NAME))

    # Override these two methods with the upstream authors' code to
    # enable full re-training on the VPS.
    # def forward(self, x):
    #     ...
    # def train_step(self, batch, optimizer):
    #     ...
