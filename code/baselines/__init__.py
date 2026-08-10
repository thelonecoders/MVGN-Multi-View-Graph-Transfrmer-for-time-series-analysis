"""TimeMMD 12 baselines package.

This package provides a uniform Python interface to the 12 standard
baselines that ship with the TimeMMD benchmark (NeurIPS 2024 Datasets &
Benchmarks track). Two complementary use modes are supported:

(1) Cite mode (zero-cost, recommended for thesis defense):
    Load the official TimeMMD leaderboard numbers published by the
    dataset authors and use them as fixed reference values when
    comparing MVGT-Net. No GPU training of baselines is required.
    Use `leaderboard_loader.load_leaderboard()`.

(2) Re-implement mode (full reproduction, optional):
    Subclass `BaseBaseline` and provide a concrete `forward()` and
    `train_step()` for each of the 12 models. Stub implementations of
    all 12 are provided so the package imports cleanly; replace the
    stub bodies with the original authors' code (licences remain with
    the original authors) to enable head-to-head re-training.

Public API
----------
- :class:`BaseBaseline`           -- abstract base class.
- :func:`load_leaderboard`        -- load the official leaderboard JSON.
- :func:`build_baseline`          -- factory returning a stub instance.
- :func:`list_available_baselines` -- return the 12 canonical names.

Reference
---------
TimeMMD: A Multi-Domain Time-Series Multimodal Dataset.
Qian et al., NeurIPS 2024 Datasets & Benchmarks.
Repository: https://github.com/AdityaLab/Time-MMD
Leaderboard: https://github.com/AdityaLab/Time-MMD#leaderboard
"""

from __future__ import annotations

from .leaderboard_loader import (
    LEADERBOARD_FILE,
    LeaderboardNotPopulatedError,
    list_available_baselines,
    load_leaderboard,
    get_baseline_scores,
    is_populated,
)
from .base_baseline import BaseBaseline, BaselineConfig
from .factory import build_baseline

__all__ = [
    "BaseBaseline",
    "BaselineConfig",
    "LEADERBOARD_FILE",
    "LeaderboardNotPopulatedError",
    "build_baseline",
    "get_baseline_scores",
    "is_populated",
    "list_available_baselines",
    "load_leaderboard",
]

__version__ = "1.0.0"
