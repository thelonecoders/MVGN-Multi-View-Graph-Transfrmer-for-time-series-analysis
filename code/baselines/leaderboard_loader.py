"""Loader for the TimeMMD official leaderboard.

Two responsibilities:

1. Parse `timemmd_leaderboard.json` (shipped with this package) into a
   Python dict.
2. Provide convenience accessors used by the experiment runner:
   - :func:`list_available_baselines` -- the 12 canonical names.
   - :func:`get_baseline_scores` -- look up scores for one baseline
     on one domain.

The shipped JSON has all `scores` fields set to ``null``. This is
intentional: it enforces the zero-hallucination contract. Populate the
JSON either by running `scripts/fetch_leaderboard.py` or by manually
pasting verified values from
https://github.com/AdityaLab/Time-MMD#leaderboard.

The loader raises :class:`LeaderboardNotPopulatedError` whenever a
caller asks for scores that are still null. This makes accidental
citation of unverified numbers impossible -- the call will fail loudly
at experiment-runner start instead of silently producing a paper with
fabricated baselines.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


LEADERBOARD_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "timemmd_leaderboard.json",
)


class LeaderboardNotPopulatedError(RuntimeError):
    """Raised when a caller asks for scores that are still null.

    This is a deliberate hard-fail: the bundle never silently cites a
    number that has not been verified against the upstream source.
    """


def load_leaderboard(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and parse the leaderboard JSON.

    Parameters
    ----------
    path : str, optional
        Path to the leaderboard JSON. Defaults to the bundled file.

    Returns
    -------
    dict
        Parsed JSON with the same structure as
        ``timemmd_leaderboard.json``.
    """
    target = path or LEADERBOARD_FILE
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_available_baselines(path: Optional[str] = None) -> List[str]:
    """Return the 12 canonical baseline names in deterministic order."""
    data = load_leaderboard(path)
    return sorted(data.get("baselines", {}).keys())


def is_populated(path: Optional[str] = None) -> bool:
    """Return True iff every baseline has non-null scores."""
    data = load_leaderboard(path)
    fetched_on = data.get("_meta", {}).get("fetched_on")
    if not fetched_on:
        return False
    for name, entry in data.get("baselines", {}).items():
        if entry.get("scores") is None:
            return False
    return True


def get_baseline_scores(
    baseline: str,
    domain: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the leaderboard scores for ``baseline`` on ``domain``.

    Parameters
    ----------
    baseline : str
        Canonical baseline name (must be in
        :func:`list_available_baselines`).
    domain : str, optional
        TimeMMD domain name. If None, return the full scores dict for
        the baseline.
    path : str, optional
        Override path to the leaderboard JSON.

    Returns
    -------
    dict
        Scores dict. If ``domain`` is None, returns the full per-domain
        dict; otherwise returns the single-domain dict
        ``{"MSE": ..., "MAE": ...}``.

    Raises
    ------
    KeyError
        If ``baseline`` or ``domain`` is not in the leaderboard.
    LeaderboardNotPopulatedError
        If scores have not been populated yet.
    """
    data = load_leaderboard(path)
    if baseline not in data.get("baselines", {}):
        raise KeyError(
            f"Unknown baseline {baseline!r}. Available: "
            f"{list_available_baselines(path)}"
        )
    entry = data["baselines"][baseline]
    if entry.get("scores") is None:
        raise LeaderboardNotPopulatedError(
            f"Scores for {baseline!r} have not been populated yet. "
            f"Run `python scripts/fetch_leaderboard.py` to fetch them "
            f"from the official TimeMMD repository, or manually paste "
            f"verified values into {LEADERBOARD_FILE}. The bundle "
            f"intentionally fails loudly rather than cite unverified "
            f"numbers."
        )
    if domain is None:
        return entry["scores"]
    if domain not in entry["scores"]:
        raise KeyError(
            f"Unknown domain {domain!r} for baseline {baseline!r}. "
            f"Available: {list(entry['scores'].keys())}"
        )
    return entry["scores"][domain]
