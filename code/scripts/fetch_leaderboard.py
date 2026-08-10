#!/usr/bin/env python3
"""Fetch the official TimeMMD leaderboard and write it to the JSON file.

The TimeMMD dataset authors publish the per-baseline, per-domain MSE/MAE
numbers in the README of their GitHub repository
(https://github.com/AdityaLab/Time-MMD). This script:

1. Fetches the README.md over HTTPS (no authentication required).
2. Extracts the leaderboard table.
3. Parses MSE/MAE per baseline per domain.
4. Writes the verified values into ``baselines/timemmd_leaderboard.json``
   (overwriting the shipped schema-only file), sets
   ``_meta.fetched_on`` to the current ISO-8601 timestamp, and sets
   ``_meta.fetched_by`` to the running user.

Usage
-----
    python scripts/fetch_leaderboard.py
    python scripts/fetch_leaderboard.py --output /tmp/lb.json
    python scripts/fetch_leaderboard.py --offline-readme path/to/README.md

If the repository structure changes and the parser can no longer find
the table, the script fails loudly with a clear error message and
suggests the manual paste-in fallback (see the LEADERBOARD_CITATION
guide). The shipped JSON is never silently modified.

Dependencies
------------
- ``requests`` (already in requirements.txt for the optional
  ``download_dataset.py`` script). For an offline / air-gapped VPS,
  use ``--offline-readme`` after manually downloading the README.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "baselines", "timemmd_leaderboard.json",
)
DEFAULT_OUTPUT = os.path.normpath(DEFAULT_OUTPUT)

LEADERBOARD_URL = "https://raw.githubusercontent.com/AdityaLab/Time-MMD/main/README.md"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _fetch_readme(offline_path: Optional[str]) -> str:
    if offline_path:
        with open(offline_path, "r", encoding="utf-8") as fh:
            return fh.read()
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            "ERROR: 'requests' is not installed. Install it with\n"
            "    pip install requests\n"
            "or re-run with --offline-readme <path>.\n"
        )
        raise SystemExit(2) from exc
    resp = requests.get(LEADERBOARD_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def _find_leaderboard_section(readme: str) -> str:
    """Return the markdown text of the leaderboard section.

    The TimeMMD README headers vary slightly across revisions, so we
    match a tolerant regex. If the section can't be located, raise.
    """
    pattern = re.compile(
        r"(?ims)^#+\s*(leaderboard|evaluation results|benchmark results).*?"
        r"(?=^#+\s|\Z)"
    )
    matches = list(pattern.finditer(readme))
    if not matches:
        raise SystemExit(
            "ERROR: could not locate the leaderboard section in the "
            "TimeMMD README. The repository structure may have changed. "
            "Falling back to manual paste-in is required -- see "
            "baselines/LEADERBOARD_CITATION.md for instructions."
        )
    # Use the longest match (most likely the actual table).
    best = max(matches, key=lambda m: len(m.group(0)))
    return best.group(0)


def _parse_markdown_table(section: str) -> List[Dict[str, str]]:
    """Parse a markdown table into a list of row dicts."""
    rows: List[Dict[str, str]] = []
    headers: List[str] = []
    in_table = False
    for line in section.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not headers:
            headers = cells
            in_table = True
            continue
        if not in_table:
            continue
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            # separator row
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _coerce_score(value: str) -> Optional[float]:
    if value in ("", "-", "N/A", "n/a"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: %(default)s)",
    )
    p.add_argument(
        "--offline-readme",
        dest="offline_readme",
        default=None,
        help="Path to a locally-saved copy of the TimeMMD README.md "
             "(for air-gapped VPS use).",
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Parse and print the leaderboard to stdout without "
             "modifying the JSON file.",
    )
    args = p.parse_args(argv)

    print(f"[1/4] Fetching README from {args.offline_readme or LEADERBOARD_URL}")
    readme = _fetch_readme(args.offline_readme)

    print("[2/4] Locating leaderboard section")
    section = _find_leaderboard_section(readme)

    print("[3/4] Parsing leaderboard table")
    rows = _parse_markdown_table(section)
    if not rows:
        raise SystemExit(
            "ERROR: leaderboard section was found but contained no "
            "parseable table. The TimeMMD README format may have "
            "changed -- please fall back to manual paste-in (see "
            "baselines/LEADERBOARD_CITATION.md)."
        )
    print(f"      parsed {len(rows)} rows")

    # Convert rows into the per-baseline-per-domain structure used by
    # the shipped JSON. We expect each row to be one baseline with
    # per-domain MSE/MAE columns.
    new_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        bname = row.get("Model") or row.get("Baseline") or row.get("Method")
        if not bname:
            continue
        bname = bname.strip()
        per_domain: Dict[str, Dict[str, float]] = {}
        for col, val in row.items():
            if col in ("Model", "Baseline", "Method"):
                continue
            # Column convention used by TimeMMD: "Domain_MSE" / "Domain_MAE"
            # or "Domain (MSE/MAE)" -- we accept both.
            m = re.match(r"^(.+?)\s*[\(\[]?(MSE|MAE)[\)\]]?$", col, re.I)
            if not m:
                continue
            domain = m.group(1).strip()
            metric = m.group(2).upper()
            score = _coerce_score(val)
            if score is None:
                continue
            per_domain.setdefault(domain, {})[metric] = score
        if per_domain:
            new_scores[bname] = per_domain

    if args.dry_run:
        print("[4/4] dry-run -- printing parsed scores, JSON unchanged")
        print(json.dumps(new_scores, indent=2, sort_keys=True))
        return 0

    print(f"[4/4] Writing verified scores to {args.output}")
    with open(args.output, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for bname, scores in new_scores.items():
        if bname in data.get("baselines", {}):
            data["baselines"][bname]["scores"] = scores
        else:
            # Unknown baseline name from upstream -- add a new entry.
            data.setdefault("baselines", {})[bname] = {
                "family": "(added by fetch_leaderboard.py)",
                "paper": "",
                "url": "",
                "code_repo": "",
                "scores": scores,
            }
    data.setdefault("_meta", {})["fetched_on"] = (
        _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )
    data["_meta"]["fetched_by"] = getpass.getuser()
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
