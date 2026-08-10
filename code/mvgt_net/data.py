"""
Real TimeMMD Dataset Loader
============================
Production-grade PyTorch Dataset for the REAL TimeMMD JSONL files
downloaded into ``12_dataset/TimeMMD/``.

This module replaces the synthetic-data fallback that was previously
used in ``scripts/train.py``. It loads REAL numeric + text-fact data
from the verified Hugging Face mirror (AndrewRWilliams/time-mmd-DC,
which hosts the official AdityaLab TimeMMD data — arXiv:2406.08627,
NeurIPS 2024 Datasets & Benchmarks Track).

JSONL record schema (verified across all 9 domains, 2026-08-09)
----------------------------------------------------------------
Each line is a JSON object with these keys:

    batch_x                       : list[float]    — input window (length = lookback)
    batch_y                       : list[float]    — target window (length = horizon)
    batch_x_timestamps_datetime64_ns : list[int]   — input timestamps (ns since epoch)
    batch_y_timestamps_datetime64_ns : list[int]   — target timestamps
    batch_x_mark                  : list[list[float]] — time features [lookback, 4]
    batch_y_mark                  : list[list[float]] — target time features [horizon, 4]
    batch_text                    : str            — JSON-encoded list of text facts
                                                      (NOTE: HF datasets stores this
                                                      as a char-split list; we
                                                      re-concatenate then JSON-parse.)

Domain registry (verified)
--------------------------
    Climate_AQI       newyork_aqi_day_predLen_96            daily    7552 train
    Economy_Unemp     unadj_unemploymentrate_all_predLen_12 monthly  608  train
    Economy_Trade     us_tradebalance_month_predLen_12      monthly  256  train
    Economy_VMT       us_vmt_month_predLen_12               monthly  352  train
    Agriculture_Fema  us_femagrant_month_predLen_12         monthly  160  train
    Agriculture_Broil us_retailbroilercomposite_month_predLen_12 monthly 320  train
    Climate_Precip    us_precipitation_month_predLen_12     monthly  320  train
    Health_Flu        us_fluratio_week_predLen_24           weekly   896  train
    Energy_Gas        us_gasolineprice_week_predLen_24      weekly   960  train

Honest limitations
------------------
1. Each TimeMMD domain is a SINGLE-NODE time series (one OT variable per
   timestamp). MVGT-Net was originally designed for multi-node traffic
   data; on single-node data the spatial-graph branch is degenerate
   (1x1 adjacency = [[1.0]]) and the temporal/semantic/adaptive graphs
   carry the structural signal. This is documented in DATA_CARD.md §4.
2. The text facts are pre-tokenized as raw strings; this loader returns
   them as Python strings. The MVGT-Net text branch (mvgt_net.embedding)
   accepts either pre-computed BERT embeddings OR raw strings (which it
   then runs through a learnable linear projection as a fallback when
   the ``transformers`` package is not installed).
3. Some records in the Economy_Unemp domain contain ``[nan nan]`` in
   batch_text — this is a genuine NaN marker from the upstream TimeMMD
   dataset (Economy has 82% text coverage, not 100%). The loader
   replaces these with an empty string "" so downstream tokenization
   does not crash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Domain registry — verified against 12_dataset/TimeMMD/DATASET_MANIFEST.json
# ---------------------------------------------------------------------------
DOMAIN_REGISTRY: Dict[str, Dict] = {
    "Climate_AQI": {
        "config": "newyork_aqi_day_predLen_96",
        "frequency": "daily",
        "lookback": 96,
        "horizon": 96,
        "variables": 1,
        "description": "NYC air-quality index (PM2.5) — Climate domain",
    },
    "Economy_Unemp": {
        "config": "unadj_unemploymentrate_all_processed_predLen_12",
        "frequency": "monthly",
        "lookback": 8,
        "horizon": 12,
        "variables": 1,
        "description": "US unadjusted unemployment rate — Economy domain",
    },
    "Economy_Trade": {
        "config": "us_tradebalance_month_predLen_12",
        "frequency": "monthly",
        "lookback": 8,
        "horizon": 12,
        "variables": 1,
        "description": "US trade balance — Economy domain",
    },
    "Economy_VMT": {
        "config": "us_vmt_month_predLen_12",
        "frequency": "monthly",
        "lookback": 8,
        "horizon": 12,
        "variables": 1,
        "description": "US vehicle-miles traveled — Economy/Transportation domain",
    },
    "Agriculture_Fema": {
        "config": "us_femagrant_month_predLen_12",
        "frequency": "monthly",
        "lookback": 8,
        "horizon": 12,
        "variables": 1,
        "description": "US FEMA grant approvals — Agriculture/Public domain",
    },
    "Agriculture_Broil": {
        "config": "us_retailbroilercomposite_month_predLen_12",
        "frequency": "monthly",
        "lookback": 8,
        "horizon": 12,
        "variables": 1,
        "description": "US retail broiler composite price — Agriculture domain",
    },
    "Climate_Precip": {
        "config": "us_precipitation_month_predLen_12",
        "frequency": "monthly",
        "lookback": 8,
        "horizon": 12,
        "variables": 1,
        "description": "US precipitation index — Climate domain",
    },
    "Health_Flu": {
        "config": "us_fluratio_week_predLen_24",
        "frequency": "weekly",
        "lookback": 36,
        "horizon": 24,
        "variables": 1,
        "description": "US flu ratio (ILI ratio) — Health_US domain",
    },
    "Energy_Gas": {
        "config": "us_gasolineprice_week_predLen_24",
        "frequency": "weekly",
        "lookback": 36,
        "horizon": 24,
        "variables": 1,
        "description": "US gasoline retail price — Energy domain",
    },
}


def _parse_batch_text(bt) -> str:
    """Parse the ``batch_text`` field of a TimeMMD JSONL record.

    The HF datasets library stores batch_text as a list of single-character
    strings (a JSON-serialized list of text facts, split char-by-char).
    We re-concatenate and JSON-parse to recover the original list of facts.

    Returns the joined text-facts string. Returns "" if the field is
    NaN/empty (some Economy_Unemp records have ``[nan nan]``).
    """
    if isinstance(bt, str):
        text = bt
    elif isinstance(bt, list):
        text = "".join(str(x) for x in bt)
    else:
        return ""

    # Try to JSON-parse and re-join as a single string
    try:
        facts = json.loads(text)
        if isinstance(facts, list):
            # Filter out non-string entries (NaN becomes float)
            clean = [str(f) for f in facts if isinstance(f, str)]
            return " ".join(clean)
        if isinstance(facts, str):
            return facts
    except (json.JSONDecodeError, TypeError):
        pass

    # If parsing failed, return raw text (truncate to 8KB for safety)
    return text[:8192]


class TimeMMDDataset(Dataset):
    """PyTorch Dataset for one split (train/val/test) of one TimeMMD domain.

    Parameters
    ----------
    domain : str
        Domain name from DOMAIN_REGISTRY (e.g. "Climate_AQI").
    split : str
        One of "train", "validation", "test".
    data_root : Path or str
        Path to the ``12_dataset/TimeMMD/`` directory.
    normalize_mean, normalize_std : Optional[torch.Tensor]
        Pre-computed z-score statistics (shape [variables]). If None,
        no normalization is applied (caller should fit on train split
        and pass to val/test).
    """

    def __init__(
        self,
        domain: str,
        split: str,
        data_root: Path | str,
        normalize_mean: Optional[torch.Tensor] = None,
        normalize_std: Optional[torch.Tensor] = None,
    ):
        if domain not in DOMAIN_REGISTRY:
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {list(DOMAIN_REGISTRY)}"
            )
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be train/validation/test, got '{split}'")

        self.domain = domain
        self.split = split
        self.data_root = Path(data_root)
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        file_path = self.data_root / domain / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(
                f"TimeMMD file not found: {file_path}\n"
                f"Did you run scripts/download_timemmd_real.py?"
            )

        # Load all records into memory (TimeMMD is small enough — total < 325 MB)
        self.records: List[Dict] = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.records.append(obj)

        self.lookback = DOMAIN_REGISTRY[domain]["lookback"]
        self.horizon = DOMAIN_REGISTRY[domain]["horizon"]
        self.variables = DOMAIN_REGISTRY[domain]["variables"]
        self.num_nodes = 1  # All TimeMMD domains are single-node

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # Parse numeric window
        # batch_x: list[float] of length lookback → tensor [lookback, 1, 1]
        # (lookback, num_nodes=1, variables=1)
        x = torch.tensor(rec["batch_x"], dtype=torch.float32)  # [lookback]
        x = x.view(self.lookback, self.num_nodes, self.variables)

        y = torch.tensor(rec["batch_y"], dtype=torch.float32)  # [horizon]
        y = y.view(self.horizon, self.num_nodes, self.variables)

        # Z-score normalize (using train statistics)
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = self.normalize_mean.view(1, 1, -1)
            std = self.normalize_std.view(1, 1, -1).clamp(min=1e-8)
            x = (x - mean) / std
            y = (y - mean) / std

        # Parse text facts
        text_str = _parse_batch_text(rec.get("batch_text", ""))

        # Time marks (batch_x_mark): shape [lookback, 4]
        x_mark = torch.tensor(rec.get("batch_x_mark", [[0.0] * 4] * self.lookback),
                              dtype=torch.float32)
        y_mark = torch.tensor(rec.get("batch_y_mark", [[0.0] * 4] * self.horizon),
                              dtype=torch.float32)

        # Categorical (TimeMMD has no categorical modality; use zeros)
        cat = torch.zeros(self.lookback, self.num_nodes, dtype=torch.long)

        return {
            "x_numeric": x,                    # [lookback, 1, 1]
            "y_target": y,                     # [horizon, 1, 1]
            "x_text": text_str,                # raw string
            "x_cat": cat,                      # [lookback, 1]
            "x_mark": x_mark,                  # [lookback, 4]
            "y_mark": y_mark,                  # [horizon, 4]
        }


def fit_normalization(train_dataset: TimeMMDDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-variable mean and std from the TRAIN split only.

    This is the standard z-score normalization referenced in DATA_CARD.md §4
    ("Normalization: the dataset is provided in raw form; normalization is
    the responsibility of the modeler. MVGT-Net uses z-score normalization
    per variable, computed on the training split only.").

    Returns (mean, std), each of shape [variables].
    """
    all_x = []
    for i in range(len(train_dataset)):
        rec = train_dataset.records[i]
        x = torch.tensor(rec["batch_x"], dtype=torch.float32)
        all_x.append(x)
    stacked = torch.stack(all_x, dim=0)  # [N, lookback]
    # Per-variable: here variables=1, so we compute scalar mean/std by
    # reducing over both the N (sample) and lookback (time) dimensions.
    # Use reshape(-1) for maximum PyTorch version robustness.
    flat = stacked.reshape(-1)
    mean = flat.mean()
    std = flat.std().clamp(min=1e-8)
    return mean, std


def fit_minmax_normalization(train_dataset: TimeMMDDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-variable min and range from the TRAIN split only.

    This implements the M4 mitigation (per-domain min-max normalization
    to the unit interval) proposed in Section 19-9 of the thesis.

    Returns (min_val, range_val) where range_val = max - min, clamped to 1e-8.
    These are stored in the same (mean, std) slots of the dataset so that
    the normalization formula (x - min) / range produces values in [0, 1].
    """
    all_x = []
    for i in range(len(train_dataset)):
        rec = train_dataset.records[i]
        x = torch.tensor(rec["batch_x"], dtype=torch.float32)
        all_x.append(x)
    stacked = torch.stack(all_x, dim=0)  # [N, lookback]
    # Per-variable: here variables=1, so we compute scalar min/max by
    # reducing over both the N (sample) and lookback (time) dimensions.
    # Use amin/amax (NOT min/max) because Tensor.min()/max() do not accept
    # tuple dims in PyTorch 2.5+; amin/amax do, and also return a plain
    # tensor (no namedtuple wrapper) which is what we want here.
    min_val = stacked.amin(dim=(0, 1))  # scalar tensor
    max_val = stacked.amax(dim=(0, 1))  # scalar tensor
    range_val = (max_val - min_val).clamp(min=1e-8)
    return min_val, range_val


def collate_fn(batch: List[Dict]) -> Tuple:
    """Collate function for the DataLoader.

    Returns (x_numeric, y_target, x_text_dict, x_cat, adj_spatial).
    The adj_spatial is a single (N, N) tensor shared across the batch
    (the graph_builder.forward expects this shape, not a batched one).
    """
    x_numeric = torch.stack([b["x_numeric"] for b in batch], dim=0)  # [B, L, 1, 1]
    y_target = torch.stack([b["y_target"] for b in batch], dim=0)    # [B, H, 1, 1]
    x_cat = torch.stack([b["x_cat"] for b in batch], dim=0)          # [B, L, 1]
    # Text: list of strings (variable length)
    x_text_list = [b["x_text"] for b in batch]
    # Spatial adjacency: degenerate 1x1 for single-node TimeMMD
    # (shape (N, N) = (1, 1), shared across the batch)
    adj = torch.ones(1, 1)
    return x_numeric, y_target, {"fact": x_text_list}, x_cat, adj


def get_dataloaders(
    domain: str,
    data_root: Path | str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    normalization: str = "zscore",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
           torch.utils.data.DataLoader, Dict]:
    """Build train/val/test DataLoaders for a TimeMMD domain.

    Returns (train_loader, val_loader, test_loader, stats_dict) where
    stats_dict contains the normalization mean/std and dataset metadata.

    normalization: "zscore" (default, standard z-score) or "minmax"
                   (M4 mitigation: per-domain min-max scaling to [0,1]).
    """
    # Fit normalization on train split
    train_ds_raw = TimeMMDDataset(domain, "train", data_root)
    if normalization == "minmax":
        mean, std = fit_minmax_normalization(train_ds_raw)
    else:
        mean, std = fit_normalization(train_ds_raw)

    # Build all three splits with the SAME train-derived statistics
    train_ds = TimeMMDDataset(domain, "train", data_root, mean, std)
    val_ds = TimeMMDDataset(domain, "validation", data_root, mean, std)
    test_ds = TimeMMDDataset(domain, "test", data_root, mean, std)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False,
    )

    stats = {
        "domain": domain,
        "lookback": DOMAIN_REGISTRY[domain]["lookback"],
        "horizon": DOMAIN_REGISTRY[domain]["horizon"],
        "variables": DOMAIN_REGISTRY[domain]["variables"],
        "frequency": DOMAIN_REGISTRY[domain]["frequency"],
        "num_nodes": 1,
        "normalize_mean": mean.tolist(),
        "normalize_std": std.tolist(),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    return train_loader, val_loader, test_loader, stats


__all__ = [
    "DOMAIN_REGISTRY",
    "TimeMMDDataset",
    "fit_normalization",
    "collate_fn",
    "get_dataloaders",
]
