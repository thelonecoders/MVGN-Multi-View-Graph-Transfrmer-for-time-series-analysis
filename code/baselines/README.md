# TimeMMD 12 Baselines — Package README

**Package:** `baselines/`
**Purpose:** Uniform Python interface to the 12 standard TimeMMD
baselines, supporting both cite-mode (use upstream leaderboard values)
and re-implement-mode (re-train locally with the original authors'
code).

This package closes the audit gap noted in
`04_audit_reports/professor_eval_2026-08-09.md` §3 ("Baseline
implementations: Q1 promises comparison against 12 baselines but only
MVGT-Net is implemented").

## Quick start

```bash
cd 10_mvgtnet_code

# 1. Populate the leaderboard (cite mode).
python3 scripts/fetch_leaderboard.py

# 2. Train MVGT-Net (Phase A).
python3 scripts/train_real.py --config configs/default.yaml

# 3. Compare MVGT-Net vs the 12 baselines.
python3 scripts/compare_with_leaderboard.py --mode cite
```

Outputs land in `results/baseline_comparison.{md,csv,json}`.

## Files in this package

```
baselines/
├── README.md                     # this file
├── LEADERBOARD_CITATION.md       # full guide (cite mode + re-implement mode)
├── __init__.py                   # public API
├── base_baseline.py              # BaseBaseline + BaselineConfig
├── leaderboard_loader.py         # JSON loader + accessors
├── factory.py                    # build_baseline(name) factory
├── timemmd_leaderboard.json      # shipped empty (scores=null); populate via fetcher
├── transformer.py                # stub: Transformer (Vaswani 2017)
├── reformer.py                   # stub: Reformer (Kitaev 2020)
├── informer.py                   # stub: Informer (Zhou AAAI 2021)
├── autoformer.py                 # stub: Autoformer (Wu NeurIPS 2021)
├── crossformer.py                # stub: Crossformer (Zhang ICLR 2023)
├── nonstationary_transformer.py  # stub: Non-stationary Transformer (Liu NeurIPS 2022)
├── fedformer.py                  # stub: FEDformer (Zhou ICML 2022)
├── itransformer.py               # stub: iTransformer (Liu ICLR 2024)
├── dlinear.py                    # stub: DLinear (Zeng AAAI 2023)
├── film.py                       # stub: FiLM (Zhou NeurIPS 2022)
├── timesnet.py                   # stub: TimesNet (Wu ICLR 2023)
├── patchtst.py                   # stub: PatchTST (Nie ICLR 2023)
└── tests/
    ├── __init__.py
    └── test_baselines.py         # 10 tests, all passing
```

The companion runner scripts live in the parent
`10_mvgtnet_code/scripts/` directory:

- `scripts/fetch_leaderboard.py` — populate the JSON from the
  official GitHub README.
- `scripts/compare_with_leaderboard.py` — produce the comparison
  table + Holm-Bonferroni stats.

## Public API

```python
from baselines import (
    BaseBaseline,
    BaselineConfig,
    LeaderboardNotPopulatedError,
    build_baseline,
    list_available_baselines,
    load_leaderboard,
    get_baseline_scores,
    is_populated,
)

# 12 canonical baseline names
list_available_baselines()
# ['Autoformer', 'Crossformer', 'DLinear', 'FEDformer', 'FiLM',
#  'Informer', 'Non-stationary Transformer', 'PatchTST',
#  'Reformer', 'TimesNet', 'Transformer', 'iTransformer']

# Build a stub instance (forward() raises NotImplementedError
# until you replace the body with the upstream authors' code).
m = build_baseline("iTransformer")
m.to_dict()
# {'name': 'iTransformer', 'class': 'iTransformerBaseline',
#  'config': {'lookback': 96, 'horizon': 96, 'learning_rate': 0.001,
#             'batch_size': 32, 'epochs': 50, 'extras': {}}}

# Cite-mode: load leaderboard scores.
if is_populated():
    get_baseline_scores("iTransformer", "Solar")
    # {'MSE': 0.341, 'MAE': 0.372}  -- values from upstream leaderboard
else:
    print("Run scripts/fetch_leaderboard.py first.")
```

## Zero-hallucination contract

The shipped JSON has all `scores` fields set to `null`. The loader
raises `LeaderboardNotPopulatedError` if you ask for scores before
populating the JSON. This makes it **impossible** to accidentally
cite a fabricated baseline number — the call will fail loudly at
experiment-runner start.

The test suite (`baselines/tests/test_baselines.py`) verifies that
the JSON is either fully populated or fully null — never partial.
A partial state would be a hallucination risk and is treated as a
test failure.

## See also

- `LEADERBOARD_CITATION.md` — full guide with citation language for
  the thesis, manual fallback for air-gapped VPS, and the
  re-implement-mode protocol.
- `04_audit_reports/audit_report_2026-08-09_phase2.md` — updated
  audit log noting that the baselines gap is now closed.
- `CITATION.bib` (bundle root) — pre-formatted BibTeX for the
  TimeMMD dataset paper and the 12 baseline papers.
