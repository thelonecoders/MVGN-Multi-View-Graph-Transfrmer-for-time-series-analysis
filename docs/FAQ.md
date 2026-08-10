# FAQ — ST-LLM-Plus VPS Code Bundle

Frequently asked questions, with verified answers.

---

## Q1. Can I run this on a GPU with less than 12 GB VRAM?

**Yes, with caveats.**

| GPU | VRAM | Feasibility | Required changes |
|-----|------|-------------|------------------|
| RTX 4090 | 24 GB | ✅ Full speed | None — larger batch size possible (64 instead of 32) |
| RTX 3090 | 24 GB | ✅ Full speed | None |
| RTX 3080 Ti | 12 GB | ✅ Recommended | None — this is the target hardware |
| RTX 3080 | 10 GB | ✅ Works | Reduce batch_size to 24 in `configs/default.yaml` |
| RTX 3070 | 8 GB | ⚠️ Tight | Reduce batch_size to 16; enable QLoRA |
| RTX 3060 | 12 GB | ✅ Works | None |
| RTX 3060 | 6 GB | ⚠️ Very tight | Reduce batch_size to 8; enable QLoRA; reduce hidden_dim to 32 |
| GTX 1080 Ti | 11 GB | ⚠️ Works but slow | Reduce batch_size to 24; no AMP fp16 (Pascal arch) |
| CPU only | 0 GB | ❌ Too slow | Use `--device cpu --smoke-test` for testing only |

To enable QLoRA:
1. Uncomment `bitsandbytes==0.43.3` in `code/requirements.txt`
2. Run `pip install -r code/requirements.txt`
3. Set `use_qlora: true` in `code/configs/default.yaml`

---

## Q2. Can I use a different LLM than bert-base-uncased?

**Yes, but you need to verify VRAM fits.**

| LLM | Params | VRAM (batch 32 + LoRA + AMP fp16) | Fits in 12 GB? |
|-----|--------|-----------------------------------|----------------|
| bert-base-uncased | 110 M | ~4.6 GB | ✅ Yes (recommended) |
| bert-large-uncased | 340 M | ~9.2 GB | ⚠️ Tight |
| distilbert-base-uncased | 66 M | ~3.4 GB | ✅ Yes |
| roberta-base | 125 M | ~4.8 GB | ✅ Yes |
| gpt2-base | 124 M | ~4.8 GB | ✅ Yes |
| Llama-3-8B | 8 B | ~22 GB (even with QLoRA 4-bit) | ❌ No |
| Mistral-7B | 7 B | ~19 GB (even with QLoRA 4-bit) | ❌ No |

To change the LLM, edit `code/configs/default.yaml`:
```yaml
llm:
  model_name: bert-base-uncased   # change to e.g. distilbert-base-uncased
  ...
```

Larger LLMs (Llama-3-8B, Mistral-7B) require RTX 3090/4090 (24 GB) or
A100 (40 GB) — they do NOT fit in 12 GB VRAM even with QLoRA 4-bit.

---

## Q3. Why does the validation split omit `batch_y_timestamps_datetime64_ns`?

This is a **verified upstream quirk** in the official TimeMMD dataset. All 9
`validation.jsonl` files on the Hugging Face mirror
(`AndrewRWilliams/time-mmd-DC`) omit this key, while train and test splits
include it.

We verified this by:
1. Downloading the upstream Climate_AQI validation file directly from HF
2. Inspecting its first record (the key is absent)
3. Comparing SHA-256 with our local copy (they match byte-for-byte)

This is **NOT** a download bug. The MVGT-Net data loader handles this
gracefully (the key is optional for validation). Documented in:
- `12_dataset/readme/README.md` §"Upstream quirk"
- `15_engineering_details/DATASET_MIRROR_COMPARISON.md` §4.3

---

## Q4. Why does Economy_Unemp contain `"[nan nan]"` in batch_text?

This is a **genuine upstream marker** for records that have no text facts
available. Approximately 18% of Economy_Unemp records contain this literal
string instead of a proper text array.

The MVGT-Net data loader converts `"[nan nan]"` to an empty string, which
the text encoder then handles as a "no text" case (the text branch is
skipped for that record).

Documented in:
- `12_dataset/readme/README.md` §"Economy_Unemp text coverage"
- `10_mvgtnet_code/DATA_CARD.md` §2

---

## Q5. Can I train on a single domain only?

**Yes.** Use the `--domain` flag:
```bash
./run/run_step3_train_single_domain.sh Climate_AQI
# Or:
python scripts/train_real.py --domain Climate_AQI --device cuda --epochs 100
```

Valid domain names (from `DOMAIN_REGISTRY`):
- `Climate_AQI`
- `Climate_Precip`
- `Economy_Trade`
- `Economy_Unemp`
- `Economy_VMT`
- `Agriculture_Fema`
- `Agriculture_Broil`
- `Health_Flu`
- `Energy_Gas`

---

## Q6. How long does training take on RTX 3080 Ti?

Per-domain training time (estimated, with early stopping patience=15):

| Domain | Train samples | Lookback | Horizon | Estimated time |
|--------|---------------|----------|---------|----------------|
| Climate_AQI | 7,552 | 96 | 96 | 12–22 min |
| Economy_Trade | 256 | 8 | 12 | 2–4 min |
| Economy_Unemp | 608 | 8 | 12 | 3–5 min |
| Economy_VMT | 352 | 8 | 12 | 2–4 min |
| Agriculture_Fema | 160 | 8 | 12 | 1–3 min |
| Agriculture_Broil | 320 | 8 | 12 | 2–4 min |
| Climate_Precip | 320 | 8 | 12 | 2–4 min |
| Health_Flu | 896 | 36 | 24 | 4–7 min |
| Energy_Gas | 960 | 36 | 24 | 4–7 min |

**Total for all 9 domains:** ~32–60 minutes (with early stopping). Without
early stopping, ~52–85 minutes (always runs full 100 epochs).

See `docs/RESOURCE_ESTIMATE.md` for the full derivation.

---

## Q7. What if I don't want to use Docker?

**You don't have to.** Docker is optional. The default path uses a Python
venv:

```bash
./run/run_step0_install.sh    # creates venv + installs deps
./run/run_step1_download_dataset.sh
./run/run_step2_smoke_test.sh
./run/run_step4_train_all_domains.sh
./run/run_step5_analyses.sh
```

Docker is provided as an alternative for users who prefer containerized
environments or who want to avoid polluting the host system with packages.

---

## Q8. Can I resume a training run that was interrupted?

**Yes.** The training script saves `latest.pt` after every epoch:
```bash
python scripts/train_real.py --domain Climate_AQI --resume
```

This loads `code/checkpoints/Climate_AQI/latest.pt` and continues from the
last completed epoch. The optimizer state, scheduler state, and RNG state
are all restored.

---

## Q9. What are the DIAGNOSTIC labels in Phase F outputs?

The 5 Phase F analysis scripts (`latency_carbon.py`,
`robustness_analysis.py`, `scaling_analysis.py`, `cross_domain_transfer.py`,
`hyperparameter_search.py`) run on a **synthetic mini-batch** by default,
not on the full trained model.

This means their outputs are valid for:
- ✅ Sanity-checking that the analysis pipelines run end-to-end
- ✅ Verifying that the plotting code produces correctly-shaped figures
- ✅ Testing the script interfaces (CLI flags, file outputs)

But they are **NOT** valid for:
- ❌ Reporting thesis-grade performance numbers
- ❌ Making quantitative claims about model robustness/scaling/transfer
- ❌ Comparing against baselines

To produce thesis-grade numbers, point each script at the trained
checkpoints:
```bash
python scripts/robustness_analysis.py \
    --checkpoint code/checkpoints/Climate_AQI/best.pt \
    --domain Climate_AQI \
    --device cuda
```

This is honestly documented in:
- `code/MODEL_CARD.md` §7.2 and §7.3
- Thesis Chapter 19
- `14_engineering_analyses/README.md`
- `README.md` §12 "Honest limitations"

---

## Q10. Why are the requirements.txt versions pinned?

For **reproducibility**. The bundle's training protocol (Chapter 18) requires
that anyone, anywhere, running the same code with the same config on the
same hardware produces bit-identical results (within floating-point
tolerance).

If we used `torch>=2.0.0`, a user running today might get torch 2.4.1 while
a user running next year might get torch 2.7.0 — and the training results
could differ subtly due to internal PyTorch changes (e.g., cuDNN algorithm
selection, kernel implementations).

By pinning to `torch==2.4.1`, we guarantee that everyone uses the exact
same library versions.

---

## Q11. Can I contribute improvements to the bundle?

**Yes.** The bundle is open source under Apache 2.0. To contribute:

1. Fork the repository (if hosted on GitHub)
2. Create a feature branch: `git checkout -b my-improvement`
3. Make your changes
4. Run all tests: `cd code && python -m pytest tests/ -v`
5. Run the bundle integrity verifier: `./verify_bundle_integrity.sh`
6. Run pre-commit: `cd code && pre-commit run --all-files`
7. Submit a pull request with a clear description of the change

For bug reports, include:
- The full pipeline log (`logs/pipeline_<timestamp>.log`)
- The output of `./verify_bundle_integrity.sh`
- The output of `python scripts/environment_fingerprint.py`
- The exact command you ran
- The expected vs. actual behavior

---

## Q12. How do I cite this bundle?

See `CITATION.cff` (machine-readable) or `CITATION.bib` (BibTeX) at the
bundle root. The BibTeX entries are also in `README.md` §13.

---

## Q13. Is the bundle compatible with Apple Silicon (M1/M2/M3)?

**Partially.** The Python code runs on Apple Silicon via MPS (Metal
Performance Shaders), but:
- Ranger21 may not be fully compatible with MPS
- CUDA-specific code paths will not run
- Performance is significantly slower than NVIDIA GPUs

To run on Apple Silicon:
```bash
# In configs/default.yaml, set:
# device: mps
# mixed_precision: null   # MPS does not support AMP fp16

python scripts/train_real.py --domain Economy_Trade --device mps --epochs 10
```

For production training, use an NVIDIA GPU (RTX 3080 Ti or better).

---

## Q14. What if the Hugging Face mirror goes offline?

The bundle includes the **full real dataset** (322 MB, 27 JSONL files) in
`12_dataset/TimeMMD/`. You do NOT need to re-download it from Hugging Face
unless you delete the local copy.

If the HF mirror goes offline AND you've deleted the local copy:
1. Try the upstream AdityaLab repo: `https://github.com/AdityaLab/TimeMMD`
   (Note: as of 2026-08-09 this returns HTTP 404 — verify before relying on it)
2. Contact the TimeMMD authors directly (see the NeurIPS 2024 paper)
3. Use the bundled `dataset_downloader/download_chunked.py` with retry logic

---

## Q15. Can I use this bundle for a different dataset?

**Yes, with modifications.** The MVGT-Net architecture is general-purpose
for multivariate time-series forecasting with text modalities. To use a
different dataset:

1. Convert your dataset to the TimeMMD JSONL format (see
   `12_dataset/readme/README.md` for the schema)
2. Add your domain to `DOMAIN_REGISTRY` in `code/mvgt_net/data.py`
3. Update `code/configs/default.yaml` to reference your domain
4. Run `./run/run_step3_train_single_domain.sh YourDomain`

The schema requires each JSONL line to be a JSON object with:
- `batch_x`: array of shape `(lookback, num_features)`
- `batch_y`: array of shape `(horizon, num_features)`
- `batch_x_timestamps`: array of shape `(lookback,)`
- `batch_y_timestamps`: array of shape `(horizon,)`
- `batch_x_mark`: array of time features (optional)
- `batch_y_mark`: array of time features (optional)
- `batch_text`: array of strings (optional, can be `"[nan nan]"` for no text)
