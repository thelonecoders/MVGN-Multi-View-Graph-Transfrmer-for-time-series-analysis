#!/usr/bin/env python3
"""
Robust chunked downloader for TimeMMD.

Strategy:
  1. For each (domain, split) file, get expected size from HF API.
  2. Use curl with --continue-at - --max-time 100 to download in chunks.
     Each chunk runs in <2 minutes (under bash-tool timeout).
  3. Re-invoke this script until all files are complete and verified.

Usage:
    python3 /home/z/my-project/scripts/download_chunked.py
    # re-run until "ALL FILES COMPLETE" is printed.

After each call, prints progress: which files are done, which still need chunks.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/home/z/my-project/download/ST-LLM-Plus_Thesis_Bundle/12_dataset/TimeMMD")
HF_RESOLVE = "https://huggingface.co/datasets/AndrewRWilliams/time-mmd-DC/resolve/main"
HF_API = "https://huggingface.co/api/datasets/AndrewRWilliams/time-mmd-DC/tree/main/data"

DOMAINS = [
    ("Climate_AQI",       "newyork_aqi_day_predLen_96",             "daily",   "NYC air-quality index (PM2.5) — Climate domain"),
    ("Economy_Unemp",     "unadj_unemploymentrate_all_processed_predLen_12", "monthly", "US unadjusted unemployment rate — Economy domain"),
    ("Economy_Trade",     "us_tradebalance_month_predLen_12",       "monthly", "US trade balance — Economy domain"),
    ("Economy_VMT",       "us_vmt_month_predLen_12",                "monthly", "US vehicle-miles traveled — Economy/Transportation domain"),
    ("Agriculture_Fema",  "us_femagrant_month_predLen_12",          "monthly", "US FEMA grant approvals — Agriculture/Public domain"),
    ("Agriculture_Broil", "us_retailbroilercomposite_month_predLen_12", "monthly", "US retail broiler composite price — Agriculture domain"),
    ("Climate_Precip",    "us_precipitation_month_predLen_12",      "monthly", "US precipitation index — Climate domain"),
    ("Health_Flu",        "us_fluratio_week_predLen_24",            "weekly",  "US flu ratio (ILI ratio) — Health_US domain"),
    ("Energy_Gas",        "us_gasolineprice_week_predLen_24",       "weekly",  "US gasoline retail price — Energy domain"),
]
# Wrap as full dicts for metadata lookup in the manifest writer
DOMAINS_FULL = [
    {"domain": d[0], "config": d[1], "frequency": d[2], "description": d[3],
     "text_coverage": "real text facts per record (varies per domain — see DATA_CARD.md)"}
    for d in DOMAINS
]
SPLITS = ["train", "validation", "test"]


def get_directory_listing(config: str) -> list:
    """List files in data/<config>/ via HF API. Returns list of entries."""
    url = f"{HF_API}/{config}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "timemmd-chunked/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except Exception as e:
        print(f"  [warn] API listing failed for {config}: {e}")
        return []


def get_expected_size(config: str, split: str) -> int | None:
    """Get expected file size from HF API by listing the directory."""
    entries = get_directory_listing(config)
    target = f"{split}.jsonl"
    for entry in entries:
        if entry.get("path", "").endswith(target):
            lfs = entry.get("lfs") or {}
            return lfs.get("size") or entry.get("size")
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def download_chunk(url: str, dest: Path, max_time: int = 100) -> bool:
    """Download one chunk (max max_time seconds) using curl with resume.
    Returns True if download completed (curl exit 0), False otherwise."""
    cmd = [
        "curl", "-sSL",
        "--continue-at", "-",       # resume
        "--max-time", str(max_time),
        "--retry", "0",
        "--connect-timeout", "15",
        "-H", "User-Agent: timemmd-chunked/1.0",
        "-o", str(dest),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 30)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"  [curl error] {e}")
        return False


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build the file list with expected sizes (cached to a JSON to avoid re-fetching)
    cache_path = BASE_DIR / ".sizes_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            sizes_cache = json.load(f)
    else:
        sizes_cache = {}

    # Refresh any missing sizes
    print("=" * 70)
    print("Building file manifest (fetching expected sizes from HF API)...")
    print("=" * 70)
    for domain, config, _, _ in DOMAINS:
        for split in SPLITS:
            key = f"{config}/{split}"
            if key not in sizes_cache:
                sz = get_expected_size(config, split)
                if sz is not None:
                    sizes_cache[key] = sz
                    print(f"  {key}: {sz:,}B")
                else:
                    print(f"  {key}: SIZE UNKNOWN (will rely on curl)")
    with open(cache_path, "w") as f:
        json.dump(sizes_cache, f, indent=2)

    # 2. For each file, check if complete; if not, download a chunk
    print("\n" + "=" * 70)
    print("Checking / downloading files (one chunk per call, ~100s max)")
    print("=" * 70)

    pending = []
    completed = []
    for domain, config, _, _ in DOMAINS:
        domain_dir = BASE_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            key = f"{config}/{split}"
            url = f"{HF_RESOLVE}/data/{config}/{split}.jsonl"
            dest = domain_dir / f"{split}.jsonl"
            expected = sizes_cache.get(key)

            if dest.exists():
                actual = dest.stat().st_size
            else:
                actual = 0

            if expected and actual >= expected:
                # File complete — verify and record
                sha = sha256_file(dest)
                n = count_lines(dest)
                completed.append({
                    "domain": domain, "config": config, "split": split,
                    "path": str(dest), "bytes": actual,
                    "sha256": sha, "records": n,
                })
                print(f"  [DONE] {domain}/{split}.jsonl  "
                      f"size={actual:,}B  records={n}  sha256={sha[:12]}…")
            else:
                pending.append((domain, config, split, url, dest, expected, actual))
                need_str = f"{expected:,}B" if expected else "?"
                print(f"  [TODO] {domain}/{split}.jsonl  "
                      f"have={actual:,}B  need={need_str}")

    # 3. Process pending files: download multiple per call, within a ~90s budget
    if pending:
        print(f"\nPending: {len(pending)} file(s). Downloading...")
        BUDGET_SECONDS = 90
        t_start = time.time()
        files_done_this_call = 0
        for domain, config, split, url, dest, expected, actual in pending:
            elapsed = time.time() - t_start
            if elapsed >= BUDGET_SECONDS:
                print(f"  [budget] {elapsed:.1f}s elapsed; stopping this call.")
                break
            remaining = BUDGET_SECONDS - elapsed
            # Cap each file's max-time to remaining budget
            chunk_time = min(80, int(remaining))
            need_str = f"{expected:,}B" if expected else "?"
            print(f"  → {domain}/{split}.jsonl  (have={actual:,}B, need={need_str}, "
                  f"budget_left={remaining:.0f}s)")
            ok = download_chunk(url, dest, max_time=chunk_time)
            new_actual = dest.stat().st_size if dest.exists() else 0
            print(f"    result: ok={ok}  have now={new_actual:,}B")
            if expected and new_actual >= expected:
                sha = sha256_file(dest)
                n = count_lines(dest)
                print(f"    [DONE] {domain}/{split}.jsonl  "
                      f"size={new_actual:,}B  records={n}  sha256={sha[:12]}…")
                files_done_this_call += 1
            else:
                print(f"    [PARTIAL] need more chunks for this file.")
                # If this file is large and ran out of time, stop here.
                break

        elapsed = time.time() - t_start
        print(f"\n>>> Call summary: {files_done_this_call} file(s) completed in {elapsed:.1f}s <<<")
        # Re-run script to continue if any pending remain
        sys.exit(0)

    # 4. All files complete — write final manifest
    print("\n" + "=" * 70)
    print("ALL FILES COMPLETE — writing final manifest")
    print("=" * 70)

    # Group by domain
    by_domain = {}
    for c in completed:
        by_domain.setdefault(c["domain"], []).append(c)

    all_manifests = []
    total_bytes = 0
    total_files = 0
    for domain, config, _, _ in DOMAINS:
        files = by_domain.get(domain, [])
        dom_total = sum(f["bytes"] for f in files)
        total_bytes += dom_total
        total_files += len(files)
        # Build per-file URL list (avoid Python closure gotcha with loop variable)
        file_entries = []
        for f in files:
            file_url = f"{HF_RESOLVE}/data/{config}/{f['split']}.jsonl"
            file_entries.append({
                "split": f["split"],
                "path": f["path"],
                "bytes": f["bytes"],
                "sha256": f["sha256"],
                "records": f["records"],
                "url": file_url,
            })
        # Look up domain metadata
        domain_meta = next((d for d in DOMAINS_FULL if d["domain"] == domain), {})
        manifest = {
            "domain": domain,
            "config": config,
            "frequency": domain_meta.get("frequency"),
            "description": domain_meta.get("description"),
            "text_coverage": domain_meta.get("text_coverage"),
            "source_repo": "https://huggingface.co/datasets/AndrewRWilliams/time-mmd-DC",
            "upstream_repo": "https://github.com/AdityaLab/TimeMMD",
            "license": "ODC-By v1.0",
            "files": file_entries,
            "total_bytes": dom_total,
        }
        with open(BASE_DIR / domain / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        all_manifests.append(manifest)

    top = {
        "dataset": "TimeMMD",
        "version": "1.0 (HF mirror AndrewRWilliams/time-mmd-DC, snapshot 2026-08-09)",
        "upstream_arxiv": "https://arxiv.org/abs/2406.08627",
        "upstream_repo": "https://github.com/AdityaLab/TimeMMD",
        "hf_mirror": "https://huggingface.co/datasets/AndrewRWilliams/time-mmd-DC",
        "license": "ODC-By v1.0",
        "no_synthetic_data": True,
        "no_placeholders": True,
        "domains": all_manifests,
        "total_bytes": total_bytes,
        "total_files": total_files,
        "download_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }
    out = BASE_DIR / "DATASET_MANIFEST.json"
    with open(out, "w") as f:
        json.dump(top, f, indent=2)

    print(f"\nTotal: {total_files} files, {total_bytes:,} bytes "
          f"({total_bytes / (1 << 20):.1f} MiB)")
    print(f"Manifest written: {out}")

    # Final per-domain summary
    print("\nPer-domain summary:")
    for m in all_manifests:
        print(f"  {m['domain']:<22} {m['total_bytes']:>12,}B  "
              f"({m['total_bytes'] / (1 << 20):>7.1f} MiB)")


if __name__ == "__main__":
    main()
