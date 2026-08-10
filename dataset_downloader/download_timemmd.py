#!/usr/bin/env python3
"""
TimeMMD Dataset Download Script
================================
Downloads the TimeMMD dataset from the official AdityaLab repository.

TimeMMD is a multi-domain, multimodal (numeric + textual) dataset for
time-series analysis, released at NeurIPS 2024.

Official repository: https://github.com/AdityaLab/TimeMMD
Paper: Xue, Wang, Salim, et al. "TimeMMD: A Multi-Domain Multimodal Dataset
       for Time Series Analysis." NeurIPS 2024 Datasets & Benchmarks.

Usage:
    python download_timemmd.py                    # Download all 9 domains
    python download_timemmd.py --domain Environment  # Download a single domain
    python download_timemmd.py --list             # List available domains
    python download_timemmd.py --dry-run          # Show what would be downloaded

Requirements:
    pip install gdown  # For Google Drive downloads

Note:
    The dataset is approximately 2 GB. Download may take 10-30 minutes
    depending on your internet connection. The script will create a
    TimeMMD/ directory with subdirectories for each domain.

Honest disclosure:
    This script was NOT run during thesis preparation because:
    1. The dataset is hosted on Google Drive (requires gdown, not just wget)
    2. Download requires ~2 GB of bandwidth and disk space
    3. Training on the full dataset requires GPU resources beyond this environment
    The script is provided for the user to run after downloading this bundle.
"""
import argparse
import os
import sys
import json
from pathlib import Path

# TimeMMD domain configuration
# Source: https://github.com/AdityaLab/TimeMMD
TIMEMMD_DOMAINS = {
    "Environment": {
        "frequency": "daily",
        "variables": 1,
        "text_coverage": "61%",
        "description": "Environmental monitoring (air quality, temperature, humidity)",
    },
    "Climate": {
        "frequency": "daily",
        "variables": 5,
        "text_coverage": "61%",
        "description": "Climate indicators (temperature, precipitation, wind, etc.)",
    },
    "Economy": {
        "frequency": "monthly",
        "variables": 3,
        "text_coverage": "82%",
        "description": "Macroeconomic indicators (GDP, CPI, unemployment, etc.)",
    },
    "Energy": {
        "frequency": "weekly",
        "variables": 4,
        "text_coverage": "moderate",
        "description": "Energy consumption by PADD region (US Petroleum Administration for Defense Districts)",
    },
    "Health_AFR": {
        "frequency": "weekly",
        "variables": 2,
        "text_coverage": "low",
        "description": "Health indicators for African countries",
    },
    "Health_US": {
        "frequency": "weekly",
        "variables": 2,
        "text_coverage": "low",
        "description": "Health indicators for US states",
    },
    "Agriculture": {
        "frequency": "monthly",
        "variables": 1,
        "text_coverage": "82%",
        "description": "Agricultural production indicators",
    },
    "Traffic": {
        "frequency": "hourly",
        "variables": 3,
        "text_coverage": "none",
        "description": "Traffic flow data (compatible with ST-LLM+ NYCTaxi/CHBike)",
    },
    "Stock": {
        "frequency": "daily",
        "variables": 5,
        "text_coverage": "none",
        "description": "Stock market indicators (OHLCV format)",
    },
}

# Google Drive file ID for the full TimeMMD dataset
# (This is a placeholder; the actual ID must be obtained from the official repo)
TIMEMMD_GDRIVE_ID = "1TIME_MMD_GDRIVE_ID_PLACEHOLDER"  # Replace with actual ID
TIMEMMD_REPO_URL = "https://github.com/AdityaLab/TimeMMD"


def list_domains():
    """Print the list of available TimeMMD domains."""
    print("=" * 70)
    print("TimeMMD Domains (9 total, organized in 10 folders)")
    print("=" * 70)
    print(f"{'Domain':<15} {'Frequency':<10} {'Variables':<10} {'Text':<10} Description")
    print("-" * 70)
    for name, info in TIMEMMD_DOMAINS.items():
        print(f"{name:<15} {info['frequency']:<10} {info['variables']:<10} "
              f"{info['text_coverage']:<10} {info['description']}")
    print("=" * 70)
    print(f"\nOfficial repository: {TIMEMMD_REPO_URL}")
    print(f"Paper: NeurIPS 2024 Datasets & Benchmarks Track")


def download_domain(domain: str, output_dir: Path, dry_run: bool = False):
    """Download a single TimeMMD domain.

    Args:
        domain: domain name (must be in TIMEMMD_DOMAINS)
        output_dir: base output directory
        dry_run: if True, print what would be downloaded without actually downloading
    """
    if domain not in TIMEMMD_DOMAINS:
        print(f"ERROR: Unknown domain '{domain}'")
        print(f"Available domains: {list(TIMEMMD_DOMAINS.keys())}")
        return False

    info = TIMEMMD_DOMAINS[domain]
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading domain: {domain}")
    print(f"  Frequency: {info['frequency']}")
    print(f"  Variables: {info['variables']}")
    print(f"  Text coverage: {info['text_coverage']}")
    print(f"  Description: {info['description']}")
    print(f"  Output directory: {domain_dir}")

    if dry_run:
        print("  [DRY RUN] No actual download performed.")
        return True

    # Try gdown first (for Google Drive)
    try:
        import gdown
        print(f"  Attempting download via gdown (Google Drive)...")
        # In practice, you would need the specific file ID for each domain.
        # The official repo provides these in its README.
        # For now, we print instructions.
        print(f"  NOTE: To download, visit {TIMEMMD_REPO_URL}")
        print(f"  and follow the instructions in the README to obtain the Google Drive file IDs.")
        print(f"  Then update this script with the actual IDs and re-run.")
        return False
    except ImportError:
        print(f"  gdown not installed. Install with: pip install gdown")
        print(f"  Then visit {TIMEMMD_REPO_URL} for manual download instructions.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download TimeMMD dataset")
    parser.add_argument("--domain", default=None,
                       help="Specific domain to download (default: all)")
    parser.add_argument("--output", default="./TimeMMD",
                       help="Output directory (default: ./TimeMMD)")
    parser.add_argument("--list", action="store_true",
                       help="List available domains and exit")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without actually downloading")
    args = parser.parse_args()

    if args.list:
        list_domains()
        return

    output_dir = Path(args.output)
    print(f"Output directory: {output_dir.resolve()}")

    if args.domain:
        download_domain(args.domain, output_dir, args.dry_run)
    else:
        print(f"\nDownloading all {len(TIMEMMD_DOMAINS)} domains...")
        for domain in TIMEMMD_DOMAINS:
            download_domain(domain, output_dir, args.dry_run)

    print("\n" + "=" * 70)
    print("IMPORTANT NOTICE")
    print("=" * 70)
    print("This script is a TEMPLATE. The actual TimeMMD dataset is hosted on")
    print("Google Drive and requires manual file ID lookup from the official repo:")
    print(f"  {TIMEMMD_REPO_URL}")
    print()
    print("Honest disclosure: The TimeMMD dataset was NOT downloaded during")
    print("thesis preparation. The thesis presents design and code only;")
    print("empirical validation requires the user to download the dataset and")
    print("run the training pipeline (10_mvgtnet_code/scripts/train.py).")
    print()
    print("After downloading, place each domain's data in:")
    print(f"  {output_dir}/<DomainName>/")
    print("Each domain should contain:")
    print("  - numeric.csv or numeric.parquet (numeric time series)")
    print("  - text.json or text.csv (time-aligned text reports)")
    print("  - metadata.json (domain metadata: frequency, variables, etc.)")


if __name__ == "__main__":
    main()
