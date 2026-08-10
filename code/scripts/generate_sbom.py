#!/usr/bin/env python3
"""generate_sbom.py — Generate a Software Bill of Materials (SBOM) for the
MVGT-Net code package.

Outputs an SBOM in two formats:
  1. SPDX JSON (spdx-2.3) — the international standard for SBOMs.
  2. CSV — human-readable flat list of all components.

The SBOM covers:
  - Python runtime dependencies (from requirements.txt, pinned versions)
  - Python dev dependencies (from requirements-dev.txt)
  - Docker base image (from Dockerfile)
  - Pre-built model weights (bert-base-uncased)
  - Dataset (TimeMMD, with license + source)

Usage:
    python3 scripts/generate_sbom.py [--output-dir OUTPUT_DIR]

Outputs:
    <output-dir>/sbom.spdx.json   — SPDX 2.3 JSON
    <output-dir>/sbom.csv         — CSV flat list
    <output-dir>/sbom.sha256      — SHA-256 of both files

Default output-dir: ./sbom/ (created if missing)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent  # 10_mvgtnet_code/
BUNDLE_ROOT = CODE_ROOT.parent  # ST-LLM-Plus_Thesis_Bundle/

REQUIREMENTS_FILES = [
    (CODE_ROOT / "requirements.txt", "runtime"),
    (CODE_ROOT / "requirements-dev.txt", "development"),
]

DOCKERFILE = CODE_ROOT / "Dockerfile"

# Known external assets (manually curated — these are not pip packages)
EXTERNAL_ASSETS = [
    {
        "name": "bert-base-uncased",
        "version": "1.0.0",
        "supplier": "Hugging Face (google-bert/bert-base-uncased)",
        "license": "Apache-2.0",
        "source": "https://huggingface.co/google-bert/bert-base-uncased",
        "purpose": "Text encoder LLM (110M params)",
    },
    {
        "name": "TimeMMD",
        "version": "1.0.0",
        "supplier": "Liu et al., NeurIPS 2024 D&B Track",
        "license": "ODC-By-1.0",
        "source": "Hugging Face mirror AndrewRWilliams/time-mmd-DC",
        "purpose": "Multimodal time-series dataset (9 domains, 17,504 records)",
    },
    {
        "name": "NVIDIA PyTorch base image",
        "version": "23.10-py3",
        "supplier": "NVIDIA Corporation",
        "license": "NVIDIA Deep Learning Container License",
        "source": "nvcr.io/nvidia/pytorch:23.10-py3",
        "purpose": "Docker base image with CUDA 12.2 + cuDNN 8.9",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_requirements(path: Path) -> list[dict]:
    """Parse a requirements.txt file into a list of {name, version, source} dicts."""
    if not path.exists():
        return []
    components = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match: package==version (with optional comments)
            m = re.match(r"^([a-zA-Z0-9_-]+)\s*==\s*([0-9a-zA-Z.\-+]+)", line)
            if m:
                components.append({
                    "name": m.group(1),
                    "version": m.group(2),
                    "source": f"https://pypi.org/project/{m.group(1)}/{m.group(2)}/",
                    "license": "see-pypi",
                    "purpose": "runtime" if "runtime" in str(path) else "development",
                })
    return components


def parse_dockerfile(path: Path) -> str | None:
    """Extract the base image from a Dockerfile."""
    if not path.exists():
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("FROM "):
                # Take the first FROM directive
                return line.split()[1]
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SPDX JSON generation
# ---------------------------------------------------------------------------

def build_spdx(packages: list[dict], bundle_version: str) -> dict:
    """Build an SPDX 2.3 JSON document."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "ST-LLM-Plus-MVGT-Net-SBOM",
        "documentNamespace": f"https://spdx.org/spdxdocs/st-llm-plus-mvgt-net-{bundle_version}-{now.replace(':', '').replace('-', '')}",
        "creationInfo": {
            "created": now,
            "creators": [
                "Tool: generate_sbom.py (manual)",
                "Organization: ST-LLM-Plus Thesis Bundle",
            ],
            "licenseListVersion": "3.21",
        },
        "packages": [],
        "relationships": [],
    }
    for i, pkg in enumerate(packages):
        spdx["packages"].append({
            "name": pkg["name"],
            "SPDXID": f"SPDXRef-Package-{i:04d}",
            "versionInfo": pkg["version"],
            "supplier": f"Organization: {pkg.get('supplier', 'unknown')}",
            "downloadLocation": pkg.get("source", "NOASSERTION"),
            "filesAnalyzed": False,
            "licenseConcluded": pkg.get("license", "NOASSERTION"),
            "licenseDeclared": pkg.get("license", "NOASSERTION"),
            "copyrightText": "NOASSERTION",
            "description": pkg.get("purpose", ""),
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{pkg['name']}@{pkg['version']}" if pkg.get("source", "").startswith("https://pypi.org") else f"pkg:generic/{pkg['name']}@{pkg['version']}",
                }
            ],
        })
        spdx["relationships"].append({
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": f"SPDXRef-Package-{i:04d}",
        })
    return spdx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SBOM (SPDX 2.3 JSON + CSV) for MVGT-Net.")
    ap.add_argument("--output-dir", default=str(CODE_ROOT / "sbom"),
                    help="Output directory (default: ./sbom/)")
    ap.add_argument("--bundle-version", default="4.0.0",
                    help="Bundle version string for SPDX document namespace.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all components
    all_components: list[dict] = []
    for req_file, purpose in REQUIREMENTS_FILES:
        pkgs = parse_requirements(req_file)
        for p in pkgs:
            p["purpose"] = purpose
        all_components.extend(pkgs)
    all_components.extend(EXTERNAL_ASSETS)

    # Build SPDX JSON
    spdx = build_spdx(all_components, args.bundle_version)
    spdx_path = out_dir / "sbom.spdx.json"
    with open(spdx_path, "w") as fh:
        json.dump(spdx, fh, indent=2)

    # Build CSV
    csv_path = out_dir / "sbom.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "version", "license", "source", "supplier", "purpose"])
        writer.writeheader()
        for c in all_components:
            writer.writerow({k: c.get(k, "") for k in ["name", "version", "license", "source", "supplier", "purpose"]})

    # SHA-256 manifest
    sha_path = out_dir / "sbom.sha256"
    with open(sha_path, "w") as fh:
        fh.write(f"{sha256_file(spdx_path)}  {spdx_path.name}\n")
        fh.write(f"{sha256_file(csv_path)}  {csv_path.name}\n")

    print(f"[OK] SBOM generated:")
    print(f"     SPDX JSON: {spdx_path} ({spdx_path.stat().st_size:,} bytes)")
    print(f"     CSV:       {csv_path} ({csv_path.stat().st_size:,} bytes)")
    print(f"     SHA-256:   {sha_path}")
    print(f"     Components: {len(all_components)} ({sum(1 for c in all_components if c.get('source','').startswith('https://pypi.org'))} pip + {len(EXTERNAL_ASSETS)} external)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
