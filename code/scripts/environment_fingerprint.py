#!/usr/bin/env python3
"""
environment_fingerprint.py — Capture the exact runtime environment at the
start of a training run, for reproducibility auditing.

Records:
  - Python version + implementation
  - OS + kernel + architecture
  - CPU info (model, cores, frequency)
  - RAM (total + available)
  - Disk (free space on the bundle's partition)
  - GPU (model, VRAM, CUDA version, driver version)
  - All installed Python packages + versions (pip freeze)
  - PyTorch-specific info (CUDA available, cuDNN version, device count)
  - Bundle code git SHA (if available) or directory mtime
  - Config SHA-256 (hash of the YAML config file used)
  - Dataset SHA-256 (hash of the dataset manifest)

Output: logs/env_<timestamp>.json

Usage:
  python scripts/environment_fingerprint.py
  python scripts/environment_fingerprint.py --config configs/default.yaml
  python scripts/environment_fingerprint.py --output custom_env.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
BUNDLE_ROOT = CODE_ROOT.parent
LOG_DIR = BUNDLE_ROOT / "logs"


def safe_run(cmd: list[str]) -> str:
    """Run a command and return its stdout, or an error string."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"[ERROR rc={result.returncode}] {result.stderr.strip()[:200]}"
    except FileNotFoundError:
        return f"[NOT FOUND] {cmd[0]}"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] {' '.join(cmd)}"
    except Exception as e:
        return f"[EXCEPTION] {type(e).__name__}: {e}"


def get_python_info() -> Dict[str, Any]:
    return {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
    }


def get_os_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    # Linux-specific: read /etc/os-release
    if platform.system() == "Linux":
        os_release = Path("/etc/os-release")
        if os_release.exists():
            data = {}
            for line in os_release.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k] = v.strip('"')
            out["os_release"] = data
    return out


def get_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "physical_cores": os.cpu_count(),
        "logical_cores": os.cpu_count(),
    }
    # Linux: lscpu
    lscpu = safe_run(["lscpu"])
    if not lscpu.startswith("["):
        for line in lscpu.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().lower().replace(" ", "_")
                v = v.strip()
                if k in ("model_name", "architecture", "cpu(s)",
                         "thread(s)_per_core", "core(s)_per_socket",
                         "socket(s)", "cpu_max_mhz", "cpu_min_mhz"):
                    info[k] = v
    return info


def get_ram_info() -> Dict[str, Any]:
    if platform.system() != "Linux":
        return {"note": "RAM info only collected on Linux"}
    free = safe_run(["free", "--bytes"])
    if free.startswith("["):
        return {"note": "free command not available"}
    lines = free.splitlines()
    if len(lines) < 2:
        return {"note": "unexpected free output"}
    headers = lines[0].split()
    values = lines[1].split()
    if "total" in headers and "available" in headers:
        total_idx = headers.index("total")
        avail_idx = headers.index("available")
        return {
            "total_bytes": int(values[total_idx]),
            "available_bytes": int(values[avail_idx]),
            "total_human": human_bytes(int(values[total_idx])),
            "available_human": human_bytes(int(values[avail_idx])),
        }
    return {"raw": free}


def get_disk_info(path: Path) -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(str(path))
        return {
            "path": str(path),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "total_human": human_bytes(usage.total),
            "used_human": human_bytes(usage.used),
            "free_human": human_bytes(usage.free),
        }
    except Exception as e:
        return {"error": str(e)}


def get_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_human": human_bytes(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })
            # nvidia-smi for driver version
            nvidia_smi = safe_run(["nvidia-smi", "--query-gpu=driver_version",
                                   "--format=csv,noheader"])
            if not nvidia_smi.startswith("["):
                info["driver_version"] = nvidia_smi.splitlines()[0]
    except ImportError:
        info["torch"] = "not installed"
    return info


def get_pip_freeze() -> list[str]:
    """Return the output of `pip freeze` as a list of lines."""
    out = safe_run([sys.executable, "-m", "pip", "freeze"])
    if out.startswith("["):
        return [out]
    return out.splitlines()


def get_code_git_sha() -> Dict[str, Any]:
    """Return the git SHA of the code directory, or its mtime if not a git repo."""
    git_sha = safe_run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"])
    if git_sha.startswith("["):
        # Not a git repo — fall back to mtime
        return {
            "git_available": False,
            "directory_mtime": dt.datetime.fromtimestamp(
                CODE_ROOT.stat().st_mtime, tz=dt.timezone.utc
            ).isoformat(timespec="seconds"),
        }
    return {
        "git_available": True,
        "git_sha": git_sha,
        "git_branch": safe_run(["git", "-C", str(CODE_ROOT), "rev-parse",
                                "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(
            safe_run(["git", "-C", str(CODE_ROOT), "status", "--porcelain"])
        ),
    }


def hash_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=CODE_ROOT / "configs" / "default.yaml",
        help="Path to the YAML config file (for SHA-256 hashing).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: logs/env_<timestamp>.json).",
    )
    p.add_argument(
        "--dataset-manifest", type=Path,
        default=BUNDLE_ROOT / "code" / "data" / "TimeMMD" / "DATASET_MANIFEST.json",
        help="Path to the dataset manifest (for SHA-256 hashing).",
    )
    args = p.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = LOG_DIR / f"env_{ts}.json"

    print(f"Capturing environment fingerprint to: {args.output}", file=sys.stderr)

    fingerprint: Dict[str, Any] = {
        "captured_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "bundle_root": str(BUNDLE_ROOT),
        "code_root": str(CODE_ROOT),
        "python": get_python_info(),
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "disk": get_disk_info(BUNDLE_ROOT),
        "gpu": get_gpu_info(),
        "pip_freeze": get_pip_freeze(),
        "code": get_code_git_sha(),
        "config": {
            "path": str(args.config),
            "sha256": hash_file(args.config),
        },
        "dataset_manifest": {
            "path": str(args.dataset_manifest),
            "sha256": hash_file(args.dataset_manifest),
            "exists": args.dataset_manifest.exists(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Done.", file=sys.stderr)
    print(f"  Python:    {fingerprint['python']['version_info']['major']}."
          f"{fingerprint['python']['version_info']['minor']}."
          f"{fingerprint['python']['version_info']['micro']}", file=sys.stderr)
    print(f"  OS:        {fingerprint['os']['system']} "
          f"{fingerprint['os']['release']}", file=sys.stderr)
    print(f"  CPU cores: {fingerprint['cpu']['physical_cores']}",
          file=sys.stderr)
    if fingerprint["gpu"]["cuda_available"]:
        for dev in fingerprint["gpu"]["devices"]:
            print(f"  GPU:       {dev['name']} "
                  f"({dev['total_memory_human']})", file=sys.stderr)
    else:
        print("  GPU:       (CUDA not available)", file=sys.stderr)
    print(f"  Packages:  {len(fingerprint['pip_freeze'])} installed",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
