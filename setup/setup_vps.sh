#!/usr/bin/env bash
#==============================================================================
# setup_vps.sh — One-shot VPS provisioning for the ST-LLM-Plus VPS Code Bundle
#==============================================================================
#
# This is a thin wrapper around run/run_step0_install.sh that also installs
# a few additional system packages recommended for VPS administration:
#   - htop, tmux, vim (for interactive debugging during long training runs)
#   - rsync (for transferring checkpoints off the VPS)
#   - jq (for pretty-printing JSON metrics)
#
# Usage:
#   chmod +x setup/setup_vps.sh
#   ./setup/setup_vps.sh
#
# After this script completes, run:
#   ./run/run_pipeline.sh
#
# Time: 8-12 minutes (same as run_step0_install.sh)
# Disk: ~3.7 GB (apt + venv + recommended utilities)
# RAM peak: 1.5 GB
#
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "===================================================================="
echo "  VPS Provisioning: ST-LLM-Plus VPS Code Bundle"
echo "  Bundle root: ${BUNDLE_ROOT}"
echo "===================================================================="

# Step A: Install recommended VPS admin utilities (optional, fast)
echo ""
echo "--- Installing recommended VPS utilities (htop, tmux, vim, rsync, jq) ---"
sudo apt-get update -qq || true
sudo apt-get install -y -qq htop tmux vim rsync jq 2>&1 | tail -3

# Step B: Delegate to run/run_step0_install.sh for the heavy lifting
echo ""
"${BUNDLE_ROOT}/run/run_step0_install.sh"

# Done
echo ""
echo "===================================================================="
echo "  VPS provisioning complete."
echo "  Next step: ./run/run_pipeline.sh"
echo "===================================================================="
exit 0
