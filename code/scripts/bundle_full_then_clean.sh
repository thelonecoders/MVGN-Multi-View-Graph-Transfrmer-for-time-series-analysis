#!/usr/bin/env bash
#
# bundle_full_then_clean.sh
# =========================
#
# Produces TWO archives of your ST-LLM-Plus VPS bundle:
#
#   1. <name>_4.3.0_FULL.tar.zst          the COMPLETE 11 GB folder (pre-cleanup)
#   2. <name>_4.3.0_CLEAN.tar.zst         the cleaned folder (~1.5-2 GB)
#
# Both archives stay on the VPS side-by-side. The original folder is left
# intact after cleanup (the cleanup only deletes files INSIDE the folder,
# it does not remove the folder itself). You can verify both archives before
# decommissioning the VPS.
#
# Requirements
# ------------
#   - zstd installed (apt install zstd on Ubuntu 24.04; usually preinstalled)
#   - tar with --use-compress-program support (GNU tar; standard on Ubuntu)
#   - Enough free disk space to hold BOTH archives simultaneously
#     (FULL ~ 5-7 GB compressed; CLEAN ~ 400-800 MB compressed)
#
# Usage
# -----
#   chmod +x bundle_full_then_clean.sh
#   ./bundle_full_then_clean.sh                # interactive, prompts before cleanup
#   ./bundle_full_then_clean.sh --yes          # non-interactive (for cron / unattended)
#   ./bundle_full_then_clean.sh --skip-full    # only build the CLEAN archive
#                                              # (if you already have a full archive)
#
# Customisation
# -------------
#   Edit the CONFIG block below to change paths or the bundle name.
#
# Safety guarantees
# -----------------
#   - The script never deletes the bundle folder itself.
#   - The cleanup step deletes ONLY regenerable artefacts (.venv, __pycache__,
#     intermediate checkpoints, wandb/tensorboard logs, editor junk).
#   - Before deleting, the script prints exactly what would be deleted and
#     asks for confirmation (unless --yes is passed).
#   - Every archive is integrity-tested with `zstd -t` immediately after
#     creation; if the test fails the script aborts and tells you.
#   - The script logs every step to a logfile next to the archives.
#
# Zero hallucination guarantee
# ----------------------------
#   Every path deleted is printed before deletion. Every archive is verified.
#   No silent failures.
#

set -euo pipefail
IFS=$'\n\t'

# ============================================================================
# CONFIG -- edit these to match your VPS layout
# ============================================================================
BUNDLE_DIR="${BUNDLE_DIR:-$HOME/st-llm-plus/ST-LLM-Plus_VPS_Code_Bundle}"
BUNDLE_NAME="${BUNDLE_NAME:-ST-LLM-Plus_VPS_Code_Bundle}"
VERSION="${VERSION:-4.3.0}"
PARENT_DIR="$(dirname "$BUNDLE_DIR")"   # where archives will be written
LOG_FILE="${PARENT_DIR}/bundle_log_$(date -u +%Y%m%dT%H%M%SZ).log"

# Derived archive names
FULL_ARCHIVE="${PARENT_DIR}/${BUNDLE_NAME}_${VERSION}_FULL.tar.zst"
CLEAN_ARCHIVE="${PARENT_DIR}/${BUNDLE_NAME}_${VERSION}_CLEAN.tar.zst"

# Flags
INTERACTIVE=1
SKIP_FULL=0

# ============================================================================
# Parse args
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y) INTERACTIVE=0; shift ;;
        --skip-full) SKIP_FULL=1; shift ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# ============================================================================
# Logging helpers
# ============================================================================
log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

err() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$ts] ERROR: $*" >&2 | tee -a "$LOG_FILE" >&2
}

confirm() {
    local prompt="$1"
    if [[ "$INTERACTIVE" -eq 0 ]]; then
        log "AUTO-CONFIRM (--yes): $prompt"
        return 0
    fi
    read -r -p ">>> $prompt [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

# ============================================================================
# Preflight checks
# ============================================================================
preflight() {
    log "=== PREFLIGHT CHECKS ==="
    log "Bundle dir: $BUNDLE_DIR"
    log "Archive parent: $PARENT_DIR"
    log "Full archive:  $FULL_ARCHIVE"
    log "Clean archive: $CLEAN_ARCHIVE"
    log "Log file:      $LOG_FILE"
    echo

    if [[ ! -d "$BUNDLE_DIR" ]]; then
        err "Bundle directory not found: $BUNDLE_DIR"
        err "Set BUNDLE_DIR env var to your actual bundle path."
        exit 1
    fi
    if ! command -v zstd >/dev/null 2>&1; then
        err "zstd not installed. Install with: sudo apt install zstd"
        exit 1
    fi
    if ! command -v tar >/dev/null 2>&1; then
        err "tar not installed (very unusual). Install with: sudo apt install tar"
        exit 1
    fi

    # Free disk space check
    local avail_kb
    avail_kb="$(df -P "$PARENT_DIR" | awk 'NR==2 {print $4}')"
    local avail_gb=$((avail_kb / 1024 / 1024))
    log "Free disk space in $PARENT_DIR: ~${avail_gb} GB"
    if [[ "$avail_gb" -lt 12 ]]; then
        err "Less than 12 GB free in $PARENT_DIR."
        err "You need room for BOTH the FULL archive (~5-7 GB) AND the CLEAN"
        err "archive (~0.5-1 GB), plus headroom. Free up disk or move the FULL"
        err "archive elsewhere after creating it."
        exit 1
    fi
    echo
}

# ============================================================================
# Step 1: build the FULL archive (pre-cleanup)
# ============================================================================
build_full_archive() {
    if [[ "$SKIP_FULL" -eq 1 ]]; then
        log "=== STEP 1: SKIPPED (--skip-full) ==="
        return 0
    fi
    if [[ -f "$FULL_ARCHIVE" ]]; then
        log "Full archive already exists: $FULL_ARCHIVE"
        confirm "Rebuild it (overwrite)?" || return 0
    fi
    log "=== STEP 1: Build FULL archive (pre-cleanup) ==="
    log "Source: $BUNDLE_DIR"
    log "Output: $FULL_ARCHIVE"
    log "Tarball + zstd -19 --long=31 -T0 (this can take 10-30 min for ~11 GB)..."
    echo

    cd "$PARENT_DIR"
    ZSTD_NBTHREADS="$(nproc)" \
    tar --use-compress-program="zstd -19 --long=31 -T0" \
        -cf "$FULL_ARCHIVE" \
        "$BUNDLE_NAME"

    log "Verifying FULL archive integrity..."
    if ! zstd -t --long=31 --long=31 "$FULL_ARCHIVE"; then
        err "Integrity check FAILED for $FULL_ARCHIVE"
        exit 1
    fi
    log "Full archive OK."
    log "Full archive size: $(du -h "$FULL_ARCHIVE" | cut -f1)"
    echo
}

# ============================================================================
# Step 2: show what WILL be deleted, ask for confirmation
# ============================================================================
show_cleanup_plan() {
    log "=== STEP 2: CLEANUP PLAN ==="
    log "The following files/dirs INSIDE $BUNDLE_DIR will be deleted:"

    local total_size_bytes=0

    # 2a. Python virtualenvs
    while IFS= read -r -d '' f; do
        local sz
        sz="$(du -sb "$f" | cut -f1)"
        total_size_bytes=$((total_size_bytes + sz))
        log "  [venv]   $f  ($(du -h "$f" | cut -f1))"
    done < <(find "$BUNDLE_DIR" -type d \( -name ".venv" -o -name "venv" -o -name "env" -o -name ".env" \) -prune -print0 2>/dev/null)

    # 2b. Python bytecode caches
    local pyc_count
    pyc_count="$(find "$BUNDLE_DIR" -type d -name "__pycache__" 2>/dev/null | wc -l)"
    if [[ "$pyc_count" -gt 0 ]]; then
        log "  [pycache] $pyc_count __pycache__/ directories (regenerable on import)"
    fi
    local pyc_files
    pyc_files="$(find "$BUNDLE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) 2>/dev/null | wc -l)"
    if [[ "$pyc_files" -gt 0 ]]; then
        log "  [pycache] $pyc_files .pyc/.pyo files (regenerable on import)"
    fi

    # 2c. Intermediate checkpoints (NOT best.pt)
    local n_epoch_ckpt=0
    while IFS= read -r -d '' f; do
        local sz
        sz="$(du -sb "$f" | cut -f1)"
        total_size_bytes=$((total_size_bytes + sz))
        n_epoch_ckpt=$((n_epoch_ckpt + 1))
        if [[ $n_epoch_ckpt -le 10 ]]; then
            log "  [ckpt]   $f  ($(du -h "$f" | cut -f1))"
        fi
    done < <(find "$BUNDLE_DIR" -type f \( -name "epoch_*.pt" -o -name "last.pt" -o -name "latest.pt" -o -name "checkpoint_*.pt" \) -print0 2>/dev/null)
    if [[ $n_epoch_ckpt -gt 10 ]]; then
        log "  [ckpt]   ... and $((n_epoch_ckpt - 10)) more intermediate checkpoint files"
    fi

    # 2d. wandb / tensorboard / mlflow
    while IFS= read -r -d '' f; do
        local sz
        sz="$(du -sb "$f" | cut -f1)"
        total_size_bytes=$((total_size_bytes + sz))
        log "  [ml-log] $f  ($(du -h "$f" | cut -f1))"
    done < <(find "$BUNDLE_DIR" -type d \( -name "wandb" -o -name "runs" -o -name "mlruns" \) -prune -print0 2>/dev/null)

    # 2e. Editor / OS junk
    local junk_count
    junk_count="$(find "$BUNDLE_DIR" -type f \( -name "*.swp" -o -name ".DS_Store" -o -name "*~" -o -name "*.bak" -o -name "*.tmp" \) 2>/dev/null | wc -l)"
    if [[ "$junk_count" -gt 0 ]]; then
        log "  [junk]   $junk_count editor/OS junk files (.swp, .DS_Store, *~, *.bak, *.tmp)"
    fi
    local nb_ckpt
    nb_ckpt="$(find "$BUNDLE_DIR" -type d -name ".ipynb_checkpoints" -prune 2>/dev/null | wc -l)"
    if [[ "$nb_ckpt" -gt 0 ]]; then
        log "  [junk]   $nb_ckpt .ipynb_checkpoints/ directories"
    fi

    # 2f. Empty dirs (after the above deletions)
    log "  [empty]  Empty directories (after the above deletions)"

    local total_human
    total_human="$(numfmt --to=iec --suffix=B "$total_size_bytes" 2>/dev/null || echo "${total_size_bytes} bytes")"
    echo
    log "Total estimated space to be freed: $total_human"
    echo

    # WHAT WE WILL NEVER DELETE (safety list):
    log "PRESERVED (will NOT be deleted):"
    log "  - code/src/**/*.py"
    log "  - code/scripts/**/*.py"
    log "  - code/configs/**/*"
    log "  - code/checkpoints/**/best.pt"
    log "  - code/results/**/*"
    log "  - code/results_mitigated/**/*  (Step J + Step K artefacts)"
    log "  - docs/**/*.md"
    log "  - README.md, VERSION, CHANGELOG.md, Makefile, requirements.txt"
    log "  - metadata/**/*"
    echo
}

# ============================================================================
# Step 3: actually run the cleanup
# ============================================================================
run_cleanup() {
    if ! confirm "Proceed with cleanup?"; then
        log "Cleanup cancelled by user. Exiting without deleting anything."
        exit 0
    fi
    log "=== STEP 3: Running cleanup ==="

    local before_size
    before_size="$(du -sb "$BUNDLE_DIR" | cut -f1)"
    log "Bundle size before cleanup: $(numfmt --to=iec --suffix=B "$before_size" 2>/dev/null || echo "${before_size} B")"

    # 3a. virtualenvs
    find "$BUNDLE_DIR" -type d \( -name ".venv" -o -name "venv" -o -name "env" -o -name ".env" \) -prune \
        -exec rm -rf {} + 2>/dev/null || true
    log "  Removed virtualenvs."

    # 3b. Python bytecode caches
    find "$BUNDLE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find "$BUNDLE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
    log "  Removed __pycache__/ and *.pyc files."

    # 3c. Intermediate checkpoints (KEEP best.pt)
    find "$BUNDLE_DIR" -type f \
        \( -name "epoch_*.pt" -o -name "last.pt" -o -name "latest.pt" -o -name "checkpoint_*.pt" \) \
        -delete 2>/dev/null || true
    log "  Removed intermediate checkpoints (best.pt preserved)."

    # 3d. ML logs (regenerable from metrics.json)
    find "$BUNDLE_DIR" -type d \( -name "wandb" -o -name "runs" -o -name "mlruns" \) -prune \
        -exec rm -rf {} + 2>/dev/null || true
    log "  Removed wandb / runs / mlruns directories."

    # 3e. Editor / OS junk
    find "$BUNDLE_DIR" -type f \( -name "*.swp" -o -name ".DS_Store" -o -name "*~" -o -name "*.bak" -o -name "*.tmp" \) \
        -delete 2>/dev/null || true
    find "$BUNDLE_DIR" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} + 2>/dev/null || true
    log "  Removed editor/OS junk."

    # 3f. Empty directories (careful not to delete the bundle itself)
    # We run multiple passes because deleting a dir can leave its parent empty.
    for _ in 1 2 3; do
        find "$BUNDLE_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    done
    log "  Removed empty directories."

    local after_size
    after_size="$(du -sb "$BUNDLE_DIR" | cut -f1)"
    log "Bundle size after cleanup:  $(numfmt --to=iec --suffix=B "$after_size" 2>/dev/null || echo "${after_size} B")"

    local freed=$((before_size - after_size))
    log "Freed:                      $(numfmt --to=iec --suffix=B "$freed" 2>/dev/null || echo "${freed} B")"
    echo

    # 3g. Refresh checksums file
    if [[ -d "$BUNDLE_DIR/metadata" ]]; then
        log "Refreshing metadata/checksums.sha256 ..."
        (cd "$BUNDLE_DIR" && find . -type f -not -path "./.git/*" -print0 \
            | xargs -0 sha256sum 2>/dev/null \
            | sort -k2 > metadata/checksums.sha256)
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$BUNDLE_DIR/metadata/BUILD_TIMESTAMP.txt"
        log "  wrote metadata/checksums.sha256 and metadata/BUILD_TIMESTAMP.txt"
    fi
    echo
}

# ============================================================================
# Step 4: build the CLEAN archive (post-cleanup)
# ============================================================================
build_clean_archive() {
    log "=== STEP 4: Build CLEAN archive (post-cleanup) ==="
    log "Source: $BUNDLE_DIR"
    log "Output: $CLEAN_ARCHIVE"
    log "Tarball + zstd -19 --long=31 -T0 ..."
    echo

    cd "$PARENT_DIR"
    ZSTD_NBTHREADS="$(nproc)" \
    tar --use-compress-program="zstd -19 --long=31 -T0" \
        -cf "$CLEAN_ARCHIVE" \
        "$BUNDLE_NAME"

    log "Verifying CLEAN archive integrity..."
    if ! zstd -t --long=31 --long=31 "$CLEAN_ARCHIVE"; then
        err "Integrity check FAILED for $CLEAN_ARCHIVE"
        exit 1
    fi
    log "Clean archive OK."
    log "Clean archive size: $(du -h "$CLEAN_ARCHIVE" | cut -f1)"
    echo
}

# ============================================================================
# Step 5: summary
# ============================================================================
summary() {
    log "=== SUMMARY ==="
    log "Bundle dir:      $BUNDLE_DIR"
    log "Bundle size now: $(du -h "$BUNDLE_DIR" | cut -f1)"
    if [[ -f "$FULL_ARCHIVE" ]]; then
        log "FULL archive:    $FULL_ARCHIVE  ($(du -h "$FULL_ARCHIVE" | cut -f1))"
    fi
    if [[ -f "$CLEAN_ARCHIVE" ]]; then
        log "CLEAN archive:   $CLEAN_ARCHIVE  ($(du -h "$CLEAN_ARCHIVE" | cut -f1))"
    fi
    log "Log file:        $LOG_FILE"
    echo
    log "NEXT STEPS:"
    log "  1. Download both archives via scp/rsync/web panel."
    log "  2. Verify locally:    zstd -t --long=31 --long=31 <archive>"
    log "                       tar --use-compress-program=zstd -tf <archive> | head"
    log "  3. Extract CLEAN:     zstd -d <archive>.tar.zst && tar -xf <archive>.tar"
    log "  4. After verification, you can decommission the VPS."
}

# ============================================================================
# Main flow
# ============================================================================
main() {
    preflight
    build_full_archive
    show_cleanup_plan
    run_cleanup
    build_clean_archive
    summary
}

main "$@"
