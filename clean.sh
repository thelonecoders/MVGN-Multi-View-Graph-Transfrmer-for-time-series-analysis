#!/usr/bin/env bash
# clean.sh — Remove all generated / runtime artifacts from the VPS Code Bundle.
#
# Use this script to reset the bundle to a pristine state before a fresh
# pipeline run, or before re-zipping the bundle for distribution.
#
# By default, this script ASKS for confirmation before deleting. Use --force
# to skip the confirmation prompt.
#
# Usage:
#   ./clean.sh                # interactive (asks for confirmation)
#   ./clean.sh --force        # non-interactive, no confirmation
#   ./clean.sh --dry-run      # show what would be deleted, delete nothing
#   ./clean.sh --help         # show this help message
#
# What gets removed:
#   - code/.venv/                (Python virtual environment, ~3.3 GB)
#   - code/data/                 (downloaded dataset, 322 MB)
#   - code/checkpoints/          (trained model weights, up to 1.4 GB)
#   - code/results/              (training metrics + predictions)
#   - code/14_engineering_analyses/  (Phase F outputs)
#   - code/bert_cache/           (BERT-base-uncased weights, 440 MB)
#   - logs/                      (pipeline log files)
#   - code/__pycache__/          (Python bytecode caches)
#   - code/**/__pycache__/       (recursive bytecode caches)
#   - code/.pytest_cache/        (pytest cache)
#   - code/.mypy_cache/          (mypy cache)
#   - code/.ruff_cache/          (ruff cache)
#
# What is PRESERVED:
#   - All source code (mvgt_net/, scripts/, tests/)
#   - All configs (configs/)
#   - All docs (docs/, README.md)
#   - All shell scripts (run/, setup/)
#   - All dataset downloaders (dataset_downloader/)
#   - All metadata (metadata/, MANIFEST.json, checksums.sha256)
#   - LICENSE, CITATION.cff, CITATION.bib, .gitignore

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# Directories to remove (if they exist).
DIRS_TO_REMOVE=(
    "code/.venv"
    "code/data"
    "code/checkpoints"
    "code/results"
    "code/14_engineering_analyses"
    "code/bert_cache"
    "logs"
    "code/.pytest_cache"
    "code/.mypy_cache"
    "code/.ruff_cache"
)

# Glob patterns to remove (recursively).
GLOBS_TO_REMOVE=(
    "code/**/__pycache__"
    "code/**/*.pyc"
    "code/**/*.pyo"
    "code/**/*.egg-info"
)

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------

FORCE=false
DRY_RUN=false

for arg in "$@"; do
    case "${arg}" in
        --force|-f)
            FORCE=true
            ;;
        --dry-run|-n)
            DRY_RUN=true
            ;;
        --help|-h)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Try: ${0} --help"
            exit 2
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Compute total size of artifacts to remove
# ----------------------------------------------------------------------------

total_size_bytes=0
for d in "${DIRS_TO_REMOVE[@]}"; do
    full="${BUNDLE_ROOT}/${d}"
    if [ -d "${full}" ]; then
        size=$(du -sb "${full}" 2>/dev/null | awk '{print $1}' || echo 0)
        total_size_bytes=$((total_size_bytes + size))
    fi
done

human_size() {
    local bytes=$1
    if [ ${bytes} -ge 1073741824 ]; then
        echo "$((bytes / 1073741824)) GiB"
    elif [ ${bytes} -ge 1048576 ]; then
        echo "$((bytes / 1048576)) MiB"
    elif [ ${bytes} -ge 1024 ]; then
        echo "$((bytes / 1024)) KiB"
    else
        echo "${bytes} B"
    fi
}

# ----------------------------------------------------------------------------
# Confirmation prompt
# ----------------------------------------------------------------------------

if ${DRY_RUN}; then
    echo "DRY RUN: no files will be deleted."
    echo ""
elif ! ${FORCE}; then
    echo "About to remove generated artifacts from:"
    echo "  ${BUNDLE_ROOT}"
    echo ""
    echo "Estimated total size to free: $(human_size ${total_size_bytes})"
    echo ""
    echo "Directories to remove:"
    for d in "${DIRS_TO_REMOVE[@]}"; do
        full="${BUNDLE_ROOT}/${d}"
        if [ -d "${full}" ]; then
            size=$(du -sh "${full}" 2>/dev/null | awk '{print $1}' || echo "?")
            printf "  [DIR]  %-50s %s\n" "${d}" "${size}"
        fi
    done
    echo ""
    echo "Glob patterns to remove:"
    for g in "${GLOBS_TO_REMOVE[@]}"; do
        printf "  [GLOB] %s\n" "${g}"
    done
    echo ""
    read -r -p "Are you sure? Type 'yes' to confirm: " response
    if [ "${response}" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# Execute deletion
# ----------------------------------------------------------------------------

removed_count=0
freed_bytes=0

for d in "${DIRS_TO_REMOVE[@]}"; do
    full="${BUNDLE_ROOT}/${d}"
    if [ -d "${full}" ]; then
        size=$(du -sb "${full}" 2>/dev/null | awk '{print $1}' || echo 0)
        if ${DRY_RUN}; then
            echo "  [DRY-RUN] Would remove: ${d} ($(human_size ${size}))"
        else
            rm -rf "${full}"
            echo "  [REMOVED] ${d} ($(human_size ${size}))"
        fi
        removed_count=$((removed_count + 1))
        freed_bytes=$((freed_bytes + size))
    fi
done

# Glob-based removal (use find for ** patterns)
for g in "${GLOBS_TO_REMOVE[@]}"; do
    # Convert ** to find syntax
    find_pattern="${g//\*\*/.}"
    find_pattern="${find_pattern//\*/[^/]*}"
    while IFS= read -r match; do
        if [ -n "${match}" ] && [ -e "${match}" ]; then
            if ${DRY_RUN}; then
                echo "  [DRY-RUN] Would remove: ${match}"
            else
                rm -rf "${match}"
                echo "  [REMOVED] ${match}"
            fi
        fi
    done < <(find "${BUNDLE_ROOT}" -path "${BUNDLE_ROOT}/${find_pattern}" 2>/dev/null || true)
done

echo ""
if ${DRY_RUN}; then
    echo "Dry run complete. ${removed_count} directories would be removed,"
    echo "freeing approximately $(human_size ${freed_bytes})."
else
    echo "Cleanup complete. ${removed_count} directories removed,"
    echo "freed approximately $(human_size ${freed_bytes})."
    echo ""
    echo "The bundle is now in a pristine state. To re-run the pipeline:"
    echo "  ./run/run_pipeline.sh"
fi
