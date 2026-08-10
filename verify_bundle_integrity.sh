#!/usr/bin/env bash
# verify_bundle_integrity.sh — End-to-end integrity verification for the
# ST-LLM-Plus VPS Code Bundle.
#
# Verifies:
#   1. SHA-256 checksums of all bundle source files
#   2. Syntax of all shell scripts (bash -n)
#   3. Syntax of all Python files (python -m py_compile)
#   4. YAML config files parse correctly
#   5. MANIFEST.json exists and is well-formed
#   6. Critical files are present (README, LICENSE, CITATION.cff, requirements.txt)
#   7. Python package imports cleanly (if venv exists)
#   8. Unit tests pass (if venv exists)
#
# Usage:
#   ./verify_bundle_integrity.sh
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
META_DIR="${BUNDLE_ROOT}/metadata"
CHECKSUMS="${META_DIR}/checksums.sha256"
MANIFEST="${META_DIR}/MANIFEST.json"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

check_pass() { green "  [PASS] $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
check_fail() { red   "  [FAIL] $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
check_skip() { yellow "  [SKIP] $1"; SKIP_COUNT=$((SKIP_COUNT + 1)); }

section() { bold ""; bold "=== $1 ==="; }

# ----------------------------------------------------------------------------
# 1. SHA-256 checksum verification
# ----------------------------------------------------------------------------

section "1. SHA-256 checksum verification"

if [ -f "${CHECKSUMS}" ]; then
    cd "${BUNDLE_ROOT}"
    if sha256sum -c "${CHECKSUMS}" --status 2>/dev/null; then
        check_pass "All $(wc -l < "${CHECKSUMS}") SHA-256 checksums match"
    else
        # Show which files failed
        cd "${BUNDLE_ROOT}"
        local_fails=$(sha256sum -c "${CHECKSUMS}" 2>/dev/null | grep -v ': OK$' || true)
        if [ -n "${local_fails}" ]; then
            red "${local_fails}"
        fi
        check_fail "Some SHA-256 checksums do not match (see above)"
    fi
    cd "${SCRIPT_DIR}"
else
    check_fail "checksums.sha256 not found at ${CHECKSUMS}"
fi

# ----------------------------------------------------------------------------
# 2. Shell script syntax check
# ----------------------------------------------------------------------------

section "2. Shell script syntax check (bash -n)"

shell_scripts=( "${BUNDLE_ROOT}"/run/*.sh "${BUNDLE_ROOT}"/setup/*.sh )
shell_ok=true
for script in "${shell_scripts[@]}"; do
    if [ -f "${script}" ]; then
        if bash -n "${script}" 2>/dev/null; then
            : # ok
        else
            red "  Syntax error in: ${script}"
            bash -n "${script}" || true
            shell_ok=false
        fi
    fi
done
if ${shell_ok}; then
    check_pass "All ${#shell_scripts[@]} shell scripts pass bash -n"
else
    check_fail "Some shell scripts have syntax errors"
fi

# ----------------------------------------------------------------------------
# 3. Python file syntax check
# ----------------------------------------------------------------------------

section "3. Python file syntax check (py_compile)"

python_files=$(find "${BUNDLE_ROOT}/code" -name '*.py' -type f 2>/dev/null || true)
python_ok=true
python_count=0
if [ -n "${python_files}" ]; then
    while IFS= read -r pyfile; do
        python_count=$((python_count + 1))
        if ! python3 -m py_compile "${pyfile}" 2>/dev/null; then
            red "  Compile error in: ${pyfile}"
            python3 -m py_compile "${pyfile}" || true
            python_ok=false
        fi
    done <<< "${python_files}"
fi
# Clean up __pycache__ directories created by py_compile
find "${BUNDLE_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
if ${python_ok} && [ ${python_count} -gt 0 ]; then
    check_pass "All ${python_count} Python files compile cleanly"
elif [ ${python_count} -eq 0 ]; then
    check_skip "No Python files found (expected at code/**/*.py)"
else
    check_fail "Some Python files have compile errors"
fi

# ----------------------------------------------------------------------------
# 4. YAML config parsing
# ----------------------------------------------------------------------------

section "4. YAML config parsing"

yaml_files=( "${BUNDLE_ROOT}"/code/configs/*.yaml "${BUNDLE_ROOT}"/code/configs/*.yml )
yaml_ok=true
yaml_count=0
if python3 -c "import yaml" 2>/dev/null; then
    for yfile in "${yaml_files[@]}"; do
        if [ -f "${yfile}" ]; then
            yaml_count=$((yaml_count + 1))
            if ! python3 -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "${yfile}" 2>/dev/null; then
                red "  Parse error in: ${yfile}"
                yaml_ok=false
            fi
        fi
    done
    if ${yaml_ok} && [ ${yaml_count} -gt 0 ]; then
        check_pass "All ${yaml_count} YAML configs parse correctly"
    elif [ ${yaml_count} -eq 0 ]; then
        check_skip "No YAML config files found"
    else
        check_fail "Some YAML configs have parse errors"
    fi
else
    check_skip "PyYAML not installed; cannot validate YAML configs"
fi

# ----------------------------------------------------------------------------
# 5. MANIFEST.json well-formedness
# ----------------------------------------------------------------------------

section "5. MANIFEST.json well-formedness"

if [ -f "${MANIFEST}" ]; then
    if python3 -c "import json; m=json.load(open('${MANIFEST}')); print(f'  Files: {m[\"statistics\"][\"total_files\"]}, Size: {m[\"statistics\"][\"total_size_human\"]}')" 2>/dev/null; then
        check_pass "MANIFEST.json is valid JSON"
    else
        check_fail "MANIFEST.json is not valid JSON"
    fi
else
    check_fail "MANIFEST.json not found at ${MANIFEST}"
fi

# ----------------------------------------------------------------------------
# 6. Critical files presence
# ----------------------------------------------------------------------------

section "6. Critical files presence"

critical_files=(
    "${BUNDLE_ROOT}/README.md"
    "${BUNDLE_ROOT}/LICENSE"
    "${BUNDLE_ROOT}/CITATION.cff"
    "${BUNDLE_ROOT}/CITATION.bib"
    "${BUNDLE_ROOT}/.gitignore"
    "${BUNDLE_ROOT}/code/requirements.txt"
    "${BUNDLE_ROOT}/code/pyproject.toml"
    "${BUNDLE_ROOT}/code/Dockerfile"
    "${BUNDLE_ROOT}/code/docker-compose.yml"
    "${BUNDLE_ROOT}/code/MODEL_CARD.md"
    "${BUNDLE_ROOT}/code/DATA_CARD.md"
    "${BUNDLE_ROOT}/code/mvgt_net/__init__.py"
    "${BUNDLE_ROOT}/code/scripts/train_real.py"
    "${BUNDLE_ROOT}/run/run_pipeline.sh"
    "${BUNDLE_ROOT}/run/run_step0_install.sh"
)
critical_ok=true
for cf in "${critical_files[@]}"; do
    if [ ! -f "${cf}" ]; then
        red "  Missing: ${cf}"
        critical_ok=false
    fi
done
if ${critical_ok}; then
    check_pass "All ${#critical_files[@]} critical files present"
else
    check_fail "Some critical files are missing"
fi

# ----------------------------------------------------------------------------
# 7. Python package import (if venv exists)
# ----------------------------------------------------------------------------

section "7. Python package import (mvgt_net)"

VENV="${BUNDLE_ROOT}/code/.venv/bin/activate"
if [ -f "${VENV}" ]; then
    # shellcheck source=/dev/null
    source "${VENV}"
    cd "${BUNDLE_ROOT}/code"
    if python3 -c "import mvgt_net; print(f'  mvgt_net exports: {len(dir(mvgt_net))} symbols')" 2>/dev/null; then
        check_pass "mvgt_net package imports cleanly"
    else
        check_fail "mvgt_net package failed to import"
    fi
    cd "${SCRIPT_DIR}"
    deactivate 2>/dev/null || true
else
    check_skip "Virtual environment not yet created (run run_step0_install.sh first)"
fi

# ----------------------------------------------------------------------------
# 8. Unit tests (if venv exists)
# ----------------------------------------------------------------------------

section "8. Unit tests"

if [ -f "${VENV}" ]; then
    # shellcheck source=/dev/null
    source "${VENV}"
    cd "${BUNDLE_ROOT}/code"
    if python3 -m pytest tests/ -q --tb=line 2>&1 | tail -5; then
        check_pass "All unit tests pass"
    else
        check_fail "Some unit tests failed"
    fi
    cd "${SCRIPT_DIR}"
    deactivate 2>/dev/null || true
else
    check_skip "Virtual environment not yet created; cannot run unit tests"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

section "Summary"
green "  Passed: ${PASS_COUNT}"
red   "  Failed: ${FAIL_COUNT}"
yellow "  Skipped: ${SKIP_COUNT}"

if [ ${FAIL_COUNT} -eq 0 ]; then
    green ""
    green "=========================================="
    green "  ALL CHECKS PASSED"
    green "=========================================="
    exit 0
else
    red ""
    red "=========================================="
    red "  ${FAIL_COUNT} CHECK(S) FAILED"
    red "=========================================="
    exit 1
fi
