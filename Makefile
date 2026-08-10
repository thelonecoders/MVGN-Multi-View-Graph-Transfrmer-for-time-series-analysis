# Makefile for the ST-LLM-Plus VPS Code Bundle.
#
# Convenience targets for common operations. All targets are phony (they
# delegate to the underlying shell scripts in run/ and setup/).
#
# Usage:
#   make help           # show all available targets
#   make install        # install dependencies (creates venv)
#   make download       # download the dataset
#   make smoke          # run smoke tests
#   make train DOMAIN=Economy_Trade EPOCHS=100   # train one domain
#   make train-all      # train all 9 domains
#   make analyses       # run Phase F engineering analyses
#   make verify         # verify bundle integrity
#   make clean          # remove all generated artifacts
#   make figures        # generate thesis figures from results
#   make fingerprint    # capture runtime environment fingerprint
#   make determinism    # verify training is deterministic

.PHONY: help install download smoke train train-all analyses \
        verify clean figures fingerprint determinism \
        test lint format docker-build docker-run

# Default shell
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Bundle root = directory containing this Makefile
BUNDLE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Variables (overridable from command line)
DOMAIN ?= Climate_AQI
EPOCHS ?= 100
DEVICE ?= $(shell python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)

help: ## Show this help message
	@echo "ST-LLM-Plus VPS Code Bundle — Makefile targets"
	@echo ""
	@echo "Usage: make <target> [VAR=value]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (overridable):"
	@echo "  DOMAIN=$(DOMAIN)        # TimeMMD domain for train target"
	@echo "  EPOCHS=$(EPOCHS)        # Number of training epochs"
	@echo "  DEVICE=$(DEVICE)        # cuda or cpu"

install: ## Install dependencies (creates Python venv)
	@echo "[make install] Running run_step0_install.sh ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_step0_install.sh

download: ## Download the REAL TimeMMD dataset (322 MB, 9 domains)
	@echo "[make download] Running run_step1_download_dataset.sh ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_step1_download_dataset.sh

smoke: ## Run smoke tests (11 unit tests + 2-epoch training)
	@echo "[make smoke] Running run_step2_smoke_test.sh ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_step2_smoke_test.sh

train: ## Train a single domain (DOMAIN=xxx EPOCHS=yyy)
	@echo "[make train] Training domain=$(DOMAIN) epochs=$(EPOCHS) device=$(DEVICE) ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_step3_train_single_domain.sh $(DOMAIN) $(EPOCHS)

train-all: ## Train all 9 TimeMMD domains sequentially
	@echo "[make train-all] Running run_step4_train_all_domains.sh ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_step4_train_all_domains.sh

analyses: ## Run all 5 Phase F engineering analyses
	@echo "[make analyses] Running run_step5_analyses.sh ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_step5_analyses.sh

verify: ## Verify bundle integrity (8 checks)
	@echo "[make verify] Running verify_bundle_integrity.sh ..."
	@cd "$(BUNDLE_ROOT)" && ./verify_bundle_integrity.sh

clean: ## Remove all generated artifacts (venv, data, checkpoints, results, logs)
	@echo "[make clean] Running clean.sh --force ..."
	@cd "$(BUNDLE_ROOT)" && ./clean.sh --force

figures: ## Generate thesis figures from training results
	@echo "[make figures] Generating figures from results/*/metrics.json ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		source .venv/bin/activate && \
		python scripts/make_figure.py \
			--metrics-glob 'results/*/metrics.json' \
			--output-dir figures

fingerprint: ## Capture runtime environment fingerprint
	@echo "[make fingerprint] Capturing environment to logs/env_*.json ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		source .venv/bin/activate && \
		python scripts/environment_fingerprint.py

determinism: ## Verify training is deterministic under seed=42
	@echo "[make determinism] Running verify_determinism.py ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		source .venv/bin/activate && \
		python scripts/verify_determinism.py --domain Economy_Trade --epochs 3

test: ## Run unit tests (pytest)
	@echo "[make test] Running pytest ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		source .venv/bin/activate && \
		python -m pytest tests/ -v

lint: ## Run ruff linter
	@echo "[make lint] Running ruff check ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		source .venv/bin/activate 2>/dev/null || true && \
		ruff check mvgt_net scripts tests || echo "(ruff not installed; run: pip install ruff==0.6.9)"

format: ## Run ruff formatter
	@echo "[make format] Running ruff format ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		source .venv/bin/activate 2>/dev/null || true && \
		ruff format mvgt_net scripts tests || echo "(ruff not installed; run: pip install ruff==0.6.9)"

docker-build: ## Build the Docker image
	@echo "[make docker-build] Building mvgtnet:latest ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		docker build -t mvgtnet:latest .

docker-run: ## Run training inside Docker
	@echo "[make docker-run] Running train in Docker (domain=$(DOMAIN), epochs=$(EPOCHS)) ..."
	@cd "$(BUNDLE_ROOT)/code" && \
		docker run --rm --gpus all \
			-v "$$(pwd)/data:/workspace/mvgt_net/data:ro" \
			-v "$$(pwd)/checkpoints:/workspace/mvgt_net/checkpoints" \
			-v "$$(pwd)/results:/workspace/mvgt_net/results" \
			mvgtnet:latest \
			python scripts/train_real.py --domain $(DOMAIN) --epochs $(EPOCHS) --device cuda

# Convenience: run the full pipeline end-to-end
pipeline: ## Run the full pipeline (install + download + smoke + train-all + analyses)
	@echo "[make pipeline] Running run_pipeline.sh (full pipeline) ..."
	@cd "$(BUNDLE_ROOT)" && ./run/run_pipeline.sh
