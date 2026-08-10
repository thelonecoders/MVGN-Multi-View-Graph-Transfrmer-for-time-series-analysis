# MVGT-Net: Multi-View Graph Transformer for Time-Series Forecasting
# Dockerfile for reproducible training environment
# Pins all library versions per the reproducibility protocol (Chapter 18)
# Base: NVIDIA PyTorch 23.10 image with CUDA 12.2 and cuDNN 8.9

FROM nvcr.io/nvidia/pytorch:23.10-py3

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl build-essential fonts-dejavu fonts-noto \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python packages (pinned versions for reproducibility)
# Tier 1: ST-LLM+ inherited (torch 2.1.0, ranger21 0.1.5)
# Tier 2: MVGT-Net additions (transformers 4.36.2, wandb 0.16.1)
# Tier 3: Eval + uncertainty (scikit-learn 1.3.2, statsmodels 0.14.0)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Working directory
WORKDIR /workspace/mvgt_net

# Copy source code
COPY . /workspace/mvgt_net/

# Set Python path
ENV PYTHONPATH=/workspace/mvgt_net:${PYTHONPATH}

# Determinism flags (per Chapter 18 protocol)
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=42

# Default command: run smoke tests
CMD ["python", "-m", "pytest", "tests/test_smoke.py", "-v"]
