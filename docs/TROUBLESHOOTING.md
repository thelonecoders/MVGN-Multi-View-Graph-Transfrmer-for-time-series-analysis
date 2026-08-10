# TROUBLESHOOTING — ST-LLM-Plus VPS Code Bundle

Common issues encountered when running the bundle on a VPS, with verified
solutions. Organized by phase of execution.

---

## Step 0: Environment setup

### `nvidia-smi: command not found`

**Cause:** NVIDIA driver not installed on the VPS.

**Fix:**
```bash
# Ubuntu 24.04
sudo apt update
sudo apt install -y nvidia-driver-550-server
sudo reboot

# After reboot, verify:
nvidia-smi
```

If the VPS is a cloud instance (AWS, GCP, Azure), the NVIDIA driver must be
pre-installed by the provider. Check the instance type — only GPU instance
types (e.g., `g4dn.xlarge`, `n1-standard-4 + T4`) come with driver support.

### `torch.cuda.is_available()` returns `False`

**Cause:** PyTorch was installed with the CPU wheel, or the CUDA version
mismatches the driver.

**Fix:**
```bash
# Check driver-supported CUDA version
nvidia-smi | grep "CUDA Version"

# Reinstall PyTorch with the correct CUDA wheel
pip uninstall -y torch torchvision torchaudio
# For CUDA 12.1 (most common):
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Verify:
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True 12.1
```

### `pip install` fails with `error: Microsoft Visual C++ 14.0 is required`

**Cause:** This is a Windows error. The bundle is designed for Ubuntu 24.04.

**Fix:** Use Ubuntu 24.04 LTS (or 22.04 with manual CUDA 12.1 install). The
bundle does not support Windows natively. On Windows, use WSL2 + Ubuntu 24.04.

### `ranger21` install fails with `error: command 'gcc' failed`

**Cause:** Build tools not installed.

**Fix:**
```bash
sudo apt install -y build-essential python3-dev
pip install ranger21==0.1.0
```

### Out-of-memory during `pip install torch`

**Cause:** VPS has < 4 GB RAM; PyTorch wheel download + install needs ~3 GB.

**Fix:**
```bash
# Add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Verify:
free -h

# Then retry pip install
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 1: Dataset download

### `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co')`

**Cause:** No internet access, or firewall blocking Hugging Face.

**Fix:**
```bash
# Test connectivity:
curl -sI https://huggingface.co | head -1
# Expected: HTTP/1.1 200 OK

# If behind a proxy, set:
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# If HF is blocked in your region, use the chunked downloader:
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle
python3 dataset_downloader/download_chunked.py
```

### `ValueError: SHA-256 mismatch for Climate_AQI/train.jsonl`

**Cause:** Downloaded file is corrupted (network issue or partial download).

**Fix:**
```bash
# Delete the corrupted file and re-download
rm code/data/TimeMMD/Climate_AQI/train.jsonl
rm code/data/TimeMMD/DATASET_MANIFEST.json  # forces re-download of manifest
./run/run_step1_download_dataset.sh
```

### `FileNotFoundError: code/data/TimeMMD/DATASET_MANIFEST.json`

**Cause:** Dataset not yet downloaded.

**Fix:**
```bash
./run/run_step1_download_dataset.sh
```

---

## Step 2: Smoke tests

### `ModuleNotFoundError: No module named 'mvgt_net'`

**Cause:** Python path not set; venv not activated.

**Fix:**
```bash
cd /path/to/ST-LLM-Plus_VPS_Code_Bundle/code
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m pytest tests/ -v
```

### `RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB`

**Cause:** GPU has < 4 GB VRAM free (other processes using it).

**Fix:**
```bash
# Check GPU memory:
nvidia-smi

# Kill any process using the GPU:
sudo fuser -v /dev/nvidia*
# Or:
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill -9 {}

# If the GPU has < 8 GB VRAM total (e.g., RTX 3060 6GB), use smoke config:
cd code
source .venv/bin/activate
python scripts/train.py --config configs/smoke.yaml --smoke-test --device cuda
```

### `ImportError: cannot import name 'Ranger21' from 'ranger21'`

**Cause:** ranger21 version mismatch.

**Fix:**
```bash
pip install --force-reinstall ranger21==0.1.0
```

If ranger21 still fails to import, the training script will fall back to
AdamW automatically — this is documented behavior, not a bug.

---

## Step 3: Training

### `RuntimeError: Input and target must have the same number of elements`

**Cause:** Lookback/horizon mismatch between data and config.

**Fix:** Use the domain's verified lookback/horizon from `DOMAIN_REGISTRY`:
```python
from mvgt_net.data import DOMAIN_REGISTRY
print(DOMAIN_REGISTRY["Climate_AQI"])
# Expected: {'lookback': 96, 'horizon': 96, 'frequency': 'day', ...}
```

Do NOT override these in `configs/default.yaml` unless you know what you're
doing.

### Training is extremely slow (> 30 min/epoch on Climate_AQI)

**Cause:** Most likely running on CPU instead of GPU.

**Fix:**
```bash
# Verify GPU is being used:
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device 0:', torch.cuda.get_device_name(0))
"

# Train with explicit --device cuda:
python scripts/train_real.py --domain Climate_AQI --device cuda --epochs 100
```

If GPU is correctly detected but training is still slow, check:
1. `num_workers=0` in DataLoader (set to 4 for speedup)
2. `mixed_precision: fp16` in config (should be on by default)
3. `batch_size` is appropriate for VRAM (32 for 12 GB, 16 for 8 GB)

### `KeyError: 'batch_y_timestamps_datetime64_ns'` in validation

**Cause:** Known upstream quirk — all 9 validation.jsonl files omit this key.
This is documented in `12_dataset/readme/README.md` and
`15_engineering_details/DATASET_MIRROR_COMPARISON.md`.

**Fix:** The MVGT-Net data loader handles this gracefully. If you see this
error, you are using an outdated data loader. Update to the latest version:
```bash
cd code
git pull  # or re-extract the bundle
```

### `KeyError: 'batch_text'` in Economy_Unemp

**Cause:** Some Economy_Unemp records contain `"[nan nan]"` instead of a
proper text array. This is an upstream marker for "no text facts available".

**Fix:** The data loader converts `"[nan nan]"` to an empty string. If you
see this error, update to the latest version (see above).

---

## Step 4: Phase F analyses

### `ModuleNotFoundError: No module named 'matplotlib'`

**Cause:** matplotlib not installed.

**Fix:**
```bash
pip install matplotlib==3.9.2
```

### Phase F outputs are blank or have NaN values

**Cause:** Phase F scripts run on a synthetic mini-batch by default, not on
trained checkpoints.

**Fix:** This is documented behavior. To produce thesis-grade numbers, point
each script at the trained checkpoints:
```bash
python scripts/robustness_analysis.py \
    --checkpoint code/checkpoints/Climate_AQI/best.pt \
    --domain Climate_AQI \
    --device cuda
```

---

## Step 5: Docker path

### `docker: Error response from daemon: could not select device driver ""`

**Cause:** NVIDIA Container Toolkit not installed.

**Fix:**
```bash
# Ubuntu 24.04
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## General

### `Permission denied` when running shell scripts

**Cause:** Scripts not executable.

**Fix:**
```bash
chmod +x run/*.sh setup/*.sh verify_bundle_integrity.sh clean.sh
```

### `bash: bad interpreter: No such file or directory`

**Cause:** Script has Windows line endings (CRLF instead of LF).

**Fix:**
```bash
sudo apt install -y dos2unix
dos2unix run/*.sh setup/*.sh
```

### Logs are filling up the disk

**Cause:** `logs/` directory grows unbounded.

**Fix:**
```bash
# Delete logs older than 7 days:
find logs/ -name '*.log' -mtime +7 -delete

# Or use the clean script to reset everything:
./clean.sh --force
```

---

## Still stuck?

If none of the above solves your issue:

1. Check the full pipeline log: `logs/pipeline_<timestamp>.log`
2. Run the bundle integrity verifier: `./verify_bundle_integrity.sh`
3. Re-read the relevant section of `README.md`
4. Consult `docs/FAQ.md` for less common questions
5. Read the source code — every module has a comprehensive docstring
