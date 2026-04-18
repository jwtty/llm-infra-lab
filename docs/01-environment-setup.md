# 01 — Environment Setup: NVIDIA Driver, Anaconda, PyTorch

This doc records the first milestone of my LLM infra engineer learning path:
get a Linux box with an NVIDIA GPU into a state where I can run a PyTorch
script on the GPU and observe GPU utilization.

Tested on **Ubuntu 24.04 LTS** with an NVIDIA GPU. Adjust commands if you are
on a different distro.

---

## 0. Prerequisites

- An Ubuntu/Debian machine with an NVIDIA GPU (consumer or datacenter).
- `sudo` privileges.
- Working internet access.
- Secure Boot **disabled** in BIOS (or be ready to sign the kernel module —
  the open driver can be a pain otherwise).

Quick sanity checks:

```bash
# Confirm the GPU is on the PCI bus
lspci | grep -i nvidia

# Kernel + distro info
uname -a
lsb_release -a

# Make sure no nouveau driver is loaded (it conflicts with nvidia)
lsmod | grep nouveau
```

If `nouveau` is loaded, blacklist it before installing the NVIDIA driver:

```bash
sudo tee /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
sudo update-initramfs -u
sudo reboot
```

---

## 1. Install the NVIDIA GPU Driver

### 1.1 Install build prerequisites

```bash
sudo apt update
sudo apt install -y build-essential dkms linux-headers-$(uname -r) pkg-config
```

### 1.2 Inspect what Ubuntu recommends (optional)

```bash
ubuntu-drivers devices
```

Example output:

```
driver   : nvidia-driver-550 - distro non-free recommended
driver   : nvidia-driver-535 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

### 1.3 Install

This is what I actually used. It picks the recommended NVIDIA driver for
your GPU and installs it via apt + DKMS, so kernel updates keep working:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

> Prefer this over the `.run` installer. It plays nicely with kernel
> updates via DKMS and is trivial to uninstall.

If you need a specific version instead (e.g. to match a CUDA toolkit), you
can install one explicitly:

```bash
sudo apt install -y nvidia-driver-580
sudo reboot
```

### 1.5 Verify the driver with `nvidia-smi`

```bash
nvidia-smi
```

Expected (numbers will differ):

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          Off |   00000001:00:00.0 Off |                    0 |
| N/A   34C    P0             43W /  300W |       0MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

What to confirm:

- `Driver Version` shows the version you installed.
- `CUDA Version` is the **driver-bundled** CUDA runtime (not your toolkit).
- The GPU appears with the correct name and memory.

Live monitoring (refreshes every 1s):

```bash
nvidia-smi -l 1
# or, more compact:
watch -n 1 nvidia-smi
```

Useful one-off queries:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
```

---

## 2. Install Anaconda (or Miniconda)

I use Miniconda — Anaconda is fine but heavier. Pick one.

### 2.1 Download and run the installer

```bash
pushd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
popd
```

For full Anaconda instead:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p $HOME/anaconda3
```

### 2.2 Initialize the shell

```bash
$HOME/miniconda3/bin/conda init bash
exec $SHELL -l
```

Verify:

```bash
conda --version
which python
```

### 2.3 Create a dedicated env for the LLM infra lab

Pin Python to a version PyTorch officially supports (3.11 is a safe pick today):

```bash
conda create -n llm-infra python=3.11 -y
conda activate llm-infra
```

Optional QoL packages:

```bash
conda install -y ipython jupyter numpy
```

---

## 3. Install PyTorch with CUDA support

> Always grab the exact command from <https://pytorch.org/get-started/locally/>
> for your CUDA version. The snippet below is current as of CUDA 12.4.

```bash
# Inside the activated llm-infra env
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify the install picked up CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version (built):", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

If `torch.cuda.is_available()` is `False`:

- Driver too old for the CUDA wheel you installed → upgrade driver, or
  install a wheel built against an older CUDA (e.g. `cu121`).
- You're inside a container or WSL without GPU passthrough.
- You installed the CPU-only wheel by accident
  (`pip show torch` will show `torch` but importing won't see CUDA).

---

## 4. Run a small GPU script and observe utilization

The script lives at [`scripts/gpu_smoke_test.py`](../scripts/gpu_smoke_test.py).

It does a sustained matrix multiplication loop on the GPU so you can see
`nvidia-smi` light up.

### 4.1 Run it

In one terminal:

```bash
conda activate llm-infra
python scripts/gpu_smoke_test.py
```

In a second terminal, watch the GPU:

```bash
watch -n 1 nvidia-smi
```

You should see:

- `GPU-Util` jump from `0%` to something high (often 90–100%).
- `Memory-Usage` grow by a few hundred MiB to a few GiB depending on the
  matrix size you picked.
- A `python` process listed under `Processes` at the bottom of `nvidia-smi`.

### 4.2 What to look for

| Symptom | Likely cause |
|---|---|
| `GPU-Util` stays at 0% but memory grows | tensors created on GPU but no kernels running — check the loop ran |
| OOM (`CUDA out of memory`) | matrix size too big for VRAM — reduce `N` in the script |
| Util oscillates 0/100 rapidly | host↔device transfer bottleneck — keep tensors on GPU |
| Process not listed in `nvidia-smi` | running on CPU; re-check `torch.cuda.is_available()` |

---

## 5. Cleanup / next steps

To leave the env:

```bash
conda deactivate
```

To remove it entirely:

```bash
conda env remove -n llm-infra
```
