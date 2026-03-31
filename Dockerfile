# --------------------------------------------------------------------------
# SAM 3 - RunPod Serverless Dockerfile
# --------------------------------------------------------------------------
# Base: NVIDIA CUDA 12.4 + cuDNN + Ubuntu 22.04
# Python 3.11, PyTorch 2.4.1, HuggingFace Transformers (SAM3 PR)
# --------------------------------------------------------------------------

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System packages -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        ffmpeg \
        git \
        curl \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip setuptools wheel

# ---- Install pget for fast weight downloads --------------------------------
RUN curl -o /usr/local/bin/pget -L \
    "https://github.com/replicate/pget/releases/download/v0.11.0/pget_linux_x86_64" \
    && chmod +x /usr/local/bin/pget

WORKDIR /src

# ---- Python dependencies ---------------------------------------------------
COPY requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r /src/requirements.txt \
    && pip install --no-cache-dir runpod

# ---- Pre-download model weights (baked into image for instant cold starts) -
RUN pget -xf https://weights.replicate.delivery/default/facebook/sam3/model.tar /src/checkpoints

# ---- Copy application code -------------------------------------------------
COPY handler.py /src/handler.py
COPY predict.py /src/predict.py

# ---- Entrypoint ------------------------------------------------------------
CMD ["python", "-u", "/src/handler.py"]
