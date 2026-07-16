# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG CATVTON_REPO=https://github.com/Zheng-Chong/CatVTON.git
ARG CATVTON_REF=7818397f25613beedb3d861a34769f607cfcf3b1
ARG BAKE_MODEL_WEIGHTS=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CATVTON_SOURCE_DIR=/opt/CatVTON \
    HF_HOME=/models/huggingface \
    OUTPUT_JPEG_QUALITY=90

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt

# Keep the base image's torch 2.1.2 and install only its matching CUDA 12.1
# torchvision/torchaudio wheels after the worker dependencies.
RUN python -m pip install --no-cache-dir \
      torchvision==0.16.2 \
      torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu121

# Keep these as separate layers so RunPod identifies the exact failing stage.
RUN git clone "$CATVTON_REPO" "$CATVTON_SOURCE_DIR"

RUN git -C "$CATVTON_SOURCE_DIR" checkout --detach "$CATVTON_REF" \
    && echo "CatVTON root contents:" \
    && ls -la "$CATVTON_SOURCE_DIR" \
    && echo "Detectron2 contents:" \
    && ls -la "$CATVTON_SOURCE_DIR/detectron2"

RUN grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]|[=<>!~]|$)' \
      "$CATVTON_SOURCE_DIR/requirements.txt" > /tmp/catvton-requirements.txt

RUN python -m pip install --no-cache-dir -r /tmp/catvton-requirements.txt

RUN python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.__version__.split('+')[0] == '2.1.2'; assert torchvision.__version__.split('+')[0] == '0.16.2'"

RUN if [ -f "$CATVTON_SOURCE_DIR/detectron2/setup.py" ] || \
       [ -f "$CATVTON_SOURCE_DIR/detectron2/pyproject.toml" ]; then \
      python -m pip install --no-cache-dir -e "$CATVTON_SOURCE_DIR/detectron2"; \
    else \
      echo "ERROR: detectron2 exists but is not an installable Python package" >&2; \
      find "$CATVTON_SOURCE_DIR/detectron2" -maxdepth 2 -type f | head -100; \
      exit 1; \
    fi

COPY . .

# Optional build-time weight baking. Default startup download keeps the image smaller.
RUN if [ "$BAKE_MODEL_WEIGHTS" = "true" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('zhengchong/CatVTON'); snapshot_download('booksforcharlie/stable-diffusion-inpainting'); snapshot_download('stabilityai/sd-vae-ft-mse')"; \
    fi

RUN python -m compileall -q /app \
    && useradd --create-home --uid 10001 worker \
    && mkdir -p /models/huggingface \
    && chown -R worker:worker /app /models

USER worker
CMD ["python", "-u", "handler.py"]
