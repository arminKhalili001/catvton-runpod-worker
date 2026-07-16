# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG CATVTON_REPO=https://github.com/Zheng-Chong/CatVTON.git
ARG CATVTON_REF=999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e
ARG DETECTRON2_REF=bcfd464d0c810f0442d91a349c0f6df945467143
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
    && ls -la "$CATVTON_SOURCE_DIR"

RUN python -c "import torch, torchvision, diffusers, transformers, huggingface_hub; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('huggingface_hub', huggingface_hub.__version__); assert torch.__version__.split('+')[0] == '2.1.2'; assert torchvision.__version__.split('+')[0] == '0.16.2'; assert diffusers.__version__ == '0.29.2'; assert transformers.__version__ == '4.27.3'; assert huggingface_hub.__version__ == '0.23.4'"

# model_loader.py initializes AutoMasker, whose DensePose implementation imports
# Detectron2. Install Detectron2 and DensePose from one pinned official revision.
RUN python -m pip install --no-cache-dir \
      "git+https://github.com/facebookresearch/detectron2.git@$DETECTRON2_REF" \
    && python -m pip install --no-cache-dir \
      "git+https://github.com/facebookresearch/detectron2.git@$DETECTRON2_REF#subdirectory=projects/DensePose"

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
