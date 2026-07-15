# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG CATVTON_REPO=https://github.com/Zheng-Chong/CatVTON.git
ARG CATVTON_REF=main
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
    pip install --no-cache-dir -r requirements.txt

# Clone official code, keep the selected ref configurable, and install its vendored Detectron2.
RUN git clone --filter=blob:none "$CATVTON_REPO" "$CATVTON_SOURCE_DIR" \
    && git -C "$CATVTON_SOURCE_DIR" checkout "$CATVTON_REF" \
    && pip install --no-cache-dir -e "$CATVTON_SOURCE_DIR/detectron2"

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
