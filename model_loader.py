"""Process-wide CatVTON model lifecycle."""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)
_MODEL: Optional["ModelBundle"] = None
_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    automasker: Any
    mask_processor: Any
    device: str


def initialize_model() -> ModelBundle:
    """Load CatVTON and mask models exactly once for this worker process."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL

        import torch

        if not torch.cuda.is_available():
            LOGGER.critical("CUDA is unavailable; CatVTON worker cannot start")
            raise RuntimeError("CUDA-capable GPU is required to start the CatVTON worker.")

        source_dir = Path(os.getenv("CATVTON_SOURCE_DIR", "/opt/CatVTON"))
        if not source_dir.is_dir():
            raise RuntimeError(f"Official CatVTON source not found at {source_dir}")
        source = str(source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)

        from diffusers.image_processor import VaeImageProcessor
        from huggingface_hub import snapshot_download
        from model.cloth_masker import AutoMasker
        from model.pipeline import CatVTONPipeline

        model_id = os.getenv("CATVTON_MODEL_ID", "zhengchong/CatVTON")
        base_model = os.getenv("CATVTON_BASE_MODEL", "booksforcharlie/stable-diffusion-inpainting")
        revision = os.getenv("CATVTON_MODEL_REVISION") or None
        LOGGER.info("Loading CatVTON on %s (model=%s)", torch.cuda.get_device_name(0), model_id)
        repo_path = snapshot_download(repo_id=model_id, revision=revision)
        pipeline = CatVTONPipeline(
            base_ckpt=base_model,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=torch.float16,
            device="cuda",
            skip_safety_check=os.getenv("CATVTON_SKIP_SAFETY_CHECK", "false").lower() == "true",
            use_tf32=True,
        )
        automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device="cuda",
        )
        mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        _MODEL = ModelBundle(pipeline, automasker, mask_processor, "cuda")
        LOGGER.info("CatVTON worker is ready")
        return _MODEL


def get_model() -> ModelBundle:
    if _MODEL is None:
        raise RuntimeError("Model is not initialized. Call initialize_model() during worker startup.")
    return _MODEL
