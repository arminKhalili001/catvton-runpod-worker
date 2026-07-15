"""Small, stable inference interface around the official CatVTON pipeline."""

from __future__ import annotations

import os
import threading
from typing import Any

from image_utils import resize_contain, resize_cover
from model_loader import get_model

_INFERENCE_LOCK = threading.Lock()
_CATEGORY_MAP = {"upper_body": "upper", "lower_body": "lower", "dress": "overall"}


def generate_tryon(
    person_image: Any,
    garment_image: Any,
    garment_category: str,
    seed: int,
    steps: int,
) -> Any:
    """Generate one 768x1024 try-on image using the startup-loaded model."""
    import torch

    width = int(os.getenv("CATVTON_WIDTH", "768"))
    height = int(os.getenv("CATVTON_HEIGHT", "1024"))
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise RuntimeError("CATVTON_WIDTH and CATVTON_HEIGHT must be positive multiples of 8.")

    bundle = get_model()
    person = resize_cover(person_image, (width, height))
    garment = resize_contain(garment_image, (width, height))
    generator = torch.Generator(device=bundle.device).manual_seed(seed)

    # The official pipeline and preprocessing models share GPU state and are not thread-safe.
    with _INFERENCE_LOCK, torch.inference_mode():
        mask = bundle.automasker(person, _CATEGORY_MAP[garment_category])["mask"]
        mask = bundle.mask_processor.blur(mask, blur_factor=9)
        result = bundle.pipeline(
            image=person,
            condition_image=garment,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=float(os.getenv("CATVTON_GUIDANCE_SCALE", "2.5")),
            height=height,
            width=width,
            generator=generator,
        )[0]
    return result.convert("RGB")
