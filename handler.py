"""RunPod Serverless entrypoint."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from image_utils import decode_image, encode_jpeg
from schemas import APIError, TryOnRequest, error_response

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _generate_tryon(*args: Any, **kwargs: Any) -> Any:
    from inference import generate_tryon
    from model_loader import initialize_model

    initialize_model()
    return generate_tryon(*args, **kwargs)


def _cleanup_cuda() -> None:
    """Collect the CUDA allocator only when explicitly requested or memory is tight."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        low_memory = total_bytes > 0 and free_bytes / total_bytes < 0.10
        if low_memory or os.getenv("EMPTY_CUDA_CACHE_AFTER_JOB", "false").lower() == "true":
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.warning("CUDA cleanup check failed", exc_info=False)


def handler(job: Any) -> dict[str, Any]:
    total_started = time.perf_counter()
    job_id = job.get("id", "unknown") if isinstance(job, dict) else "unknown"
    person = garment = result = None
    inference_attempted = False
    try:
        request = TryOnRequest.from_job(job)
        person = decode_image(request.person_image_base64, "person_image_base64")
        garment = decode_image(request.garment_image_base64, "garment_image_base64")
        LOGGER.info("Starting try-on job id=%s category=%s steps=%d", job_id, request.garment_category, request.steps)

        inference_started = time.perf_counter()
        inference_attempted = True
        result = _generate_tryon(person, garment, request.garment_category, request.seed, request.steps)
        inference_seconds = time.perf_counter() - inference_started
        quality = int(os.getenv("OUTPUT_JPEG_QUALITY", "90"))
        encoded = encode_jpeg(result, quality=quality)
        total_seconds = time.perf_counter() - total_started
        LOGGER.info("Completed try-on job id=%s inference_seconds=%.3f total_seconds=%.3f", job_id, inference_seconds, total_seconds)
        return {
            "status": "completed",
            "output": {
                "image_base64": encoded,
                "mime_type": "image/jpeg",
                "width": result.width,
                "height": result.height,
                "seed": request.seed,
            },
            "metrics": {
                "inference_seconds": round(inference_seconds, 3),
                "total_seconds": round(total_seconds, 3),
            },
        }
    except APIError as exc:
        LOGGER.warning("Rejected job id=%s code=%s", job_id, exc.code)
        return error_response(exc)
    except Exception:
        LOGGER.exception("Try-on job failed internally id=%s", job_id)
        return error_response(APIError("INFERENCE_ERROR", "The try-on request could not be processed."))
    finally:
        for image in (person, garment, result):
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
        if inference_attempted:
            _cleanup_cuda()


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
