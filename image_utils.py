"""Safe decoding, normalization, resizing and encoding for request images."""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any, Final

from schemas import APIError


MAX_IMAGE_BYTES: Final = 10 * 1024 * 1024
ALLOWED_FORMATS: Final = frozenset({"JPEG", "PNG", "WEBP"})
_MAX_BASE64_CHARS: Final = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 128


def _load_pillow() -> tuple[Any, Any, type[BaseException]]:
    """Import Pillow only when an image operation is actually requested."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    Image.MAX_IMAGE_PIXELS = 50_000_000
    return Image, ImageOps, UnidentifiedImageError


def _strip_data_url(value: str) -> str:
    if not value.lower().startswith("data:"):
        return value
    try:
        header, payload = value.split(",", 1)
    except ValueError as exc:
        raise APIError("INVALID_IMAGE", "Malformed image data URL.") from exc
    if ";base64" not in header.lower():
        raise APIError("INVALID_IMAGE", "Image data URL must use base64 encoding.")
    return payload


def decode_image(value: str, field_name: str) -> Any:
    """Decode and verify a base64 JPEG/PNG/WEBP image, returning detached RGB pixels."""
    payload = "".join(_strip_data_url(value).split())
    if len(payload) > _MAX_BASE64_CHARS:
        raise APIError("IMAGE_TOO_LARGE", f"{field_name} exceeds the 10 MB limit.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise APIError("INVALID_IMAGE", f"{field_name} is not valid base64.") from exc
    if not raw:
        raise APIError("INVALID_IMAGE", f"{field_name} is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise APIError("IMAGE_TOO_LARGE", f"{field_name} exceeds the 10 MB limit.")

    Image, ImageOps, UnidentifiedImageError = _load_pillow()
    try:
        # Treat Pillow's pixel-count warning as an error to prevent compressed image bombs.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                image_format = opened.format
                opened.verify()
            if image_format not in ALLOWED_FORMATS:
                raise APIError("UNSUPPORTED_IMAGE_FORMAT", f"{field_name} must be JPEG, PNG, or WEBP.")
            with Image.open(io.BytesIO(raw)) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB").copy()
    except APIError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise APIError("INVALID_IMAGE", f"{field_name} is not a valid image.") from exc


def resize_cover(image: Any, size: tuple[int, int]) -> Any:
    """Resize while preserving ratio, then center-crop to exactly size."""
    Image, ImageOps, _ = _load_pillow()
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def resize_contain(image: Any, size: tuple[int, int]) -> Any:
    """Resize while preserving ratio and pad with white to exactly size."""
    Image, ImageOps, _ = _load_pillow()
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def encode_jpeg(image: Any, quality: int = 90) -> str:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
