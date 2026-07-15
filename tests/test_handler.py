from __future__ import annotations

import base64
import builtins
import importlib
import io
from pathlib import Path
import sys

import pytest
from PIL import Image

import handler
from image_utils import MAX_IMAGE_BYTES


def image_b64(fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def valid_job(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "person_image_base64": image_b64(),
        "garment_image_base64": image_b64(),
        "garment_category": "upper_body",
        "seed": 42,
        "steps": 30,
    }
    data.update(overrides)
    return {"input": data}


@pytest.fixture(autouse=True)
def no_cuda_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handler, "_cleanup_cuda", lambda: None)


def test_handler_import_does_not_require_heavy_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__
    blocked = {"PIL", "torch", "runpod", "diffusers", "huggingface_hub", "model"}

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(f"blocked test dependency: {name}")
        return real_import(name, *args, **kwargs)

    saved = {name: sys.modules.pop(name, None) for name in ("handler", "image_utils")}
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        imported = importlib.import_module("handler")
        assert callable(imported.handler)
    finally:
        sys.modules.pop("handler", None)
        sys.modules.pop("image_utils", None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


def test_runpod_entrypoint_is_direct_and_guarded() -> None:
    source = Path(handler.__file__).read_text(encoding="utf-8")
    assert source.rstrip().endswith(
        'if __name__ == "__main__":\n'
        "    import runpod\n"
        '    runpod.serverless.start({"handler": handler})'
    )


def test_missing_person_image() -> None:
    job = valid_job()
    del job["input"]["person_image_base64"]  # type: ignore[index]
    assert handler.handler(job)["error"]["code"] == "MISSING_PERSON_IMAGE"


def test_validation_error_never_reaches_cuda_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup = pytest.fail
    monkeypatch.setattr(handler, "_cleanup_cuda", lambda: cleanup("CUDA cleanup ran before inference"))
    assert handler.handler({"input": {}})["error"]["code"] == "MISSING_PERSON_IMAGE"


def test_invalid_base64_does_not_load_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    import image_utils

    monkeypatch.setattr(
        image_utils,
        "_load_pillow",
        lambda: pytest.fail("Pillow loaded before base64 validation"),
    )
    response = handler.handler(valid_job(person_image_base64="not!!base64"))
    assert response["error"]["code"] == "INVALID_IMAGE"


def test_missing_garment_image() -> None:
    job = valid_job()
    del job["input"]["garment_image_base64"]  # type: ignore[index]
    assert handler.handler(job)["error"]["code"] == "MISSING_GARMENT_IMAGE"


def test_invalid_base64() -> None:
    response = handler.handler(valid_job(person_image_base64="not!!base64"))
    assert response["error"]["code"] == "INVALID_IMAGE"


def test_unsupported_image_format() -> None:
    response = handler.handler(valid_job(person_image_base64=image_b64("GIF")))
    assert response["error"]["code"] == "UNSUPPORTED_IMAGE_FORMAT"


def test_image_too_large() -> None:
    oversized = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode("ascii")
    response = handler.handler(valid_job(person_image_base64=oversized))
    assert response["error"]["code"] == "IMAGE_TOO_LARGE"


def test_invalid_category() -> None:
    response = handler.handler(valid_job(garment_category="shoes"))
    assert response["error"]["code"] == "INVALID_CATEGORY"


def test_success_with_mocked_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handler, "_generate_tryon", lambda *args, **kwargs: Image.new("RGB", (768, 1024), "red"))
    response = handler.handler(valid_job())
    assert response["status"] == "completed"
    assert response["output"]["mime_type"] == "image/jpeg"
    assert response["output"]["width"] == 768
    assert response["output"]["height"] == 1024
    assert base64.b64decode(response["output"]["image_base64"]).startswith(b"\xff\xd8")


def test_internal_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal path /models/private")

    monkeypatch.setattr(handler, "_generate_tryon", fail)
    response = handler.handler(valid_job())
    assert response == {
        "status": "failed",
        "error": {"code": "INFERENCE_ERROR", "message": "The try-on request could not be processed."},
    }
    assert "private" not in str(response)


@pytest.mark.parametrize("field,value", [("seed", -1), ("seed", True), ("steps", 0), ("steps", 101)])
def test_numeric_validation(field: str, value: object) -> None:
    response = handler.handler(valid_job(**{field: value}))
    assert response["status"] == "failed"
