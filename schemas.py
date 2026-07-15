"""Request validation and public API error types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_CATEGORIES = frozenset({"upper_body", "lower_body", "dress"})
MIN_STEPS = 1
MAX_STEPS = 100
MIN_SEED = 0
MAX_SEED = 2**32 - 1


class APIError(Exception):
    """An expected error that is safe to return to the client."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TryOnRequest:
    person_image_base64: str
    garment_image_base64: str
    garment_category: str
    prompt: str
    seed: int
    steps: int

    @classmethod
    def from_job(cls, job: Any) -> "TryOnRequest":
        if not isinstance(job, Mapping) or not isinstance(job.get("input"), Mapping):
            raise APIError("INVALID_INPUT", "Job must contain an 'input' object.")
        data = job["input"]

        person = data.get("person_image_base64")
        garment = data.get("garment_image_base64")
        if not isinstance(person, str) or not person.strip():
            raise APIError("MISSING_PERSON_IMAGE", "person_image_base64 is required.")
        if not isinstance(garment, str) or not garment.strip():
            raise APIError("MISSING_GARMENT_IMAGE", "garment_image_base64 is required.")

        category = data.get("garment_category", "upper_body")
        if category not in ALLOWED_CATEGORIES:
            allowed = ", ".join(sorted(ALLOWED_CATEGORIES))
            raise APIError("INVALID_CATEGORY", f"garment_category must be one of: {allowed}.")

        seed = data.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not MIN_SEED <= seed <= MAX_SEED:
            raise APIError("INVALID_SEED", f"seed must be an integer from {MIN_SEED} to {MAX_SEED}.")

        steps = data.get("steps", 30)
        if isinstance(steps, bool) or not isinstance(steps, int) or not MIN_STEPS <= steps <= MAX_STEPS:
            raise APIError("INVALID_STEPS", f"steps must be an integer from {MIN_STEPS} to {MAX_STEPS}.")

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str) or len(prompt) > 1000:
            raise APIError("INVALID_PROMPT", "prompt must be a string no longer than 1000 characters.")

        return cls(person.strip(), garment.strip(), category, prompt, seed, steps)


def error_response(error: APIError) -> dict[str, Any]:
    return {"status": "failed", "error": {"code": error.code, "message": error.message}}
