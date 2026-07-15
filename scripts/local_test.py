"""Call the real handler with two local image files (requires a CUDA environment)."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handler import handler  # noqa: E402
from model_loader import initialize_model  # noqa: E402


def file_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local CatVTON handler request")
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--garment", type=Path, required=True)
    parser.add_argument("--category", choices=("upper_body", "lower_body", "dress"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("output/result.jpg"))
    args = parser.parse_args()

    initialize_model()
    response = handler({
        "id": "local-test",
        "input": {
            "person_image_base64": file_b64(args.person),
            "garment_image_base64": file_b64(args.garment),
            "garment_category": args.category,
            "prompt": "",
            "seed": args.seed,
            "steps": args.steps,
        },
    })
    if response.get("status") != "completed":
        print(json.dumps(response, indent=2))
        raise SystemExit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base64.b64decode(response["output"]["image_base64"]))
    printable = {**response, "output": {**response["output"], "image_base64": "<omitted>"}}
    print(json.dumps(printable, indent=2))
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
