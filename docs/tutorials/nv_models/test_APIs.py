#!/usr/bin/env python3
"""Test NVIDIA Parse, Embedding, and VLM APIs for PaperQA.

This script is a CLI version of test_APIs.ipynb with support for:
- local self-hosted endpoints (e.g. localhost NIM)
- remote hosted inference endpoints (any OpenAI-compatible URL)

Examples:
  # Local Parse + local Embedding + local VLM
  python test_APIs.py --mode all \
    --parse-base-url http://localhost:8002/v1 \
    --embedding-base-url http://localhost:8003/v1 \
    --vlm-base-url http://localhost:8004/v1 \
    --parse-api-key not-needed --embedding-api-key dummy --vlm-api-key dummy

  # Local Parse + hosted Embedding/VLM
  NVIDIA_INFERENCE_KEY=... python test_APIs.py --mode all \
    --parse-base-url http://localhost:8002/v1 \
    --embedding-base-url https://inference-api.nvidia.com/v1 \
    --vlm-base-url https://inference-api.nvidia.com/v1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

DEFAULT_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/c/c3/"
    "LibreOffice_Writer_6.3.png"
)
DEFAULT_IMAGE_PATH = "sample_image.png"
DEFAULT_QUERY = "Describe the scene"
DEFAULT_EMBED_TEXT = "What is the civil caseload in South Dakota courts?"
DEFAULT_PARSE_MODEL = "nvidia/nemotron-parse"
DEFAULT_EMBED_MODEL = "nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_VLM_MODEL = "nvidia/nvidia/nemotron-nano-12b-v2-vl"

IMAGE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def _norm_base_url(url: str) -> str:
    return url.rstrip("/")


def _encode_image_base64(image_path: Path) -> str:
    ext = image_path.suffix.lstrip(".").lower()
    if ext not in IMAGE_MIME:
        raise ValueError(f"Unsupported image format: {ext}. Use one of {list(IMAGE_MIME)}")
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{IMAGE_MIME[ext]};base64,{b64}"


def _download_sample_image(image_path: Path, image_url: str, force: bool) -> None:
    if image_path.exists() and not force:
        print(f"Using existing image: {image_path}")
        return
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(image_url, headers=headers, timeout=60)
    response.raise_for_status()
    image_path.write_bytes(response.content)
    print(f"Downloaded image to: {image_path}")


def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=_norm_base_url(base_url), api_key=api_key)


def test_parse(
    image_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> bool:
    print("\n=== Testing Parse API ===")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    try:
        client = _make_client(base_url, api_key)
        image_url = _encode_image_base64(image_path)

        completion = client.chat.completions.create(
            model=model,
            tools=[{"type": "function", "function": {"name": "markdown_bbox"}}],
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        raw = completion.model_dump() if hasattr(completion, "model_dump") else completion
        print("Parse request: Success")
        print(json.dumps(raw, indent=2, default=str)[:3000])

        if completion.choices and completion.choices[0].message.tool_calls:
            args = completion.choices[0].message.tool_calls[0].function.arguments
            parsed: Any = json.loads(args)
            text = parsed.get("text", parsed) if isinstance(parsed, dict) else parsed
            print("--- Extracted text preview ---")
            print(str(text)[:1000])
        return True
    except Exception as exc:
        print(f"Parse request: Failed - {exc}")
        return False


def test_embedding(
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    input_type: str,
) -> bool:
    print("\n=== Testing Embedding API ===")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    try:
        client = _make_client(base_url, api_key)
        response = client.embeddings.create(
            input=[text],
            model=model,
            encoding_format="float",
            extra_body={"input_type": input_type},
        )
        vector = response.data[0].embedding
        print("Embedding request: Success")
        print(f"Vector dim: {len(vector)}")
        print(f"Vector preview: {vector[:8]}")
        return True
    except Exception as exc:
        print(f"Embedding request: Failed - {exc}")
        return False


def test_vlm(
    image_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    max_tokens: int,
    temperature: float,
) -> bool:
    print("\n=== Testing VLM API ===")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    try:
        client = _make_client(base_url, api_key)
        image_url = _encode_image_base64(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "/think"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print("VLM request: Success")
        content = response.choices[0].message.content if response.choices else ""
        print(f"Response preview: {str(content)[:1000]}")
        return True
    except Exception as exc:
        print(f"VLM request: Failed - {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "parse", "embedding", "vlm"),
        default="all",
        help="Which API test(s) to run.",
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH, help="Image path for parse/vlm tests.")
    parser.add_argument(
        "--image-url",
        default=DEFAULT_IMAGE_URL,
        help="Download URL for sample image if --download-sample-image is used.",
    )
    parser.add_argument(
        "--download-sample-image",
        action="store_true",
        help="Download sample image if missing (or always with --force-download).",
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download image even if it exists.")

    parser.add_argument("--parse-base-url", default="http://localhost:8002/v1")
    parser.add_argument("--embedding-base-url", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--vlm-base-url", default="https://inference-api.nvidia.com/v1")

    parser.add_argument("--parse-model", default=DEFAULT_PARSE_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)

    parser.add_argument(
        "--parse-api-key",
        default=os.getenv("PQA_PARSE_API_KEY", "not-needed"),
        help="API key for parse endpoint (default from PQA_PARSE_API_KEY or 'not-needed').",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.getenv("PQA_EMBEDDING_API_KEY", os.getenv("NVIDIA_INFERENCE_KEY", "")),
        help="API key for embedding endpoint (default PQA_EMBEDDING_API_KEY or NVIDIA_INFERENCE_KEY).",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=os.getenv("PQA_VLM_API_KEY", os.getenv("NVIDIA_INFERENCE_KEY", "")),
        help="API key for VLM endpoint (default PQA_VLM_API_KEY or NVIDIA_INFERENCE_KEY).",
    )

    parser.add_argument("--embed-text", default=DEFAULT_EMBED_TEXT)
    parser.add_argument("--embed-input-type", default="query", choices=("query", "passage"))
    parser.add_argument("--vlm-query", default=DEFAULT_QUERY)
    parser.add_argument("--parse-max-tokens", type=int, default=256)
    parser.add_argument("--vlm-max-tokens", type=int, default=2048)
    parser.add_argument("--vlm-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path)

    if args.download_sample_image:
        _download_sample_image(image_path, args.image_url, args.force_download)

    if args.mode in ("parse", "vlm") and not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Pass --download-sample-image or provide --image-path to an existing image.")
        return 2

    results: dict[str, bool] = {}

    if args.mode in ("all", "parse"):
        results["parse"] = test_parse(
            image_path=image_path,
            base_url=args.parse_base_url,
            api_key=args.parse_api_key,
            model=args.parse_model,
            max_tokens=args.parse_max_tokens,
        )

    if args.mode in ("all", "embedding"):
        results["embedding"] = test_embedding(
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key,
            model=args.embedding_model,
            text=args.embed_text,
            input_type=args.embed_input_type,
        )

    if args.mode in ("all", "vlm"):
        results["vlm"] = test_vlm(
            image_path=image_path,
            base_url=args.vlm_base_url,
            api_key=args.vlm_api_key,
            model=args.vlm_model,
            query=args.vlm_query,
            max_tokens=args.vlm_max_tokens,
            temperature=args.vlm_temperature,
        )

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
