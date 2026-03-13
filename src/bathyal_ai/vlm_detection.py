"""Gemini 3 Flash bounding box detection fallback for when Megalodon finds nothing."""

from __future__ import annotations

import io
import logging
import os
from typing import Sequence

from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DETECTION_PROMPT = (
    "Detect all marine organisms visible in this underwater image. "
    "Return a bounding box for every distinct organism you can identify. "
    "Each bounding box should use the format [ymin, xmin, ymax, xmax] "
    "with coordinates normalized to the range 0-1000."
)


class BoundingBox(BaseModel):
    box_2d: list[int] = Field(min_length=4, max_length=4)
    label: str


class DetectionResult(BaseModel):
    detections: list[BoundingBox]


def is_available() -> bool:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def convert_gemini_box(
    box_normalized: Sequence[int], width: int, height: int
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = box_normalized
    left = int(x0 * width / 1000)
    top = int(y0 * height / 1000)
    right = int(x1 * width / 1000)
    bottom = int(y1 * height / 1000)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _image_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class GeminiDetectionClient:
    def __init__(self, model: str = "gemini-3-flash-preview") -> None:
        from google import genai

        self.client = genai.Client()
        self.model = model

    def detect(self, image: Image.Image) -> list[dict]:
        from google.genai import types

        image_bytes = _image_to_jpeg_bytes(image)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    DETECTION_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=DetectionResult,
                ),
            )
        except Exception:
            logger.warning("Gemini detection API call failed", exc_info=True)
            return []

        try:
            result = DetectionResult.model_validate_json(response.text)
        except Exception:
            logger.warning("Failed to parse Gemini detection response: %s", response.text[:500])
            return []

        detections: list[dict] = []
        for box in result.detections:
            coords = box.box_2d
            if len(coords) != 4:
                continue
            if any(c < 0 or c > 1000 for c in coords):
                continue
            detections.append({"box_normalized": coords, "label": box.label})

        return detections
