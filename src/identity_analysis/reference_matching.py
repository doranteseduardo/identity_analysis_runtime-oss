"""Experimental comparison of catalog reference patches."""

from __future__ import annotations

import base64
import hashlib
import io
from itertools import product

import numpy as np
from PIL import Image


def decode_reference_image(patch: dict) -> Image.Image:
    image = patch.get("image") or {}
    encoded = image.get("data")
    if not encoded:
        raise ValueError("reference patch does not contain image data")
    expected_hash = image.get("sha256")
    actual_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("reference patch payload hash mismatch")
    try:
        payload = base64.b64decode(encoded, validate=True)
        reference = Image.open(io.BytesIO(payload))
        reference.load()
    except (ValueError, OSError) as error:
        raise ValueError("reference patch is not a valid encoded image") from error
    return reference.convert("L")


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left.astype(np.float32) - float(left.mean())
    right_centered = right.astype(np.float32) - float(right.mean())
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator <= 1e-8:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.sum(left_centered * right_centered) / denominator, -1, 1))


def gradient_magnitude(values: np.ndarray) -> np.ndarray:
    vertical, horizontal = np.gradient(values.astype(np.float32))
    return np.hypot(horizontal, vertical)


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    intensity = normalized_correlation(reference, candidate)
    gradient = normalized_correlation(
        gradient_magnitude(reference), gradient_magnitude(candidate)
    )
    mae_similarity = 1.0 - float(
        np.mean(np.abs(reference.astype(np.float32) - candidate.astype(np.float32)))
        / 255.0
    )
    score = (
        0.35 * ((intensity + 1.0) / 2.0)
        + 0.45 * ((gradient + 1.0) / 2.0)
        + 0.20 * mae_similarity
    )
    return {
        "experimentalScore": float(np.clip(score, 0, 1)),
        "intensityCorrelation": intensity,
        "gradientCorrelation": gradient,
        "maeSimilarity": float(np.clip(mae_similarity, 0, 1)),
    }


def candidate_box(
    bounds: list[float],
    image_size: tuple[int, int],
    scale: float,
    offset_x: float,
    offset_y: float,
) -> tuple[int, int, int, int] | None:
    image_width, image_height = image_size
    left, top, right, bottom = bounds
    center_x = ((left + right) / 2 + offset_x * (right - left)) * image_width
    center_y = ((top + bottom) / 2 + offset_y * (bottom - top)) * image_height
    width = (right - left) * image_width * scale
    height = (bottom - top) * image_height * scale
    box = (
        max(0, round(center_x - width / 2)),
        max(0, round(center_y - height / 2)),
        min(image_width, round(center_x + width / 2)),
        min(image_height, round(center_y + height / 2)),
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def compare_reference_patch(
    document: Image.Image,
    patch: dict,
    scales: tuple[float, ...] = (0.9, 1.0, 1.1),
    offsets: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2),
) -> dict:
    reference_image = decode_reference_image(patch)
    reference = np.asarray(reference_image, dtype=np.uint8)
    grayscale = document.convert("L")
    best = None
    evaluated = 0
    for scale, offset_x, offset_y in product(scales, offsets, offsets):
        box = candidate_box(
            patch["bounds"], grayscale.size, scale, offset_x, offset_y
        )
        if box is None:
            continue
        candidate_image = grayscale.crop(box).resize(
            reference_image.size, Image.Resampling.BILINEAR
        )
        candidate = np.asarray(candidate_image, dtype=np.uint8)
        metrics = comparison_metrics(reference, candidate)
        evaluated += 1
        result = {
            **metrics,
            "scale": scale,
            "offset": {"x": offset_x, "y": offset_y},
            "candidateBoundsPixels": list(box),
        }
        if best is None or result["experimentalScore"] > best["experimentalScore"]:
            best = result
    if best is None:
        raise ValueError("reference patch does not define a usable document region")
    return {
        "status": "experimental_metric_only",
        "referenceNumber": patch.get("number"),
        "referenceBounds": patch["bounds"],
        "referenceSha256": (patch.get("image") or {}).get("sha256"),
        "lightType": patch.get("lightType"),
        "evaluatedCandidates": evaluated,
        **best,
    }


def compare_layout_reference_patches(
    document: Image.Image,
    layout: dict,
    light_types: set[int] | None = None,
) -> list[dict]:
    allowed_lights = light_types or {6, 24}
    return [
        compare_reference_patch(document, patch)
        for patch in layout.get("referencePatches", [])
        if patch.get("lightType") in allowed_lights
        and (patch.get("image") or {}).get("data")
    ]
