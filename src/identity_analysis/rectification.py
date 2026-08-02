"""Document corner detection and perspective rectification."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .onnx_runtime import create_session

OUTPUT_SIZE = (1600, 1008)


def map_rectified_bounds_to_source(
    bounds: list[float],
    source_corners: list[list[float]],
    source_size: tuple[int, int],
    rotation: int = 0,
) -> list[float]:
    top_left, top_right, bottom_right, bottom_left = np.asarray(
        source_corners, dtype=np.float64
    )

    def project(horizontal: float, vertical: float) -> np.ndarray:
        return (
            (1 - horizontal) * (1 - vertical) * top_left
            + horizontal * (1 - vertical) * top_right
            + horizontal * vertical * bottom_right
            + (1 - horizontal) * vertical * bottom_left
        )

    left, top, right, bottom = bounds
    if rotation == 90:
        left, top, right, bottom = 1 - bottom, left, 1 - top, right
    elif rotation == 180:
        left, top, right, bottom = 1 - right, 1 - bottom, 1 - left, 1 - top
    elif rotation == 270:
        left, top, right, bottom = top, 1 - right, bottom, 1 - left
    elif rotation != 0:
        raise ValueError(f"unsupported rectification rotation: {rotation}")
    points = np.asarray(
        [project(left, top), project(right, top), project(right, bottom), project(left, bottom)]
    )
    width, height = source_size
    return [
        float(np.clip(points[:, 0].min() / width, 0, 1)),
        float(np.clip(points[:, 1].min() / height, 0, 1)),
        float(np.clip(points[:, 0].max() / width, 0, 1)),
        float(np.clip(points[:, 1].max() / height, 0, 1)),
    ]


@lru_cache(maxsize=4)
def rectification_session(resource_string: str) -> ort.InferenceSession:
    root = Path(resource_string)
    return create_session((root / "models" / "document_corners.onnx").read_bytes())


def rectify_document(resource: Path, image: Image.Image) -> tuple[Image.Image, dict]:
    resized = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    tensor = np.asarray(resized, dtype=np.uint8)[None]
    session = rectification_session(str(resource.resolve()))
    corners = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]

    corners[:, 0] *= image.width / 512
    corners[:, 1] *= image.height / 512
    top_left, top_right, bottom_right, bottom_left = corners
    quad = tuple(top_left) + tuple(bottom_left) + tuple(bottom_right) + tuple(top_right)
    rectified = image.transform(
        OUTPUT_SIZE, Image.Transform.QUAD, quad, Image.Resampling.BICUBIC
    )
    return rectified, {
        "model": "models/document_corners.onnx",
        "sourceCorners": [[float(x), float(y)] for x, y in corners],
        "outputSize": list(OUTPUT_SIZE),
    }
