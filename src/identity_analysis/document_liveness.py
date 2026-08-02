"""Config-driven preprocessing operators for document liveness."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image


INTERPOLATION = {
    "INTER_LINEAR": Image.Resampling.BILINEAR,
    "INTER_AREA": Image.Resampling.BOX,
    "INTER_NEAREST": Image.Resampling.NEAREST,
}


@dataclass(frozen=True)
class DocumentDetection:
    kind: str
    box: tuple[float, float, float, float]
    score: float = 1.0


ModelRunner = Callable[[str, str, str, np.ndarray], np.ndarray]

DOCUMENT_VALIDATION_ERRORS = frozenset(
    {
        "DOCUMENT_NOT_FOUND",
        "DOCUMENT_PHOTO_NOT_FOUND",
        "DOCUMENT_CROPPED",
    }
)

DOCUMENT_QUALITY_WARNINGS = frozenset(
    {
        "RELATIVE_DOCUMENT_SIZE_LOWER_THAN_10_PERCENT",
        "DOCUMENT_BORDERS_OUTSIDE_OF_FRAME",
        "MULTIPLE_DOCUMENTS_IN_FRAME",
        "DOCUMENT_TOO_CLOSE_TO_BORDER",
        "IMAGE_TOO_BLURRY",
        "IMAGE_IS_TOO_COMPRESSED",
        "POOR_IMAGE_EXPOSURE",
        "GLARE_ON_IMAGE",
    }
)


def _detection(
    detections: list[DocumentDetection], kind: str
) -> DocumentDetection | None:
    matches = [item for item in detections if item.kind == kind]
    return max(matches, key=lambda item: item.score) if matches else None


def _crop_by_detection(
    image: Image.Image, detection: DocumentDetection, padding: float
) -> Image.Image:
    left, top, right, bottom = detection.box
    width = right - left
    height = bottom - top
    horizontal = width * padding / 100.0
    vertical = height * padding / 100.0
    return image.crop(
        (
            max(0, round(left - horizontal)),
            max(0, round(top - vertical)),
            min(image.width, round(right + horizontal)),
            min(image.height, round(bottom + vertical)),
        )
    )


def _resize(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    interpolation = config.get("interpolation_mode", "INTER_LINEAR")
    if interpolation not in INTERPOLATION:
        raise ValueError(f"unsupported interpolation mode: {interpolation}")
    return image.resize(
        (int(config["target_width"]), int(config["target_height"])),
        INTERPOLATION[interpolation],
    )


def _orientation_degrees(value: str | int) -> int:
    if isinstance(value, int):
        return value % 360
    mapping = {"k0": 0, "k90": 90, "k180": 180, "k270": 270}
    if value not in mapping:
        raise ValueError(f"unsupported orientation: {value}")
    return mapping[value]


def _stack_detections(
    image: Image.Image,
    detections: list[DocumentDetection],
    config: dict[str, Any],
) -> np.ndarray:
    channels = []
    size = (int(config["target_width"]), int(config["target_height"]))
    interpolation = INTERPOLATION[config.get("interpolation_mode", "INTER_LINEAR")]
    required = bool(config.get("detections_required", True))
    for crop_config in config["detections_to_align_by_bbox"]:
        detection = _detection(detections, crop_config["detection_to_use"])
        if detection is None:
            if required:
                raise ValueError(
                    f"required detection missing: {crop_config['detection_to_use']}"
                )
            channels.append(np.zeros((size[1], size[0]), dtype=np.float32))
            continue
        crop = _crop_by_detection(
            image, detection, float(crop_config.get("pad_percentage", 0))
        ).resize(size, interpolation)
        if config.get("apply_bw_transform", False):
            crop = crop.convert("L")
        channel = np.asarray(crop, dtype=np.float32)
        if channel.ndim == 3:
            channel = channel.mean(axis=2)
        channels.append(channel)
    return np.stack(channels, axis=-1)


def apply_preprocessors(
    image: Image.Image,
    preprocessors: list[dict[str, Any]],
    detections: list[DocumentDetection] | None = None,
    orientation: str | int = "k0",
) -> np.ndarray:
    detections = detections or []
    current: Image.Image | np.ndarray = image.convert("RGB")
    for config in preprocessors:
        operation = config["type"]
        if operation == "align_document_by_bbox":
            detection = _detection(detections, config["detection_to_use"])
            if detection is None:
                raise ValueError(
                    f"required detection missing: {config['detection_to_use']}"
                )
            if not isinstance(current, Image.Image):
                current = Image.fromarray(np.asarray(current, dtype=np.uint8))
            current = _crop_by_detection(
                current, detection, float(config.get("pad_percentage", 0))
            )
        elif operation == "resize":
            if not isinstance(current, Image.Image):
                current = Image.fromarray(np.asarray(current, dtype=np.uint8))
            current = _resize(current, config)
        elif operation == "lead_to_orientation":
            if not isinstance(current, Image.Image):
                current = Image.fromarray(np.asarray(current, dtype=np.uint8))
            current = current.rotate(_orientation_degrees(orientation), expand=True)
        elif operation == "normalize":
            pixels = np.asarray(current, dtype=np.float32)
            current = (pixels - np.asarray(config["mean"], dtype=np.float32)) / np.asarray(
                config["std"], dtype=np.float32
            )
        elif operation == "convert_to_float":
            current = np.asarray(current, dtype=np.float32)
        elif operation == "stack_detections_by_bbox":
            if not isinstance(current, Image.Image):
                current = Image.fromarray(np.asarray(current, dtype=np.uint8))
            current = _stack_detections(current, detections, config)
        else:
            raise ValueError(f"unsupported document liveness preprocessor: {operation}")
    return np.asarray(current, dtype=np.float32)


def evaluate_thresholds(
    values: np.ndarray, thresholds: list[float]
) -> list[dict[str, float | bool | int]]:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(flattened) != len(thresholds):
        raise ValueError(
            f"threshold count {len(thresholds)} does not match output count {len(flattened)}"
        )
    return [
        {
            "index": index,
            "value": float(value),
            "threshold": float(threshold),
            "margin": float(value - threshold),
            "aboveThreshold": bool(value >= threshold),
        }
        for index, (value, threshold) in enumerate(zip(flattened, thresholds))
    ]


def run_configured_model(
    image: Image.Image,
    config: dict[str, Any],
    runner: ModelRunner,
    detections: list[DocumentDetection] | None = None,
    orientation: str | int = "k0",
) -> dict[str, Any]:
    required = ("model_name", "nnet_input_name", "nnet_output_name")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"model configuration is missing: {', '.join(missing)}")
    tensor = apply_preprocessors(
        image,
        config.get("preprocessors", []),
        detections=detections,
        orientation=orientation,
    )
    raw_output = np.asarray(
        runner(
            config["model_name"],
            config["nnet_input_name"],
            config["nnet_output_name"],
            tensor,
        ),
        dtype=np.float32,
    )
    result = {
        "status": "diagnostic_only",
        "model": config["model_name"],
        "inputName": config["nnet_input_name"],
        "outputName": config["nnet_output_name"],
        "inputShape": list(tensor.shape),
        "outputShape": list(raw_output.shape),
        "rawOutput": [float(value) for value in raw_output.reshape(-1)],
    }
    if "output_thresholds" in config:
        result["thresholdResults"] = evaluate_thresholds(
            raw_output, config["output_thresholds"]
        )
    return result


def validate_document_geometry(
    image_size: tuple[int, int],
    detections: list[DocumentDetection],
    config: dict[str, Any],
) -> dict[str, Any]:
    width, height = image_size
    documents = [item for item in detections if item.kind == "kGenericDocument"]
    violations = []
    maximum = int(config.get("max_number_of_documents", 1))
    if len(documents) > maximum:
        violations.append(
            {
                "code": "too_many_documents",
                "observed": len(documents),
                "maximum": maximum,
            }
        )
    minimum_padding = float(config.get("min_document_padding_px", 0))
    for index, document in enumerate(documents):
        left, top, right, bottom = document.box
        observed = min(left, top, width - right, height - bottom)
        if observed < minimum_padding:
            violations.append(
                {
                    "code": "insufficient_document_padding",
                    "documentIndex": index,
                    "observed": float(observed),
                    "minimum": minimum_padding,
                }
            )
    return {
        "valid": not violations,
        "documentCount": len(documents),
        "violations": violations,
    }


def build_document_liveness_result(
    pipeline: str,
    liveness_score: float | None,
    liveness_probability: float | None,
    status_code: str | None = None,
    image_quality_warnings: list[str] | tuple[str, ...] = (),
    threshold: float = 0.5,
    calibration: str | None = None,
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("liveness threshold must be between 0 and 1")
    if status_code is not None and status_code not in DOCUMENT_VALIDATION_ERRORS:
        raise ValueError(f"unknown document validation status: {status_code}")
    unknown_warnings = sorted(set(image_quality_warnings) - DOCUMENT_QUALITY_WARNINGS)
    if unknown_warnings:
        raise ValueError(f"unknown document quality warnings: {unknown_warnings}")
    if status_code is None:
        if liveness_score is None or liveness_probability is None:
            raise ValueError("successful liveness result requires score and probability")
        if not 0 <= liveness_probability <= 1:
            raise ValueError("liveness probability must be between 0 and 1")
        decision = "live" if liveness_probability > threshold else "spoof"
    else:
        decision = "not_available"
    return {
        "pipeline": pipeline,
        "calibration": calibration,
        "statusCode": status_code,
        "livenessScore": liveness_score,
        "livenessProbability": liveness_probability,
        "threshold": threshold,
        "decision": decision,
        "imageQualityWarnings": list(image_quality_warnings),
    }
