"""Facial detection and landmark inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .onnx_runtime import create_session


HEAD_POSE_MODEL = {
    30: (0.0, 0.0, 0.0),
    8: (0.0, -330.0, -65.0),
    36: (-225.0, 170.0, -135.0),
    45: (225.0, 170.0, -135.0),
    48: (-150.0, -150.0, -125.0),
    54: (150.0, -150.0, -125.0),
}
FACE_QUALITY_LIMITS = {
    "minimumFacePixels": 80,
    "minimumNormalizedSide": 0.12,
    "borderMargin": 0.01,
    "maximumAbsoluteYaw": 35.0,
    "maximumAbsolutePitch": 35.0,
    "maximumAbsoluteRoll": 30.0,
    "coveredFaceScore": 0.8,
}
LIVENESS_BLOCKING_QUALITY_WARNINGS = {
    "MULTIPLE_FACES",
    "FACE_TOO_SMALL",
    "FACE_CUTOFF",
}


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[float, float, float, float]
    confidence: float


def estimate_head_pose(landmarks: list[dict]) -> dict:
    """Estimate Euler angles from six 68-point landmarks under weak perspective."""
    points = {int(point["index"]): point["pixel"] for point in landmarks}
    missing = sorted(set(HEAD_POSE_MODEL) - set(points))
    if missing:
        raise ValueError(f"missing head-pose landmarks: {missing}")
    model = np.asarray(list(HEAD_POSE_MODEL.values()), dtype=np.float64)
    image = np.asarray([points[index] for index in HEAD_POSE_MODEL], dtype=np.float64)
    image[:, 1] *= -1.0
    model -= model.mean(axis=0)
    image -= image.mean(axis=0)
    affine = np.linalg.lstsq(model, image, rcond=None)[0].T
    first_norm = np.linalg.norm(affine[0])
    if first_norm <= 1e-8:
        raise ValueError("head-pose landmarks are geometrically degenerate")
    first = affine[0] / first_norm
    second = affine[1] - np.dot(affine[1], first) * first
    second_norm = np.linalg.norm(second)
    if second_norm <= 1e-8:
        raise ValueError("head-pose landmarks are geometrically degenerate")
    second /= second_norm
    third = np.cross(first, second)
    rotation = np.vstack((first, second, third))
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = np.degrees(np.arctan2(-rotation[2, 0], horizontal))
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "method": "six_point_weak_perspective",
    }


def assess_face_quality(
    detection: FaceDetection,
    image_size: tuple[int, int],
    head_pose: dict,
    face_count: int = 1,
    landmark_quality_score: float | None = None,
) -> dict:
    width, height = image_size
    x1, y1, x2, y2 = detection.box
    face_width = (x2 - x1) * width
    face_height = (y2 - y1) * height
    warnings = []

    def add(code: str, legacy_code: int) -> None:
        warnings.append({"code": code, "legacyCode": legacy_code})

    if face_count > 1:
        add("MULTIPLE_FACES", -500)
    if (
        min(face_width, face_height) < FACE_QUALITY_LIMITS["minimumFacePixels"]
        or min(x2 - x1, y2 - y1) < FACE_QUALITY_LIMITS["minimumNormalizedSide"]
    ):
        add("FACE_TOO_SMALL", -100)
    margin = FACE_QUALITY_LIMITS["borderMargin"]
    if x1 <= margin or y1 <= margin or x2 >= 1.0 - margin or y2 >= 1.0 - margin:
        add("FACE_CUTOFF", -200)
    if (
        abs(head_pose["yaw"]) > FACE_QUALITY_LIMITS["maximumAbsoluteYaw"]
        or abs(head_pose["pitch"]) > FACE_QUALITY_LIMITS["maximumAbsolutePitch"]
        or abs(head_pose["roll"]) > FACE_QUALITY_LIMITS["maximumAbsoluteRoll"]
    ):
        add("EXCESSIVE_POSE", -300)
    if (
        landmark_quality_score is not None
        and landmark_quality_score >= FACE_QUALITY_LIMITS["coveredFaceScore"]
    ):
        add("COVERED_FACE", -400)
    return {
        "status": "review" if warnings else "pass",
        "warnings": warnings,
        "livenessEligible": not any(
            warning["code"] in LIVENESS_BLOCKING_QUALITY_WARNINGS
            for warning in warnings
        ),
        "policy": "portable_geometry_v1",
    }


def generate_ssd_anchors() -> np.ndarray:
    feature_maps = (32, 16, 8, 4, 2, 1)
    scales = np.linspace(0.2, 0.95, len(feature_maps))
    anchors = []
    for layer, feature_map in enumerate(feature_maps):
        if layer == 0:
            dimensions = ((0.1, 0.1), (0.2 / np.sqrt(2), 0.2 * np.sqrt(2)), (0.2 * np.sqrt(2), 0.2 / np.sqrt(2)))
        else:
            scale = scales[layer]
            next_scale = scales[layer + 1] if layer + 1 < len(scales) else 1.0
            dimensions = [
                (scale, scale),
                (np.sqrt(scale * next_scale), np.sqrt(scale * next_scale)),
            ]
            for ratio in (2.0, 0.5, 3.0, 1.0 / 3.0):
                dimensions.append((scale / np.sqrt(ratio), scale * np.sqrt(ratio)))
        for row in range(feature_map):
            for column in range(feature_map):
                center_y = (row + 0.5) / feature_map
                center_x = (column + 0.5) / feature_map
                for height, width in dimensions:
                    anchors.append((center_y, center_x, height, width))
    result = np.asarray(anchors, dtype=np.float32)
    if result.shape != (5118, 4):
        raise RuntimeError(f"unexpected SSD anchor count: {result.shape}")
    return result


def decode_ssd_boxes(encodings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encodings, dtype=np.float32).reshape(-1, 4)
    center_y = encoded[:, 0] / 10.0 * anchors[:, 2] + anchors[:, 0]
    center_x = encoded[:, 1] / 10.0 * anchors[:, 3] + anchors[:, 1]
    height = np.exp(encoded[:, 2] / 5.0) * anchors[:, 2]
    width = np.exp(encoded[:, 3] / 5.0) * anchors[:, 3]
    return np.column_stack(
        (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2)
    )


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        intersection_x1 = np.maximum(x1[current], x1[order[1:]])
        intersection_y1 = np.maximum(y1[current], y1[order[1:]])
        intersection_x2 = np.minimum(x2[current], x2[order[1:]])
        intersection_y2 = np.minimum(y2[current], y2[order[1:]])
        intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
            0.0, intersection_y2 - intersection_y1
        )
        union = areas[current] + areas[order[1:]] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = order[1:][overlap <= threshold]
    return keep


def _openvino_core():
    try:
        from openvino import Core
    except ImportError as error:
        raise RuntimeError(
            "OpenVINO is required for the independently extracted facial IR models"
        ) from error
    return Core()


class _ModelRunner:
    def __init__(self, model: Path):
        self.model = model
        if model.suffix.lower() == ".onnx":
            self.backend = "onnxruntime"
            self.session = create_session(str(model))
            self.input_name = self.session.get_inputs()[0].name
        else:
            core = _openvino_core()
            self.backend = "openvino"
            self.session = core.compile_model(core.read_model(str(model)), "CPU")
            self.input_name = self.session.input(0)

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.backend == "onnxruntime":
            return [
                np.asarray(value)
                for value in self.session.run(None, {self.input_name: tensor})
            ]
        result = self.session({self.input_name: tensor})
        return [np.asarray(result[output]) for output in self.session.outputs]


def _image_tensor(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return np.transpose(pixels, (2, 0, 1))[None, ...]


class OpenVINOFaceDetector:
    def __init__(self, model: Path):
        self.runner = _ModelRunner(model)
        self.anchors = generate_ssd_anchors()

    def detect(
        self,
        image: Image.Image,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.3,
    ) -> list[FaceDetection]:
        tensor = _image_tensor(image, (512, 512))
        arrays = self.runner.run(tensor)
        scores = next(array for array in arrays if array.shape[-1] == 2).reshape(-1, 2)[:, 1]
        encodings = next(array for array in arrays if array.shape[-1] == 4).reshape(-1, 4)
        boxes = np.clip(decode_ssd_boxes(encodings, self.anchors), 0.0, 1.0)
        selected = np.flatnonzero(scores >= score_threshold)
        dimensions = boxes[selected, 2:] - boxes[selected, :2]
        aspect_ratios = dimensions[:, 0] / np.maximum(dimensions[:, 1], 1e-6)
        plausible = (
            (dimensions[:, 0] >= 0.08)
            & (dimensions[:, 1] >= 0.08)
            & (aspect_ratios >= 0.5)
            & (aspect_ratios <= 1.5)
        )
        selected = selected[plausible]
        kept = non_maximum_suppression(boxes[selected], scores[selected], iou_threshold)
        return [
            FaceDetection(tuple(float(value) for value in boxes[selected[index]]), float(scores[selected[index]]))
            for index in kept
        ]


class OpenVINOLandmarkQuality:
    def __init__(self, model: Path):
        self.runner = _ModelRunner(model)

    def infer(self, image: Image.Image, detection: FaceDetection, padding: float = 0.15) -> dict:
        width, height = image.size
        x1, y1, x2, y2 = detection.box
        box_width, box_height = x2 - x1, y2 - y1
        crop_box = (
            max(0, (x1 - box_width * padding) * width),
            max(0, (y1 - box_height * padding) * height),
            min(width, (x2 + box_width * padding) * width),
            min(height, (y2 + box_height * padding) * height),
        )
        crop = image.crop(crop_box)
        tensor = _image_tensor(crop, (224, 224))
        outputs = self.runner.run(tensor)
        landmark_outputs = [value.reshape(-1) for value in outputs if value.size == 68]
        quality_output = next(value for value in outputs if value.size == 1)
        x_coordinates, y_coordinates = landmark_outputs
        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        landmarks = []
        for index, (x_coordinate, y_coordinate) in enumerate(
            zip(x_coordinates, y_coordinates)
        ):
            pixel_x = crop_box[0] + float(x_coordinate) * crop_width
            pixel_y = crop_box[1] + float(y_coordinate) * crop_height
            landmarks.append(
                {
                    "index": index,
                    "cropNormalized": [float(x_coordinate), float(y_coordinate)],
                    "imageNormalized": [pixel_x / width, pixel_y / height],
                    "pixel": [round(pixel_x), round(pixel_y)],
                }
            )
        return {
            "landmarkConvention": "68-point x-vector plus y-vector",
            "landmarks": landmarks,
            "qualityScore": float(quality_output.reshape(-1)[0]),
            "qualitySemantics": "coverage_sensitive_experimental",
            "qualityThreshold": FACE_QUALITY_LIMITS["coveredFaceScore"],
            "headPose": estimate_head_pose(landmarks),
        }


FaceDetector = OpenVINOFaceDetector
LandmarkQuality = OpenVINOLandmarkQuality


def analyze_faces(
    image: Image.Image,
    detector: FaceDetector,
    landmark_model: LandmarkQuality | None = None,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.3,
) -> dict:
    detections = detector.detect(image, score_threshold, iou_threshold)
    width, height = image.size
    faces = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        face = {
            "confidence": detection.confidence,
            "normalizedBox": list(detection.box),
            "pixelBox": [
                round(x1 * width),
                round(y1 * height),
                round(x2 * width),
                round(y2 * height),
            ],
        }
        if landmark_model is not None:
            landmark_result = landmark_model.infer(image, detection)
            face["landmarksAndQuality"] = landmark_result
            face["headPose"] = landmark_result["headPose"]
            face["quality"] = assess_face_quality(
                detection,
                image.size,
                landmark_result["headPose"],
                len(detections),
                landmark_result["qualityScore"],
            )
        faces.append(face)
    return {
        "engine": "independent-onnx-facial-pipeline",
        "imageSize": [width, height],
        "faceCount": len(faces),
        "faces": faces,
        "livenessDecision": "use_/v1/face/liveness",
        "recognitionDecision": "use_/v1/face/compare",
        "limitations": [
            "Head pose is a portable geometric estimate, not a native-angle reproduction.",
            "The coverage-sensitive scalar uses a portable threshold derived from controlled perturbations.",
            "Passive liveness and recognition are exposed by specialized endpoints.",
        ],
    }
