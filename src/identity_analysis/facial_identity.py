"""Passive liveness and face-recognition inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .face_engines import FaceDetection, FaceDetector, LandmarkQuality
from .onnx_runtime import create_session


LIVENESS_WEIGHTS = np.asarray(
    [
        0.16606101027849307,
        0.09698286874516233,
        0.1259597090153954,
        0.11798336510870011,
        0.24542562822020086,
        0.15819160876456914,
        0.0893958098674789,
    ],
    dtype=np.float64,
)
LIVENESS_BIAS = -3.8685429515789105
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
RECOGNITION_THRESHOLD = 0.67
FACE_TEMPLATE_LENGTH = 512
FACE_TEMPLATE_BYTES = FACE_TEMPLATE_LENGTH * 4
RECOGNITION_TEMPLATE = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


class ONNXSession:
    def __init__(self, model: Path):
        self.model = model
        self.session = create_session(str(model))
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.height = int(model_input.shape[2])
        self.width = int(model_input.shape[3])

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run(None, {self.input_name: tensor})[0])


def _pixel_box(
    image: Image.Image, detection: FaceDetection
) -> tuple[float, float, float, float]:
    width, height = image.size
    x1, y1, x2, y2 = detection.box
    return x1 * width, y1 * height, x2 * width, y2 * height


def _expanded_crop(
    image: Image.Image,
    detection: FaceDetection,
    factor: float,
    square_from_height: bool,
) -> Image.Image:
    x1, y1, x2, y2 = _pixel_box(image, detection)
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    box_width = int(x2 - x1)
    box_height = int(y2 - y1)
    crop_width = int(box_height * factor) if square_from_height else int(box_width * factor)
    crop_height = int(box_height * factor)
    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(image.width - 1, center_x + crop_width // 2)
    bottom = min(image.height - 1, center_y + crop_height // 2)
    return image.crop((left, top, right + 1, bottom + 1))


def _liveness_tensor(
    image: Image.Image, detection: FaceDetection, index: int, size: tuple[int, int]
) -> np.ndarray:
    if index in (0, 1):
        crop = image
    elif index in (2, 4):
        crop = _expanded_crop(image, detection, 1.1, False)
    elif index in (3, 5):
        crop = _expanded_crop(image, detection, 1.45, True)
    else:
        crop = _expanded_crop(image, detection, 0.7, True)
    pixels = np.asarray(
        crop.convert("RGB").resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    tensor = np.transpose(pixels, (2, 0, 1)) / 255.0
    if index in (0, 1, 2, 4):
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor[None, ...].astype(np.float32)


def combine_liveness_logits(logits: list[float]) -> tuple[float, float]:
    if len(logits) != len(LIVENESS_WEIGHTS):
        raise ValueError("the liveness ensemble requires seven logits")
    linear_score = float(np.dot(LIVENESS_WEIGHTS, logits) + LIVENESS_BIAS)
    divisor = 3.0 if linear_score > 0.0 else 7.0
    calibrated = 0.5 + 0.5 * (linear_score + 0.2) / divisor
    return linear_score, float(np.clip(calibrated, 0.0, 1.0))


def liveness_decision(
    score: float,
    real_threshold: float,
    spoof_threshold: float,
) -> str:
    if not 0.0 <= spoof_threshold <= real_threshold <= 1.0:
        raise ValueError(
            "liveness thresholds must satisfy 0 <= spoof threshold <= real threshold <= 1"
        )
    if score > real_threshold:
        return "real"
    if score <= spoof_threshold:
        return "spoof"
    return "review"


class PassiveLivenessEngine:
    def __init__(self, model_root: Path):
        model_paths = sorted(model_root.glob("*/model.onnx"))
        if len(model_paths) != 7:
            raise ValueError(f"expected seven liveness models, found {len(model_paths)}")
        self.models = [ONNXSession(path) for path in model_paths]
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.models),
            thread_name_prefix="face-liveness",
        )

    def _run_models(self, tensors: list[np.ndarray]) -> list[np.ndarray]:
        model_inputs = list(zip(self.models, tensors))
        try:
            return list(
                self.executor.map(
                    lambda pair: pair[0].run(pair[1]),
                    model_inputs,
                )
            )
        except RuntimeError as error:
            if "can't start new thread" not in str(error):
                raise
            return [model.run(tensor) for model, tensor in model_inputs]

    def infer(
        self,
        image: Image.Image,
        detection: FaceDetection,
        threshold: float = 0.37,
        spoof_threshold: float = 0.25,
    ) -> dict:
        tensors = [
            _liveness_tensor(
                image, detection, index, (model.width, model.height)
            )
            for index, model in enumerate(self.models)
        ]
        outputs = self._run_models(tensors)
        logits = [float(output.reshape(-1)[0]) for output in outputs]
        linear_score, score = combine_liveness_logits(logits)
        return {
            "decision": liveness_decision(score, threshold, spoof_threshold),
            "score": score,
            "threshold": threshold,
            "spoofThreshold": spoof_threshold,
            "thresholdSource": "configured_policy",
            "linearScore": linear_score,
            "modelLogits": logits,
            "ensemble": "recovered_seven_model_weighted_calibration",
        }


def _five_landmarks(landmark_result: dict) -> np.ndarray:
    points = np.asarray(
        [point["pixel"] for point in landmark_result["landmarks"]], dtype=np.float64
    )
    return np.asarray(
        [
            points[36:42].mean(axis=0),
            points[42:48].mean(axis=0),
            points[30],
            points[48],
            points[54],
        ]
    )


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (source_x, source_y), (target_x, target_y) in zip(source, target):
        rows.extend(
            ([source_x, -source_y, 1.0, 0.0], [source_y, source_x, 0.0, 1.0])
        )
        values.extend((target_x, target_y))
    scale_x, scale_y, translate_x, translate_y = np.linalg.lstsq(
        np.asarray(rows), np.asarray(values), rcond=None
    )[0]
    return np.asarray(
        [
            [scale_x, -scale_y, translate_x],
            [scale_y, scale_x, translate_y],
            [0.0, 0.0, 1.0],
        ]
    )


def align_recognition_face(image: Image.Image, landmarks: dict) -> Image.Image:
    transform = similarity_transform(_five_landmarks(landmarks), RECOGNITION_TEMPLATE)
    inverse = np.linalg.inv(transform)
    coefficients = (
        inverse[0, 0],
        inverse[0, 1],
        inverse[0, 2],
        inverse[1, 0],
        inverse[1, 1],
        inverse[1, 2],
    )
    return image.convert("RGB").transform(
        (112, 112), Image.Transform.AFFINE, coefficients, Image.Resampling.BILINEAR
    )


class FaceRecognitionEngine:
    def __init__(
        self,
        model: Path,
        detector: FaceDetector,
        landmarks: LandmarkQuality,
    ):
        self.model = ONNXSession(model)
        self.detector = detector
        self.landmarks = landmarks

    def embedding_for_detection(
        self, image: Image.Image, detection: FaceDetection
    ) -> dict:
        landmark_result = self.landmarks.infer(image, detection)
        aligned = align_recognition_face(image, landmark_result)
        bgr = np.asarray(aligned, dtype=np.float32)[..., ::-1].copy()
        tensor = np.transpose(bgr, (2, 0, 1))[None, ...]
        vector = self.model.run(tensor).reshape(-1).astype(np.float32)
        return {
            "detection": detection,
            "landmarksAndQuality": landmark_result,
            "vector": vector,
        }

    def embedding(self, image: Image.Image) -> dict:
        detections = self.detector.detect(image)
        if len(detections) != 1:
            raise ValueError(f"expected one face, found {len(detections)}")
        return self.embedding_for_detection(image, detections[0])

    def embedding_pair(
        self,
        first: Image.Image,
        second: Image.Image,
        first_detection: FaceDetection | None = None,
    ) -> tuple[dict, dict]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = (
                executor.submit(
                    self.embedding_for_detection,
                    first,
                    first_detection,
                )
                if first_detection is not None
                else executor.submit(self.embedding, first)
            )
            second_future = executor.submit(self.embedding, second)
            return first_future.result(), second_future.result()

    def compare(
        self,
        first: Image.Image,
        second: Image.Image,
        threshold: float = RECOGNITION_THRESHOLD,
    ) -> dict:
        first_result, second_result = self.embedding_pair(first, second)
        result = compare_face_templates(
            first_result["vector"], second_result["vector"], threshold
        )
        result["faceBoxes"] = [
            list(first_result["detection"].box),
            list(second_result["detection"].box),
        ]
        result["templateVectors"] = [
            first_result["vector"],
            second_result["vector"],
        ]
        return result


def encode_face_template(vector: np.ndarray) -> bytes:
    normalized = np.asarray(vector, dtype="<f4").reshape(-1)
    if len(normalized) != FACE_TEMPLATE_LENGTH or not np.isfinite(normalized).all():
        raise ValueError("face template must contain 512 finite float32 values")
    if float(np.linalg.norm(normalized)) == 0.0:
        raise ValueError("face template must have a nonzero norm")
    return normalized.tobytes()


def decode_face_template(payload: bytes) -> np.ndarray:
    if len(payload) != FACE_TEMPLATE_BYTES:
        raise ValueError(f"face template must contain exactly {FACE_TEMPLATE_BYTES} bytes")
    vector = np.frombuffer(payload, dtype="<f4").copy()
    if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) == 0.0:
        raise ValueError("face template must contain finite values with a nonzero norm")
    return vector


def compare_face_templates(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
    threshold: float = RECOGNITION_THRESHOLD,
) -> dict:
    first_vector = np.asarray(first_vector, dtype=np.float32).reshape(-1)
    second_vector = np.asarray(second_vector, dtype=np.float32).reshape(-1)
    if (
        len(first_vector) != FACE_TEMPLATE_LENGTH
        or len(second_vector) != FACE_TEMPLATE_LENGTH
    ):
        raise ValueError("face templates must contain 512 values")
    if not np.isfinite(first_vector).all() or not np.isfinite(second_vector).all():
        raise ValueError("face templates must contain finite values")
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    if denominator == 0.0:
        raise ValueError("face templates must have a nonzero norm")
    cosine = float(np.dot(first_vector, second_vector) / denominator)
    score = (cosine + 1.0) / 2.0
    return {
        "decision": "same_person" if score > threshold else "different_person",
        "score": score,
        "threshold": threshold,
        "cosineSimilarity": cosine,
        "embeddingLength": len(first_vector),
    }


def identity_verification_decision(
    identity_decision: str,
    liveness_decision: str | None,
    document_capture_decision: str | None = None,
) -> dict:
    checks = {
        "portraitMatch": identity_decision,
        "selfieLiveness": liveness_decision or "not_available",
        "documentCapture": document_capture_decision or "not_available",
    }
    if identity_decision == "different_person":
        decision = "fail"
    elif liveness_decision == "spoof":
        decision = "fail"
    elif document_capture_decision in {"review", "spoof"}:
        decision = "review"
    elif liveness_decision is None:
        decision = "not_available"
    elif document_capture_decision in {None, "not_available"}:
        decision = "review"
    elif (
        identity_decision == "same_person"
        and liveness_decision == "real"
        and document_capture_decision in {"pass", "real"}
    ):
        decision = "pass"
    else:
        decision = "review"
    return {
        "decision": decision,
        "checks": checks,
        "scope": "portrait_identity_selfie_liveness_and_document_capture",
    }
