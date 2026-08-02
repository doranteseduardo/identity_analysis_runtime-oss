"""Document quality and capture-risk inference."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageStat

from .onnx_runtime import create_session

def softmax(values: np.ndarray) -> list[float]:
    shifted = values - np.max(values)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return [float(value) for value in probabilities]


@lru_cache(maxsize=4)
def focus_session(resource_string: str) -> ort.InferenceSession:
    root = Path(resource_string)
    return create_session((root / "models" / "focus_device.onnx").read_bytes())


def focus_model(resource: Path, image: Image.Image) -> dict:
    prepared = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    tensor = (np.asarray(prepared, dtype=np.float32) / 255.0)[None, :, :, None]
    session = focus_session(str(resource.resolve()))
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    probabilities = softmax(output)
    predicted_index = int(np.argmax(probabilities))
    labels = ("blurred", "focused")
    return {
        "model": "models/focus_device.onnx",
        "rawOutput": [float(value) for value in output],
        "classProbabilities": {
            labels[index]: probability for index, probability in enumerate(probabilities)
        },
        "predictedClass": labels[predicted_index],
        "classMappingEvidence": "Differential inference: Gaussian blur consistently selects class 0; the sharp source selects class 1.",
    }


def image_statistics(image: Image.Image) -> dict:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    luminance = ImageStat.Stat(gray)
    edge_stats = ImageStat.Stat(edges)
    return {
        "meanLuminance": float(luminance.mean[0] / 255.0),
        "luminanceStdDev": float(luminance.stddev[0] / 255.0),
        "edgeMean": float(edge_stats.mean[0] / 255.0),
    }


@lru_cache(maxsize=4)
def liveness_sessions(resource_string: str) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    root = Path(resource_string)
    electronic = create_session(
        (root / "models" / "electronic_device.onnx").read_bytes()
    )
    moire = create_session((root / "models" / "moire.onnx").read_bytes())
    return electronic, moire


def warm_up_quality(resource: Path) -> None:
    focus_session(str(resource.resolve()))
    if (resource / "models" / "electronic_device.onnx").exists() and (resource / "models" / "moire.onnx").exists():
        liveness_sessions(str(resource.resolve()))


def document_liveness(resource: Path, image: Image.Image) -> dict:
    electronic, moire = liveness_sessions(str(resource.resolve()))
    rgb = image.convert("RGB")
    electronic_image = np.asarray(
        rgb.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8
    )[:, :, ::-1].copy()[None]
    electronic_output = electronic.run(None, {"input": electronic_image})[0][0]
    electronic_class = int(np.argmax(electronic_output))

    linear = np.asarray(
        rgb.resize((668, 524), Image.Resampling.BILINEAR), dtype=np.uint8
    )[:, :, ::-1].copy()
    area = np.asarray(
        rgb.resize((668, 524), Image.Resampling.BOX), dtype=np.uint8
    )[:, :, ::-1].copy()
    moire_score = float(
        moire.run(
            None,
            {"actual_input_lin": linear, "actual_input_area": area},
        )[0][0][0]
    )
    moire_threshold = 0.529
    electronic_status = "clear" if electronic_class == 0 else "detected"
    moire_status = "clear" if moire_score <= moire_threshold else "detected"
    return {
        "decision": "pass" if electronic_status == moire_status == "clear" else "review",
        "electronicDevice": {
            "status": electronic_status,
            "rawOutput": [float(value) for value in electronic_output],
            "predictedClassIndex": electronic_class,
        },
        "moire": {
            "status": moire_status,
            "score": moire_score,
            "threshold": moire_threshold,
        },
        "scope": "document_capture",
        "evidence": "Class mapping and Moire threshold assume ONNX models matching this runtime's document-capture-PAD contract; see docs/models.md.",
    }


def analyze_quality(resource: Path, image: Image.Image) -> dict:
    has_document_pad = (
        (resource / "models" / "electronic_device.onnx").exists()
        and (resource / "models" / "moire.onnx").exists()
    )
    with ThreadPoolExecutor(max_workers=3 if has_document_pad else 2) as executor:
        focus_future = executor.submit(focus_model, resource, image)
        statistics_future = executor.submit(image_statistics, image)
        liveness_future = (
            executor.submit(document_liveness, resource, image)
            if has_document_pad
            else None
        )
        focus = focus_future.result()
        statistics = statistics_future.result()
        liveness = liveness_future.result() if liveness_future is not None else None
    result = {
        "focus": focus,
        "imageStatistics": statistics,
        "spoofingDecision": "not_available",
        "livenessDecision": "not_available",
        "capabilities": {
            "documentQuality": True,
            "documentSpoofingPAD": False,
            "faceLiveness": False,
        },
        "unavailableReasons": {
            "spoofingDecision": "Document-capture PAD models are not present in the configured asset directory.",
            "livenessDecision": "A still document image contains no live-face challenge, motion or depth input.",
        },
        "note": "Focus is a capture-quality signal and must not be promoted to a spoofing or liveness decision.",
    }
    if liveness is not None:
        result["documentLiveness"] = liveness
        result["spoofingDecision"] = liveness["decision"]
        result["capabilities"]["documentSpoofingPAD"] = True
        result["unavailableReasons"].pop("spoofingDecision")
        result["note"] = "Document-capture PAD is available; face liveness remains unavailable from a document still image."
    return result
