"""HTTP API for document and facial identity analysis."""

import base64
import binascii
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.datastructures import UploadFile

from .assets import validate_assets
from .capabilities import SDK_COMPATIBILITY, runtime_capabilities
from .document_classifier import (
    classify_document,
    document_catalog_entry,
    document_catalog_facets,
    search_document_catalog,
)
from .face_engines import FaceDetector, LandmarkQuality, analyze_faces, assess_face_quality
from .facial_identity import (
    FACE_TEMPLATE_BYTES,
    FaceRecognitionEngine,
    PassiveLivenessEngine,
    compare_face_templates,
    decode_face_template,
    encode_face_template,
    identity_verification_decision,
)
from .pipeline import (
    SUPPORTED_REQUEST_PROFILES,
    process_document,
    process_document_pages,
    process_document_pair,
    warm_up,
)
from .reference_matching import compare_layout_reference_patches
from .responses import (
    comparison_response,
    document_response,
    document_portrait_comparison_response,
    face_analysis_response,
    face_template_response,
    liveness_response,
    success,
)
from .visual_layouts import (
    barcode_regions,
    declared_layout_requirements,
    graphic_regions,
    layout_descriptor,
    layout_field_names,
    layout_page_role,
    layout_relations,
    reference_patches,
    security_regions,
    text_regions,
    visual_layout,
)


DEFAULT_ASSETS_PATH = Path(__file__).resolve().parents[2] / "assets"
ASSETS_PATH = Path(
    os.environ.get("IDENTITY_ANALYSIS_ASSETS", DEFAULT_ASSETS_PATH)
).resolve()
MAX_UPLOAD_BYTES = int(
    os.environ.get(
        "IDENTITY_ANALYSIS_MAX_UPLOAD_BYTES",
        20 * 1024 * 1024,
    )
)
MAX_DOCUMENT_PAGES = int(os.environ.get("IDENTITY_ANALYSIS_MAX_DOCUMENT_PAGES", "10"))
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
EXTRACTABLE_DOCUMENT_IMAGES = {
    "documentFrontSide",
    "portrait",
    "portraitOfChild",
    "ghostPortrait",
    "signature",
}
DOCUMENT_PORTRAIT_FACE_THRESHOLD = 0.2
WARM_UP_ENABLED = os.environ.get(
    "IDENTITY_ANALYSIS_WARMUP", "true"
).lower() not in {"0", "false", "no"}
FACE_DETECTOR_PATH = ASSETS_PATH / "facial/detector/face_detector.onnx"
FACE_LANDMARKS_PATH = ASSETS_PATH / "facial/landmarks/landmarks_quality.onnx"
FACE_LIVENESS_PATH = ASSETS_PATH / "facial/liveness"
FACE_RECOGNITION_PATH = (
    ASSETS_PATH / "facial/recognition/00_R_L_CF_V1_16GPUs/model.onnx"
)
DEFAULT_LIVENESS_THRESHOLD = float(
    os.environ.get("IDENTITY_ANALYSIS_LIVENESS_THRESHOLD", "0.37")
)
DEFAULT_LIVENESS_SPOOF_THRESHOLD = float(
    os.environ.get("IDENTITY_ANALYSIS_LIVENESS_SPOOF_THRESHOLD", "0.25")
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if WARM_UP_ENABLED:
        # Every model family is optional; start-up never fails because one of
        # them is absent.  GET /v1/capabilities reports what actually loaded.
        await run_in_threadpool(warm_up, ASSETS_PATH)
        try:
            await run_in_threadpool(facial_models)
        except Exception:  # noqa: BLE001 - facial models are an optional family
            pass
    yield

app = FastAPI(
    title="Independent Identity Analysis Runtime",
    version="0.1.0",
    description="Document OCR, capture PAD, and facial analysis with explicit capability coverage.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_processing_time(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.exception_handler(HTTPException)
async def http_error_response(_: Request, error: HTTPException) -> JSONResponse:
    codes = {
        400: "invalid_request",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "unprocessable_image",
        503: "service_unavailable",
    }
    return JSONResponse(
        status_code=error.status_code,
        content={
            "status": "error",
            "error": {
                "code": codes.get(error.status_code, "request_failed"),
                "message": str(error.detail),
            },
        },
    )


def decode_base64_image(value: str) -> bytes:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="imageBase64 is not valid base64") from error


def validate_upload(payload: bytes, filename: str) -> str:
    if not payload:
        raise HTTPException(status_code=400, detail="The image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes")
    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image extension: {suffix}",
        )
    return suffix


def request_boolean(value: Any, default: bool = False, name: str = "includeImages") -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise HTTPException(status_code=400, detail=f"{name} must be boolean")


def include_images_value(request: Request, body_value: Any = None) -> bool:
    value = body_value
    if value is None:
        value = request.query_params.get("includeImages")
    return request_boolean(value)


def include_templates_value(request: Request, body_value: Any = None) -> bool:
    value = body_value
    if value is None:
        value = request.query_params.get("includeTemplates")
    return request_boolean(value, name="includeTemplates")


def analyze_portraits_value(request: Request, body_value: Any = None) -> bool:
    value = body_value
    if value is None:
        value = request.query_params.get("analyzePortraits")
    return request_boolean(value, name="analyzePortraits")


def analyze_liveness_value(request: Request, body_value: Any = None) -> bool:
    value = body_value
    if value is None:
        value = request.query_params.get("analyzeLiveness")
    return request_boolean(value, name="analyzeLiveness")


def document_region_crops(
    result: dict,
    images_by_side: dict[str, bytes],
    default_side: str,
):
    opened = {}
    for name, region in result.get("visualRegions", {}).items():
        side = region.get("side") or default_side
        payload = images_by_side.get(side)
        bounds = region.get("box")
        if payload is None or not isinstance(bounds, list) or len(bounds) != 4:
            continue
        if side not in opened:
            image = Image.open(io.BytesIO(payload))
            image.load()
            opened[side] = image.convert("RGB")
        image = opened[side]
        left, top, right, bottom = bounds
        box = (
            max(0, min(image.width, round(float(left) * image.width))),
            max(0, min(image.height, round(float(top) * image.height))),
            max(0, min(image.width, round(float(right) * image.width))),
            max(0, min(image.height, round(float(bottom) * image.height))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        yield name, region, side, image.crop(box)


def extract_document_images(
    result: dict,
    images_by_side: dict[str, bytes],
    default_side: str,
) -> list[dict]:
    extracted = []
    for name, _, side, crop in document_region_crops(
        result, images_by_side, default_side
    ):
        if name not in EXTRACTABLE_DOCUMENT_IMAGES:
            continue
        encoded = io.BytesIO()
        crop.save(encoded, format="JPEG", quality=90, optimize=True)
        extracted.append(
            {
                "type": name,
                "side": side,
                "mediaType": "image/jpeg",
                "imageBase64": base64.b64encode(encoded.getvalue()).decode("ascii"),
                "width": crop.width,
                "height": crop.height,
            }
        )
    return extracted


def analyze_document_portraits(
    result: dict,
    images_by_side: dict[str, bytes],
    default_side: str,
    detector: FaceDetector,
) -> None:
    for _, region, _, crop in document_region_crops(
        result, images_by_side, default_side
    ):
        if not region.get("faceExpected"):
            continue
        detections = detector.detect(crop, DOCUMENT_PORTRAIT_FACE_THRESHOLD)
        region["facePresence"] = {
            "expected": True,
            "detected": bool(detections),
            "count": len(detections),
            "threshold": DOCUMENT_PORTRAIT_FACE_THRESHOLD,
            "confidence": (
                max(detection.confidence for detection in detections)
                if detections
                else None
            ),
            "status": "pass" if detections else "review",
        }


def compare_document_portrait(
    document_result: dict,
    document_payload: bytes,
    selfie: Image.Image,
    engine: FaceRecognitionEngine,
    threshold: float,
    passive_liveness: PassiveLivenessEngine | None = None,
    liveness_threshold: float = DEFAULT_LIVENESS_THRESHOLD,
    liveness_spoof_threshold: float = DEFAULT_LIVENESS_SPOOF_THRESHOLD,
) -> dict:
    portrait = next(
        (
            (region, crop)
            for name, region, _, crop in document_region_crops(
                document_result,
                {"document": document_payload},
                "document",
            )
            if name == "portrait" and region.get("faceExpected")
        ),
        None,
    )
    if portrait is None:
        raise ValueError("recognized document has no declared portrait region")
    region, crop = portrait
    detections = engine.detector.detect(crop, DOCUMENT_PORTRAIT_FACE_THRESHOLD)
    if not detections:
        raise ValueError("no face detected in declared document portrait")
    document_detection = max(detections, key=lambda detection: detection.confidence)
    document_embedding, selfie_embedding = engine.embedding_pair(
        crop,
        selfie,
        document_detection,
    )
    classification_candidates = (
        (document_result.get("documentClassification") or {}).get("candidates") or []
    )
    result = compare_face_templates(
        document_embedding["vector"], selfie_embedding["vector"], threshold
    )
    result.update(
        {
            "documentProfile": document_result.get("recognitionProfile"),
            "layoutIdentifier": document_result.get("requestedDocumentIdentifier")
            or (
                classification_candidates[0].get("documentIdentifier")
                if classification_candidates
                else None
            ),
            "portraitBox": region.get("box"),
            "documentFaceBox": list(document_detection.box),
            "documentDetectionConfidence": document_detection.confidence,
            "documentDetectionThreshold": DOCUMENT_PORTRAIT_FACE_THRESHOLD,
            "selfieFaceBox": list(selfie_embedding["detection"].box),
        }
    )
    if passive_liveness is not None:
        landmark_result = selfie_embedding["landmarksAndQuality"]
        quality = assess_face_quality(
            selfie_embedding["detection"],
            selfie.size,
            landmark_result["headPose"],
            1,
            landmark_result.get("qualityScore"),
        )
        if not quality["livenessEligible"]:
            liveness = {
                "decision": "review",
                "threshold": liveness_threshold,
                "spoofThreshold": liveness_spoof_threshold,
            }
        else:
            liveness = passive_liveness.infer(
                selfie,
                selfie_embedding["detection"],
                liveness_threshold,
                liveness_spoof_threshold,
            )
        liveness["quality"] = quality
        liveness["headPose"] = landmark_result["headPose"]
        result["selfieLiveness"] = liveness
    result["verification"] = identity_verification_decision(
        result["decision"],
        (result.get("selfieLiveness") or {}).get("decision"),
        (document_result.get("qualitySignals") or {}).get("spoofingDecision"),
    )
    return result


def multipart_upload(form: Any, name: str) -> UploadFile:
    value = form.get(name)
    if isinstance(value, UploadFile):
        return value
    raise HTTPException(
        status_code=400,
        detail=f"Multipart field '{name}' is required",
    )


def multipart_uploads(form: Any, name: str) -> list[UploadFile]:
    values = form.getlist(name)
    if values and all(isinstance(value, UploadFile) for value in values):
        return values
    raise HTTPException(
        status_code=400,
        detail=f"Multipart field '{name}' is required",
    )


async def read_image(request: Request) -> tuple[bytes, str, str, bool, bool]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = multipart_upload(form, "image")
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
        profile = str(form.get("profile", "auto_research"))
        return (
            payload,
            upload.filename or "upload.jpg",
            profile,
            include_images_value(request, form.get("includeImages")),
            analyze_portraits_value(request, form.get("analyzePortraits")),
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict) or not isinstance(body.get("imageBase64"), str):
            raise HTTPException(status_code=400, detail="JSON field 'imageBase64' is required")
        filename = body.get("filename", "upload.jpg")
        if not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="filename must be a string")
        profile = body.get("profile", "auto_research")
        if not isinstance(profile, str):
            raise HTTPException(status_code=400, detail="profile must be a string")
        return (
            decode_base64_image(body["imageBase64"]),
            filename,
            profile,
            include_images_value(request, body.get("includeImages")),
            analyze_portraits_value(request, body.get("analyzePortraits")),
        )
    raise HTTPException(
        status_code=415,
        detail="Use application/json or multipart/form-data",
    )


def requested_field_names(value: Any) -> set[str] | None:
    if value is None or value == [] or value == "":
        return None
    values = value if isinstance(value, list) else [value]
    if not all(isinstance(item, str) for item in values):
        raise HTTPException(status_code=400, detail="fields must contain strings")
    names = {
        name.strip()
        for item in values
        for name in item.split(",")
        if name.strip()
    }
    if not names:
        return None
    if len(names) > 64:
        raise HTTPException(status_code=422, detail="fields cannot contain more than 64 names")
    return names


def optional_document_identifier(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail=f"{name} must be an integer"
        ) from error


async def read_layout_ocr_request(
    request: Request,
) -> tuple[bytes, str, bool, bool, set[str] | None]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = multipart_upload(form, "image")
        return (
            await upload.read(MAX_UPLOAD_BYTES + 1),
            upload.filename or "upload.jpg",
            include_images_value(request, form.get("includeImages")),
            analyze_portraits_value(request, form.get("analyzePortraits")),
            requested_field_names(form.getlist("fields")),
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict) or not isinstance(body.get("imageBase64"), str):
            raise HTTPException(status_code=400, detail="JSON field 'imageBase64' is required")
        filename = body.get("filename", "upload.jpg")
        if not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="filename must be a string")
        return (
            decode_base64_image(body["imageBase64"]),
            filename,
            include_images_value(request, body.get("includeImages")),
            analyze_portraits_value(request, body.get("analyzePortraits")),
            requested_field_names(body.get("fields")),
        )
    raise HTTPException(
        status_code=415,
        detail="Use application/json or multipart/form-data",
    )


async def read_reference_metrics_request(request: Request) -> tuple[bytes, str, int]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = multipart_upload(form, "image")
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
        identifier = form.get("documentIdentifier")
        filename = upload.filename or "upload.jpg"
    elif content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict) or not isinstance(body.get("imageBase64"), str):
            raise HTTPException(status_code=400, detail="JSON field 'imageBase64' is required")
        payload = decode_base64_image(body["imageBase64"])
        identifier = body.get("documentIdentifier")
        filename = body.get("filename", "upload.jpg")
        if not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="filename must be a string")
    else:
        raise HTTPException(
            status_code=415,
            detail="Use application/json or multipart/form-data",
        )
    try:
        document_identifier = int(identifier)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail="documentIdentifier must be an integer"
        ) from error
    return payload, filename, document_identifier


def classification_top_k(value: Any) -> int:
    if value is None or value == "":
        return 5
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail="topK must be an integer")
    try:
        top_k = int(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="topK must be an integer") from error
    if not 1 <= top_k <= 25:
        raise HTTPException(status_code=422, detail="topK must be between 1 and 25")
    return top_k


async def read_classification_request(request: Request) -> tuple[bytes, str, int]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = multipart_upload(form, "image")
        return (
            await upload.read(MAX_UPLOAD_BYTES + 1),
            upload.filename or "upload.jpg",
            classification_top_k(form.get("topK")),
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict) or not isinstance(body.get("imageBase64"), str):
            raise HTTPException(status_code=400, detail="JSON field 'imageBase64' is required")
        filename = body.get("filename", "upload.jpg")
        if not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="filename must be a string")
        return (
            decode_base64_image(body["imageBase64"]),
            filename,
            classification_top_k(body.get("topK")),
        )
    raise HTTPException(
        status_code=415,
        detail="Use application/json or multipart/form-data",
    )


@lru_cache(maxsize=1)
def facial_models() -> tuple[FaceDetector, LandmarkQuality]:
    return FaceDetector(FACE_DETECTOR_PATH), LandmarkQuality(FACE_LANDMARKS_PATH)


@lru_cache(maxsize=1)
def liveness_model() -> PassiveLivenessEngine:
    return PassiveLivenessEngine(FACE_LIVENESS_PATH)


@lru_cache(maxsize=1)
def recognition_model() -> FaceRecognitionEngine:
    detector, landmarks = facial_models()
    return FaceRecognitionEngine(FACE_RECOGNITION_PATH, detector, landmarks)


def request_threshold(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="threshold must be numeric") from error
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    return threshold


def request_liveness_thresholds(
    threshold_value: Any,
    spoof_threshold_value: Any,
) -> tuple[float, float]:
    threshold = request_threshold(threshold_value, DEFAULT_LIVENESS_THRESHOLD)
    default_spoof_threshold = min(DEFAULT_LIVENESS_SPOOF_THRESHOLD, threshold)
    spoof_threshold = request_threshold(
        spoof_threshold_value,
        default_spoof_threshold,
    )
    if spoof_threshold > threshold:
        raise HTTPException(
            status_code=400,
            detail="spoofThreshold must be less than or equal to threshold",
        )
    return threshold, spoof_threshold


async def read_face_pair(
    request: Request,
) -> tuple[tuple[bytes, str], tuple[bytes, str], float, bool]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        first = multipart_upload(form, "firstImage")
        second = multipart_upload(form, "secondImage")
        return (
            (await first.read(MAX_UPLOAD_BYTES + 1), first.filename or "first.jpg"),
            (await second.read(MAX_UPLOAD_BYTES + 1), second.filename or "second.jpg"),
            request_threshold(form.get("threshold"), 0.67),
            include_templates_value(request, form.get("includeTemplates")),
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        first_value = body.get("firstImageBase64")
        second_value = body.get("secondImageBase64")
        if not isinstance(first_value, str) or not isinstance(second_value, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    "JSON fields 'firstImageBase64' and 'secondImageBase64' "
                    "are required"
                ),
            )
        first_name = body.get("firstFilename", "first.jpg")
        second_name = body.get("secondFilename", "second.jpg")
        if not isinstance(first_name, str) or not isinstance(second_name, str):
            raise HTTPException(status_code=400, detail="filenames must be strings")
        return (
            (decode_base64_image(first_value), first_name),
            (decode_base64_image(second_value), second_name),
            request_threshold(body.get("threshold"), 0.67),
            include_templates_value(request, body.get("includeTemplates")),
        )
    raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")


async def read_template_pair(request: Request) -> tuple[np.ndarray, np.ndarray, float]:
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        raise HTTPException(status_code=415, detail="Use application/json")
    try:
        body: Any = await request.json()
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from error
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    values = (body.get("template1Base64"), body.get("template2Base64"))
    if not all(isinstance(value, str) for value in values):
        raise HTTPException(
            status_code=400,
            detail="JSON fields 'template1Base64' and 'template2Base64' are required",
        )
    try:
        payloads = [
            base64.b64decode(
                value.split(",", 1)[1]
                if value.startswith("data:") and "," in value
                else value,
                validate=True,
            )
            for value in values
        ]
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="face template is not valid base64") from error
    try:
        templates = [decode_face_template(payload) for payload in payloads]
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return templates[0], templates[1], request_threshold(body.get("threshold"), 0.67)


async def read_document_pair(
    request: Request,
) -> tuple[
    tuple[bytes, str],
    tuple[bytes, str],
    str,
    bool,
    bool,
    int | None,
    int | None,
]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        front = multipart_upload(form, "frontImage")
        back = multipart_upload(form, "backImage")
        return (
            (await front.read(MAX_UPLOAD_BYTES + 1), front.filename or "front.jpg"),
            (await back.read(MAX_UPLOAD_BYTES + 1), back.filename or "back.jpg"),
            str(form.get("profile", "auto_research")),
            include_images_value(request, form.get("includeImages")),
            analyze_portraits_value(request, form.get("analyzePortraits")),
            optional_document_identifier(
                form.get("frontDocumentIdentifier"), "frontDocumentIdentifier"
            ),
            optional_document_identifier(
                form.get("backDocumentIdentifier"), "backDocumentIdentifier"
            ),
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        if not isinstance(body.get("frontImageBase64"), str) or not isinstance(
            body.get("backImageBase64"), str
        ):
            raise HTTPException(
                status_code=400,
                detail="JSON fields 'frontImageBase64' and 'backImageBase64' are required",
            )
        front_name = body.get("frontFilename", "front.jpg")
        back_name = body.get("backFilename", "back.jpg")
        profile = body.get("profile", "auto_research")
        if not all(isinstance(value, str) for value in (front_name, back_name, profile)):
            raise HTTPException(status_code=400, detail="filenames and profile must be strings")
        return (
            (decode_base64_image(body["frontImageBase64"]), front_name),
            (decode_base64_image(body["backImageBase64"]), back_name),
            profile,
            include_images_value(request, body.get("includeImages")),
            analyze_portraits_value(request, body.get("analyzePortraits")),
            optional_document_identifier(
                body.get("frontDocumentIdentifier"), "frontDocumentIdentifier"
            ),
            optional_document_identifier(
                body.get("backDocumentIdentifier"), "backDocumentIdentifier"
            ),
        )
    raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")


async def read_document_pages(
    request: Request,
) -> tuple[list[tuple[bytes, str]], str, bool, bool, list[int | None]]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploads = multipart_uploads(form, "images")
        if not 2 <= len(uploads) <= MAX_DOCUMENT_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Provide between 2 and {MAX_DOCUMENT_PAGES} document pages",
            )
        pages = [
            (await upload.read(MAX_UPLOAD_BYTES + 1), upload.filename or f"page-{index}.jpg")
            for index, upload in enumerate(uploads, 1)
        ]
        raw_identifiers = form.getlist("documentIdentifiers")
        if raw_identifiers and len(raw_identifiers) != len(pages):
            raise HTTPException(
                status_code=400,
                detail="documentIdentifiers must align with images",
            )
        identifiers = (
            [
                optional_document_identifier(
                    value, f"documentIdentifiers[{index}]"
                )
                for index, value in enumerate(raw_identifiers)
            ]
            if raw_identifiers
            else [None] * len(pages)
        )
        return (
            pages,
            str(form.get("profile", "auto_research")),
            include_images_value(request, form.get("includeImages")),
            analyze_portraits_value(request, form.get("analyzePortraits")),
            identifiers,
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict) or not isinstance(body.get("images"), list):
            raise HTTPException(status_code=400, detail="JSON field 'images' must be an array")
        if not 2 <= len(body["images"]) <= MAX_DOCUMENT_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Provide between 2 and {MAX_DOCUMENT_PAGES} document pages",
            )
        pages = []
        identifiers = []
        for index, item in enumerate(body["images"], 1):
            if not isinstance(item, dict) or not isinstance(item.get("imageBase64"), str):
                raise HTTPException(
                    status_code=400,
                    detail=f"images[{index - 1}].imageBase64 is required",
                )
            filename = item.get("filename", f"page-{index}.jpg")
            if not isinstance(filename, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"images[{index - 1}].filename must be a string",
                )
            pages.append((decode_base64_image(item["imageBase64"]), filename))
            identifiers.append(
                optional_document_identifier(
                    item.get("documentIdentifier"),
                    f"images[{index - 1}].documentIdentifier",
                )
            )
        profile = body.get("profile", "auto_research")
        if not isinstance(profile, str):
            raise HTTPException(status_code=400, detail="profile must be a string")
        return (
            pages,
            profile,
            include_images_value(request, body.get("includeImages")),
            analyze_portraits_value(request, body.get("analyzePortraits")),
            identifiers,
        )
    raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")


async def read_document_portrait_pair(
    request: Request,
) -> tuple[
    tuple[bytes, str],
    tuple[bytes, str],
    str,
    int | None,
    float,
    bool,
    float,
    float,
]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        document = multipart_upload(form, "documentImage")
        selfie = multipart_upload(form, "selfieImage")
        liveness_thresholds = request_liveness_thresholds(
            form.get("livenessThreshold"),
            form.get("livenessSpoofThreshold"),
        )
        return (
            (
                await document.read(MAX_UPLOAD_BYTES + 1),
                document.filename or "document.jpg",
            ),
            (await selfie.read(MAX_UPLOAD_BYTES + 1), selfie.filename or "selfie.jpg"),
            str(form.get("profile", "auto_research")),
            optional_document_identifier(
                form.get("documentIdentifier"), "documentIdentifier"
            ),
            request_threshold(form.get("threshold"), 0.67),
            analyze_liveness_value(request, form.get("analyzeLiveness")),
            *liveness_thresholds,
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        if not isinstance(body.get("documentImageBase64"), str) or not isinstance(
            body.get("selfieImageBase64"), str
        ):
            raise HTTPException(
                status_code=400,
                detail="JSON fields 'documentImageBase64' and 'selfieImageBase64' are required",
            )
        document_name = body.get("documentFilename", "document.jpg")
        selfie_name = body.get("selfieFilename", "selfie.jpg")
        profile = body.get("profile", "auto_research")
        if not all(
            isinstance(value, str)
            for value in (document_name, selfie_name, profile)
        ):
            raise HTTPException(
                status_code=400,
                detail="filenames and profile must be strings",
            )
        liveness_thresholds = request_liveness_thresholds(
            body.get("livenessThreshold"),
            body.get("livenessSpoofThreshold"),
        )
        return (
            (decode_base64_image(body["documentImageBase64"]), document_name),
            (decode_base64_image(body["selfieImageBase64"]), selfie_name),
            profile,
            optional_document_identifier(
                body.get("documentIdentifier"), "documentIdentifier"
            ),
            request_threshold(body.get("threshold"), 0.67),
            analyze_liveness_value(request, body.get("analyzeLiveness")),
            *liveness_thresholds,
        )
    raise HTTPException(
        status_code=415,
        detail="Use application/json or multipart/form-data",
    )


async def read_liveness_image(
    request: Request,
) -> tuple[bytes, str, float, float]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = multipart_upload(form, "image")
        thresholds = request_liveness_thresholds(
            form.get("threshold"),
            form.get("spoofThreshold"),
        )
        return (
            await upload.read(MAX_UPLOAD_BYTES + 1),
            upload.filename or "upload.jpg",
            *thresholds,
        )
    if content_type.startswith("application/json"):
        try:
            body: Any = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(body, dict) or not isinstance(body.get("imageBase64"), str):
            raise HTTPException(status_code=400, detail="JSON field 'imageBase64' is required")
        filename = body.get("filename", "upload.jpg")
        if not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="filename must be a string")
        thresholds = request_liveness_thresholds(
            body.get("threshold"),
            body.get("spoofThreshold"),
        )
        return (
            decode_base64_image(body["imageBase64"]),
            filename,
            *thresholds,
        )
    raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")


@app.get("/health")
def health() -> dict:
    try:
        manifest = validate_assets(ASSETS_PATH)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Assets unavailable: {error}") from error
    return success(
        {"service": "healthy", "assetFormatVersion": manifest["formatVersion"]}
    )


@app.get("/v1/capabilities")
def capabilities() -> dict:
    return success(
        {**SDK_COMPATIBILITY, "runtime": runtime_capabilities(ASSETS_PATH)},
        omit_empty=False,
    )


@app.get("/v1/document/catalog")
def document_catalog(
    q: str | None = None,
    countryCode: str | None = None,
    documentType: str | None = None,
    documentFormat: str | None = None,
    includeDeprecated: bool = False,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be zero or greater")
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    matches = search_document_catalog(
        ASSETS_PATH,
        query=q,
        country_code=countryCode,
        document_type=documentType,
        document_format=documentFormat,
        include_deprecated=includeDeprecated,
    )
    items = [
        document_catalog_item(document)
        for document in matches[offset : offset + limit]
    ]
    return success(
        {
            "total": len(matches),
            "offset": offset,
            "limit": limit,
            "items": items,
        },
        omit_empty=False,
    )


def document_catalog_item(document: dict) -> dict:
    identifier = document["identifier"]
    return {
        "identifier": identifier,
        "name": document.get("caption"),
        "country": document.get("country"),
        "countryCodes": document.get("isoCodes") or [],
        "type": (document.get("documentType") or {}).get("name"),
        "format": (document.get("documentFormat") or {}).get("name"),
        "edition": document.get("year"),
        "series": document.get("series"),
        "jurisdictionCodes": document.get("stateCodes") or [],
        "issuedFrom": document.get("issuedFrom"),
        "issuedTo": document.get("issuedTo"),
        "deprecated": bool(document.get("deprecated")),
        "hasMrz": bool((document.get("mrz") or {}).get("present")),
        "hasBarcode": bool((document.get("barcode") or {}).get("present")),
        "layoutAvailable": visual_layout(ASSETS_PATH, identifier) is not None,
        "pageRole": layout_page_role(ASSETS_PATH, identifier),
        "layoutEvidencePath": f"/v1/document/layout/{identifier}/evidence",
    }


@app.get("/v1/document/catalog/facets")
def document_catalog_filter_facets(includeDeprecated: bool = False) -> dict:
    return success(
        document_catalog_facets(
            ASSETS_PATH, include_deprecated=includeDeprecated
        ),
        omit_empty=False,
    )


@app.get("/v1/document/catalog/{document_identifier}")
def document_catalog_detail(document_identifier: int) -> dict:
    document = document_catalog_entry(ASSETS_PATH, document_identifier)
    if document is None:
        raise HTTPException(status_code=404, detail="Unknown documentIdentifier")
    return success(document_catalog_item(document), omit_empty=False)


@app.post("/v1/document/classify")
async def classify_document_image(request: Request) -> dict:
    payload, filename, top_k = await read_classification_request(request)
    validate_upload(payload, filename)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            classification = await run_in_threadpool(
                classify_document,
                ASSETS_PATH,
                image.convert("RGB"),
                top_k,
            )
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=422, detail=f"Image processing failed: {error}"
        ) from error
    candidates = []
    for candidate in classification["candidates"]:
        document = candidate.get("document")
        item = (
            document_catalog_item(document)
            if document
            else {
                "identifier": candidate["documentIdentifier"],
                "metadataAvailable": False,
            }
        )
        item["confidence"] = candidate["confidence"]
        item.setdefault("metadataAvailable", True)
        candidates.append(item)
    return success(
        {
            "classCount": classification["classCount"],
            "requestedCandidates": top_k,
            "returnedCandidates": len(candidates),
            "candidates": candidates,
        },
        omit_empty=False,
    )


@app.post("/v1/document/layout/{document_identifier}/ocr")
async def recognize_document_layout(
    document_identifier: int, request: Request
) -> dict:
    payload, filename, include_images, analyze_portraits, fields = (
        await read_layout_ocr_request(request)
    )
    layout = visual_layout(ASSETS_PATH, document_identifier)
    if layout is None:
        raise HTTPException(status_code=404, detail="Unknown documentIdentifier")
    unknown_fields = sorted((fields or set()) - layout_field_names(layout))
    if unknown_fields:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown layout fields: {', '.join(unknown_fields)}",
        )
    suffix = validate_upload(payload, filename)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
            image_file.write(payload)
            temporary_path = Path(image_file.name)
        result = await run_in_threadpool(
            process_document,
            temporary_path,
            ASSETS_PATH,
            "auto_research",
            document_identifier,
            fields,
        )
        if include_images:
            result["extractedImages"] = extract_document_images(
                result, {"document": payload}, "document"
            )
        if analyze_portraits:
            detector, _ = facial_models()
            await run_in_threadpool(
                analyze_document_portraits,
                result,
                {"document": payload},
                "document",
                detector,
            )
        return document_response(result)
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=422, detail=f"Image processing failed: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.get("/v1/document/layout/{document_identifier}/evidence")
def document_layout_evidence(document_identifier: int) -> dict:
    layout = visual_layout(ASSETS_PATH, document_identifier)
    if layout is None:
        raise HTTPException(status_code=404, detail="Unknown documentIdentifier")
    text = text_regions(layout)
    graphics = graphic_regions(layout)
    barcodes = barcode_regions(layout)
    regions = security_regions(layout)
    patches = reference_patches(layout)
    return success(
        {
            "documentIdentifier": document_identifier,
            "document": layout_descriptor(layout),
            "dimensionsMm": layout.get("dimensionsMm"),
            "relations": layout_relations(layout),
            "regions": {
                "text": {"count": len(text), "items": text},
                "graphics": {"count": len(graphics), "items": graphics},
                "barcodes": {"count": len(barcodes), "items": barcodes},
            },
            "securityRegions": {
                "count": len(regions),
                "items": regions,
            },
            "referencePatches": {
                "count": len(patches),
                "items": patches,
            },
            "declaredRequirements": declared_layout_requirements(layout),
            "authenticityDecision": None,
        },
        omit_empty=False,
    )


@app.post("/v1/document/reference-metrics")
async def document_reference_metrics(request: Request) -> dict:
    payload, filename, document_identifier = await read_reference_metrics_request(
        request
    )
    validate_upload(payload, filename)
    layout = visual_layout(ASSETS_PATH, document_identifier)
    if layout is None:
        raise HTTPException(status_code=422, detail="Unknown documentIdentifier")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            metrics = await run_in_threadpool(
                compare_layout_reference_patches, image.convert("RGB"), layout
            )
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=422, detail=f"Image processing failed: {error}"
        ) from error
    if not metrics:
        raise HTTPException(
            status_code=422,
            detail="The document layout has no visible-light reference patches",
        )
    return success(
        {
            "documentIdentifier": document_identifier,
            "status": "metric_only",
            "decision": None,
            "patchCount": len(metrics),
            "patches": metrics,
        },
        omit_empty=False,
    )


@app.post("/v1/ocr")
async def recognize(request: Request) -> dict:
    payload, filename, profile, include_images, analyze_portraits = await read_image(
        request
    )
    if profile not in SUPPORTED_REQUEST_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported profile; choose one of {sorted(SUPPORTED_REQUEST_PROFILES)}",
        )
    suffix = validate_upload(payload, filename)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
            image_file.write(payload)
            temporary_path = Path(image_file.name)
        result = await run_in_threadpool(
            process_document, temporary_path, ASSETS_PATH, profile
        )
        if include_images:
            result["extractedImages"] = extract_document_images(
                result, {"document": payload}, "document"
            )
        if analyze_portraits:
            detector, _ = facial_models()
            await run_in_threadpool(
                analyze_document_portraits,
                result,
                {"document": payload},
                "document",
                detector,
            )
        return document_response(result)
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/v1/ocr/pair")
async def recognize_pair(request: Request) -> dict:
    (
        front,
        back,
        profile,
        include_images,
        analyze_portraits,
        front_identifier,
        back_identifier,
    ) = await read_document_pair(request)
    if profile not in SUPPORTED_REQUEST_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported profile; choose one of {sorted(SUPPORTED_REQUEST_PROFILES)}",
        )
    if (front_identifier is not None or back_identifier is not None) and profile != "auto_research":
        raise HTTPException(
            status_code=422,
            detail="Per-side document identifiers require profile auto_research",
        )
    for side, identifier in (
        ("front", front_identifier),
        ("back", back_identifier),
    ):
        if identifier is not None and visual_layout(ASSETS_PATH, identifier) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown {side}DocumentIdentifier",
            )
    front_suffix = validate_upload(*front)
    back_suffix = validate_upload(*back)
    temporary_paths = []
    try:
        for payload, suffix in ((front[0], front_suffix), (back[0], back_suffix)):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
                image_file.write(payload)
                temporary_paths.append(Path(image_file.name))
        arguments = [
            temporary_paths[0],
            temporary_paths[1],
            ASSETS_PATH,
            profile,
        ]
        if front_identifier is not None or back_identifier is not None:
            arguments.extend([front_identifier, back_identifier])
        result = await run_in_threadpool(process_document_pair, *arguments)
        if include_images:
            result["extractedImages"] = extract_document_images(
                result,
                {"front": front[0], "back": back[0]},
                "front",
            )
        if analyze_portraits:
            detector, _ = facial_models()
            await run_in_threadpool(
                analyze_document_portraits,
                result,
                {"front": front[0], "back": back[0]},
                "front",
                detector,
            )
        return document_response(result)
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


@app.post("/v1/ocr/pages")
async def recognize_pages(request: Request) -> dict:
    pages, profile, include_images, analyze_portraits, identifiers = (
        await read_document_pages(request)
    )
    if profile not in SUPPORTED_REQUEST_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported profile; choose one of {sorted(SUPPORTED_REQUEST_PROFILES)}",
        )
    if any(identifier is not None for identifier in identifiers) and profile != "auto_research":
        raise HTTPException(
            status_code=422,
            detail="Per-page document identifiers require profile auto_research",
        )
    for index, identifier in enumerate(identifiers):
        if identifier is not None and visual_layout(ASSETS_PATH, identifier) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown images[{index}].documentIdentifier",
            )
    validated = [(payload, validate_upload(payload, filename)) for payload, filename in pages]
    temporary_paths = []
    try:
        for payload, suffix in validated:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
                image_file.write(payload)
                temporary_paths.append(Path(image_file.name))
        arguments = [temporary_paths, ASSETS_PATH, profile]
        if any(identifier is not None for identifier in identifiers):
            arguments.append(identifiers)
        result = await run_in_threadpool(process_document_pages, *arguments)
        if include_images:
            result["extractedImages"] = extract_document_images(
                result,
                {
                    f"page_{index}": payload
                    for index, (payload, _) in enumerate(pages, 1)
                },
                "page_1",
            )
        if analyze_portraits:
            detector, _ = facial_models()
            await run_in_threadpool(
                analyze_document_portraits,
                result,
                {
                    f"page_{index}": payload
                    for index, (payload, _) in enumerate(pages, 1)
                },
                "page_1",
                detector,
            )
        return document_response(result)
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


@app.post("/v1/face/analyze")
async def analyze_face(request: Request) -> dict:
    payload, filename, _, _, _ = await read_image(request)
    validate_upload(payload, filename)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            detector, landmarks = facial_models()
            result = await run_in_threadpool(
                analyze_faces, image.copy(), detector, landmarks
            )
            return face_analysis_response(result)
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error


@app.post("/v1/document/portrait/compare")
async def compare_document_portrait_to_selfie(request: Request) -> dict:
    (
        document,
        selfie,
        profile,
        document_identifier,
        threshold,
        analyze_liveness,
        liveness_threshold,
        liveness_spoof_threshold,
    ) = await read_document_portrait_pair(request)
    if profile not in SUPPORTED_REQUEST_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported profile; choose one of {sorted(SUPPORTED_REQUEST_PROFILES)}",
        )
    if document_identifier is not None:
        if profile != "auto_research":
            raise HTTPException(
                status_code=422,
                detail="documentIdentifier requires profile auto_research",
            )
        if visual_layout(ASSETS_PATH, document_identifier) is None:
            raise HTTPException(status_code=404, detail="Unknown documentIdentifier")
    document_suffix = validate_upload(*document)
    validate_upload(*selfie)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=document_suffix, delete=False
        ) as image_file:
            image_file.write(document[0])
            temporary_path = Path(image_file.name)
        arguments = [temporary_path, ASSETS_PATH, profile]
        if document_identifier is not None:
            arguments.append(document_identifier)
        document_result = await run_in_threadpool(process_document, *arguments)
        with Image.open(io.BytesIO(selfie[0])) as selfie_image:
            prepared_selfie = selfie_image.convert("RGB")
        result = await run_in_threadpool(
            compare_document_portrait,
            document_result,
            document[0],
            prepared_selfie,
            recognition_model(),
            threshold,
            liveness_model() if analyze_liveness else None,
            liveness_threshold,
            liveness_spoof_threshold,
        )
        return document_portrait_comparison_response(result)
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Image processing failed: {error}",
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/v1/face/liveness")
async def face_liveness(request: Request) -> dict:
    payload, filename, threshold, spoof_threshold = await read_liveness_image(request)
    validate_upload(payload, filename)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            prepared = image.convert("RGB")
        detector, landmarks = facial_models()
        detections = await run_in_threadpool(detector.detect, prepared)
        if not detections:
            raise ValueError("expected one face, found 0")
        detection = detections[0]
        landmark_result = await run_in_threadpool(
            landmarks.infer, prepared, detection
        )
        quality = assess_face_quality(
            detection,
            prepared.size,
            landmark_result["headPose"],
            len(detections),
            landmark_result.get("qualityScore"),
        )
        if not quality["livenessEligible"]:
            result = {
                "decision": "review",
                "threshold": threshold,
                "spoofThreshold": spoof_threshold,
            }
        else:
            result = await run_in_threadpool(
                liveness_model().infer,
                prepared,
                detection,
                threshold,
                spoof_threshold,
            )
        result["headPose"] = landmark_result["headPose"]
        result["quality"] = quality
        return liveness_response(result, list(detection.box))
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error


@app.post("/v1/face/compare")
async def compare_faces(request: Request) -> dict:
    first, second, threshold, include_templates = await read_face_pair(request)
    validate_upload(*first)
    validate_upload(*second)
    try:
        with Image.open(io.BytesIO(first[0])) as first_image:
            prepared_first = first_image.convert("RGB")
        with Image.open(io.BytesIO(second[0])) as second_image:
            prepared_second = second_image.convert("RGB")
        result = await run_in_threadpool(
            recognition_model().compare, prepared_first, prepared_second, threshold
        )
        if include_templates:
            result["templates"] = [
                {
                    "templateBase64": base64.b64encode(
                        encode_face_template(vector)
                    ).decode("ascii"),
                    "format": "float32-le",
                    "length": len(vector),
                    "byteLength": FACE_TEMPLATE_BYTES,
                }
                for vector in result["templateVectors"]
            ]
        return comparison_response(result)
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error


@app.post("/v1/face/template")
async def extract_face_template(request: Request) -> dict:
    payload, filename, _, _, _ = await read_image(request)
    validate_upload(payload, filename)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            result = await run_in_threadpool(
                recognition_model().embedding, image.convert("RGB")
            )
        encoded = encode_face_template(result["vector"])
        return face_template_response(
            {
                "templateBase64": base64.b64encode(encoded).decode("ascii"),
                "length": len(result["vector"]),
                "byteLength": FACE_TEMPLATE_BYTES,
                "faceBox": list(result["detection"].box),
            }
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {error}") from error


@app.post("/v1/face/template/compare")
async def compare_face_templates_endpoint(request: Request) -> dict:
    first, second, threshold = await read_template_pair(request)
    result = compare_face_templates(first, second, threshold)
    return comparison_response(result)


def main() -> None:
    uvicorn.run(
        "identity_analysis.api:app",
        host=os.environ.get(
            "IDENTITY_ANALYSIS_HOST", "0.0.0.0"
        ),
        port=int(
            os.environ.get(
                "IDENTITY_ANALYSIS_PORT", "8000"
            )
        ),
    )
