"""HTTP surface for facial analysis and document OCR.

Expectations come from running the endpoints against the synthetic fixtures in
``examples/samples``.  Note the passive-liveness endpoint correctly reports
``spoof`` for the procedurally drawn selfie: it is a rendering, not a live
capture.
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from identity_analysis.api import app
import identity_analysis.api as api_module
from identity_analysis.face_engines import FaceDetection

from conftest import (
    BACK_LAYOUT,
    FRONT_LAYOUT,
    ID_BACK,
    ID_FRONT,
    PASSPORT_1,
    SELFIE,
    requires_assets,
)


def assert_face_contract(result: dict) -> None:
    assert result["status"] == "ok"
    data = result["data"]
    assert data["faceCount"] == 1
    assert len(data["faces"][0]["landmarks"]) == 68
    assert set(data["faces"][0]) == {
        "confidence",
        "box",
        "landmarks",
        "headPose",
        "quality",
    }
    assert data["faces"][0]["quality"]["status"] == "pass"


def test_capabilities_distinguish_implemented_and_blocked_engines():
    with TestClient(app) as client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200
    lifecycle = response.json()["data"]["engineLifecycle"]
    assert lifecycle["faceLiveness"]["endpoint"] == "/v1/face/liveness"
    assert lifecycle["faceRecognition"]["endpoint"] == "/v1/face/compare"
    assert lifecycle["faceRecognition"]["templateEndpoint"] == "/v1/face/template"
    assert lifecycle["faceRecognition"]["templateComparisonEndpoint"] == (
        "/v1/face/template/compare"
    )
    assert lifecycle["documentLiveness"]["endpoint"] is None
    classification = response.json()["data"]["visualClassification"]
    assert classification["catalogHintSemantics"]["restExposed"] is True
    assert classification["layoutMetadata"]["status"] == "implemented"
    assert classification["documentCatalog"]["status"] == "implemented"
    assert classification["documentCatalog"]["facetsEndpoint"].endswith("/facets")
    assert classification["standaloneClassification"]["status"] == "implemented"
    assert classification["explicitLayoutRecognition"]["status"] == "implemented"
    assert classification["orientedFieldRecognition"]["status"] == "implemented"
    assert (
        classification["declaredFieldPreprocessing"]["status"]
        == "implemented_with_original_fallback"
    )
    assert (
        classification["barcodeFormatRouting"]["status"]
        == "implemented_with_multiformat_fallback"
    )
    assert (
        classification["fieldMetadataSemantics"]["status"]
        == "preserved_without_guessed_decisions"
    )
    assert classification["portraitFacePresence"]["status"] == "implemented_opt_in"
    assert classification["securityMetadataSemantics"]["restExposed"] is True
    assert classification["referencePatchMatching"]["restExposed"] is True
    recognition = response.json()["data"]["engineLifecycle"]["faceRecognition"]
    assert recognition["documentPortraitEndpoint"] == "/v1/document/portrait/compare"
    assert recognition["documentPortraitOptionalLiveness"] is True
    assert (
        recognition["documentPortraitVerificationPolicy"]["pass"]
        == "same_person_and_real_and_document_capture_pass"
    )
    runtime = response.json()["data"]["runtime"]
    assert runtime["documentation"] == "docs/models.md"
    assert isinstance(runtime["features"], dict)
    assert all(isinstance(value, bool) for value in runtime["features"].values())


def test_ocr_defaults_to_automatic_profile(monkeypatch):
    captured = {}

    def fake_process_document(image_path, assets_path, profile):
        captured["profile"] = profile
        return {"recognitionProfile": "TEST"}

    monkeypatch.setattr(api_module, "process_document", fake_process_document)
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            json={"filename": "document.jpg", "imageBase64": encoded},
        )

    assert response.status_code == 200
    assert captured["profile"] == "auto_research"


def test_api_reports_internal_processing_time() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200
    assert response.headers["server-timing"].startswith("app;dur=")
    assert float(response.headers["x-process-time-ms"]) >= 0


@requires_assets
def test_ocr_returns_minimal_standard_contract(assets):
    with TestClient(app) as client, ID_BACK.open("rb") as image:
        response = client.post(
            "/v1/ocr",
            files={"image": ("card.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["data"]["document"]["profile"] == "ICAO-TD1"
    assert result["data"]["holder"]["surname"] == "SPECIMEN"
    serialized = str(result)
    for internal_key in ("ContainerList", "fieldList", "recognizedFields", "sdkCompatibility"):
        assert internal_key not in serialized


def test_ocr_rejects_ambiguous_file_field():
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            "/v1/ocr",
            files={"file": ("card.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Multipart field 'image' is required"
    )


@requires_assets
def test_layout_ocr_returns_catalog_portrait_geometry_in_original_coordinates(
    catalog_api,
):
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            files={"image": ("card.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    regions = response.json()["data"]["regions"]
    assert regions["portrait"] == {
        "box": [0.055, 0.3, 0.3, 0.7],
        "coordinateSpace": "original_image",
        "faceExpected": True,
    }
    assert "ghostPortrait" in regions


@requires_assets
def test_layout_ocr_can_return_extracted_images(catalog_api) -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            files={
                "image": ("card.jpg", image, "image/jpeg"),
                "includeImages": (None, "true"),
            },
        )

    assert response.status_code == 200
    images = response.json()["data"]["images"]
    portrait = next(item for item in images if item["type"] == "portrait")
    assert portrait["side"] == "document"
    assert portrait["mediaType"] == "image/jpeg"
    decoded = Image.open(io.BytesIO(base64.b64decode(portrait["imageBase64"])))
    assert decoded.size == (portrait["width"], portrait["height"])
    assert portrait["width"] > 100
    assert portrait["height"] > 100


@requires_assets
def test_layout_ocr_returns_every_declared_graphic_region(catalog_api) -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            files={
                "image": ("card.jpg", image, "image/jpeg"),
                "includeImages": (None, "true"),
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["profile"] == f"CATALOG-P{FRONT_LAYOUT}"
    assert {item["type"] for item in data["images"]} == {
        "portrait",
        "ghostPortrait",
        "signature",
    }
    assert all(item["side"] == "document" for item in data["images"])
    assert all(item["width"] > 0 and item["height"] > 0 for item in data["images"])
    assert all(base64.b64decode(item["imageBase64"]) for item in data["images"])


@requires_assets
def test_layout_ocr_accepts_include_images_as_query_parameter(catalog_api) -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr?includeImages=true",
            files={"image": ("card.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    assert any(
        item["type"] == "portrait" for item in response.json()["data"]["images"]
    )


def test_ocr_rejects_invalid_include_images() -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            json={
                "filename": "card.jpg",
                "imageBase64": encoded,
                "includeImages": "sometimes",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "includeImages must be boolean"


def test_extracted_images_use_the_region_side_source() -> None:
    front_stream = io.BytesIO()
    back_stream = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(front_stream, format="PNG")
    Image.new("RGB", (200, 120), "blue").save(back_stream, format="PNG")
    result = {
        "visualRegions": {
            "portrait": {
                "box": [0.25, 0.25, 0.75, 0.75],
                "side": "back",
            }
        }
    }

    images = api_module.extract_document_images(
        result,
        {"front": front_stream.getvalue(), "back": back_stream.getvalue()},
        "front",
    )

    assert images[0]["side"] == "back"
    assert images[0]["width"] == 100
    assert images[0]["height"] == 60
    crop = Image.open(io.BytesIO(base64.b64decode(images[0]["imageBase64"])))
    assert crop.getpixel((50, 30))[2] > 240


@requires_assets
def test_layout_ocr_can_analyze_declared_portrait_presence(
    monkeypatch, catalog_api
) -> None:
    class Detector:
        def detect(self, image, *args, **kwargs):
            assert image.width > 0
            assert image.height > 0
            return [FaceDetection((0.1, 0.1, 0.9, 0.9), 0.97)]

    monkeypatch.setattr(api_module, "facial_models", lambda: (Detector(), None))
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            files={
                "image": ("card.jpg", image, "image/jpeg"),
                "analyzePortraits": (None, "true"),
            },
        )

    assert response.status_code == 200
    presence = response.json()["data"]["regions"]["portrait"]["facePresence"]
    assert presence == {
        "expected": True,
        "detected": True,
        "count": 1,
        "threshold": 0.2,
        "confidence": 0.97,
        "status": "pass",
    }
    assert response.json()["data"]["validation"]["livenessDecision"] == "not_available"


def test_ocr_rejects_invalid_analyze_portraits() -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            json={
                "filename": "card.jpg",
                "imageBase64": encoded,
                "analyzePortraits": "sometimes",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "analyzePortraits must be boolean"


@requires_assets
def test_ocr_pair_fuses_front_and_back_multipart(catalog_api):
    with (
        TestClient(app) as client,
        ID_FRONT.open("rb") as front,
        ID_BACK.open("rb") as back,
    ):
        response = client.post(
            "/v1/ocr/pair",
            files={
                "frontImage": ("front.jpg", front, "image/jpeg"),
                "backImage": ("back.jpg", back, "image/jpeg"),
                "frontDocumentIdentifier": (None, str(FRONT_LAYOUT)),
                "backDocumentIdentifier": (None, str(BACK_LAYOUT)),
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["profile"] == "PAIRED-DOCUMENT"
    assert data["document"]["number"] == "ZZ7654321"
    assert data["holder"]["givenNames"] == "ALEX TAYLOR"
    assert data["pairing"]["decision"] == "matched"
    assert data["machineReadable"]["type"] == "TD1"
    assert data["pairing"]["expectedRelatedDocuments"] == [
        {"name": "Specimen Identity Card (2024) Back"}
    ]
    assert data["pairing"]["relatedSideBarcodeCount"] == 0
    assert data["regions"]["portrait"]["side"] == "front"


def test_ocr_pair_accepts_prefixed_and_raw_base64(monkeypatch):
    captured = {}

    def fake_process_pair(front_path, back_path, assets_path, profile):
        captured["profile"] = profile
        captured["front"] = front_path.read_bytes()
        captured["back"] = back_path.read_bytes()
        return {"recognitionProfile": "PAIRED-DOCUMENT", "pairing": {"decision": "review"}}

    monkeypatch.setattr(api_module, "process_document_pair", fake_process_pair)
    front = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    back = base64.b64encode(ID_BACK.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/pair",
            json={
                "frontFilename": "front.jpg",
                "backFilename": "back.jpg",
                "frontImageBase64": f"data:image/jpeg;base64,{front}",
                "backImageBase64": back,
            },
        )

    assert response.status_code == 200
    assert captured["profile"] == "auto_research"
    assert captured["front"] == ID_FRONT.read_bytes()
    assert captured["back"] == ID_BACK.read_bytes()


@requires_assets
def test_ocr_pair_accepts_independent_layout_identifiers(catalog_api):
    with (
        TestClient(app) as client,
        ID_FRONT.open("rb") as front,
        ID_BACK.open("rb") as back,
    ):
        response = client.post(
            "/v1/ocr/pair",
            files={
                "frontImage": ("front.jpg", front, "image/jpeg"),
                "backImage": ("back.jpg", back, "image/jpeg"),
                "frontDocumentIdentifier": (None, str(FRONT_LAYOUT)),
                "backDocumentIdentifier": (None, str(BACK_LAYOUT)),
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["pairing"]["decision"] == "matched"
    assert [side["layoutIdentifier"] for side in data["pairing"]["sides"]] == [
        FRONT_LAYOUT,
        BACK_LAYOUT,
    ]
    assert all("classification" not in side for side in data["pairing"]["sides"])


def test_ocr_pair_rejects_unknown_or_incompatible_layout_identifier(catalog_api):
    front = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    back = base64.b64encode(
        ID_BACK.read_bytes()
    ).decode("ascii")
    with TestClient(app) as client:
        unknown = client.post(
            "/v1/ocr/pair",
            json={
                "frontImageBase64": front,
                "backImageBase64": back,
                "frontDocumentIdentifier": 999999999,
            },
        )
        incompatible_profile = client.post(
            "/v1/ocr/pair",
            json={
                "frontImageBase64": front,
                "backImageBase64": back,
                "profile": "mex_ine",
                "frontDocumentIdentifier": FRONT_LAYOUT,
            },
        )

    assert unknown.status_code == 404
    assert incompatible_profile.status_code == 422


def test_ocr_pages_accepts_repeated_multipart_files(monkeypatch):
    captured = {}

    def fake_process_pages(paths, assets_path, profile):
        captured["profile"] = profile
        captured["payloads"] = [path.read_bytes() for path in paths]
        return {
            "recognitionProfile": "MULTI-PAGE-DOCUMENT",
            "pageProcessing": {
                "decision": "matched",
                "pageCount": len(paths),
                "matchedPageCount": len(paths),
            },
        }

    monkeypatch.setattr(api_module, "process_document_pages", fake_process_pages)
    front_payload = ID_FRONT.read_bytes()
    back_payload = ID_BACK.read_bytes()
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/pages",
            files=[
                ("images", ("page-1.jpg", front_payload, "image/jpeg")),
                ("images", ("page-2.jpg", back_payload, "image/jpeg")),
                ("images", ("page-3.jpg", back_payload, "image/jpeg")),
            ],
        )

    assert response.status_code == 200
    assert captured["profile"] == "auto_research"
    assert captured["payloads"] == [front_payload, back_payload, back_payload]
    assert response.json()["data"]["pages"]["pageCount"] == 3


def test_ocr_pages_accepts_json_image_array(monkeypatch):
    captured = {}

    def fake_process_pages(paths, assets_path, profile):
        captured["count"] = len(paths)
        return {
            "recognitionProfile": "MULTI-PAGE-DOCUMENT",
            "pageProcessing": {"decision": "review", "pageCount": len(paths)},
        }

    monkeypatch.setattr(api_module, "process_document_pages", fake_process_pages)
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/pages",
            json={
                "images": [
                    {"filename": "one.jpg", "imageBase64": encoded},
                    {
                        "filename": "two.jpg",
                        "imageBase64": f"data:image/jpeg;base64,{encoded}",
                    },
                ]
            },
        )

    assert response.status_code == 200
    assert captured["count"] == 2
    assert response.json()["data"]["pages"]["decision"] == "review"


def test_ocr_pages_accepts_per_page_layout_identifiers(monkeypatch, catalog_api):
    captured = {}

    def fake_process_pages(paths, assets_path, profile, identifiers):
        captured["identifiers"] = identifiers
        return {
            "recognitionProfile": "MULTI-PAGE-DOCUMENT",
            "pageProcessing": {
                "decision": "matched",
                "pageCount": len(paths),
                "matchedPageCount": len(paths),
                "pages": [
                    {
                        "side": f"page_{index}",
                        "recognized": True,
                        "profile": "EXACT",
                        "layoutIdentifier": identifier,
                    }
                    for index, identifier in enumerate(identifiers, 1)
                ],
            },
        }

    monkeypatch.setattr(api_module, "process_document_pages", fake_process_pages)
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/pages",
            json={
                "images": [
                    {
                        "imageBase64": encoded,
                        "documentIdentifier": FRONT_LAYOUT,
                    },
                    {
                        "imageBase64": encoded,
                        "documentIdentifier": BACK_LAYOUT,
                    },
                ]
            },
        )

    assert response.status_code == 200
    assert captured["identifiers"] == [FRONT_LAYOUT, BACK_LAYOUT]
    assert [
        page["layoutIdentifier"]
        for page in response.json()["data"]["pages"]["items"]
    ] == [FRONT_LAYOUT, BACK_LAYOUT]


def test_ocr_pages_validates_identifier_alignment_and_values(monkeypatch, catalog_api):
    def unexpected_processing(*args, **kwargs):
        raise AssertionError("invalid requests must fail before processing")

    monkeypatch.setattr(api_module, "process_document_pages", unexpected_processing)
    payload = ID_FRONT.read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    with TestClient(app) as client:
        misaligned = client.post(
            "/v1/ocr/pages",
            files=[
                ("images", ("one.jpg", payload, "image/jpeg")),
                ("images", ("two.jpg", payload, "image/jpeg")),
                ("documentIdentifiers", (None, str(FRONT_LAYOUT))),
            ],
        )
        unknown = client.post(
            "/v1/ocr/pages",
            json={
                "images": [
                    {"imageBase64": encoded, "documentIdentifier": 999999999},
                    {"imageBase64": encoded},
                ]
            },
        )
        incompatible_profile = client.post(
            "/v1/ocr/pages",
            json={
                "profile": "mex_ine",
                "images": [
                    {
                        "imageBase64": encoded,
                        "documentIdentifier": FRONT_LAYOUT,
                    },
                    {"imageBase64": encoded},
                ],
            },
        )

    assert misaligned.status_code == 400
    assert unknown.status_code == 404
    assert incompatible_profile.status_code == 422


def test_api_errors_use_standard_contract():
    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr",
            json={"filename": "document.jpg", "imageBase64": "not-base64"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "error": {
            "code": "invalid_request",
            "message": "imageBase64 is not valid base64",
        },
    }


@requires_assets
def test_face_analyze_accepts_multipart_file():
    with TestClient(app) as client, SELFIE.open("rb") as image:
        response = client.post(
            "/v1/face/analyze",
            files={"image": ("selfie.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    assert_face_contract(response.json())


@requires_assets
def test_face_analyze_accepts_prefixed_base64():
    encoded = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/analyze",
            json={
                "filename": "selfie.jpg",
                "imageBase64": f"data:image/jpeg;base64,{encoded}",
            },
        )

    assert response.status_code == 200
    assert_face_contract(response.json())


@requires_assets
def test_face_liveness_accepts_raw_base64():
    encoded = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/liveness",
            json={"filename": "selfie.jpg", "imageBase64": encoded},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    # A procedurally drawn face is not a live capture, and passive PAD says so.
    assert result["data"]["decision"] == "spoof"
    assert result["data"]["score"] < 0.25
    assert set(result["data"]) == {
        "decision",
        "score",
        "threshold",
        "spoofThreshold",
        "face",
        "headPose",
        "quality",
    }
    assert result["data"]["threshold"] == 0.37
    assert result["data"]["spoofThreshold"] == 0.25


@requires_assets
def test_face_liveness_accepts_multipart_file():
    with TestClient(app) as client, SELFIE.open("rb") as image:
        response = client.post(
            "/v1/face/liveness",
            files={"image": ("selfie.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["data"]["decision"] == "spoof"


def test_face_liveness_rejects_inverted_review_band():
    encoded = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/liveness",
            json={
                "filename": "selfie.jpg",
                "imageBase64": encoded,
                "threshold": 0.25,
                "spoofThreshold": 0.37,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "spoofThreshold must be less than or equal to threshold"
    )


@requires_assets
def test_face_liveness_preserves_legacy_zero_threshold():
    encoded = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/liveness",
            json={
                "filename": "selfie.jpg",
                "imageBase64": encoded,
                "threshold": 0.0,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["threshold"] == 0.0
    assert response.json()["data"]["spoofThreshold"] == 0.0


def test_face_liveness_returns_review_before_pad_for_multiple_faces(monkeypatch):
    class Detector:
        def detect(self, image):
            return [
                FaceDetection((0.2, 0.2, 0.7, 0.8), 0.99),
                FaceDetection((0.1, 0.1, 0.3, 0.4), 0.95),
            ]

    class Landmarks:
        def infer(self, image, detection):
            return {"headPose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}}

    monkeypatch.setattr(api_module, "facial_models", lambda: (Detector(), Landmarks()))
    monkeypatch.setattr(
        api_module,
        "liveness_model",
        lambda: (_ for _ in ()).throw(AssertionError("PAD must not run")),
    )
    encoded = base64.b64encode(SELFIE.read_bytes()).decode("ascii")

    with TestClient(app) as client:
        response = client.post(
            "/v1/face/liveness",
            json={"filename": "selfie.jpg", "imageBase64": encoded},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["decision"] == "review"
    assert "score" not in data
    assert data["quality"]["warnings"] == [
        {"code": "MULTIPLE_FACES", "legacyCode": -500}
    ]


def test_face_liveness_runs_pad_for_soft_quality_warnings(monkeypatch):
    class Detector:
        def detect(self, image):
            return [FaceDetection((0.2, 0.2, 0.7, 0.8), 0.99)]

    class Landmarks:
        def infer(self, image, detection):
            return {
                "headPose": {"yaw": 36.0, "pitch": 0.0, "roll": 0.0},
                "qualityScore": 0.9,
            }

    class Liveness:
        def infer(self, image, detection, threshold, spoof_threshold):
            return {
                "decision": "real",
                "score": 0.9,
                "threshold": threshold,
                "spoofThreshold": spoof_threshold,
            }

    monkeypatch.setattr(api_module, "facial_models", lambda: (Detector(), Landmarks()))
    monkeypatch.setattr(api_module, "liveness_model", lambda: Liveness())
    encoded = base64.b64encode(SELFIE.read_bytes()).decode("ascii")

    with TestClient(app) as client:
        response = client.post(
            "/v1/face/liveness",
            json={"filename": "selfie.jpg", "imageBase64": encoded},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["decision"] == "real"
    assert data["score"] == 0.9
    assert data["quality"]["status"] == "review"
    assert data["quality"]["warnings"] == [
        {"code": "EXCESSIVE_POSE", "legacyCode": -300},
        {"code": "COVERED_FACE", "legacyCode": -400},
    ]


@requires_assets
def test_face_compare_accepts_multipart_files():
    with (
        TestClient(app) as client,
        SELFIE.open("rb") as selfie,
        ID_FRONT.open("rb") as identity_document,
    ):
        response = client.post(
            "/v1/face/compare",
            files={
                "firstImage": ("selfie.jpg", selfie, "image/jpeg"),
                "secondImage": ("identity.jpg", identity_document, "image/jpeg"),
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["data"]["decision"] == "same_person"
    assert result["data"]["score"] > result["data"]["threshold"]
    assert set(result["data"]) == {"decision", "score", "threshold", "faces"}


@requires_assets
def test_document_portrait_compare_uses_declared_crop(catalog_api) -> None:
    with (
        TestClient(app) as client,
        ID_FRONT.open("rb") as document,
        SELFIE.open("rb") as selfie,
    ):
        response = client.post(
            "/v1/document/portrait/compare",
            files={
                "documentImage": ("document.jpg", document, "image/jpeg"),
                "selfieImage": ("selfie.jpg", selfie, "image/jpeg"),
                "documentIdentifier": (None, str(FRONT_LAYOUT)),
                "analyzeLiveness": (None, "true"),
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["decision"] == "same_person"
    assert data["score"] > data["threshold"]
    assert data["verification"]["checks"]["portraitMatch"] == "same_person"
    assert data["verification"]["scope"] == (
        "portrait_identity_selfie_liveness_and_document_capture"
    )
    assert data["document"]["layoutIdentifier"] == FRONT_LAYOUT
    assert data["document"]["portrait"]["box"] == [0.055, 0.3, 0.3, 0.7]
    assert data["document"]["portrait"]["detectionThreshold"] == 0.2
    assert data["document"]["portrait"]["detectionConfidence"] > 0.2
    assert len(data["selfie"]["faceBox"]) == 4
    assert data["selfie"]["liveness"]["quality"]["status"] == "pass"


@requires_assets
def test_document_portrait_compare_rejects_unknown_layout(catalog_api) -> None:
    encoded_document = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    encoded_selfie = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/document/portrait/compare",
            json={
                "documentImageBase64": encoded_document,
                "selfieImageBase64": f"data:image/jpeg;base64,{encoded_selfie}",
                "documentIdentifier": 999999999,
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Unknown documentIdentifier"


@requires_assets
def test_document_portrait_compare_rejects_invalid_liveness_option() -> None:
    encoded_document = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    encoded_selfie = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/document/portrait/compare",
            json={
                "documentImageBase64": encoded_document,
                "selfieImageBase64": encoded_selfie,
                "analyzeLiveness": "sometimes",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "analyzeLiveness must be boolean"


@requires_assets
def test_face_compare_accepts_prefixed_and_raw_base64():
    selfie = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    identity_document = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/compare",
            json={
                "firstFilename": "selfie.jpg",
                "secondFilename": "identity.jpg",
                "firstImageBase64": f"data:image/jpeg;base64,{selfie}",
                "secondImageBase64": identity_document,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["decision"] == "same_person"


@requires_assets
def test_face_compare_can_return_templates_without_reinference() -> None:
    selfie = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    identity_document = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/compare?includeTemplates=true",
            json={
                "firstFilename": "selfie.jpg",
                "secondFilename": "identity.jpg",
                "firstImageBase64": selfie,
                "secondImageBase64": identity_document,
            },
        )

    assert response.status_code == 200
    templates = response.json()["data"]["templates"]
    assert len(templates) == 2
    for template in templates:
        assert template["format"] == "float32-le"
        assert template["length"] == 512
        assert template["byteLength"] == 2048
        assert len(base64.b64decode(template["templateBase64"])) == 2048


def test_face_compare_rejects_invalid_include_templates() -> None:
    selfie = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/compare",
            json={
                "firstImageBase64": selfie,
                "secondImageBase64": selfie,
                "includeTemplates": "sometimes",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "includeTemplates must be boolean"


@requires_assets
def test_face_template_can_be_extracted_and_compared() -> None:
    encoded_image = base64.b64encode(SELFIE.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        extraction = client.post(
            "/v1/face/template",
            json={"filename": "selfie.jpg", "imageBase64": encoded_image},
        )
        template = extraction.json()["data"]["templateBase64"]
        comparison = client.post(
            "/v1/face/template/compare",
            json={
                "template1Base64": template,
                "template2Base64": f"data:application/octet-stream;base64,{template}",
            },
        )

    assert extraction.status_code == 200
    extracted = extraction.json()["data"]
    assert extracted["format"] == "float32-le"
    assert extracted["length"] == 512
    assert extracted["byteLength"] == 2048
    assert len(base64.b64decode(template)) == 2048
    assert comparison.status_code == 200
    compared = comparison.json()["data"]
    assert compared["decision"] == "same_person"
    assert compared["score"] == pytest.approx(1.0)
    assert compared["threshold"] == 0.67


def test_face_template_comparison_rejects_invalid_length() -> None:
    invalid = base64.b64encode(b"short").decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/face/template/compare",
            json={"template1Base64": invalid, "template2Base64": invalid},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "face template must contain exactly 2048 bytes"
    )
