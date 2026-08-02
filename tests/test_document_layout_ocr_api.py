"""Explicit-layout OCR, driven by the synthetic catalog and sample card."""

import base64

import pytest
from fastapi.testclient import TestClient

import identity_analysis.pipeline as pipeline
from identity_analysis.api import app
from identity_analysis.pipeline import process_document

from conftest import FRONT_LAYOUT, ID_FRONT


pytestmark = pytest.mark.usefixtures("catalog_api")


def test_explicit_layout_pipeline_skips_classifier(monkeypatch, catalog_assets) -> None:
    def unexpected_classification(*args, **kwargs):
        raise AssertionError("classifier must not run for an explicit layout")

    monkeypatch.setattr(pipeline, "classify_document", unexpected_classification)
    result = process_document(
        ID_FRONT,
        catalog_assets,
        "auto_research",
        FRONT_LAYOUT,
    )

    assert result["recognitionProfile"] == f"CATALOG-P{FRONT_LAYOUT}"
    assert result["recognitionProfileStatus"] == "selected_by_explicit_layout"
    assert result["requestedDocumentIdentifier"] == FRONT_LAYOUT
    assert result["surname"] == "SPECIMEN"
    assert "documentClassification" not in result


def test_explicit_layout_uses_declarative_fallback(monkeypatch, catalog_assets) -> None:
    captured = {}

    def fake_catalog_result(resource, image, candidate, layout):
        captured["identifier"] = candidate["documentIdentifier"]
        result = pipeline.unsupported_document_result()
        result["DocumentName"] = candidate["document"]["caption"]
        result["recognitionProfile"] = "CATALOG-TEST"
        return result

    monkeypatch.setattr(pipeline, "classifier_layout_hint", lambda value: None)
    monkeypatch.setattr(pipeline, "classifier_mrz_hint", lambda value: None)
    monkeypatch.setattr(pipeline, "catalog_visual_result", fake_catalog_result)
    result = process_document(
        ID_FRONT,
        catalog_assets,
        "auto_research",
        FRONT_LAYOUT,
    )

    assert captured["identifier"] == FRONT_LAYOUT
    assert result["recognitionProfile"] == "CATALOG-TEST"
    assert result["recognitionProfileStatus"] == "selected_by_explicit_layout"


def test_selective_layout_ocr_skips_machine_readable_routes(
    monkeypatch, catalog_assets
) -> None:
    def unexpected_mrz(*args, **kwargs):
        raise AssertionError("MRZ recognition must not run for selective visual OCR")

    monkeypatch.setattr(pipeline, "classifier_mrz_hint", lambda value: "td3")
    monkeypatch.setattr(pipeline, "recognize_best_two_line_mrz", unexpected_mrz)
    result = process_document(
        ID_FRONT,
        catalog_assets,
        "auto_research",
        FRONT_LAYOUT,
        {"dateOfBirth"},
    )

    assert result["dateOfBirth"] == "01/01/1990"
    assert result["requestedFields"] == ["dateOfBirth"]


def test_explicit_layout_ocr_endpoint_returns_normalized_result() -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            files={"image": ("card.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["recognized"] is True
    assert data["document"]["layoutIdentifier"] == FRONT_LAYOUT
    assert data["document"]["profile"] == f"CATALOG-P{FRONT_LAYOUT}"
    assert data["document"]["name"] == "Specimen Identity Card (2024) Front"
    assert data["holder"]["surname"] == "SPECIMEN"
    assert data["holder"]["givenNames"] == "ALEX TAYLOR"
    assert "classification" not in data["document"]


def test_explicit_layout_ocr_accepts_base64_and_returns_images() -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            json={
                "filename": "card.jpg",
                "imageBase64": f"data:image/jpeg;base64,{encoded}",
                "includeImages": True,
            },
        )

    assert response.status_code == 200
    images = response.json()["data"]["images"]
    assert {image["type"] for image in images} == {
        "portrait",
        "ghostPortrait",
        "signature",
    }


def test_explicit_layout_ocr_selects_requested_fields() -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            json={
                "filename": "card.jpg",
                "imageBase64": encoded,
                "fields": ["dateOfBirth"],
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["requestedFields"] == ["dateOfBirth"]
    assert data["dates"]["birth"] == "01/01/1990"
    assert "holder" not in data
    assert "address" not in data


def test_explicit_layout_ocr_accepts_repeated_and_comma_separated_fields() -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            files=[
                ("image", ("card.jpg", image, "image/jpeg")),
                ("fields", (None, "dateOfBirth,documentNumber")),
                ("fields", (None, "dateOfExpiry")),
            ],
        )

    assert response.status_code == 200
    assert response.json()["data"]["document"]["requestedFields"] == [
        "dateOfBirth",
        "dateOfExpiry",
        "documentNumber",
    ]


def test_explicit_layout_ocr_rejects_unknown_fields() -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            f"/v1/document/layout/{FRONT_LAYOUT}/ocr",
            json={"imageBase64": encoded, "fields": ["notAField"]},
        )

    assert response.status_code == 422
    assert "notAField" in response.json()["error"]["message"]


def test_explicit_layout_ocr_rejects_unknown_identifier() -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            "/v1/document/layout/999999999/ocr",
            files={"image": ("card.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 404
