import base64
import hashlib
import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from identity_analysis.api import app
from identity_analysis.reference_matching import (
    compare_layout_reference_patches,
    compare_reference_patch,
    decode_reference_image,
)
from identity_analysis.visual_layouts import reference_patches, visual_layout

from conftest import BACK_LAYOUT, FIXTURE_CATALOG, FRONT_LAYOUT, ID_FRONT


def encoded_patch(image: Image.Image, bounds: list[float]) -> dict:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return {
        "number": 1,
        "bounds": bounds,
        "lightType": 6,
        "image": {
            "format": ".PNG",
            "dpi": 300,
            "data": encoded,
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        },
    }


def synthetic_document() -> tuple[Image.Image, dict]:
    document = Image.new("L", (200, 120), 230)
    draw = ImageDraw.Draw(document)
    draw.rectangle((60, 30, 119, 69), fill=40)
    draw.line((63, 34, 115, 64), fill=220, width=4)
    draw.ellipse((82, 40, 98, 56), fill=120)
    reference = document.crop((60, 30, 120, 70))
    return document, encoded_patch(reference, [0.3, 0.25, 0.6, 7 / 12])


def test_reference_patch_exact_region_scores_high() -> None:
    document, patch = synthetic_document()

    result = compare_reference_patch(document, patch)

    assert result["status"] == "experimental_metric_only"
    assert result["evaluatedCandidates"] == 75
    assert result["experimentalScore"] > 0.99
    assert result["scale"] == 1.0
    assert result["offset"] == {"x": 0.0, "y": 0.0}


def test_reference_patch_detects_visible_change() -> None:
    document, patch = synthetic_document()
    altered = document.copy()
    ImageDraw.Draw(altered).rectangle((60, 30, 119, 69), fill=230)

    exact = compare_reference_patch(document, patch)
    changed = compare_reference_patch(altered, patch)

    assert changed["experimentalScore"] < exact["experimentalScore"] - 0.15


def test_catalog_reference_payload_decodes_and_matches_hash() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)
    patch = reference_patches(layout, include_image_data=True)[0]

    image = decode_reference_image(patch)

    assert image.size == (48, 24)
    assert image.mode == "L"


def test_layout_matching_filters_non_visible_lights() -> None:
    document, visible = synthetic_document()
    infrared = {**visible, "number": 2, "lightType": 128}

    results = compare_layout_reference_patches(
        document, {"referencePatches": [visible, infrared]}
    )

    assert [result["referenceNumber"] for result in results] == [1]


def test_reference_metrics_endpoint_returns_metrics_without_decision(
    catalog_only_api,
) -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            "/v1/document/reference-metrics",
            files={
                "image": ("card.jpg", image, "image/jpeg"),
                "documentIdentifier": (None, str(FRONT_LAYOUT)),
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "metric_only"
    assert data["decision"] is None
    assert data["patchCount"] == len(data["patches"])
    assert data["patchCount"] > 0


def test_reference_metrics_endpoint_requires_exact_layout(catalog_only_api) -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            "/v1/document/reference-metrics",
            files={
                "image": ("card.jpg", image, "image/jpeg"),
                "documentIdentifier": (None, "999999999"),
            },
        )

    assert response.status_code == 422


def test_reference_metrics_endpoint_accepts_prefixed_base64(catalog_only_api) -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/document/reference-metrics",
            json={
                "filename": "card.jpg",
                "imageBase64": f"data:image/jpeg;base64,{encoded}",
                "documentIdentifier": FRONT_LAYOUT,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["decision"] is None


def test_layout_evidence_endpoint_returns_metadata_without_image_payloads(
    catalog_only_api,
) -> None:
    with TestClient(app) as client:
        response = client.get(f"/v1/document/layout/{FRONT_LAYOUT}/evidence")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["documentIdentifier"] == FRONT_LAYOUT
    assert data["document"]["name"] == "Specimen Identity Card (2024) Front"
    assert data["document"]["type"] == "IdentityCard"
    assert data["relations"]["childIdentifiers"] == [BACK_LAYOUT]
    assert data["regions"]["text"]["count"] == len(data["regions"]["text"]["items"])
    assert data["regions"]["graphics"]["count"] == len(
        data["regions"]["graphics"]["items"]
    )
    assert data["regions"]["barcodes"]["count"] == len(
        data["regions"]["barcodes"]["items"]
    )
    assert data["regions"]["text"]["count"] > 0
    text_region = data["regions"]["text"]["items"][0]
    assert isinstance(text_region["comparisonMode"], int)
    assert text_region["usedForComparison"] is bool(text_region["comparisonMode"])
    assert "colorType" in text_region
    assert "fontLayer" in text_region
    assert "backgroundRemoval" in text_region
    assert data["regions"]["graphics"]["count"] > 0
    assert data["securityRegions"]["count"] == len(
        data["securityRegions"]["items"]
    )
    assert data["referencePatches"]["count"] == len(
        data["referencePatches"]["items"]
    )
    assert data["securityRegions"]["count"] > 0
    assert data["referencePatches"]["count"] > 0
    assert data["authenticityDecision"] is None
    assert data["declaredRequirements"]["capture"]["requiredLightMask"] == 6
    assert data["declaredRequirements"]["electronicDocument"]["chipPage"] == 0
    assert all(
        "data" not in patch["image"]
        for patch in data["referencePatches"]["items"]
    )


def test_layout_evidence_endpoint_exposes_declared_barcode_regions(
    catalog_only_api,
) -> None:
    with TestClient(app) as client:
        response = client.get(f"/v1/document/layout/{BACK_LAYOUT}/evidence")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["regions"]["barcodes"]["count"] == 3
    assert {item["codeType"] for item in data["regions"]["barcodes"]["items"]} == {
        1,
        14,
        99,
    }


def test_layout_evidence_endpoint_rejects_unknown_identifier(catalog_only_api) -> None:
    with TestClient(app) as client:
        response = client.get("/v1/document/layout/999999999/evidence")

    assert response.status_code == 404
