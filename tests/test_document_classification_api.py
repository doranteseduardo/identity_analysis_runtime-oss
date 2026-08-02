import base64

from fastapi.testclient import TestClient

import identity_analysis.api as api_module
from identity_analysis.api import app

from conftest import ID_FRONT, requires_assets


@requires_assets
def test_classification_endpoint_returns_ranked_normalized_candidates(assets) -> None:
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            "/v1/document/classify",
            files={
                "image": ("card.jpg", image, "image/jpeg"),
                "topK": (None, "3"),
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["classCount"] > 0
    assert data["requestedCandidates"] == 3
    assert data["returnedCandidates"] == 3
    assert len(data["candidates"]) == 3
    confidences = [candidate["confidence"] for candidate in data["candidates"]]
    assert confidences == sorted(confidences, reverse=True)
    assert "sourceMember" not in data["candidates"][0]


@requires_assets
def test_classification_endpoint_accepts_prefixed_base64(assets) -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/v1/document/classify",
            json={
                "filename": "card.jpg",
                "imageBase64": f"data:image/jpeg;base64,{encoded}",
                "topK": 2,
            },
        )

    assert response.status_code == 200
    assert response.json()["data"]["returnedCandidates"] == 2


def test_classification_endpoint_validates_top_k() -> None:
    encoded = base64.b64encode(ID_FRONT.read_bytes()).decode("ascii")
    with TestClient(app) as client:
        too_small = client.post(
            "/v1/document/classify",
            json={"imageBase64": encoded, "topK": 0},
        )
        too_large = client.post(
            "/v1/document/classify",
            json={"imageBase64": encoded, "topK": 26},
        )
        invalid = client.post(
            "/v1/document/classify",
            json={"imageBase64": encoded, "topK": "many"},
        )

    assert too_small.status_code == 422
    assert too_large.status_code == 422
    assert invalid.status_code == 422


def test_classification_endpoint_preserves_unnamed_output(monkeypatch) -> None:
    def fake_classification(root, image, top_k):
        return {
            "classCount": 12,
            "candidates": [
                {
                    "documentIdentifier": 1895998652,
                    "confidence": 0.75,
                    "document": None,
                }
            ],
        }

    monkeypatch.setattr(api_module, "classify_document", fake_classification)
    with TestClient(app) as client, ID_FRONT.open("rb") as image:
        response = client.post(
            "/v1/document/classify",
            files={"image": ("card.jpg", image, "image/jpeg")},
        )

    candidate = response.json()["data"]["candidates"][0]
    assert candidate == {
        "identifier": 1895998652,
        "metadataAvailable": False,
        "confidence": 0.75,
    }
