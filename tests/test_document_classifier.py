import numpy as np
from PIL import Image

from identity_analysis.document_classifier import (
    classifier_available,
    classify_document,
    preprocess_document_classifier,
)
from identity_analysis.pipeline import (
    classifier_mrz_hint,
    classifier_layout_hint,
    classifier_visual_layout_candidate,
)

from conftest import FIXTURE_CATALOG, ID_FRONT, requires_assets


def test_preprocess_document_classifier_uses_expected_tensor_layout() -> None:
    image = Image.new("RGB", (40, 20), (10, 80, 220))

    tensor = preprocess_document_classifier(image)

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32
    assert abs(float(tensor.mean())) < 1e-5
    assert abs(float(tensor.std()) - 1.0) < 1e-5


@requires_assets
def test_classifier_returns_ranked_candidates_from_installed_model(assets) -> None:
    image = Image.open(ID_FRONT)

    classification = classify_document(assets, image, top_k=5)

    assert classification["classCount"] > 0
    assert len(classification["candidates"]) == 5
    confidences = [candidate["confidence"] for candidate in classification["candidates"]]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= confidence <= 1.0 for confidence in confidences)
    assert all(
        isinstance(candidate["documentIdentifier"], int)
        for candidate in classification["candidates"]
    )


def test_classifier_availability_is_reported_not_raised() -> None:
    # The synthetic fixture catalog ships layouts and catalog metadata but no
    # classifier weights, which must degrade to "unavailable".
    assert classifier_available(FIXTURE_CATALOG) is False


def test_declarative_layout_requires_unambiguous_classification() -> None:
    document = {"caption": "Example"}
    accepted = {
        "candidates": [
            {"documentIdentifier": 1, "confidence": 0.8, "document": document},
            {"documentIdentifier": 2, "confidence": 0.1, "document": document},
        ]
    }
    ambiguous = {
        "candidates": [
            {"documentIdentifier": 1, "confidence": 0.7, "document": document},
            {"documentIdentifier": 2, "confidence": 0.6, "document": document},
        ]
    }

    assert classifier_visual_layout_candidate(accepted)["documentIdentifier"] == 1
    assert classifier_visual_layout_candidate(ambiguous) is None


def test_classifier_mrz_hint_uses_presence_format_and_document_type() -> None:
    def classification(document_format: str, document_type: str, present: bool = True):
        return {
            "candidates": [
                {
                    "documentIdentifier": 1,
                    "confidence": 0.9,
                    "document": {
                        "mrz": {"present": present},
                        "documentFormat": {"name": document_format},
                        "documentType": {"name": document_type},
                    },
                },
                {
                    "documentIdentifier": 2,
                    "confidence": 0.05,
                    "document": {"caption": "Other"},
                },
            ]
        }

    assert classifier_mrz_hint(classification("ID1", "IdentityCard")) == "td1"
    assert classifier_mrz_hint(classification("ID2", "IdentityCard")) == "td2"
    assert classifier_mrz_hint(classification("ID3", "Passport")) == "td3"
    assert classifier_mrz_hint(classification("ID3", "Visa")) == "mrv"
    assert classifier_mrz_hint(classification("ID2", "Visa", False)) == "mrv"
    assert classifier_mrz_hint(classification("ID3", "Passport", False)) == "td3"
    assert classifier_mrz_hint(classification("ID3", "PassportPage", False)) is None
