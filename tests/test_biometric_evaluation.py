import base64
import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import identity_analysis.biometric_evaluation as evaluation
from identity_analysis.facial_identity import encode_face_template


def records(genuine: list[float], impostor: list[float]) -> list[dict]:
    return [
        *({"samePerson": True, "score": score} for score in genuine),
        *({"samePerson": False, "score": score} for score in impostor),
    ]


def test_evaluate_scores_reports_perfect_separation() -> None:
    report = evaluation.evaluate_scores(
        records([0.9, 0.8], [0.2, 0.1]),
        target_fars=[0.0, 0.01],
    )

    assert report["recordCount"] == 4
    assert report["genuineCount"] == 2
    assert report["impostorCount"] == 2
    assert report["rocAuc"] == pytest.approx(1.0)
    assert report["equalErrorRate"] == pytest.approx(0.0)
    assert report["equalErrorThreshold"] == pytest.approx(0.5)
    assert report["operatingPoints"][0]["falseAcceptRate"] == 0.0
    assert report["operatingPoints"][0]["trueAcceptRate"] == 1.0


def test_evaluate_scores_reports_overlapping_distribution() -> None:
    report = evaluation.evaluate_scores(
        records([0.9, 0.4], [0.6, 0.1]),
        target_fars=[0.0],
    )

    assert report["rocAuc"] == pytest.approx(0.75)
    assert report["equalErrorRate"] == pytest.approx(0.5)
    assert report["equalErrorThreshold"] == pytest.approx(0.5)
    assert report["operatingPoints"][0]["threshold"] == pytest.approx(0.75)
    assert report["operatingPoints"][0]["trueAcceptRate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("same_person", True),
        ("genuine", True),
        ("1", True),
        ("different_person", False),
        ("impostor", False),
        ("0", False),
    ],
)
def test_parse_label(value, expected) -> None:
    assert evaluation.parse_label(value) is expected


def test_evaluate_scores_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="genuine and impostor"):
        evaluation.evaluate_scores(records([0.9], []))


def test_load_score_csv(tmp_path: Path) -> None:
    source = tmp_path / "scores.csv"
    source.write_text(
        "pairIdentifier,samePerson,score\n"
        "g1,true,0.9\n"
        "i1,false,0.2\n",
        encoding="utf-8",
    )

    mode, loaded = evaluation.load_evaluation_records(source)

    assert mode == "score"
    assert loaded == [
        {"samePerson": True, "score": 0.9, "pairIdentifier": "g1"},
        {"samePerson": False, "score": 0.2, "pairIdentifier": "i1"},
    ]


def test_load_template_csv_computes_scores(tmp_path: Path) -> None:
    source = tmp_path / "templates.csv"
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    opposite = -first
    encoded_first = base64.b64encode(encode_face_template(first)).decode("ascii")
    encoded_opposite = base64.b64encode(
        encode_face_template(opposite)
    ).decode("ascii")
    with source.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["samePerson", "template1Base64", "template2Base64"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "samePerson": "true",
                "template1Base64": encoded_first,
                "template2Base64": encoded_first,
            }
        )
        writer.writerow(
            {
                "samePerson": "false",
                "template1Base64": encoded_first,
                "template2Base64": encoded_opposite,
            }
        )

    mode, loaded = evaluation.load_evaluation_records(source)

    assert mode == "template"
    assert loaded[0]["score"] == pytest.approx(1.0)
    assert loaded[1]["score"] == pytest.approx(0.0)


def test_load_image_csv_reuses_cached_embeddings(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "images.csv"
    source.write_text(
        "samePerson,image1,image2\n"
        "true,first.png,first.png\n"
        "false,first.png,second.png\n",
        encoding="utf-8",
    )
    Image.new("RGB", (2, 2), (255, 0, 0)).save(tmp_path / "first.png")
    Image.new("RGB", (2, 2), (0, 0, 255)).save(tmp_path / "second.png")
    calls = []

    class StubModel:
        def __init__(self, *args):
            pass

    class StubRecognition:
        def __init__(self, *args):
            pass

        def embedding(self, image):
            calls.append(image.getpixel((0, 0)))
            vector = np.zeros(512, dtype=np.float32)
            vector[0 if image.getpixel((0, 0))[0] else 1] = 1.0
            return {"vector": vector}

    monkeypatch.setattr(evaluation, "FaceDetector", StubModel)
    monkeypatch.setattr(evaluation, "LandmarkQuality", StubModel)
    monkeypatch.setattr(evaluation, "FaceRecognitionEngine", StubRecognition)

    mode, loaded = evaluation.load_evaluation_records(
        source, tmp_path / "assets"
    )

    assert mode == "image"
    assert len(calls) == 2
    assert loaded[0]["score"] == pytest.approx(1.0)
    assert loaded[1]["score"] == pytest.approx(0.5)
