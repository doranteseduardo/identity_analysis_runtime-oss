"""Biometric face-comparison evaluation and threshold calibration."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .face_engines import FaceDetector, LandmarkQuality
from .facial_identity import (
    FaceRecognitionEngine,
    compare_face_templates,
    decode_face_template,
)


DEFAULT_TARGET_FARS = (0.001, 0.01)


def parse_label(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "same", "same_person", "genuine"}:
        return True
    if normalized in {
        "0",
        "false",
        "no",
        "different",
        "different_person",
        "impostor",
    }:
        return False
    raise ValueError(f"unsupported samePerson value: {value!r}")


def threshold_candidates(scores: np.ndarray) -> list[float]:
    unique = np.unique(scores)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return [1.0, *[float(value) for value in midpoints[::-1]], 0.0]


def operating_point(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    accepted = scores > threshold
    genuine = labels
    impostor = ~labels
    true_accepts = int(np.count_nonzero(accepted & genuine))
    false_rejects = int(np.count_nonzero(~accepted & genuine))
    false_accepts = int(np.count_nonzero(accepted & impostor))
    true_rejects = int(np.count_nonzero(~accepted & impostor))
    genuine_count = int(np.count_nonzero(genuine))
    impostor_count = int(np.count_nonzero(impostor))
    return {
        "threshold": float(threshold),
        "falseAcceptRate": false_accepts / impostor_count,
        "falseRejectRate": false_rejects / genuine_count,
        "trueAcceptRate": true_accepts / genuine_count,
        "trueRejectRate": true_rejects / impostor_count,
        "counts": {
            "trueAccept": true_accepts,
            "falseReject": false_rejects,
            "falseAccept": false_accepts,
            "trueReject": true_rejects,
        },
    }


def score_summary(values: np.ndarray) -> dict:
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def evaluate_scores(
    records: Iterable[dict],
    target_fars: Iterable[float] = DEFAULT_TARGET_FARS,
) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("evaluation requires at least one record")
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    labels = np.asarray([bool(row["samePerson"]) for row in rows], dtype=bool)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite values between zero and one")
    genuine_count = int(np.count_nonzero(labels))
    impostor_count = len(rows) - genuine_count
    if genuine_count == 0 or impostor_count == 0:
        raise ValueError("evaluation requires genuine and impostor records")

    curve = [
        operating_point(scores, labels, threshold)
        for threshold in threshold_candidates(scores)
    ]
    eer_point = min(
        curve,
        key=lambda point: (
            abs(point["falseAcceptRate"] - point["falseRejectRate"]),
            point["falseAcceptRate"] + point["falseRejectRate"],
        ),
    )
    eer = (
        eer_point["falseAcceptRate"] + eer_point["falseRejectRate"]
    ) / 2.0

    roc_coordinates = sorted(
        {
            (point["falseAcceptRate"], point["trueAcceptRate"])
            for point in curve
        }
        | {(0.0, 0.0), (1.0, 1.0)}
    )
    # NumPy 2.0 removed `trapz` in favor of `trapezoid`; `getattr(np, "trapezoid",
    # np.trapz)` still crashes on NumPy 2.x because the fallback argument is
    # evaluated eagerly, before the attribute lookup result is known.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    auc = float(
        integrate(
            [coordinate[1] for coordinate in roc_coordinates],
            [coordinate[0] for coordinate in roc_coordinates],
        )
    )

    targets = []
    for raw_target in target_fars:
        target = float(raw_target)
        if not 0.0 <= target <= 1.0:
            raise ValueError("target FAR values must be between zero and one")
        eligible = [
            point for point in curve if point["falseAcceptRate"] <= target
        ]
        selected = max(
            eligible,
            key=lambda point: (
                point["trueAcceptRate"],
                point["falseAcceptRate"],
                -point["threshold"],
            ),
        )
        targets.append({"targetFalseAcceptRate": target, **selected})

    genuine_scores = scores[labels]
    impostor_scores = scores[~labels]
    return {
        "recordCount": len(rows),
        "genuineCount": genuine_count,
        "impostorCount": impostor_count,
        "decisionRule": "same_person_when_score_greater_than_threshold",
        "scoreRange": [0.0, 1.0],
        "distributions": {
            "genuine": score_summary(genuine_scores),
            "impostor": score_summary(impostor_scores),
        },
        "rocAuc": auc,
        "equalErrorRate": eer,
        "equalErrorThreshold": eer_point["threshold"],
        "operatingPoints": targets,
        "curve": curve,
    }


def input_mode(fieldnames: list[str] | None) -> str:
    fields = set(fieldnames or [])
    if {"samePerson", "score"} <= fields:
        return "score"
    if {"samePerson", "template1Base64", "template2Base64"} <= fields:
        return "template"
    if {"samePerson", "image1", "image2"} <= fields:
        return "image"
    raise ValueError(
        "CSV requires samePerson and either score, template1Base64/template2Base64, or image1/image2"
    )


def load_evaluation_records(
    source: Path,
    assets: Path | None = None,
) -> tuple[str, list[dict]]:
    source = Path(source).resolve()
    with source.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        mode = input_mode(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError("evaluation CSV contains no records")

    engine = None
    if mode == "image":
        if assets is None:
            raise ValueError("--assets is required for image-pair evaluation")
        root = Path(assets).resolve()
        detector = FaceDetector(root / "facial/detector/face_detector.onnx")
        landmarks = LandmarkQuality(
            root / "facial/landmarks/landmarks_quality.onnx"
        )
        engine = FaceRecognitionEngine(
            root / "facial/recognition/00_R_L_CF_V1_16GPUs/model.onnx",
            detector,
            landmarks,
        )

    @lru_cache(maxsize=None)
    def image_template(path_value: str) -> np.ndarray:
        path = Path(path_value)
        if not path.is_absolute():
            path = source.parent / path
        with Image.open(path) as image:
            return engine.embedding(image.convert("RGB"))["vector"]

    records = []
    for line_number, row in enumerate(rows, start=2):
        try:
            label = parse_label(row["samePerson"])
            if mode == "score":
                score = float(row["score"])
            elif mode == "template":
                first = decode_face_template(
                    base64.b64decode(row["template1Base64"], validate=True)
                )
                second = decode_face_template(
                    base64.b64decode(row["template2Base64"], validate=True)
                )
                score = compare_face_templates(first, second)["score"]
            else:
                first = image_template(row["image1"])
                second = image_template(row["image2"])
                score = compare_face_templates(first, second)["score"]
        except (OSError, ValueError) as error:
            raise ValueError(f"CSV line {line_number}: {error}") from error
        records.append(
            {
                "samePerson": label,
                "score": score,
                "pairIdentifier": row.get("pairIdentifier") or None,
            }
        )
    return mode, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Evaluation CSV")
    parser.add_argument("--assets", type=Path, help="Runtime assets for image pairs")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--target-far",
        action="append",
        type=float,
        dest="target_fars",
        help="Target false-accept rate; may be repeated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mode, records = load_evaluation_records(args.input, args.assets)
    report = {
        "input": str(args.input.resolve()),
        "inputMode": mode,
        **evaluate_scores(records, args.target_fars or DEFAULT_TARGET_FARS),
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
