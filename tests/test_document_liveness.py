import numpy as np
import pytest
from PIL import Image

from identity_analysis.document_liveness import (
    DOCUMENT_QUALITY_WARNINGS,
    DOCUMENT_VALIDATION_ERRORS,
    DocumentDetection,
    apply_preprocessors,
    build_document_liveness_result,
    evaluate_thresholds,
    run_configured_model,
    validate_document_geometry,
)


def test_align_resize_and_normalize_follow_config_order():
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[20:80, 50:150] = [127, 64, 32]
    image = Image.fromarray(pixels)
    detection = DocumentDetection("kGenericDocument", (50, 20, 150, 80))

    result = apply_preprocessors(
        image,
        [
            {
                "type": "align_document_by_bbox",
                "detection_to_use": "kGenericDocument",
                "pad_percentage": 0,
            },
            {
                "type": "resize",
                "target_width": 20,
                "target_height": 10,
                "interpolation_mode": "INTER_NEAREST",
            },
            {"type": "normalize", "mean": [127, 64, 32], "std": [1, 2, 4]},
        ],
        [detection],
    )

    assert result.shape == (10, 20, 3)
    assert np.all(result == 0)


def test_stack_ghost_and_photo_uses_zero_for_optional_missing_detection():
    image = Image.new("RGB", (100, 100), "white")
    photo = DocumentDetection("kPhoto", (20, 20, 80, 80))
    result = apply_preprocessors(
        image,
        [
            {
                "type": "stack_detections_by_bbox",
                "target_width": 16,
                "target_height": 12,
                "interpolation_mode": "INTER_NEAREST",
                "detections_to_align_by_bbox": [
                    {"detection_to_use": "kGhostPhoto", "pad_percentage": 50},
                    {"detection_to_use": "kPhoto", "pad_percentage": 0},
                ],
                "apply_bw_transform": True,
                "detections_required": False,
            },
            {"type": "convert_to_float"},
        ],
        [photo],
    )

    assert result.shape == (12, 16, 2)
    assert np.all(result[..., 0] == 0)
    assert np.all(result[..., 1] == 255)


def test_missing_required_detection_is_explicit():
    with pytest.raises(ValueError, match="kPhoto"):
        apply_preprocessors(
            Image.new("RGB", (10, 10)),
            [
                {
                    "type": "align_document_by_bbox",
                    "detection_to_use": "kPhoto",
                }
            ],
        )


def test_configured_runner_preserves_raw_outputs_and_thresholds():
    calls = []

    def runner(model_name, input_name, output_name, tensor):
        calls.append((model_name, input_name, output_name, tensor.shape))
        return np.asarray([[-2.0, -0.5, -1.7, 0.75]], dtype=np.float32)

    result = run_configured_model(
        Image.new("RGB", (16, 16), "white"),
        {
            "model_name": "quality.onnx",
            "nnet_input_name": "input",
            "nnet_output_name": "output",
            "preprocessors": [
                {
                    "type": "resize",
                    "target_width": 8,
                    "target_height": 6,
                    "interpolation_mode": "INTER_NEAREST",
                }
            ],
            "output_thresholds": [-1.8, -0.9, -1.6, 0.5],
        },
        runner,
    )

    assert calls == [("quality.onnx", "input", "output", (6, 8, 3))]
    assert result["status"] == "diagnostic_only"
    assert result["rawOutput"] == pytest.approx([-2.0, -0.5, -1.7, 0.75])
    assert [item["aboveThreshold"] for item in result["thresholdResults"]] == [
        False,
        True,
        False,
        True,
    ]


def test_threshold_count_must_match_output_count():
    with pytest.raises(ValueError, match="threshold count"):
        evaluate_thresholds(np.asarray([1.0, 2.0]), [0.5])


def test_basic_geometry_validator_reports_each_confirmed_violation():
    detections = [
        DocumentDetection("kGenericDocument", (1, 10, 90, 90)),
        DocumentDetection("kGenericDocument", (10, 10, 80, 80)),
    ]

    result = validate_document_geometry(
        (100, 100),
        detections,
        {"max_number_of_documents": 1, "min_document_padding_px": 2},
    )

    assert result["valid"] is False
    assert result["documentCount"] == 2
    assert [item["code"] for item in result["violations"]] == [
        "too_many_documents",
        "insufficient_document_padding",
    ]


def test_liveness_result_contract_uses_strict_probability_threshold():
    live = build_document_liveness_result("screen-replay", 1.25, 0.5001)
    spoof = build_document_liveness_result("screen-replay", -0.1, 0.5)

    assert live["decision"] == "live"
    assert spoof["decision"] == "spoof"
    assert live["threshold"] == 0.5


def test_validation_error_suppresses_liveness_decision():
    result = build_document_liveness_result(
        "portrait-substitution",
        None,
        None,
        status_code="DOCUMENT_PHOTO_NOT_FOUND",
        image_quality_warnings=["DOCUMENT_TOO_CLOSE_TO_BORDER"],
        calibration="REGULAR",
    )

    assert result["decision"] == "not_available"
    assert result["statusCode"] in DOCUMENT_VALIDATION_ERRORS
    assert set(result["imageQualityWarnings"]) <= DOCUMENT_QUALITY_WARNINGS


def test_liveness_result_rejects_unknown_contract_values():
    with pytest.raises(ValueError, match="unknown document validation"):
        build_document_liveness_result("test", None, None, status_code="UNKNOWN")
    with pytest.raises(ValueError, match="unknown document quality"):
        build_document_liveness_result(
            "test", 0.0, 0.5, image_quality_warnings=["UNKNOWN"]
        )
