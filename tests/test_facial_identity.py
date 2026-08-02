import numpy as np
import pytest

from identity_analysis.facial_identity import (
    FACE_TEMPLATE_BYTES,
    LIVENESS_BIAS,
    LIVENESS_WEIGHTS,
    PassiveLivenessEngine,
    combine_liveness_logits,
    compare_face_templates,
    decode_face_template,
    encode_face_template,
    identity_verification_decision,
    liveness_decision,
    similarity_transform,
)


def test_liveness_falls_back_to_sequential_inference_when_threads_are_exhausted():
    class ExhaustedExecutor:
        def map(self, *_args, **_kwargs):
            raise RuntimeError("can't start new thread")

    class Model:
        def __init__(self, value):
            self.value = value

        def run(self, tensor):
            return np.asarray([self.value + float(tensor[0])], dtype=np.float32)

    engine = PassiveLivenessEngine.__new__(PassiveLivenessEngine)
    engine.models = [Model(index) for index in range(7)]
    engine.executor = ExhaustedExecutor()

    outputs = engine._run_models(
        [np.asarray([index], dtype=np.float32) for index in range(7)]
    )

    assert [float(output[0]) for output in outputs] == [
        float(index * 2) for index in range(7)
    ]


def test_liveness_ensemble_matches_recovered_formula():
    logits = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25]
    linear, score = combine_liveness_logits(logits)
    expected_linear = float(np.dot(LIVENESS_WEIGHTS, logits) + LIVENESS_BIAS)
    expected_score = np.clip(0.5 + 0.5 * (expected_linear + 0.2) / 7.0, 0, 1)

    assert np.isclose(linear, expected_linear)
    assert np.isclose(score, expected_score)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.2500, "spoof"),
        (0.2501, "review"),
        (0.3700, "review"),
        (0.3701, "real"),
    ],
)
def test_liveness_decision_uses_review_band(score, expected):
    assert liveness_decision(score, 0.37, 0.25) == expected


def test_liveness_decision_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="liveness thresholds"):
        liveness_decision(0.5, 0.25, 0.37)


def test_similarity_transform_maps_points():
    source = np.asarray([[10.0, 20.0], [30.0, 20.0], [20.0, 30.0]])
    target = np.asarray([[25.0, 47.0], [65.0, 47.0], [45.0, 67.0]])
    transform = similarity_transform(source, target)
    mapped = np.column_stack((source, np.ones(len(source)))) @ transform.T

    assert np.allclose(mapped[:, :2], target)


def test_face_template_binary_round_trip() -> None:
    vector = np.linspace(-1.0, 1.0, 512, dtype=np.float32)

    payload = encode_face_template(vector)

    assert len(payload) == FACE_TEMPLATE_BYTES
    assert np.array_equal(decode_face_template(payload), vector)


def test_identical_face_templates_compare_as_same_person() -> None:
    vector = np.linspace(-1.0, 1.0, 512, dtype=np.float32)

    result = compare_face_templates(vector, vector)

    assert result["decision"] == "same_person"
    assert result["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("identity", "liveness", "document_capture", "expected"),
    [
        ("same_person", "real", "pass", "pass"),
        ("same_person", "spoof", "pass", "fail"),
        ("different_person", "real", "pass", "fail"),
        ("different_person", None, None, "fail"),
        ("same_person", "review", "pass", "review"),
        ("same_person", "real", "review", "review"),
        ("same_person", "real", None, "review"),
        ("same_person", None, "pass", "not_available"),
    ],
)
def test_identity_verification_policy(
    identity: str,
    liveness: str | None,
    document_capture: str | None,
    expected: str,
) -> None:
    result = identity_verification_decision(
        identity,
        liveness,
        document_capture,
    )

    assert result["decision"] == expected
    assert result["checks"]["portraitMatch"] == identity
    assert result["checks"]["selfieLiveness"] == (liveness or "not_available")
    assert result["checks"]["documentCapture"] == (
        document_capture or "not_available"
    )


@pytest.mark.parametrize("payload", [b"", b"short", bytes(FACE_TEMPLATE_BYTES)])
def test_face_template_decoder_rejects_invalid_payload(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_face_template(payload)
