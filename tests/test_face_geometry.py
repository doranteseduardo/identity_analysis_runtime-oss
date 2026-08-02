import numpy as np
import pytest

from identity_analysis.face_engines import (
    HEAD_POSE_MODEL,
    FaceDetection,
    assess_face_quality,
    estimate_head_pose,
)


def rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    pitch, yaw, roll = np.radians([pitch, yaw, roll])
    rotate_x = np.asarray(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    rotate_y = np.asarray(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    rotate_z = np.asarray(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )
    return rotate_z @ rotate_y @ rotate_x


def projected_landmarks(pitch: float, yaw: float, roll: float) -> list[dict]:
    rotation = rotation_matrix(pitch, yaw, roll)
    landmarks = []
    for index, point in HEAD_POSE_MODEL.items():
        projected = rotation @ np.asarray(point)
        landmarks.append(
            {
                "index": index,
                "pixel": [float(projected[0] + 500), float(-projected[1] + 500)],
            }
        )
    return landmarks


def test_head_pose_recovers_known_weak_perspective_rotation() -> None:
    pose = estimate_head_pose(projected_landmarks(10.0, 20.0, -5.0))

    assert pose["pitch"] == pytest.approx(10.0)
    assert pose["yaw"] == pytest.approx(20.0)
    assert pose["roll"] == pytest.approx(-5.0)


def test_head_pose_rejects_missing_or_degenerate_landmarks() -> None:
    with pytest.raises(ValueError, match="missing head-pose landmarks"):
        estimate_head_pose(projected_landmarks(0.0, 0.0, 0.0)[:-1])

    collapsed = [
        {"index": index, "pixel": [100.0, 100.0]}
        for index in HEAD_POSE_MODEL
    ]
    with pytest.raises(ValueError, match="geometrically degenerate"):
        estimate_head_pose(collapsed)


def test_face_quality_reports_portable_legacy_semantics() -> None:
    quality = assess_face_quality(
        FaceDetection((0.0, 0.2, 0.08, 0.28), 0.99),
        (640, 480),
        {"yaw": 40.0, "pitch": 0.0, "roll": 0.0},
        face_count=2,
    )

    assert quality["status"] == "review"
    assert quality["livenessEligible"] is False
    assert quality["warnings"] == [
        {"code": "MULTIPLE_FACES", "legacyCode": -500},
        {"code": "FACE_TOO_SMALL", "legacyCode": -100},
        {"code": "FACE_CUTOFF", "legacyCode": -200},
        {"code": "EXCESSIVE_POSE", "legacyCode": -300},
    ]


def test_face_quality_passes_centered_frontal_face() -> None:
    quality = assess_face_quality(
        FaceDetection((0.25, 0.2, 0.75, 0.8), 0.99),
        (640, 480),
        {"yaw": 2.0, "pitch": -3.0, "roll": 1.0},
    )

    assert quality == {
        "status": "pass",
        "warnings": [],
        "livenessEligible": True,
        "policy": "portable_geometry_v1",
    }


def test_face_quality_allows_liveness_for_soft_warnings() -> None:
    quality = assess_face_quality(
        FaceDetection((0.25, 0.2, 0.75, 0.8), 0.99),
        (640, 480),
        {"yaw": 36.0, "pitch": 0.0, "roll": 0.0},
        landmark_quality_score=0.9,
    )

    assert quality["status"] == "review"
    assert quality["livenessEligible"] is True
    assert [warning["code"] for warning in quality["warnings"]] == [
        "EXCESSIVE_POSE",
        "COVERED_FACE",
    ]


def test_face_quality_maps_high_landmark_scalar_to_covered_face() -> None:
    quality = assess_face_quality(
        FaceDetection((0.25, 0.2, 0.75, 0.8), 0.99),
        (640, 480),
        {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        landmark_quality_score=0.95,
    )

    assert quality["warnings"] == [
        {"code": "COVERED_FACE", "legacyCode": -400}
    ]
