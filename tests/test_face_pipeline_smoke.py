from PIL import Image

from identity_analysis.face_engines import FaceDetector, LandmarkQuality

from conftest import SELFIE, requires_assets


@requires_assets
def test_synthetic_selfie_face_pipeline(assets):
    detector = FaceDetector(assets / "facial/detector/face_detector.onnx")
    landmarks = LandmarkQuality(assets / "facial/landmarks/landmarks_quality.onnx")
    with Image.open(SELFIE) as image:
        detections = detector.detect(image)
        result = landmarks.infer(image, detections[0])

    assert len(detections) == 1
    assert detections[0].confidence > 0.99
    assert len(result["landmarks"]) == 68
    assert all(len(point["pixel"]) == 2 for point in result["landmarks"])
    assert 0.0 <= result["qualityScore"] <= 1.0
    assert result["qualitySemantics"] == "coverage_sensitive_experimental"
    assert result["qualityThreshold"] == 0.8
    assert result["headPose"]["method"] == "six_point_weak_perspective"
    assert abs(result["headPose"]["yaw"]) < 10.0
    assert abs(result["headPose"]["pitch"]) < 10.0
    assert abs(result["headPose"]["roll"]) < 10.0
