import numpy as np

from identity_analysis.face_engines import (
    decode_ssd_boxes,
    generate_ssd_anchors,
    non_maximum_suppression,
)


def test_generate_ssd_anchors_matches_model_output():
    anchors = generate_ssd_anchors()

    assert anchors.shape == (5118, 4)
    assert np.all((anchors[:, :2] > 0) & (anchors[:, :2] < 1))


def test_zero_box_encoding_returns_anchor_bounds():
    anchors = generate_ssd_anchors()[:1]
    box = decode_ssd_boxes(np.zeros((1, 4), dtype=np.float32), anchors)[0]

    center_y, center_x, height, width = anchors[0]
    assert np.allclose(
        box,
        [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2],
    )


def test_non_maximum_suppression_removes_overlap():
    boxes = np.array([[0, 0, 1, 1], [0.05, 0.05, 0.95, 0.95], [2, 2, 3, 3]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert non_maximum_suppression(boxes, scores, 0.5) == [0, 2]
