import pytest

from identity_analysis.rectification import map_rectified_bounds_to_source


def test_map_rectified_bounds_to_axis_aligned_source() -> None:
    corners = [[100, 50], [900, 50], [900, 450], [100, 450]]

    result = map_rectified_bounds_to_source(
        [0.25, 0.25, 0.75, 0.75], corners, (1000, 500)
    )

    assert result == [0.3, 0.3, 0.7, 0.7]


def test_map_rectified_bounds_clamps_source_coordinates() -> None:
    corners = [[-10, -20], [110, -20], [110, 120], [-10, 120]]

    result = map_rectified_bounds_to_source([0, 0, 1, 1], corners, (100, 100))

    assert result == [0.0, 0.0, 1.0, 1.0]


def test_map_rotated_rectified_bounds_to_source() -> None:
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]

    result = map_rectified_bounds_to_source(
        [0.1, 0.2, 0.3, 0.4], corners, (100, 100), rotation=90
    )

    assert result == pytest.approx([0.6, 0.1, 0.8, 0.3])
