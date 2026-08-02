"""Command-line interface for facial analysis."""

import argparse
import json
from pathlib import Path

from PIL import Image

from .face_engines import FaceDetector, LandmarkQuality, analyze_faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    detector = FaceDetector(args.detector)
    landmark_model = LandmarkQuality(args.landmarks) if args.landmarks else None
    with Image.open(args.image) as image:
        result = analyze_faces(
            image,
            detector,
            landmark_model,
            args.score_threshold,
            args.iou_threshold,
        )
    result["image"] = str(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
