"""Command-line interface for document analysis."""

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import process_document


def default_assets_path() -> Path:
    configured = os.environ.get("IDENTITY_ANALYSIS_ASSETS")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "assets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="identity-document",
        description="Structured OCR for the explicitly supported document profiles.",
    )
    parser.add_argument("image", type=Path, help="ID image in HEIC, JPEG or PNG format")
    parser.add_argument(
        "--assets",
        type=Path,
        default=default_assets_path(),
        help="Prepared runtime asset directory",
    )
    parser.add_argument("--output", "-o", type=Path, help="Write JSON to this file")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    parser.add_argument(
        "--profile",
        choices=("mex_ine", "mex_passport", "icao_td1", "icao_td2", "icao_td3", "icao_mrv", "aamva_pdf417", "swe_id_2021", "auto_research"),
        default="auto_research",
        help="Explicit recognition profile",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = process_document(args.image, args.assets, args.profile)
        payload = json.dumps(
            result,
            indent=None if args.compact else 2,
            ensure_ascii=False,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0
    except Exception as error:
        print(
            json.dumps({"errorCode": 1, "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
