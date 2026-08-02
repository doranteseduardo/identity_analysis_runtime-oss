"""Command-line interface for runtime asset management."""

import argparse
import json
from pathlib import Path

from .assets import validate_assets, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="identity-assets",
        description=(
            "Describe and verify a bring-your-own ONNX asset directory "
            "(see docs/models.md)."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    manifest = subcommands.add_parser(
        "manifest", help="Write manifest.json for a directory of models"
    )
    manifest.add_argument("assets", type=Path, nargs="?", default=Path("assets"))
    validate = subcommands.add_parser("validate", help="Verify bundle hashes")
    validate.add_argument("assets", type=Path, nargs="?", default=Path("assets"))
    args = parser.parse_args()

    if args.command == "manifest":
        result = write_manifest(args.assets)
    else:
        result = validate_assets(args.assets)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
