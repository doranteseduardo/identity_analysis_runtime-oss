"""Validate a bring-your-own runtime asset directory.

This runtime ships no model weights.  An operator points
``IDENTITY_ANALYSIS_ASSETS`` at their own directory of ONNX models; the layout
that directory is expected to follow, and the ``manifest.json`` schema used
below, are documented in ``docs/models.md``.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT_VERSION = 2


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    """Describe every file under ``root`` as a verifiable manifest entry."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return {"formatVersion": MANIFEST_FORMAT_VERSION, "files": files}


def write_manifest(root: Path) -> dict:
    manifest = build_manifest(root)
    (Path(root).resolve() / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_assets(root: Path) -> dict:
    return _validate_assets(str(root.resolve()))


@lru_cache(maxsize=4)
def _validate_assets(root_string: str) -> dict:
    root = Path(root_string)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        payload = (root / entry["path"]).read_bytes()
        if sha256(payload) != entry["sha256"]:
            raise ValueError(f"Asset hash mismatch: {entry['path']}")
    return manifest
