"""Portable metadata and image-structure signals."""

from pathlib import Path

from PIL import Image


EDITOR_MARKERS = ("photoshop", "gimp", "paint", "lightroom")


def check_metadata_integrity(image_path: Path, _assets_path: Path) -> dict:
    image_path = image_path.resolve()
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        exif = image.getexif()
        software = str(exif.get(305, ""))
        format_name = image.format or "unknown"
        width, height = image.size
        has_icc = bool(image.info.get("icc_profile"))
        has_exif = bool(exif)

    editor_detected = any(marker in software.lower() for marker in EDITOR_MARKERS)
    signals = {
        "validImageStructure": True,
        "format": format_name,
        "dimensions": [width, height],
        "hasExif": has_exif,
        "hasIccProfile": has_icc,
        "software": software,
        "editorMarkerDetected": editor_detected,
    }
    return {
        "available": True,
        "decision": "review" if editor_detected else "pass",
        "checker": "portable-python-metadata-v1",
        "scope": "Decoding, EXIF software and ICC presence",
        "signals": signals,
        "isSpoofingDecision": False,
        "note": "This portable check inspects container metadata only and is not PAD.",
    }
