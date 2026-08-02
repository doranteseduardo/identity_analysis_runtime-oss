#!/usr/bin/env python3
"""Build the synthetic document catalog used by the test suite.

``tests/fixtures/catalog/document_classifier`` is an entirely hand-authored, two-layout catalog
describing the synthetic identity card rendered by
``tools/generate_synthetic_samples.py``.  It exists so the catalog-driven code
paths (``VisualLayoutCatalog``, ``barcode_regions``, ``recognize_visual_layout``,
the document catalog search/facet helpers) keep real test coverage without any
third-party catalog data.

The field- and graphic-type numbering below is this project's own: the values
are opaque integers to the runtime, which keys everything off the ``name``.

Usage::

    python tools/generate_test_catalog.py [output_directory]
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image


FRONT_IDENTIFIER = 900000001
BACK_IDENTIFIER = 900000002
BOOKLET_PAGE_1_IDENTIFIER = 900000010
BOOKLET_PAGE_2_IDENTIFIER = 900000011
RESIDENCE_FRONT_IDENTIFIER = 900000020
RESIDENCE_BACK_IDENTIFIER = 900000021

FIELD_TYPES = {
    1: ("SURNAME", "surname"),
    2: ("GIVEN_NAMES", "givenNames"),
    3: ("DATE_OF_BIRTH", "dateOfBirth"),
    4: ("SEX", "sex"),
    5: ("NATIONALITY", "nationality"),
    6: ("DOCUMENT_NUMBER", "documentNumber"),
    7: ("DATE_OF_EXPIRY", "dateOfExpiry"),
    8: ("SURNAME_AND_GIVEN_NAMES", "surnameAndGivenNames"),
    20: ("VERIFICATION_CODE", "verificationCode"),
    21: ("OTHER", "other"),
}
GRAPHIC_FIELD_TYPES = {
    101: ("PORTRAIT", "portrait"),
    102: ("GHOST_PORTRAIT", "ghostPortrait"),
    103: ("SIGNATURE", "signature"),
    104: ("BAR_CODE", "barCode"),
}

VISIBLE = {"lightType": 6, "status": 1}


def _text_field(
    number: int,
    type_number: int,
    bounds: list[float],
    mask: str | None = None,
    **extra,
) -> dict:
    official, name = FIELD_TYPES[type_number]
    return {
        "number": number,
        "type": type_number,
        "name": name,
        "officialName": official,
        "bounds": bounds,
        **({"mask": mask} if mask else {}),
        "lcid": 1033,
        "textHeight": 0.055,
        "colorType": 1,
        "fontLayer": 0,
        "inComparison": 1,
        "lowContrastText": False,
        "removeBackground": False,
        **VISIBLE,
        **extra,
    }


def _graphic(number: int, type_number: int, bounds: list[float], **extra) -> dict:
    official, name = GRAPHIC_FIELD_TYPES[type_number]
    return {
        "number": number,
        "type": type_number,
        "name": name,
        "officialName": official,
        "bounds": bounds,
        **VISIBLE,
        **extra,
    }


def _barcode(
    number: int, type_number: int, bounds: list[float], code_type: int, code_class: int
) -> dict:
    official, name = FIELD_TYPES[type_number]
    return {
        "number": number,
        "type": type_number,
        "name": name,
        "officialName": official,
        "bounds": bounds,
        "orientation": 0,
        "colorType": 1,
        "codeType": code_type,
        "codeClass": code_class,
        "pdf417Codec": 0,
        **VISIBLE,
    }


CARD_FRONT_SAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "samples" / "synthetic_id_front.jpg"
)
REFERENCE_PATCH_BOUNDS = (
    [0.340, 0.020, 0.470, 0.090],  # header lettering
    [0.060, 0.305, 0.130, 0.360],  # portrait frame corner
)


def reference_patches() -> list[dict]:
    """Crop small reference patches out of the rendered sample card."""

    if not CARD_FRONT_SAMPLE.is_file():
        return []
    patches = []
    with Image.open(CARD_FRONT_SAMPLE) as card:
        width, height = card.size
        for number, bounds in enumerate(REFERENCE_PATCH_BOUNDS, start=1):
            left, top, right, bottom = bounds
            crop = card.convert("L").crop(
                (
                    round(left * width),
                    round(top * height),
                    round(right * width),
                    round(bottom * height),
                )
            )
            crop = crop.resize((48, 24), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            crop.save(stream, format="PNG", optimize=True)
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            patches.append(
                {
                    "number": number,
                    "bounds": list(bounds),
                    "lightType": 6,
                    "master": number == 1,
                    "stored": True,
                    "image": {
                        "format": ".PNG",
                        "dpi": 300,
                        "data": encoded,
                        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    },
                }
            )
    return patches


def _value_bounds(row: int, right: float = 0.620) -> list[float]:
    """Bounds of the printed value on row ``row`` of the synthetic card front."""

    top = 0.221 + 0.098 * row
    return [0.330, round(top - 0.008, 6), right, round(top + 0.070, 6)]


def front_layout() -> dict:
    return {
        "identifier": FRONT_IDENTIFIER,
        "caption": "Specimen Identity Card (2024) Front",
        "country": "ZZT",
        "isoCodes": ["ZZT"],
        "documentType": {"value": 12, "name": "IdentityCard"},
        "documentFormat": {"value": 0, "name": "ID1"},
        "classifierLinked": False,
        "year": "2024",
        "orientation": 0,
        "dimensionsMm": {"width": 86, "height": 54},
        "twoSided": True,
        "mainDocument": True,
        "childDocuments": [BACK_IDENTIFIER],
        "pairedPages": [],
        "fields": [
            _text_field(1, 1, _value_bounds(0)),
            _text_field(2, 2, _value_bounds(1)),
            _text_field(3, 3, _value_bounds(2), "{DAY_DD}/{MONTH_DD}/{YEAR}"),
            _text_field(4, 4, _value_bounds(3, 0.450), "{Sex_MF}", inComparison=0),
            _text_field(5, 5, _value_bounds(4, 0.450), "{Country_Code}"),
            _text_field(6, 6, _value_bounds(5), "CCDDDDDDD"),
            _text_field(7, 7, _value_bounds(6), "{DAY_DD}/{MONTH_DD}/{YEAR}"),
        ],
        "graphics": [
            _graphic(1, 101, [0.055, 0.300, 0.300, 0.700], checkFacePresent=True),
            _graphic(2, 102, [0.825, 0.300, 0.955, 0.455], checkFacePresent=True),
            _graphic(3, 103, [0.625, 0.640, 0.810, 0.755]),
        ],
        "barcodes": [],
        "securityRegions": [
            {
                "number": 1,
                "name": "guillocheBackground",
                "bounds": [0.0, 0.115, 1.0, 1.0],
                "lightType": 6,
                "checkType": 1,
            }
        ],
        "referencePatches": reference_patches(),
        "assemblies": [
            {
                "target": {"type": 8, "name": "surnameAndGivenNames", "locale": 1033},
                "parts": [
                    {"kind": "field", "type": 1, "locale": 1033},
                    {"kind": "literal", "value": " "},
                    {"kind": "field", "type": 2, "locale": 1033},
                ],
            }
        ],
        "catalogHints": {
            "capture": {
                "dNecessaryLights": 6,
                "dUVExp": 4,
                "dOVIExp": 3,
                "hologramTiltType": 1,
            },
            "recognition": {
                "dOCRSTX": 20,
                "dOCRSTY": 20,
                "recognTag": 0,
                "dBarcode": 0,
                "dMRZ": 0,
            },
            "electronicDocument": {"chipPage": 0},
            "sourceReferences": {"AuthSourceType": 0},
            "authenticityConfiguration": {
                "dAuthenticity": 12345,
                "authenticity": {
                    "photoReplacementCheck": True,
                    "dWholePageLuminescense": False,
                    "dCheckPhotoHalo": True,
                    "photoEmbedType": 1,
                },
            },
        },
    }


def back_layout() -> dict:
    return {
        "identifier": BACK_IDENTIFIER,
        "caption": "Specimen Identity Card (2024) Back",
        "country": "ZZT",
        "isoCodes": ["ZZT"],
        "documentType": {"value": 12, "name": "IdentityCard"},
        "documentFormat": {"value": 0, "name": "ID1"},
        "classifierLinked": False,
        "year": "2024",
        "orientation": 0,
        "dimensionsMm": {"width": 86, "height": 54},
        "twoSided": True,
        "mainDocument": False,
        "parentIdentifier": FRONT_IDENTIFIER,
        "childDocuments": [],
        "pairedPages": [],
        "fields": [],
        "graphics": [
            _graphic(1, 104, [0.025, 0.025, 0.275, 0.175]),
            _graphic(2, 104, [0.770, 0.070, 0.980, 0.390]),
        ],
        "barcodes": [
            # Code 128 credential number, top left of the card back.
            _barcode(1, 6, [0.025, 0.025, 0.275, 0.175], 1, 1),
            # Verification QR code, top right of the card back.
            _barcode(2, 20, [0.770, 0.070, 0.980, 0.390], 14, 2),
            # Same area declared with a code type this runtime does not map, so
            # the "unknown format" branch of the region decoder stays covered.
            _barcode(3, 21, [0.760, 0.060, 0.990, 0.400], 99, 2),
        ],
        "securityRegions": [],
        "referencePatches": [],
        "assemblies": [],
        "catalogHints": {
            "capture": {"dNecessaryLights": 6},
            "recognition": {
                "dMRZ": 1,
                "dMRZFormat": 1,
                "dMRZHeight": 96,
                "dMRZWidth": 810,
                "dMRZDistanceBetweenLines": 45,
                "recognTag": "specimen-id1-back",
            },
            "electronicDocument": {},
            "sourceReferences": {},
            "authenticityConfiguration": {"dAuthenticity": 0, "authenticity": {}},
        },
    }


def booklet_page(identifier: int, other: int, ordinal: int) -> dict:
    """A minimal two-page booklet, used to exercise semantic page ordering."""

    return {
        "identifier": identifier,
        "caption": (
            "Specimen Travel Booklet (2024)"
            if ordinal == 1
            else f"Specimen Travel Booklet (2024) Page {ordinal}"
        ),
        "country": "ZZT",
        "isoCodes": ["ZZT"],
        "documentType": {"value": 20, "name": "Passport"},
        "documentFormat": {"value": 2, "name": "ID3"},
        "classifierLinked": False,
        "year": "2024",
        "orientation": 0,
        "dimensionsMm": {"width": 125, "height": 88},
        "twoSided": False,
        "mainDocument": ordinal == 1,
        "childDocuments": [],
        "pairedPages": [other],
        "fields": [],
        "graphics": [],
        "barcodes": [],
        "securityRegions": [],
        "referencePatches": [],
        "assemblies": [],
        "catalogHints": {
            "recognition": {
                "dMRZ": 1,
                "dMRZFormat": 2,
                "dMRZFontID": 1,
                "dMRZDistanceBetweenLines": 42,
                "dMRZHeight": 88,
                "dMRZHeightDiff": 4,
                "dMRZWidth": 1112,
                "dMRZWidthDiff": 20,
                "dMRZSymbolPos": 25,
                "dMRZSymbolPosDiff": 5,
            }
        },
    }


def residence_layout(identifier: int, child: int | None, side: str) -> dict:
    """An unmarked front layout whose declared child carries the "Side B" marker.

    This is what lets ``layout_page_role`` infer a front role from a related
    back layout instead of from the caption alone.
    """

    return {
        "identifier": identifier,
        "caption": f"Specimen Residence Card (2024){side}",
        "country": "ZZT",
        "isoCodes": ["ZZT"],
        "documentType": {"value": 30, "name": "ResidencePermit"},
        "documentFormat": {"value": 0, "name": "ID1"},
        "classifierLinked": False,
        "year": "2024",
        "orientation": 0,
        "dimensionsMm": {"width": 86, "height": 54},
        "twoSided": True,
        "mainDocument": child is not None,
        "childDocuments": [child] if child is not None else [],
        "pairedPages": [],
        "fields": [],
        "graphics": [],
        "barcodes": [],
        "securityRegions": [],
        "referencePatches": [],
        "assemblies": [],
        "catalogHints": {},
    }


def documents() -> list[dict]:
    return [
        {
            "identifier": FRONT_IDENTIFIER,
            "caption": "Specimen Identity Card (2024) Front",
            "country": "ZZT",
            "isoCodes": ["ZZT"],
            "documentType": {"value": 12, "name": "IdentityCard"},
            "documentFormat": {"value": 0, "name": "ID1"},
            "mrz": {"present": False, "ignored": False, "expectedProfile": None},
            "barcode": {"present": False},
            "deprecated": False,
            "year": "2024",
            "series": None,
            "stateCodes": [],
            "issuedFrom": None,
            "issuedTo": None,
        },
        {
            "identifier": BACK_IDENTIFIER,
            "caption": "Specimen Identity Card (2024) Back",
            "country": "ZZT",
            "isoCodes": ["ZZT"],
            "documentType": {"value": 12, "name": "IdentityCard"},
            "documentFormat": {"value": 0, "name": "ID1"},
            "mrz": {"present": True, "ignored": False, "expectedProfile": "td1"},
            "barcode": {"present": True},
            "deprecated": False,
            "year": "2024",
            "series": None,
            "stateCodes": [],
            "issuedFrom": None,
            "issuedTo": None,
        },
        {
            "identifier": 900000003,
            "caption": "Specimen Identity Card (2016) Front",
            "country": "ZZT",
            "isoCodes": ["ZZT"],
            "documentType": {"value": 12, "name": "IdentityCard"},
            "documentFormat": {"value": 0, "name": "ID1"},
            "mrz": {"present": False, "ignored": False, "expectedProfile": None},
            "barcode": {"present": False},
            "deprecated": True,
            "year": "2016",
            "series": None,
            "stateCodes": [],
            "issuedFrom": None,
            "issuedTo": None,
        },
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else root / "tests" / "fixtures" / "catalog" / "document_classifier"
    )
    output.mkdir(parents=True, exist_ok=True)

    layouts = {
        "schemaVersion": 2,
        "coordinateSystem": "normalized_document_rectangle",
        "fieldTypes": {
            str(number): {"officialName": official, "name": name}
            for number, (official, name) in sorted(FIELD_TYPES.items())
        },
        "graphicFieldTypes": {
            str(number): {"officialName": official, "name": name}
            for number, (official, name) in sorted(GRAPHIC_FIELD_TYPES.items())
        },
        "layouts": {
            str(FRONT_IDENTIFIER): front_layout(),
            str(BACK_IDENTIFIER): back_layout(),
            str(BOOKLET_PAGE_1_IDENTIFIER): booklet_page(
                BOOKLET_PAGE_1_IDENTIFIER, BOOKLET_PAGE_2_IDENTIFIER, 1
            ),
            str(BOOKLET_PAGE_2_IDENTIFIER): booklet_page(
                BOOKLET_PAGE_2_IDENTIFIER, BOOKLET_PAGE_1_IDENTIFIER, 2
            ),
            str(RESIDENCE_FRONT_IDENTIFIER): residence_layout(
                RESIDENCE_FRONT_IDENTIFIER, RESIDENCE_BACK_IDENTIFIER, ""
            ),
            str(RESIDENCE_BACK_IDENTIFIER): residence_layout(
                RESIDENCE_BACK_IDENTIFIER, None, " Side B"
            ),
        },
    }
    catalog = {
        "schemaVersion": 2,
        "networkName": "synthetic-test-fixture",
        "classCount": len(documents()),
        "identifierMap": [
            {
                "outputIndex": index,
                "documentIdentifier": document["identifier"],
                "document": document,
            }
            for index, document in enumerate(documents())
        ],
    }

    layouts_path = output / "visual-layouts.json.gz"
    payload = json.dumps(layouts, sort_keys=True).encode("utf-8")
    # mtime=0 keeps the archive byte-identical between runs.
    with open(layouts_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    catalog_path = output / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schemaVersion": 1,
        "engine": "synthetic-test-fixture",
        "catalog": {
            "path": "catalog.json",
            "sha256": sha256(catalog_path),
            "classCount": catalog["classCount"],
            "namedDocumentCount": catalog["classCount"],
        },
        "visualLayouts": {
            "path": "visual-layouts.json.gz",
            "sha256": sha256(layouts_path),
            "layoutCount": len(layouts["layouts"]),
            "fieldCount": sum(
                len(layout["fields"]) for layout in layouts["layouts"].values()
            ),
            "barcodeRegionCount": sum(
                len(layout["barcodes"]) for layout in layouts["layouts"].values()
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote synthetic catalog to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
