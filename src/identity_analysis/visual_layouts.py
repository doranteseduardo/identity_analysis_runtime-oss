"""Portable catalog-driven visual field layouts."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from copy import deepcopy
from datetime import date
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

from .mask_lexicons import (
    TEXT_MONTHS,
    named_token_valid,
    structural_mask_results,
    text_month_number,
)
from .ocr import resolve_ocr_locale, run as recognize_line


PERSON_NAME_FIELDS = {
    "surnameAndGivenNames",
    "surname",
    "firstSurname",
    "secondSurname",
    "givenNames",
    "fathersName",
    "mothersName",
}
LATIN_REGIONAL_MODELS = {1034, 1050, 1055, 1061, 1066}


def mask_tokens(mask: str) -> list[str]:
    return re.findall(r"\{([^}\[\"\s]+)", mask.split("|", 1)[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_bounds(bounds: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left, top, right, bottom = bounds
    return (
        max(0, min(width, round(left * width))),
        max(0, min(height, round(top * height))),
        max(0, min(width, round(right * width))),
        max(0, min(height, round(bottom * height))),
    )


def is_visible_field(field: dict) -> bool:
    return field.get("lightType") in {6, 24} and not field.get("layer")


def layout_descriptor(layout: dict) -> dict:
    document_type = layout.get("documentType") or {}
    document_format = layout.get("documentFormat") or {}
    return {
        "name": layout.get("caption"),
        "country": layout.get("country"),
        "countryCodes": list(layout.get("isoCodes") or []),
        "type": document_type.get("name"),
        "typeIdentifier": document_type.get("value"),
        "format": document_format.get("name"),
        "formatIdentifier": document_format.get("value"),
        "edition": layout.get("year"),
        "orientation": layout.get("orientation"),
        "twoSided": bool(layout.get("twoSided")),
        "mainDocument": bool(layout.get("mainDocument")),
    }


def explicit_page_role(layout: dict) -> dict | None:
    caption = layout.get("caption", "")
    if re.search(r"\bfront\s+cover\b", caption, re.IGNORECASE):
        return {
            "role": "front_cover",
            "method": "caption_marker",
            "confidence": "declared",
        }
    if re.search(r"\bback\s+cover\b", caption, re.IGNORECASE):
        return {
            "role": "back_cover",
            "method": "caption_marker",
            "confidence": "declared",
        }
    if re.search(r"\b(?:front|side\s*A)\b", caption, re.IGNORECASE):
        return {
            "role": "front",
            "method": "caption_marker",
            "confidence": "declared",
        }
    if re.search(
        r"\b(?:back|reverse|rear|side\s*B)\b", caption, re.IGNORECASE
    ):
        return {
            "role": "back",
            "method": "caption_marker",
            "confidence": "declared",
        }
    match = re.search(r"\bpage\s*([0-9]+|[A-Z])\b", caption, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).upper()
    ordinal = int(token) if token.isdigit() else ord(token) - ord("A") + 1
    return {
        "role": "numbered_page",
        "ordinal": ordinal,
        "method": "caption_marker",
        "confidence": "declared",
    }


def layout_page_role(root: Path, identifier: int) -> dict:
    layout = visual_layout(root, identifier)
    if layout is None:
        return {
            "role": "unknown",
            "method": "unavailable",
            "confidence": "none",
        }
    explicit = explicit_page_role(layout)
    if explicit:
        return explicit

    related_back_identifiers = []
    for child_identifier in layout.get("childDocuments") or []:
        child = visual_layout(root, child_identifier)
        child_role = explicit_page_role(child) if child else None
        if child_role and child_role["role"] in {"back", "back_cover"}:
            related_back_identifiers.append(child_identifier)
    if related_back_identifiers:
        return {
            "role": "front",
            "method": "related_back_layout",
            "confidence": "inferred",
            "relatedLayoutIdentifiers": related_back_identifiers,
        }

    related_page_identifiers = []
    for page_identifier in layout.get("pairedPages") or []:
        page = visual_layout(root, page_identifier)
        page_role = explicit_page_role(page) if page else None
        if (
            page_role
            and page_role["role"] == "numbered_page"
            and page_role.get("ordinal", 1) > 1
        ):
            related_page_identifiers.append(page_identifier)
    if related_page_identifiers:
        return {
            "role": "primary_page",
            "ordinal": 1,
            "method": "related_numbered_page",
            "confidence": "inferred",
            "relatedLayoutIdentifiers": related_page_identifiers,
        }

    return {
        "role": "unknown",
        "method": "unavailable",
        "confidence": "none",
    }


def text_regions(layout: dict) -> list[dict]:
    regions = []
    seen = set()
    for definition in layout.get("fields", []):
        if not is_visible_field(definition):
            continue
        key = (
            definition.get("number"),
            definition.get("name"),
            tuple(definition["bounds"]),
        )
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            {
                "number": definition.get("number"),
                "type": definition.get("type"),
                "name": definition.get("name"),
                "bounds": definition["bounds"],
                "locale": definition.get("lcid"),
                "mask": definition.get("mask"),
                "textHeight": definition.get("textHeight"),
                "colorType": definition.get("colorType"),
                "fontLayer": definition.get("fontLayer"),
                "layer": definition.get("layer"),
                "lowContrast": bool(definition.get("lowContrastText")),
                "backgroundRemoval": bool(definition.get("removeBackground")),
                "comparisonMode": definition.get("inComparison"),
                "usedForComparison": bool(definition.get("inComparison")),
            }
        )
    return regions


def graphic_regions(layout: dict, names: set[str] | None = None) -> list[dict]:
    regions = []
    seen = set()
    for definition in layout.get("graphics", []):
        if not is_visible_field(definition):
            continue
        if names is not None and definition["name"] not in names:
            continue
        key = (definition["name"], tuple(definition["bounds"]))
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            {
                "type": definition["type"],
                "name": definition["name"],
                "bounds": definition["bounds"],
                "faceExpected": bool(definition.get("checkFacePresent")),
            }
        )
    return regions


def barcode_regions(layout: dict) -> list[dict]:
    regions = []
    seen = set()
    for definition in layout.get("barcodes", []):
        if not is_visible_field(definition):
            continue
        key = tuple(definition["bounds"])
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            {
                "type": definition.get("type"),
                "name": definition.get("name"),
                "bounds": definition["bounds"],
                "orientation": definition.get("orientation"),
                "codeType": definition.get("codeType"),
                "codeClass": definition.get("codeClass"),
                "pdf417Codec": definition.get("pdf417Codec"),
            }
        )
    return regions


def security_regions(layout: dict) -> list[dict]:
    return [dict(region) for region in layout.get("securityRegions", [])]


def reference_patches(layout: dict, include_image_data: bool = False) -> list[dict]:
    patches = []
    for definition in layout.get("referencePatches", []):
        patch = dict(definition)
        image = dict(patch.get("image") or {})
        if not include_image_data:
            image.pop("data", None)
        patch["image"] = image
        patches.append(patch)
    return patches


def catalog_hints(layout: dict) -> dict:
    return deepcopy(layout.get("catalogHints", {}))


def layout_relations(layout: dict) -> dict:
    return {
        "mainDocument": bool(layout.get("mainDocument")),
        "parentIdentifier": layout.get("parentIdentifier"),
        "childIdentifiers": list(layout.get("childDocuments") or []),
        "pairedPageIdentifiers": list(layout.get("pairedPages") or []),
    }


def declared_layout_requirements(layout: dict) -> dict:
    hints = layout.get("catalogHints", {})
    capture = hints.get("capture", {})
    recognition = hints.get("recognition", {})
    electronic = hints.get("electronicDocument", {})
    sources = hints.get("sourceReferences", {})
    authenticity = hints.get("authenticityConfiguration", {})

    def present(values: dict) -> dict:
        return {key: value for key, value in values.items() if value is not None}

    raw_authenticity_parameters = authenticity.get("authenticity", {})
    authenticity_parameters = present(
        {
            "uvDullPaper": deepcopy(raw_authenticity_parameters.get("uvDullPaper")),
            "photoEmbedType": raw_authenticity_parameters.get("photoEmbedType"),
            "checkPhotoEmbedType": raw_authenticity_parameters.get(
                "checkPhotoEmbedType"
            ),
            "photoReplacementCheck": raw_authenticity_parameters.get(
                "photoReplacementCheck"
            ),
            "wholePageLuminescence": raw_authenticity_parameters.get(
                "dWholePageLuminescense"
            ),
            "checkFalseLuminescence": raw_authenticity_parameters.get(
                "dCheckFalseLuminescense"
            ),
            "hasDynamicObjects": raw_authenticity_parameters.get(
                "dHaveDynamicObjects"
            ),
            "checkFalseWatermarks": raw_authenticity_parameters.get(
                "dCheckFalseWatermarks"
            ),
            "checkBlankCaptions": raw_authenticity_parameters.get(
                "dCheckBlankCaptions"
            ),
            "checkPhotoHalo": raw_authenticity_parameters.get("dCheckPhotoHalo"),
            "checkFaceCount": raw_authenticity_parameters.get("dCheckFaceCount"),
            "kinegramModelType": raw_authenticity_parameters.get(
                "kinegramModelType"
            ),
            "fibers": deepcopy(raw_authenticity_parameters.get("fibers")),
        }
    )

    return {
        "capture": present(
            {
                "requiredLightMask": capture.get("dNecessaryLights"),
                "uvExposure": capture.get("dUVExp"),
                "opticallyVariableExposure": capture.get("dOVIExp"),
                "hologramTiltType": capture.get("hologramTiltType"),
                "backgroundDeltaExpectation": capture.get("dBackgroundDeltaExpectation"),
                "backgroundDeltaThreshold": capture.get("dBackgroundDeltaST"),
                "backgroundLightExpectation": capture.get("dBackgroundLightExpectation"),
                "backgroundLightThreshold": capture.get("dBackgroundLightST"),
            }
        ),
        "recognition": present(
            {
                "ocrSearchToleranceX": recognition.get("dOCRSTX"),
                "ocrSearchToleranceY": recognition.get("dOCRSTY"),
                "ocrExpectationX": recognition.get("dOCRExpectationX"),
                "ocrExpectationY": recognition.get("dOCRExpectationY"),
                "tag": recognition.get("recognTag"),
                "barcodeType": recognition.get("dBarcode"),
                "mrzType": recognition.get("dMRZ"),
                "mrzHeight": recognition.get("dMRZHeight"),
                "mrzHeightTolerance": recognition.get("dMRZHeightDiff"),
                "mrzWidth": recognition.get("dMRZWidth"),
                "mrzWidthTolerance": recognition.get("dMRZWidthDiff"),
                "mrzSymbolPosition": recognition.get("dMRZSymbolPos"),
                "mrzSymbolPositionTolerance": recognition.get("dMRZSymbolPosDiff"),
                "mrzLineDistance": recognition.get("dMRZDistanceBetweenLines"),
                "mrzFontIdentifier": recognition.get("dMRZFontID"),
                "mrzFormat": recognition.get("dMRZFormat"),
                "filterMask": recognition.get("recognFilterMask"),
                "cyrillicConversion": recognition.get("dCyrilicConvert"),
                "handwrittenOption": recognition.get("dHandwrittenOption"),
                "textIdentifier": recognition.get("textID"),
            }
        ),
        "electronicDocument": present(
            {
                "chipPage": electronic.get("chipPage"),
                "dataSetIdentifiers": electronic.get("dFDSID"),
                "certificateValidityPeriodDisabled": electronic.get(
                    "dDisableDSCertificateValidityPeriod"
                ),
            }
        ),
        "sourceReferences": present(
            {
                "authenticationSourceType": sources.get("AuthSourceType"),
                "authenticationDocumentIdentifier": sources.get("AuthSourceDocumentID"),
                "hologramDocumentIdentifier": sources.get("HoloSourceDocumentID"),
            }
        ),
        "authenticityConfiguration": present(
            {
                "checkMask": authenticity.get("dAuthenticity"),
                "parameters": authenticity_parameters,
            }
        ),
    }


def mrz_physical_geometry(layout: dict) -> dict | None:
    recognition = layout.get("catalogHints", {}).get("recognition", {})
    dimensions = layout.get("dimensionsMm") or {}
    document_width = dimensions.get("width")
    document_height = dimensions.get("height")
    width_units = recognition.get("dMRZWidth")
    height_units = recognition.get("dMRZHeight")
    if not all(
        isinstance(value, (int, float)) and value > 0
        for value in (document_width, document_height, width_units, height_units)
    ):
        return None
    result = {
        "widthRatio": (width_units / 10.0) / document_width,
        "heightRatio": (height_units / 10.0) / document_height,
        "sourceUnit": "tenths_of_millimeter",
        "raw": {
            key: recognition[key]
            for key in (
                "dMRZFormat",
                "dMRZFontID",
                "dMRZDistanceBetweenLines",
                "dMRZHeight",
                "dMRZHeightDiff",
                "dMRZWidth",
                "dMRZWidthDiff",
                "dMRZSymbolPos",
                "dMRZSymbolPosDiff",
            )
            if key in recognition
        },
    }
    line_distance = recognition.get("dMRZDistanceBetweenLines")
    if isinstance(line_distance, (int, float)) and line_distance > 0:
        result["lineDistanceRatio"] = (line_distance / 10.0) / document_height
    return result


class VisualLayoutCatalog:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        entry = manifest.get("visualLayouts")
        if not entry:
            raise ValueError("Document classifier assets do not contain visual layouts")
        path = self.root / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Visual layout asset hash mismatch: {entry['path']}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        self.field_types = payload["fieldTypes"]
        self.graphic_field_types = payload.get("graphicFieldTypes", {})
        self.layouts = payload["layouts"]
        self.country_codes = frozenset(
            code.upper()
            for layout in self.layouts.values()
            for code in layout.get("isoCodes", [])
            if isinstance(code, str)
            and len(code) == 3
            and code.isascii()
            and code.isalpha()
        )

    def get(self, identifier: int) -> dict | None:
        return self.layouts.get(str(identifier))


@lru_cache(maxsize=2)
def layout_catalog_for_root(root: str) -> VisualLayoutCatalog:
    return VisualLayoutCatalog(Path(root) / "document_classifier")


def optional_layout_catalog(root: Path) -> VisualLayoutCatalog | None:
    """Return the layout catalog, or ``None`` when it is not installed.

    Layout metadata is an optional asset family: a deployment that only ships
    the recognition models still serves every MRZ and barcode route, so a
    missing or unreadable catalog degrades to "unavailable" instead of raising.
    """

    try:
        return layout_catalog_for_root(str(Path(root).resolve()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def layout_catalog_available(root: Path) -> bool:
    return optional_layout_catalog(root) is not None


def visual_layout(root: Path, identifier: int) -> dict | None:
    catalog = optional_layout_catalog(root)
    return catalog.get(identifier) if catalog is not None else None


def document_country_codes(root: Path) -> frozenset[str]:
    catalog = optional_layout_catalog(root)
    return catalog.country_codes if catalog is not None else frozenset()


def warm_up_visual_layouts(root: Path) -> None:
    layout_catalog_for_root(str(Path(root).resolve()))


def layout_field_names(layout: dict) -> set[str]:
    return {
        definition["name"] for definition in layout.get("fields", [])
    } | {
        rule["target"]["name"] for rule in layout.get("assemblies", [])
    }


def requested_layout_field_dependencies(
    layout: dict, requested: set[str]
) -> set[str]:
    expanded = set(requested)
    rules = layout.get("assemblies", [])
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule["target"]["name"] not in expanded:
                continue
            for part in rule["parts"]:
                if part["kind"] != "field":
                    continue
                for definition in layout.get("fields", []):
                    if definition["type"] == part["type"] and (
                        part.get("locale") is None
                        or definition.get("lcid") == part["locale"]
                    ):
                        if definition["name"] not in expanded:
                            expanded.add(definition["name"])
                            changed = True
                for source_rule in rules:
                    target = source_rule["target"]
                    if target["type"] == part["type"] and (
                        part.get("locale") is None
                        or target.get("locale") == part["locale"]
                    ):
                        if target["name"] not in expanded:
                            expanded.add(target["name"])
                            changed = True
    return expanded


def recognize_visual_layout(
    resource: Path,
    image: Image.Image,
    layout: dict,
    field_names: set[str] | None = None,
    adaptive_search: bool = True,
) -> list[dict]:
    recognized = []
    recognition_names = (
        requested_layout_field_dependencies(layout, field_names)
        if field_names is not None
        else None
    )
    country_codes = document_country_codes(resource)
    for source_definition in layout["fields"]:
        definition = dict(source_definition)
        definition["documentOrientation"] = layout.get("orientation", 0)
        if not is_visible_field(definition):
            continue
        if (
            recognition_names is not None
            and definition["name"] not in recognition_names
        ):
            continue
        bounds = definition["bounds"]
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        result = recognize_field(
            resource,
            image,
            definition,
            country_codes,
            adaptive_search,
        )
        recognized.append(
            {
                "type": definition["type"],
                "name": definition["name"],
                "value": result["text"].strip(" <"),
                "confidence": result["confidence"],
                "bounds": definition["bounds"],
                "mask": definition.get("mask"),
                "locale": definition.get("lcid"),
                "orientationCorrection": result.get("orientationCorrection"),
                "preprocessing": result.get("preprocessing"),
                "colorType": definition.get("colorType"),
                "fontLayer": definition.get("fontLayer"),
                "layer": definition.get("layer"),
                "comparisonMode": definition.get("inComparison"),
            }
        )
    recognized.extend(assemble_visual_fields(recognized, layout.get("assemblies", [])))
    if field_names is not None:
        return [item for item in recognized if item["name"] in field_names]
    return recognized


def assembly_source(recognized: list[dict], reference: dict) -> dict | None:
    candidates = [
        item
        for item in recognized
        if item["type"] == reference["type"]
        and item.get("value")
        and (
            reference.get("locale") is None
            or item.get("locale") == reference["locale"]
        )
    ]
    return max(candidates, key=lambda item: item["confidence"], default=None)


def assemble_visual_fields(recognized: list[dict], rules: list[dict]) -> list[dict]:
    assembled = []
    for rule in rules:
        target = rule["target"]
        if any(
            item["type"] == target["type"]
            and item.get("value")
            and (
                target.get("locale") is None
                or item.get("locale") == target["locale"]
            )
            for item in recognized
        ):
            continue
        resolved = [
            assembly_source(recognized, part) if part["kind"] == "field" else None
            for part in rule["parts"]
        ]
        available_indices = [index for index, source in enumerate(resolved) if source]
        fragments = []
        used = []
        for position, index in enumerate(available_indices):
            source = resolved[index]
            if position:
                previous_index = available_indices[position - 1]
                between = rule["parts"][previous_index + 1 : index]
                missing_reference = any(
                    part["kind"] == "field" for part in between
                )
                literals = [
                    part["value"]
                    for part in between
                    if part["kind"] == "literal"
                ]
                if missing_reference:
                    safe_literals = [value for value in literals if "/" not in value]
                    if safe_literals:
                        fragments.append(safe_literals[-1])
                else:
                    fragments.extend(literals)
            fragments.append(source["value"])
            used.append(source)
        value = "".join(fragments).strip(" ,/")
        if not value or not used:
            continue
        bounds = [
            min(item["bounds"][0] for item in used),
            min(item["bounds"][1] for item in used),
            max(item["bounds"][2] for item in used),
            max(item["bounds"][3] for item in used),
        ]
        assembled.append(
            {
                "type": target["type"],
                "name": target["name"],
                "value": value,
                "confidence": sum(item["confidence"] for item in used) / len(used),
                "bounds": bounds,
                "mask": None,
                "locale": target.get("locale"),
                "derived": True,
                "derivedFrom": [item["name"] for item in used],
            }
        )
    return assembled


def expected_line_count(mask: str | None) -> int | None:
    if not mask:
        return None
    primary = mask.split("|", 1)[0]
    count = len(re.findall(r"\{(?:TEXT|STRINGS)", primary))
    if count == 1 and "{STRINGS}" in primary:
        return None
    return count or None


def candidate_windows(definition: dict) -> list[list[float]]:
    left, top, right, bottom = definition["bounds"]
    region_height = bottom - top
    text_height = float(definition.get("textHeight") or region_height)
    window_height = min(region_height, max(text_height * 1.5, region_height / 8))
    if region_height <= window_height * 1.8:
        search_top = top - text_height * 0.5
        search_bottom = bottom + text_height * 0.5
    else:
        search_top = top
        search_bottom = bottom
    search_height = search_bottom - search_top
    steps = max(3, min(10, round(search_height / max(text_height * 0.55, 0.001))))
    windows = [
        [left, search_top + (search_height - window_height) * index / (steps - 1), right,
         search_top + (search_height - window_height) * index / (steps - 1) + window_height]
        for index in range(steps)
    ]
    region_center = (top + bottom) / 2
    return sorted(
        windows,
        key=lambda bounds: abs((bounds[1] + bounds[3]) / 2 - region_center),
    )


def expected_mask_length(mask: str | None) -> int | None:
    if not mask:
        return None
    remainder = re.sub(r"\{[^}]*\}", "", mask.split("|", 1)[0])
    token_digits = sum(
        int(length)
        for length in re.findall(r"\{(\d+)D(?:\"[^\"]*\")?\}", mask.split("|", 1)[0])
    )
    length = token_digits + sum(character in "CDW" for character in remainder)
    return length or None


def numeric_date_tokens(mask: str) -> list[str] | None:
    primary = mask.split("|", 1)[0]
    tokens = mask_tokens(primary)
    date_tokens = [
        token
        for token in tokens
        if token.upper().startswith("DAY")
        or token.upper().startswith("MONTH")
        or token.upper().startswith("YEAR")
    ]
    if len(date_tokens) != 3:
        return None
    if not any(token.upper().startswith("DAY") for token in date_tokens):
        return None
    if not any(token.upper().startswith("YEAR") for token in date_tokens):
        return None
    month = next(
        (token for token in date_tokens if token.upper().startswith("MONTH")), None
    )
    if month not in {"MONTH_DD", "MONTH_D_DD", "MONTH_LZ"} | set(TEXT_MONTHS):
        return None
    return date_tokens


def valid_mask_date(mask: str, value: str) -> bool | None:
    tokens = numeric_date_tokens(mask)
    if not tokens:
        return None
    groups = re.findall(r"[\w]+", value, flags=re.UNICODE)
    if len(groups) != 3:
        return False
    components = {}
    for token, group in zip(tokens, groups):
        token_class = token.upper()
        if token_class.startswith("DAY"):
            if not group.isdigit():
                return False
            components["day"] = int(group)
        elif token_class.startswith("MONTH"):
            month = int(group) if group.isdigit() else text_month_number(token, group)
            if month is None:
                return False
            components["month"] = month
        elif token_class.startswith("YEAR"):
            if not group.isdigit():
                return False
            year = int(group)
            if len(group) == 2:
                year += 2000
            components["year"] = year
    try:
        date(components["year"], components["month"], components["day"])
    except (KeyError, ValueError):
        return False
    return True


def named_mask_results(mask: str, value: str) -> list[bool]:
    primary = mask.split("|", 1)[0]
    tokens = mask_tokens(primary)
    if len(tokens) == 1:
        result = named_token_valid(tokens[0], value)
        return [] if result is None else [result]
    groups = re.findall(r"[\w]+", value, flags=re.UNICODE)
    if len(tokens) != 3 or len(groups) != 3:
        return []
    return [
        result
        for token, group in zip(tokens, groups)
        if (result := named_token_valid(token, group)) is not None
    ]


def mask_compatibility_score(
    mask: str | None,
    value: str,
    country_codes: frozenset[str] | None = None,
) -> float:
    if not mask or not value:
        return 0.0
    score = 0.0
    expected_length = expected_mask_length(mask)
    normalized = "".join(character for character in value if character.isalnum())
    if expected_length:
        score += 0.15 - 0.03 * abs(len(normalized) - expected_length)

    remainder = re.sub(r"\{[^}]*\}", "", mask.split("|", 1)[0])
    fixed_pattern = "".join(character for character in remainder if character in "CDW")
    if fixed_pattern and len(normalized) == len(fixed_pattern):
        digit_checks = [
            character.isdigit()
            for expected, character in zip(fixed_pattern, normalized)
            if expected == "D"
        ]
        if digit_checks:
            score += 0.1 if all(digit_checks) else -0.2

    token_digit_lengths = [
        int(length)
        for length in re.findall(r"\{(\d+)D(?:\"[^\"]*\")?\}", mask.split("|", 1)[0])
    ]
    if token_digit_lengths and len(normalized) == sum(token_digit_lengths):
        score += 0.1 if normalized.isdigit() else -0.2

    valid_date = valid_mask_date(mask, value)
    if valid_date is not None:
        score += 0.25 if valid_date else -0.2
    elif "{YEAR}" in mask:
        score += 0.2 if re.fullmatch(r"(?:19|20)\d{2}", normalized) else -0.1
    if "{Sex_MF}" in mask:
        score += 0.15 if normalized.upper() in {"M", "F"} else -0.15
    if "{Country_Code}" in mask:
        valid_country_code = len(normalized) == 3 and normalized.isalpha()
        if valid_country_code and country_codes is not None:
            valid_country_code = normalized.upper() in country_codes
        score += 0.12 if valid_country_code else -0.12
    named_results = named_mask_results(mask, value)
    if named_results:
        score += 0.18 if all(named_results) else -0.18
    structure_results = structural_mask_results(mask, value)
    if structure_results:
        score += 0.2 if all(structure_results) else -0.2
    return score


def oriented_text_crops(
    crop: Image.Image, orientation: int | float | None
) -> list[tuple[Image.Image, int]]:
    degrees = int(orientation or 0) % 360
    if degrees == 0:
        return [(crop, 0)]
    if degrees == 180:
        return [(crop.rotate(180, expand=True), 180)]
    if degrees in {90, 270}:
        primary = (360 - degrees) % 360
        return [
            (crop.rotate(primary, expand=True), primary),
            (crop.rotate(degrees, expand=True), degrees),
        ]
    return [(crop, 0)]


def declared_text_variants(
    crop: Image.Image, definition: dict
) -> list[tuple[Image.Image, str]]:
    variants = [(crop, "original")]
    if definition.get("lowContrastText") or definition.get("removeBackground"):
        enhanced = ImageOps.autocontrast(ImageOps.grayscale(crop)).convert("RGB")
        variants.append((enhanced, "catalog_contrast_normalization"))
    return variants


def recognize_field(
    resource: Path,
    image: Image.Image,
    definition: dict,
    country_codes: frozenset[str] | None = None,
    adaptive_search: bool = True,
) -> dict:
    candidates = []
    expected_length = expected_mask_length(definition.get("mask"))
    mask_remainder = re.sub(
        r"\{[^}]*\}", "", (definition.get("mask") or "").split("|", 1)[0]
    )
    padding_ratios = (
        (0.0, 0.02, 0.04, 0.06)
        if mask_remainder and set(mask_remainder) <= {"D"}
        else (0.0, 0.02, 0.04, 0.06, 0.1)
    )
    requested_locale = definition.get("lcid")
    requested_model_locale = resolve_ocr_locale(resource, requested_locale)
    locales = [requested_locale]
    if (
        requested_model_locale in LATIN_REGIONAL_MODELS
        or (
            definition.get("name") in PERSON_NAME_FIELDS
            and requested_model_locale
        )
    ):
        locales.append(None)
    text_height = float(definition.get("textHeight") or 0.03)
    limit = expected_line_count(definition.get("mask"))
    if limit is None:
        region_height = definition["bounds"][3] - definition["bounds"][1]
        limit = max(1, min(6, round(region_height / max(text_height * 1.8, 0.001))))
    for bounds in candidate_windows(definition):
        window_candidate_start = len(candidates)
        width = bounds[2] - bounds[0]
        for padding_ratio in padding_ratios:
            padding_candidate_start = len(candidates)
            padding = width * padding_ratio
            padded = [bounds[0] - padding, bounds[1], bounds[2] + padding, bounds[3]]
            pixels = pixel_bounds(padded, image.size)
            if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
                continue
            crop = image.crop(pixels)
            orientation_variants = oriented_text_crops(
                crop, definition.get("documentOrientation")
            )
            for orientation_index, (
                oriented_crop,
                orientation_correction,
            ) in enumerate(orientation_variants):
                orientation_candidate_start = len(candidates)
                preprocessing_variants = declared_text_variants(
                    oriented_crop, definition
                )
                if adaptive_search and padding_ratio != 0.04:
                    preprocessing_variants = preprocessing_variants[:1]
                for preprocessing_index, (
                    recognition_crop,
                    preprocessing,
                ) in enumerate(preprocessing_variants):
                    preprocessing_candidate_start = len(candidates)
                    for locale in locales:
                        result = recognize_line(
                            resource,
                            recognition_crop,
                            False,
                            "minus-one-one",
                            locale=locale,
                        )
                        value = result["text"].strip(" <")
                        if value:
                            mask_score = mask_compatibility_score(
                                definition.get("mask"), value, country_codes
                            )
                            name_score = 0.0
                            if definition.get("name") in PERSON_NAME_FIELDS:
                                name_score = (
                                    -0.4
                                    if any(character.isdigit() for character in value)
                                    else 0.05
                                )
                            locale_score = 0.0
                            model_locale = result.get("modelLocale", 0)
                            if requested_model_locale in LATIN_REGIONAL_MODELS:
                                if model_locale == 0:
                                    locale_score = 0.2
                                elif all(ord(character) < 128 for character in value):
                                    locale_score = -0.2
                                else:
                                    locale_score = 0.05
                            candidates.append(
                                {
                                    "value": value,
                                    "confidence": result["confidence"],
                                    "score": (
                                        result["confidence"]
                                        + mask_score
                                        + name_score
                                        + locale_score
                                    ),
                                    "maskScore": mask_score,
                                    "bounds": padded,
                                    "center": (bounds[1] + bounds[3]) / 2,
                                    "orientationCorrection": orientation_correction,
                                    "preprocessing": preprocessing,
                                }
                            )
                    if adaptive_search and preprocessing_index == 0:
                        original_candidates = candidates[
                            preprocessing_candidate_start:
                        ]
                        if original_candidates:
                            best_original = max(
                                original_candidates,
                                key=lambda item: item["score"],
                            )
                            if (
                                best_original["confidence"] >= 0.99
                                and best_original["score"] >= 0.99
                                and best_original["maskScore"] >= 0
                            ):
                                break
                if (
                    adaptive_search
                    and orientation_index == 0
                    and len(orientation_variants) > 1
                ):
                    orientation_candidates = candidates[
                        orientation_candidate_start:
                    ]
                    if orientation_candidates:
                        best_orientation = max(
                            orientation_candidates,
                            key=lambda item: item["score"],
                        )
                        if (
                            best_orientation["confidence"] >= 0.995
                            and best_orientation["score"] >= 0.995
                            and best_orientation["maskScore"] >= 0
                        ):
                            break
            if adaptive_search and padding_ratio >= 0.04:
                padding_candidates = candidates[padding_candidate_start:]
                if padding_candidates:
                    best_padding = max(
                        padding_candidates, key=lambda item: item["score"]
                    )
                    if (
                        best_padding["confidence"] >= 0.995
                        and best_padding["score"] >= 0.995
                        and best_padding["maskScore"] >= 0
                    ):
                        break
        if adaptive_search and limit == 1:
            window_candidates = candidates[window_candidate_start:]
            if window_candidates:
                best = max(window_candidates, key=lambda item: item["score"])
                if (
                    best["confidence"] >= 0.995
                    and best["score"] >= 0.995
                    and best["maskScore"] >= 0
                ):
                    return {
                        "text": best["value"],
                        "confidence": best["confidence"],
                        "orientationCorrection": best["orientationCorrection"],
                        "preprocessing": best["preprocessing"],
                    }
    if not candidates:
        return {
            "text": "",
            "confidence": 0.0,
            "orientationCorrection": 0,
            "preprocessing": "original",
        }
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for candidate in candidates:
        if candidate["confidence"] < 0.18:
            continue
        if any(abs(candidate["center"] - kept["center"]) < text_height * 0.7 for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    if not selected:
        selected = candidates[:1]
    selected.sort(key=lambda item: item["center"])
    return {
        "text": "\n".join(item["value"] for item in selected),
        "confidence": sum(item["confidence"] for item in selected) / len(selected),
        "orientationCorrection": (
            selected[0]["orientationCorrection"]
            if len({item["orientationCorrection"] for item in selected}) == 1
            else [item["orientationCorrection"] for item in selected]
        ),
        "preprocessing": (
            selected[0]["preprocessing"]
            if len({item["preprocessing"] for item in selected}) == 1
            else [item["preprocessing"] for item in selected]
        ),
    }
