from pathlib import Path

from PIL import Image

import identity_analysis.pipeline as pipeline
import identity_analysis.visual_layouts as visual_layouts
from identity_analysis.visual_layouts import (
    explicit_page_role,
    layout_page_role,
    assemble_visual_fields,
    catalog_hints,
    candidate_windows,
    declared_text_variants,
    declared_layout_requirements,
    document_country_codes,
    expected_mask_length,
    graphic_regions,
    expected_line_count,
    is_visible_field,
    layout_descriptor,
    layout_field_names,
    layout_relations,
    mask_compatibility_score,
    mask_tokens,
    mrz_physical_geometry,
    oriented_text_crops,
    pixel_bounds,
    reference_patches,
    recognize_field,
    recognize_visual_layout,
    requested_layout_field_dependencies,
    security_regions,
    text_regions,
    valid_mask_date,
    visual_layout,
)
from identity_analysis.document_classifier import classify_document
from conftest import (
    FIXTURE_CATALOG,
    ID_FRONT,
    FRONT_LAYOUT,
    BACK_LAYOUT,
    requires_assets,
)
from identity_analysis.pipeline import (
    catalog_side_relation,
    catalog_visual_result,
    classifier_visual_layout_candidate,
    corroborated_mexican_birth_date,
    merge_mexican_voter_names,
    recognize_mexican_voter_name_fallback,
    recognize_mexican_voter_name_grid,
    recognize_mexican_curp_grid,
    valid_mexican_curp,
    mrz_search_ratios,
    recognized_document_visual_layout_candidate,
    should_scan_pdf417,
    should_try_td1,
)


ROOT = Path(__file__).resolve().parents[1]
BOOKLET_PAGE_1 = 900000010
BOOKLET_PAGE_2 = 900000011
RESIDENCE_FRONT = 900000020
RESIDENCE_BACK = 900000021


def recognized_field(
    field_type: int,
    name: str,
    value: str,
    confidence: float = 0.9,
    locale: int | None = None,
    bounds: list[float] | None = None,
) -> dict:
    return {
        "type": field_type,
        "name": name,
        "value": value,
        "confidence": confidence,
        "locale": locale,
        "bounds": bounds or [0.1, 0.1, 0.2, 0.2],
    }


def assembly_rule(target: dict, parts: list[dict]) -> dict:
    return {"target": target, "parts": parts}


def test_pixel_bounds_convert_normalized_geometry() -> None:
    assert pixel_bounds([0.25, 0.1, 0.75, 0.2], (1000, 500)) == (250, 50, 750, 100)


def test_pixel_bounds_are_clamped_to_image() -> None:
    assert pixel_bounds([-0.1, -0.2, 1.1, 1.2], (100, 50)) == (0, 0, 100, 50)


def test_visible_fields_exclude_non_primary_layers() -> None:
    assert is_visible_field({"lightType": 24})
    assert is_visible_field({"lightType": 6})
    assert not is_visible_field({"lightType": 24, "layer": 1})
    assert not is_visible_field({"lightType": 128})


def test_search_region_uses_text_height_and_mask() -> None:
    definition = {
        "bounds": [0.2, 0.3, 0.8, 0.5],
        "textHeight": 0.04,
        "mask": '{TEXT"8+"}^{TEXT"145+"}^{TEXT"9+"}',
    }

    assert expected_line_count(definition["mask"]) == 3
    assert expected_mask_length("{_@LEX(CHECK)@}CCCCDDDD") == 8
    assert expected_mask_length('{4D"1"}{6D"2"}') == 10
    windows = candidate_windows(definition)
    assert len(windows) > 3
    assert all(round(window[3] - window[1], 8) == 0.06 for window in windows)


def test_mask_semantics_validate_numeric_dates() -> None:
    day_first = "{DAY}/{MONTH_DD}/{YEAR}"
    year_first = "{YEAR}-{MONTH_DD}-{DAY}"

    assert valid_mask_date(day_first, "29/02/2024") is True
    assert valid_mask_date(day_first, "29/02/2023") is False
    assert valid_mask_date(year_first, "2025-12-31") is True
    assert valid_mask_date("{DAY} {MONTH} {YEAR}", "31 DEC 2025") is None
    assert valid_mask_date("{DAY}-{MONTH_SPA}-{YEAR}", "31-DIC-2025") is True
    assert valid_mask_date("{DAY}-{MONTH_SPA}-{YEAR}", "31-FEB-2025") is False
    assert mask_compatibility_score(day_first, "29/02/2024") > 0
    assert mask_compatibility_score(day_first, "41/19/2024") < 0
    assert mask_compatibility_score("{DAY_DD}", "07") > 0
    assert mask_compatibility_score("{DAY_DD}", "7") < 0
    assert mask_compatibility_score("{MONTH_DD}", "12") > 0
    assert mask_compatibility_score("{MONTH_DD}", "13") < 0
    assert mask_compatibility_score("{YEAR_DD}", "25") > 0


def test_mask_semantics_score_safe_token_classes() -> None:
    assert mask_compatibility_score("DDDDDD", "123456") > mask_compatibility_score(
        "DDDDDD", "12AB56"
    )
    assert mask_compatibility_score("{6D}", "123456") > mask_compatibility_score(
        "{6D}", "ABCDEF"
    )
    assert mask_compatibility_score("{Sex_MF}", "F") > 0
    assert mask_compatibility_score("{Sex_MF}", "X") < 0
    assert mask_compatibility_score("{Country_Code}", "SWE") > 0


def test_country_code_masks_use_portable_catalog_membership() -> None:
    codes = document_country_codes(FIXTURE_CATALOG)

    # Membership comes from the installed catalog, whatever it declares.
    assert codes == frozenset({"ZZT"})
    assert not ({"", "D<<", "PS<", "TM<"} & codes)
    assert mask_compatibility_score("{Country_Code}", "ZZT", codes) > 0
    assert mask_compatibility_score("{Country_Code}", "SWE", codes) < 0
    # Without a catalog the check falls back to shape only.
    assert mask_compatibility_score("{Country_Code}", "SWE", None) > 0


def test_mask_semantics_score_public_lexicons() -> None:
    assert mask_compatibility_score("{MONTH_SPA}", "DIC") > 0
    assert mask_compatibility_score("{MONTH_SPA}", "XYZ") < 0
    assert mask_compatibility_score("{DAY}-{MONTH_SPA}-{YEAR}", "31-DIC-2025") > 0
    assert mask_compatibility_score("{DAY}-{MONTH_SPA}-{YEAR}", "31-XYZ-2025") < 0
    assert mask_compatibility_score("{MONTH_FRA_Full}", "AOUT") > 0
    assert mask_compatibility_score("{MONTH_FRA_Full}", "AOÛT") > 0
    assert mask_compatibility_score("{STATE_CODE_USA}", "CA") > 0
    assert mask_compatibility_score("{STATE_CODE_USA}", "ZZ") < 0
    assert mask_compatibility_score("{STATE_CODE_CAN}", "QC") > 0
    assert mask_compatibility_score("{STATE_CODE_AUS}", "NSW") > 0
    assert mask_compatibility_score("{EYE_COLOR_D20}", "BLU") > 0
    assert mask_compatibility_score("{HAIR_COLOR_D20}", "BLN") > 0
    assert mask_compatibility_score("{Sex_WORD}", "FEMALE") > 0
    assert mask_compatibility_score("{Sex_WORD}", "UNKNOWN") < 0
    assert mask_compatibility_score("{Sex_МЖ}", "Ж") > 0
    assert mask_compatibility_score("{Sex_МЖ}", "F") < 0


def test_mask_parser_preserves_unicode_token_names() -> None:
    assert mask_tokens("!{Sex_МЖ}!/{Sex_MF}") == ["Sex_МЖ", "Sex_MF"]


def test_mask_semantics_score_postal_and_height_structures() -> None:
    usa = '{CITY_USA}, {STATE_CODE_USA} {POSTAL_CODE_USA_9}'
    canada = '{CITY_CAN} {STATE_CODE_CAN}^{POSTAL_CODE_CAN_P1} {POSTAL_CODE_CAN_P2}'

    assert mask_compatibility_score(usa, "Seattle, WA 98101-1234") > 0
    assert mask_compatibility_score(usa, "Seattle, WA ABCDE") < 0
    assert mask_compatibility_score(canada, "Toronto ON M5V 3A8") > 0
    assert mask_compatibility_score("{FOOTS}-{INCHES}", "6-02") > 0
    assert mask_compatibility_score("{FOOTS}-{INCHES}", "6-15") < 0


def test_mask_semantics_score_declared_measures_and_compact_classes() -> None:
    assert mask_compatibility_score("{POUNDS}", "180 LBS") > 0
    assert mask_compatibility_score("{POUNDS}", "HEAVY") < 0
    assert mask_compatibility_score("{KGS}!{MEASURE}", "82 KG") > 0
    assert mask_compatibility_score("{METRE},{CMS} {MEASURE}", "1,82") > 0
    assert mask_compatibility_score("{EYE_COLOR_2C}", "BL") > 0
    assert mask_compatibility_score("{EYE_COLOR_2C}", "BLUE") < 0
    assert mask_compatibility_score("{Blood_group}", "O+") > 0
    assert mask_compatibility_score("{Blood_group}", "X+") < 0


def test_mask_semantics_score_unicode_text_shapes() -> None:
    assert mask_compatibility_score("{WORD}", "O’CONNOR") > 0
    assert mask_compatibility_score("{WORD}", "A12") < 0
    assert mask_compatibility_score("{WORD_D}", "A12") > 0
    assert mask_compatibility_score("{Surname_ENG}", "GARCÍA-LÓPEZ") > 0
    assert mask_compatibility_score("{Given_Name_ENG}", "ANNA MARIA") > 0


def test_assembled_field_composes_available_sources() -> None:
    recognized = [
        recognized_field(8, "surname", "DOE", bounds=[0.1, 0.2, 0.3, 0.3]),
        recognized_field(9, "givenNames", "JANE", 0.8, bounds=[0.4, 0.2, 0.7, 0.3]),
    ]
    rules = [
        assembly_rule(
            {"type": 25, "name": "surnameAndGivenNames", "locale": None},
            [
                {"kind": "field", "type": 8, "name": "surname", "locale": None},
                {"kind": "literal", "value": ","},
                {"kind": "field", "type": 9, "name": "givenNames", "locale": None},
            ],
        )
    ]

    assert assemble_visual_fields(recognized, rules) == [
        {
            "type": 25,
            "name": "surnameAndGivenNames",
            "value": "DOE,JANE",
            "confidence": 0.8500000000000001,
            "bounds": [0.1, 0.2, 0.7, 0.3],
            "mask": None,
            "locale": None,
            "derived": True,
            "derivedFrom": ["surname", "givenNames"],
        }
    ]


def test_assembled_field_respects_locale_and_omits_dangling_separator() -> None:
    recognized = [
        recognized_field(76, "addressStreet", "RUA A", locale=1046),
        recognized_field(76, "addressStreet", "CALLE B", confidence=0.99, locale=3082),
        recognized_field(77, "addressCity", "LISBOA", locale=1046),
    ]
    rules = [
        assembly_rule(
            {"type": 17, "name": "address", "locale": 1046},
            [
                {"kind": "field", "type": 76, "name": "addressStreet", "locale": 1046},
                {"kind": "literal", "value": ", "},
                {"kind": "field", "type": 346, "name": "addressBuilding", "locale": 1046},
                {"kind": "literal", "value": "/"},
                {"kind": "field", "type": 189, "name": "addressUnit", "locale": 1046},
                {"kind": "literal", "value": ", "},
                {"kind": "field", "type": 77, "name": "addressCity", "locale": 1046},
            ],
        )
    ]

    result = assemble_visual_fields(recognized, rules)

    assert result[0]["value"] == "RUA A, LISBOA"
    assert result[0]["derivedFrom"] == ["addressStreet", "addressCity"]


def test_direct_target_suppresses_assembled_duplicate() -> None:
    recognized = [
        recognized_field(8, "surname", "DOE"),
        recognized_field(9, "givenNames", "JANE"),
        recognized_field(25, "surnameAndGivenNames", "DOE JANE"),
    ]
    rules = [
        assembly_rule(
            {"type": 25, "name": "surnameAndGivenNames", "locale": None},
            [
                {"kind": "field", "type": 8, "name": "surname", "locale": None},
                {"kind": "literal", "value": " "},
                {"kind": "field", "type": 9, "name": "givenNames", "locale": None},
            ],
        )
    ]

    assert assemble_visual_fields(recognized, rules) == []


def test_declared_layout_matches_catalog_geometry() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)

    assert layout is not None
    assert layout["caption"] == "Specimen Identity Card (2024) Front"
    assert layout["dimensionsMm"] == {"width": 86, "height": 54}
    assert {field["name"] for field in layout["fields"]} >= {
        "surname",
        "givenNames",
        "documentNumber",
        "dateOfBirth",
    }
    surname = next(
        field for field in layout["fields"] if field["name"] == "surname"
    )
    assert surname["bounds"] == [0.330, 0.213, 0.620, 0.291]
    graphics = {region["name"]: region for region in graphic_regions(layout)}
    assert graphics["portrait"]["bounds"] == [0.055, 0.300, 0.300, 0.700]
    assert graphics["portrait"]["faceExpected"] is True
    assert "ghostPortrait" in graphics


def test_booklet_layout_preserves_semantic_page_relations() -> None:
    layout = visual_layout(FIXTURE_CATALOG, BOOKLET_PAGE_2)

    assert layout["caption"] == "Specimen Travel Booklet (2024) Page 2"
    assert layout["pairedPages"] == [BOOKLET_PAGE_1]
    relation = catalog_side_relation(
        {
            "documentClassification": {
                "candidates": [
                    {
                        "documentIdentifier": BOOKLET_PAGE_2,
                        "confidence": 0.99,
                        "document": {"caption": layout["caption"]},
                    }
                ]
            }
        },
        FIXTURE_CATALOG,
    )
    assert relation["relationType"] == "paired_page"
    assert [item["identifier"] for item in relation["relatedDocuments"]] == layout[
        "pairedPages"
    ]


def test_catalog_preserves_declared_assembled_fields() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)

    assert layout["assemblies"][0]["target"] == {
        "type": 8,
        "locale": 1033,
        "name": "surnameAndGivenNames",
    }
    assert [part["kind"] for part in layout["assemblies"][0]["parts"]] == [
        "field",
        "literal",
        "field",
    ]


def test_catalog_preserves_security_and_reference_regions() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)

    security = security_regions(layout)
    patches = reference_patches(layout)
    patches_with_data = reference_patches(layout, include_image_data=True)

    assert security[0] == {
        "number": 1,
        "name": "guillocheBackground",
        "lightType": 6,
        "checkType": 1,
        "bounds": [0.0, 0.115, 1.0, 1.0],
    }
    assert patches[0]["bounds"] == [0.340, 0.020, 0.470, 0.090]
    assert patches[0]["image"]["format"] == ".PNG"
    assert len(patches[0]["image"]["sha256"]) == 64
    assert "data" not in patches[0]["image"]
    assert patches_with_data[0]["image"]["data"]


def test_catalog_projects_normalized_layout_metadata() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)

    descriptor = layout_descriptor(layout)
    text = text_regions(layout)
    relations = layout_relations(layout)

    assert descriptor == {
        "name": "Specimen Identity Card (2024) Front",
        "country": "ZZT",
        "countryCodes": ["ZZT"],
        "type": "IdentityCard",
        "typeIdentifier": 12,
        "format": "ID1",
        "formatIdentifier": 0,
        "edition": "2024",
        "orientation": 0,
        "twoSided": True,
        "mainDocument": True,
    }
    assert text[0]["name"] == "surname"
    assert text[0]["bounds"] == [0.330, 0.213, 0.620, 0.291]
    assert text[0]["comparisonMode"] == 1
    assert text[0]["usedForComparison"] is True
    assert "colorType" in text[0]
    assert "fontLayer" in text[0]
    assert "backgroundRemoval" in text[0]
    date_of_birth = next(item for item in text if item["name"] == "dateOfBirth")
    assert date_of_birth["mask"].startswith("{DAY_DD}")
    assert relations == {
        "mainDocument": True,
        "parentIdentifier": None,
        "childIdentifiers": [BACK_LAYOUT],
        "pairedPageIdentifiers": [],
    }

    booklet_page = visual_layout(FIXTURE_CATALOG, BOOKLET_PAGE_2)
    assert layout_relations(booklet_page)["pairedPageIdentifiers"] == [BOOKLET_PAGE_1]


def test_explicit_page_role_preserves_declared_marker() -> None:
    assert explicit_page_role({"caption": "Residence Permit Side B"}) == {
        "role": "back",
        "method": "caption_marker",
        "confidence": "declared",
    }
    assert explicit_page_role({"caption": "ePassport Page 3"}) == {
        "role": "numbered_page",
        "ordinal": 3,
        "method": "caption_marker",
        "confidence": "declared",
    }


def test_layout_page_role_infers_front_from_related_back() -> None:
    role = layout_page_role(FIXTURE_CATALOG, RESIDENCE_FRONT)

    assert role["role"] == "front"
    assert role["method"] == "related_back_layout"
    assert role["confidence"] == "inferred"
    assert role["relatedLayoutIdentifiers"] == [RESIDENCE_BACK]


def test_layout_page_role_infers_primary_from_related_numbered_page() -> None:
    role = layout_page_role(FIXTURE_CATALOG, BOOKLET_PAGE_1)

    assert role["role"] == "primary_page"
    assert role["ordinal"] == 1
    assert role["method"] == "related_numbered_page"
    assert role["confidence"] == "inferred"
    assert BOOKLET_PAGE_2 in role["relatedLayoutIdentifiers"]


def test_layout_page_role_reports_unknown_without_evidence() -> None:
    assert layout_page_role(FIXTURE_CATALOG, 999999999) == {
        "role": "unknown",
        "method": "unavailable",
        "confidence": "none",
    }


def test_requested_layout_fields_expand_assembly_dependencies() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)

    available = layout_field_names(layout)
    dependencies = requested_layout_field_dependencies(
        layout, {"surnameAndGivenNames"}
    )

    assert "surnameAndGivenNames" in available
    assert dependencies > {"surnameAndGivenNames"}
    assert {"surname", "givenNames"}.issubset(dependencies)


@requires_assets
def test_requested_layout_fields_reduce_ocr_calls(monkeypatch, catalog_assets) -> None:
    image = Image.open(ID_FRONT)
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)
    original = visual_layouts.recognize_line
    call_count = 0

    def counted_recognition(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(visual_layouts, "recognize_line", counted_recognition)
    complete = recognize_visual_layout(catalog_assets, image, layout)
    complete_calls = call_count
    call_count = 0
    selected = recognize_visual_layout(
        catalog_assets, image, layout, {"dateOfBirth"}
    )

    assert selected == [
        next(item for item in complete if item["name"] == "dateOfBirth")
    ]
    assert call_count <= complete_calls * 0.30


def test_oriented_text_crops_cover_declared_corrections() -> None:
    crop = Image.new("RGB", (20, 80))

    zero = oriented_text_crops(crop, 0)
    ninety = oriented_text_crops(crop, 90)
    one_eighty = oriented_text_crops(crop, 180)
    two_seventy = oriented_text_crops(crop, 270)

    assert [(image.size, angle) for image, angle in zero] == [((20, 80), 0)]
    assert [(image.size, angle) for image, angle in ninety] == [
        ((80, 20), 270),
        ((80, 20), 90),
    ]
    assert [(image.size, angle) for image, angle in one_eighty] == [
        ((20, 80), 180)
    ]
    assert [(image.size, angle) for image, angle in two_seventy] == [
        ((80, 20), 90),
        ((80, 20), 270),
    ]


def test_oriented_field_recognition_uses_horizontal_crop(monkeypatch) -> None:
    calls = []

    def recognize_horizontal(resource, image, *args, **kwargs):
        calls.append(image.size)
        return {
            "text": "1234" if image.width > image.height else "",
            "confidence": 0.99,
            "modelLocale": 0,
        }

    monkeypatch.setattr(visual_layouts, "recognize_line", recognize_horizontal)
    result = recognize_field(
        FIXTURE_CATALOG,
        Image.new("RGB", (20, 80)),
        {
            "name": "documentNumber",
            "bounds": [0.0, 0.0, 1.0, 1.0],
            "mask": "{4D}",
            "textHeight": 1.0,
            "documentOrientation": 90,
        },
    )

    assert result["text"] == "1234"
    assert result["orientationCorrection"] in {90, 270}
    assert calls
    assert all(width > height for width, height in calls)


def test_layout_orientation_reaches_field_recognition(monkeypatch) -> None:
    captured = {}

    def fake_recognition(resource, image, definition, *args, **kwargs):
        captured["orientation"] = definition["documentOrientation"]
        return {
            "text": "VALUE",
            "confidence": 1.0,
            "orientationCorrection": 270,
        }

    monkeypatch.setattr(visual_layouts, "recognize_field", fake_recognition)
    result = recognize_visual_layout(
        FIXTURE_CATALOG,
        Image.new("RGB", (100, 60)),
        {
            "orientation": 90,
            "fields": [
                {
                    "type": 1,
                    "name": "documentNumber",
                    "bounds": [0.1, 0.1, 0.9, 0.2],
                    "lightType": 6,
                }
            ],
            "assemblies": [],
        },
    )

    assert captured["orientation"] == 90
    assert result[0]["orientationCorrection"] == 270


def test_oriented_field_primary_direction_stops_perfect_fallback(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        visual_layouts,
        "oriented_text_crops",
        lambda crop, orientation: [
            (Image.new("RGB", (80, 20)), 270),
            (Image.new("RGB", (81, 20)), 90),
        ],
    )

    def perfect_recognition(resource, image, *args, **kwargs):
        calls.append(image.width)
        return {"text": "1234", "confidence": 1.0, "modelLocale": 0}

    monkeypatch.setattr(visual_layouts, "recognize_line", perfect_recognition)
    result = recognize_field(
        FIXTURE_CATALOG,
        Image.new("RGB", (20, 80)),
        {
            "name": "documentNumber",
            "bounds": [0.0, 0.0, 1.0, 1.0],
            "mask": "{4D}",
            "textHeight": 1.0,
            "documentOrientation": 90,
        },
    )

    assert result["text"] == "1234"
    assert result["orientationCorrection"] == 270
    assert 80 in calls
    assert 81 not in calls


def test_declared_text_variants_preserve_original_and_add_contrast() -> None:
    crop = Image.new("L", (20, 10), 110)
    for x in range(10, 20):
        for y in range(10):
            crop.putpixel((x, y), 140)

    original = declared_text_variants(crop, {})
    enhanced = declared_text_variants(crop, {"lowContrastText": True})

    assert [(image.mode, name) for image, name in original] == [("L", "original")]
    assert [name for _, name in enhanced] == [
        "original",
        "catalog_contrast_normalization",
    ]
    assert enhanced[1][0].mode == "RGB"
    assert enhanced[1][0].getextrema() == ((0, 255), (0, 255), (0, 255))


def test_declared_contrast_fallback_recognizes_low_contrast_field(
    monkeypatch,
) -> None:
    calls = []

    def recognize_contrast(resource, image, *args, **kwargs):
        extrema = image.convert("L").getextrema()
        calls.append(extrema)
        return {
            "text": "1234" if extrema == (0, 255) else "",
            "confidence": 0.99,
            "modelLocale": 0,
        }

    image = Image.new("L", (20, 10), 110)
    for x in range(10, 20):
        for y in range(10):
            image.putpixel((x, y), 140)
    monkeypatch.setattr(visual_layouts, "recognize_line", recognize_contrast)

    result = recognize_field(
        FIXTURE_CATALOG,
        image,
        {
            "name": "documentNumber",
            "bounds": [0.0, 0.0, 1.0, 1.0],
            "mask": "{4D}",
            "textHeight": 1.0,
            "lowContrastText": True,
        },
    )

    assert result["text"] == "1234"
    assert result["preprocessing"] == "catalog_contrast_normalization"
    assert (110, 140) in calls
    assert (0, 255) in calls


def test_catalog_preserves_document_hints_without_outcome_semantics() -> None:
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)

    hints = catalog_hints(layout)

    assert hints["capture"] == {
        "dNecessaryLights": 6,
        "dUVExp": 4,
        "dOVIExp": 3,
        "hologramTiltType": 1,
    }
    assert hints["recognition"] == {
        "dOCRSTX": 20,
        "dOCRSTY": 20,
        "recognTag": 0,
        "dBarcode": 0,
        "dMRZ": 0,
    }
    assert hints["electronicDocument"] == {"chipPage": 0}
    assert hints["sourceReferences"] == {"AuthSourceType": 0}
    assert hints["authenticityConfiguration"]["dAuthenticity"] == 12345
    assert hints["authenticityConfiguration"]["authenticity"][
        "photoReplacementCheck"
    ] is True

    hints["authenticityConfiguration"]["authenticity"][
        "photoReplacementCheck"
    ] = False
    assert catalog_hints(layout)["authenticityConfiguration"]["authenticity"][
        "photoReplacementCheck"
    ] is True

    requirements = declared_layout_requirements(layout)
    assert requirements["capture"] == {
        "requiredLightMask": 6,
        "uvExposure": 4,
        "opticallyVariableExposure": 3,
        "hologramTiltType": 1,
    }
    assert requirements["recognition"]["ocrSearchToleranceX"] == 20
    assert requirements["recognition"]["ocrSearchToleranceY"] == 20
    assert requirements["electronicDocument"]["chipPage"] == 0
    assert requirements["sourceReferences"]["authenticationSourceType"] == 0
    assert requirements["authenticityConfiguration"]["checkMask"] == 12345
    parameters = requirements["authenticityConfiguration"]["parameters"]
    assert parameters["photoReplacementCheck"] is True
    assert parameters["wholePageLuminescence"] is False
    assert all(not key.startswith("d") for key in parameters)


def test_catalog_mrz_geometry_converts_physical_tenths_to_ratios() -> None:
    layout = visual_layout(FIXTURE_CATALOG, BOOKLET_PAGE_1)

    geometry = mrz_physical_geometry(layout)

    assert geometry["sourceUnit"] == "tenths_of_millimeter"
    assert geometry["widthRatio"] == 111.2 / 125
    assert geometry["heightRatio"] == 8.8 / 88
    assert geometry["raw"]["dMRZFontID"] == 1
    assert geometry["raw"]["dMRZSymbolPos"] == 25


def test_guided_mrz_search_preserves_default_fallback_ratios() -> None:
    defaults = mrz_search_ratios(2)
    guided = mrz_search_ratios(
        2,
        {"widthRatio": 0.89, "heightRatio": 0.10},
    )

    assert tuple(round(value, 3) for value in guided[:3]) == (0.095, 0.11, 0.125)
    assert set(defaults).issubset(guided)


def test_recognized_document_confirms_geometry_by_semantics_not_profile() -> None:
    matching = {
        "documentIdentifier": 123,
        "confidence": 0.2,
        "document": {
            "isoCodes": ["MEX"],
            "documentType": {"name": "VotingCard"},
        },
    }
    result = {
        "DocumentName": "Voter Credential",
        "dCountryName": "Mexico",
        "recognitionProfile": "ignored-by-confirmation",
    }

    candidate = recognized_document_visual_layout_candidate(
        result, {"candidates": [matching]}
    )

    assert candidate == matching


def test_recognized_passport_confirms_issuer_family_and_format() -> None:
    candidate = {
        "documentIdentifier": 123,
        "confidence": 0.75,
        "document": {
            "isoCodes": ["BLR"],
            "documentType": {"name": "Passport"},
            "documentFormat": {"name": "ID3"},
        },
    }

    result = recognized_document_visual_layout_candidate(
        {
            "DocumentName": "Passport",
            "dCountryName": "Belarus",
            "issuingStateCode": "BLR",
        },
        {"candidates": [candidate]},
    )

    assert result == candidate


def test_recognized_document_rejects_weak_or_secondary_geometry() -> None:
    matching = {
        "documentIdentifier": 123,
        "confidence": 0.19,
        "document": {
            "isoCodes": ["MEX"],
            "documentType": {"name": "VotingCard"},
        },
    }
    unrelated = {
        "documentIdentifier": 456,
        "confidence": 0.8,
        "document": {
            "isoCodes": ["BRA"],
            "documentType": {"name": "IdentityCard"},
        },
    }
    recognized = {"DocumentName": "Voter Credential", "dCountryName": "Mexico"}

    assert recognized_document_visual_layout_candidate(
        recognized, {"candidates": [matching]}
    ) is None
    assert recognized_document_visual_layout_candidate(
        recognized, {"candidates": [unrelated, {**matching, "confidence": 0.9}]}
    ) is None


def test_pdf417_scan_is_skipped_for_confirmed_incompatible_families() -> None:
    assert not should_scan_pdf417("auto_research", "mex_ine", None)
    assert not should_scan_pdf417("auto_research", "swe_id_2021", None)
    assert not should_scan_pdf417("auto_research", None, "td3")
    assert should_scan_pdf417("auto_research", None, None)
    assert should_scan_pdf417("aamva_pdf417", "mex_ine", "td3")


def test_explicit_profiles_skip_incompatible_td1_ocr() -> None:
    assert should_try_td1("mex_ine")
    assert should_try_td1("icao_td1")
    assert should_try_td1("auto_research")
    assert not should_try_td1("icao_td2")
    assert not should_try_td1("icao_td3")
    assert not should_try_td1("icao_mrv")
    assert not should_try_td1("swe_id_2021")


def test_mexican_birth_date_requires_three_way_agreement() -> None:
    assert (
        corroborated_mexican_birth_date(
            "10.06/1988",
            "HEDC880610MSLRZL04",
            "HRDZCL88061025M700",
        )
        == "10/06/1988"
    )
    assert (
        corroborated_mexican_birth_date(
            "11/06/1988",
            "HEDC880610MSLRZL04",
            "HRDZCL88061025M700",
        )
        == "10/06/1988"
    )
    assert (
        corroborated_mexican_birth_date(
            "24/12/1980",
            "OECS901224HJCCRL03",
            "OCCRSL90122414H900",
        )
        == "24/12/1990"
    )
    assert (
        corroborated_mexican_birth_date(
            "12/07/1988",
            "HEDC880610MSLRZL04",
            "HRDZCL88061025M700",
        )
        is None
    )
    assert (
        corroborated_mexican_birth_date(
            "10/06/1988",
            "HEDC880611MSLRZL04",
            "HRDZCL88061025M700",
        )
        is None
    )
    assert (
        corroborated_mexican_birth_date(
            "31/02/1988",
            "HEDC880231MSLRZL04",
            "HRDZCL88023125M700",
        )
        is None
    )


def test_legacy_mexican_name_fallback_recognizes_separate_lines(
    monkeypatch,
) -> None:
    outputs = iter(
        [
            {"text": "LANDEROS", "confidence": 0.97},
            {"text": "PEREZ", "confidence": 0.99},
            {"text": "ALFONSO", "confidence": 0.98},
        ]
    )
    monkeypatch.setattr(pipeline, "recognize_line", lambda *_args, **_kwargs: next(outputs))

    values, confidences = recognize_mexican_voter_name_fallback(
        FIXTURE_CATALOG, Image.new("RGB", (1600, 1008), "white")
    )

    assert values == ["LANDEROS", "PEREZ", "ALFONSO"]
    assert min(confidences) >= 0.9


def test_mexican_name_merge_repairs_missing_letters_without_shortening_values() -> None:
    assert merge_mexican_voter_names(
        ["REZ", "RAMREZ", "VALERIA"],
        ["PEREZ", "RAMIREZ", "VALERIA"],
        "PERV870129MQTRML05",
    ) == ["PEREZ", "RAMIREZ", "VALERIA"]
    assert merge_mexican_voter_names(
        ["NOMBRE", "CEDENO", "YESENIA"],
        ["GOMEZ", "CEDENO", "YESENA"],
        "GOCY850331MJCMDS01",
    ) == ["GOMEZ", "CEDENO", "YESENIA"]
    assert merge_mexican_voter_names(
        ["NOMBRE", "GONZALEZ", "GASCA"],
        ["GONZALEZ", "GASCA", "KARLAVETH"],
        "GOGK870521MMCNSR07",
    ) == ["GONZALEZ", "GASCA", "KARLAVETH"]


def test_mexican_name_grid_filters_by_curp_initial_and_confidence(
    monkeypatch,
) -> None:
    outputs = iter(
        [{"text": "MARIN", "confidence": 0.99}]
        + [{"text": "WRONG", "confidence": 0.99}] * 35
        + [{"text": "JAIMES", "confidence": 0.98}]
        + [{"text": "WRONG", "confidence": 0.99}] * 35
        + [{"text": "GAMALIEL", "confidence": 0.97}]
        + [{"text": "WRONG", "confidence": 0.99}] * 35
    )
    monkeypatch.setattr(
        pipeline,
        "recognize_line",
        lambda *_args, **_kwargs: next(outputs),
    )

    values, confidences = recognize_mexican_voter_name_grid(
        FIXTURE_CATALOG,
        Image.new("RGB", (1600, 1008), "white"),
        "MAJG840205HDFRMM16",
    )

    assert values == ["MARIN", "JAIMES", "GAMALIEL"]
    assert confidences == [0.99, 0.98, 0.97]


def test_mexican_curp_grid_requires_checksum_and_voter_date(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "recognize_line",
        lambda *_args, **_kwargs: {
            "text": "AEGD030618HJCRNLA8",
            "confidence": 0.97,
        },
    )

    value, confidence = recognize_mexican_curp_grid(
        FIXTURE_CATALOG,
        Image.new("RGB", (1600, 1008), "white"),
        "AEGDEX03061814H000",
    )

    assert valid_mexican_curp(value)
    assert value == "AEGD030618HJCRNLA8"
    assert confidence == 0.97
    assert not valid_mexican_curp("AEGD030618HUCRNLA8")


@requires_assets
def test_adaptive_layout_search_matches_exhaustive_with_fewer_calls(
    monkeypatch, catalog_assets
) -> None:
    image = Image.open(ID_FRONT)
    layout = visual_layout(FIXTURE_CATALOG, FRONT_LAYOUT)
    original = visual_layouts.recognize_line
    call_count = 0

    def counted_recognition(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(visual_layouts, "recognize_line", counted_recognition)
    exhaustive = recognize_visual_layout(
        catalog_assets, image, layout, adaptive_search=False
    )
    exhaustive_calls = call_count
    call_count = 0
    adaptive = recognize_visual_layout(
        catalog_assets, image, layout, adaptive_search=True
    )

    def without_confidence(fields):
        return [
            {key: value for key, value in field.items() if key != "confidence"}
            for field in fields
        ]

    assert without_confidence(adaptive) == without_confidence(exhaustive)
    assert all(
        abs(left["confidence"] - right["confidence"]) < 0.01
        for left, right in zip(adaptive, exhaustive)
    )
    assert call_count <= exhaustive_calls * 0.85
