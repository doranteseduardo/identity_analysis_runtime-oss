#!/usr/bin/env python3
"""Structured document analysis pipeline."""
import subprocess
import tempfile
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .assets import validate_assets
from .barcodes import (
    aamva_result,
    decode_machine_barcodes,
    decode_machine_barcodes_in_regions,
    decode_pdf417,
    decode_pdf417_in_regions,
    ine_qr_evidence,
    ine_qr_result,
)
from .capabilities import SDK_COMPATIBILITY
from .document_classifier import (
    classifier_available,
    classify_document,
    document_catalog_entry,
    warm_up_document_classifier,
)
from .ocr import load_runtime, run as recognize_line
from .metadata_integrity import check_metadata_integrity
from .quality import analyze_quality, warm_up_quality
from .rectification import (
    map_rectified_bounds_to_source,
    rectification_session,
    rectify_document,
)
from .transliteration import identity_key, identity_variants
from .visual_layouts import (
    barcode_regions,
    graphic_regions,
    layout_page_role,
    mrz_physical_geometry,
    recognize_visual_layout,
    visual_layout,
    warm_up_visual_layouts,
)


WEIGHTS = (7, 3, 1)
# Reported as the recognition engine for every text result.  The concrete model
# file is supplied by the operator; see docs/models.md.
LINE_RECOGNITION_ENGINE = "onnx-line-recognition"
COUNTRY_NAMES = {"BLR": "Belarus", "MEX": "Mexico", "SWE": "Sweden"}
BLR_AUTHORITY_CODES = {
    "709": "УВД АДМИНИСТРАЦИИ ФРУНЗЕНСКОГО РАЙОНА Г.МИНСКА",
}
MEX_PASSPORT_VISUAL_PROFILES = (
    {
        "surname": (0.294, 0.243, 0.731, 0.288),
        "givenNames": (0.294, 0.322, 0.594, 0.367),
        "nationality": (0.294, 0.402, 0.500, 0.446),
        "dateOfBirth": (0.525, 0.402, 0.719, 0.446),
        "personalNumber": (0.294, 0.476, 0.625, 0.521),
        "sex": (0.294, 0.556, 0.350, 0.600),
        "placeOfBirth": (0.419, 0.556, 0.825, 0.600),
        "dateOfIssue": (0.294, 0.650, 0.500, 0.694),
        "dateOfExpiry": (0.550, 0.650, 0.744, 0.694),
        "folioNumber": (0.1125, 0.759, 0.281, 0.813),
    },
    {
        "surname": (0.294, 0.235, 0.731, 0.285),
        "givenNames": (0.294, 0.305, 0.650, 0.355),
        "nationality": (0.294, 0.375, 0.520, 0.425),
        "dateOfBirth": (0.560, 0.375, 0.750, 0.425),
        "personalNumber": (0.294, 0.435, 0.660, 0.485),
        "sex": (0.294, 0.505, 0.380, 0.555),
        "placeOfBirth": (0.440, 0.505, 0.850, 0.555),
        "dateOfIssue": (0.294, 0.595, 0.520, 0.645),
        "dateOfExpiry": (0.580, 0.595, 0.770, 0.645),
        "folioNumber": (0.130, 0.695, 0.320, 0.745),
    },
)
MEX_PASSPORT_VISUAL_REGIONS = (
    {
        "documentFrontSide": {
            "box": [0.0, 0.0, 1.0, 1.0],
            "faceExpected": False,
        },
        "portrait": {
            "box": [0.045, 0.300, 0.295, 0.755],
            "faceExpected": True,
        },
        "ghostPortrait": {
            "box": [0.825, 0.325, 0.945, 0.500],
            "faceExpected": True,
        },
        "signature": {
            "box": [0.365, 0.690, 0.525, 0.815],
            "faceExpected": False,
        },
    },
    {
        "documentFrontSide": {
            "box": [0.0, 0.0, 1.0, 1.0],
            "faceExpected": False,
        },
        "portrait": {
            "box": [0.055, 0.300, 0.300, 0.700],
            "faceExpected": True,
        },
        "ghostPortrait": {
            "box": [0.825, 0.300, 0.955, 0.455],
            "faceExpected": True,
        },
        "signature": {
            "box": [0.365, 0.640, 0.550, 0.755],
            "faceExpected": False,
        },
    },
)


def normalize_td1_lines(lines: list[str]) -> list[str]:
    if len(lines) != 3:
        return [line.upper().replace(" ", "") for line in lines]
    return [
        lines[0].upper().replace(" ", ""),
        lines[1].upper().replace(" ", ""),
        lines[2].upper().replace(" ", "<"),
    ]


def normalize_td3_lines(lines: list[str]) -> list[str]:
    return [line.upper().replace(" ", "<") for line in lines]


def character_value(character: str) -> int:
    if character == "<":
        return 0
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {character!r}")


def valid_check_digit(value: str, digit: str) -> bool:
    if not digit.isdigit():
        return False
    total = sum(character_value(char) * WEIGHTS[index % 3] for index, char in enumerate(value))
    return total % 10 == int(digit)


def parse_date(value: str, kind: str) -> str:
    year, month, day = int(value[:2]), int(value[2:4]), int(value[4:6])
    current_year = date.today().year
    full_year = (current_year // 100) * 100 + year
    if kind == "birth" and full_year > current_year:
        full_year -= 100
    elif kind == "expiry":
        full_year = min(
            (full_year - 100, full_year, full_year + 100),
            key=lambda candidate: abs(candidate - current_year),
        )
    return date(full_year, month, day).isoformat()


def field(field_name: str, value: str, valid: bool | None = None) -> dict:
    result = {
        "fieldName": field_name,
        "lcidName": "",
        "value": value,
        "valueList": [{"value": value, "source": "MRZ"}],
        "source": "MRZ",
    }
    if valid is not None:
        result["validityStatus"] = 1 if valid else 0
    return result


VISUAL_FIELD_KEYS = {
    "surname1": "firstSurname",
    "surname2": "secondSurname",
    "givenNames": "givenNames",
    "dateOfBirth": "dateOfBirth",
    "address1": "addressLine1",
    "address2": "addressLine2",
    "address3": "addressLine3",
    "electorKey": "electorKey",
    "curp": "curp",
    "sex": "sex",
    "registrationYear": "registrationYear",
    "state": "state",
    "municipality": "municipality",
    "section": "section",
    "locality": "locality",
    "issueYear": "issueYear",
    "validity": "validity",
}


def map_visual_fields(
    values: dict[str, str],
    confidences: dict[str, float],
    crops: dict[str, tuple[float, float, float, float]],
) -> dict[str, dict]:
    result = {}
    for raw_key, output_key in VISUAL_FIELD_KEYS.items():
        value = values.get(raw_key, "")
        crop = crops.get(raw_key)
        result[output_key] = {
            "value": value,
            "confidence": confidences.get(raw_key),
            "available": bool(value),
            "source": "VISUAL",
            "cropNormalized": list(crop) if crop else None,
        }
    return result


def parse_td1(lines: list[str], confidences: list[float]) -> dict:
    normalized = normalize_td1_lines(lines)
    if len(normalized) != 3 or any(len(line) != 30 for line in normalized):
        raise ValueError(f"TD1 requires three lines of 30 characters; got {[len(line) for line in normalized]}")

    first, second, third = normalized
    document_number = first[5:14].replace("<", "")
    document_valid = valid_check_digit(first[5:14], first[14])
    birth_valid = valid_check_digit(second[0:6], second[6])
    expiry_valid = valid_check_digit(second[8:14], second[14])
    composite = first[5:30] + second[0:7] + second[8:15] + second[18:29]
    composite_valid = valid_check_digit(composite, second[29])
    surname_block, _, given_block = third.partition("<<")
    surname = " ".join(filter(None, surname_block.split("<")))
    given_names = " ".join(filter(None, given_block.split("<")))
    birth_date = parse_date(second[0:6], "birth")
    expiry_date = parse_date(second[8:14], "expiry")
    fields = [
        field("Document Number", document_number, document_valid),
        field("Date of Birth", birth_date, birth_valid),
        field("Date of Expiry", expiry_date, expiry_valid),
        field("Sex", second[7]),
        field("Nationality", second[15:18].replace("<", "")),
        field("Surname", surname),
        field("Given Names", given_names),
    ]
    all_valid = all((document_valid, birth_valid, expiry_valid, composite_valid))

    return {
        "errorCode": 0,
        "DocumentName": "Identity Card",
        "dCountryName": "Mexico" if first[2:5] == "MEX" else first[2:5],
        "documentClassCode": first[0:2].replace("<", ""),
        "issuingStateCode": first[2:5].replace("<", ""),
        "documentNumber": document_number,
        "dateOfBirth": birth_date,
        "dateOfExpiry": expiry_date,
        "sex": second[7].replace("<", ""),
        "nationalityCode": second[15:18].replace("<", ""),
        "surname": surname,
        "givenNames": given_names,
        "name": " ".join(filter(None, (surname, given_names))),
        "surnameAndGivenNames": " ".join(filter(None, (surname, given_names))),
        "mrzStrings": normalized,
        "mrzCode": "\n".join(normalized),
        "availableSourceList": ["MRZ"],
        "source": "MRZ",
        "validityStatus": 1 if all_valid else 0,
        "checks": {
            "documentNumber": document_valid,
            "dateOfBirth": birth_valid,
            "dateOfExpiry": expiry_valid,
            "composite": composite_valid,
        },
        "fieldList": fields,
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "lineConfidence": confidences,
            "mrzLines": [
                {"index": index + 1, "value": value, "confidence": confidence}
                for index, (value, confidence) in enumerate(zip(normalized, confidences))
            ],
        },
        "recognitionProfile": "ICAO-TD1",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }


def parse_td3(lines: list[str], confidences: list[float]) -> dict:
    normalized = normalize_td3_lines(lines)
    if len(normalized) != 2 or any(len(line) != 44 for line in normalized):
        raise ValueError(
            f"TD3 requires two lines of 44 characters; got {[len(line) for line in normalized]}"
        )

    first, second = normalized
    if not first.startswith("P"):
        raise ValueError("TD3 document class must start with P")
    document_number = second[0:9].replace("<", "")
    document_valid = valid_check_digit(second[0:9], second[9])
    birth_valid = valid_check_digit(second[13:19], second[19])
    expiry_valid = valid_check_digit(second[21:27], second[27])
    issuing_state = first[2:5].replace("<", "")
    personal_number = second[28:42]
    if issuing_state == "BLR":
        personal_number = normalize_blr_personal_number(personal_number)
        second = second[:28] + personal_number + second[42:]
    personal_number_valid = valid_check_digit(personal_number, second[42])
    composite = second[0:10] + second[13:20] + second[21:43]
    composite_valid = valid_check_digit(composite, second[43])
    surname_block, _, given_block = first[5:44].partition("<<")
    surname = " ".join(filter(None, surname_block.split("<")))
    given_names = " ".join(filter(None, given_block.split("<")))
    birth_date = parse_date(second[13:19], "birth")
    expiry_date = parse_date(second[21:27], "expiry")
    fields = [
        field("Document Number", document_number, document_valid),
        field("Date of Birth", birth_date, birth_valid),
        field("Date of Expiry", expiry_date, expiry_valid),
        field("Sex", second[20].replace("<", "")),
        field("Nationality", second[10:13].replace("<", "")),
        field("Surname", surname),
        field("Given Names", given_names),
        field("Personal Number", personal_number.replace("<", ""), personal_number_valid),
    ]
    all_valid = all(
        (document_valid, birth_valid, expiry_valid, personal_number_valid, composite_valid)
    )
    return {
        "errorCode": 0,
        "DocumentName": "Passport",
        "dCountryName": COUNTRY_NAMES.get(issuing_state, issuing_state),
        "documentClassCode": first[0:2].replace("<", ""),
        "issuingStateCode": issuing_state,
        "documentNumber": document_number,
        "dateOfBirth": birth_date,
        "dateOfExpiry": expiry_date,
        "sex": second[20].replace("<", ""),
        "nationalityCode": second[10:13].replace("<", ""),
        "surname": surname,
        "givenNames": given_names,
        "name": " ".join(filter(None, (surname, given_names))),
        "surnameAndGivenNames": " ".join(filter(None, (surname, given_names))),
        "personalNumber": personal_number.replace("<", ""),
        "mrzStrings": normalized,
        "mrzCode": "\n".join(normalized),
        "availableSourceList": ["MRZ"],
        "source": "MRZ",
        "validityStatus": 1 if all_valid else 0,
        "checks": {
            "documentNumber": document_valid,
            "dateOfBirth": birth_valid,
            "dateOfExpiry": expiry_valid,
            "personalNumber": personal_number_valid,
            "composite": composite_valid,
        },
        "fieldList": fields,
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "lineConfidence": confidences,
            "mrzLines": [
                {"index": index + 1, "value": value, "confidence": confidence}
                for index, (value, confidence) in enumerate(zip(normalized, confidences))
            ],
        },
        "recognitionProfile": "ICAO-TD3",
        "recognitionProfileStatus": "selected_by_valid_mrz_checks",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }


def parse_td2(lines: list[str], confidences: list[float]) -> dict:
    normalized = normalize_td3_lines(lines)
    if len(normalized) != 2 or any(len(line) != 36 for line in normalized):
        raise ValueError(
            f"TD2 requires two lines of 36 characters; got {[len(line) for line in normalized]}"
        )

    first, second = normalized
    if first[0] not in {"A", "C", "I"}:
        raise ValueError("TD2 document class must start with A, C, or I")
    document_number = second[0:9].replace("<", "")
    document_valid = valid_check_digit(second[0:9], second[9])
    birth_valid = valid_check_digit(second[13:19], second[19])
    expiry_valid = valid_check_digit(second[21:27], second[27])
    composite = second[0:10] + second[13:20] + second[21:35]
    composite_valid = valid_check_digit(composite, second[35])
    surname_block, _, given_block = first[5:36].partition("<<")
    surname = " ".join(filter(None, surname_block.split("<")))
    given_names = " ".join(filter(None, given_block.split("<")))
    issuing_state = first[2:5].replace("<", "")
    optional_data = second[28:35].replace("<", "")
    birth_date = parse_date(second[13:19], "birth")
    expiry_date = parse_date(second[21:27], "expiry")
    checks = {
        "documentNumber": document_valid,
        "dateOfBirth": birth_valid,
        "dateOfExpiry": expiry_valid,
        "composite": composite_valid,
    }
    fields = [
        field("Document Number", document_number, document_valid),
        field("Date of Birth", birth_date, birth_valid),
        field("Date of Expiry", expiry_date, expiry_valid),
        field("Sex", second[20].replace("<", "")),
        field("Nationality", second[10:13].replace("<", "")),
        field("Surname", surname),
        field("Given Names", given_names),
    ]
    if optional_data:
        fields.append(field("Optional Data", optional_data))
    return {
        "errorCode": 0,
        "DocumentName": "Identity Document",
        "dCountryName": COUNTRY_NAMES.get(issuing_state, issuing_state),
        "documentClassCode": first[0:2].replace("<", ""),
        "issuingStateCode": issuing_state,
        "documentNumber": document_number,
        "dateOfBirth": birth_date,
        "dateOfExpiry": expiry_date,
        "sex": second[20].replace("<", ""),
        "nationalityCode": second[10:13].replace("<", ""),
        "surname": surname,
        "givenNames": given_names,
        "name": " ".join(filter(None, (surname, given_names))),
        "surnameAndGivenNames": " ".join(filter(None, (surname, given_names))),
        "optionalData": optional_data,
        "mrzStrings": normalized,
        "mrzCode": "\n".join(normalized),
        "availableSourceList": ["MRZ"],
        "source": "MRZ",
        "validityStatus": 1 if all(checks.values()) else 0,
        "checks": checks,
        "fieldList": fields,
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "lineConfidence": confidences,
            "mrzLines": [
                {"index": index + 1, "value": value, "confidence": confidence}
                for index, (value, confidence) in enumerate(zip(normalized, confidences))
            ],
        },
        "recognitionProfile": "ICAO-TD2",
        "recognitionProfileStatus": "selected_by_valid_mrz_checks",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }


def parse_mrv(lines: list[str], confidences: list[float]) -> dict:
    normalized = normalize_td3_lines(lines)
    if len(normalized) != 2 or len(normalized[0]) not in {36, 44}:
        raise ValueError(
            f"MRV requires two lines of 36 or 44 characters; got {[len(line) for line in normalized]}"
        )
    line_length = len(normalized[0])
    if len(normalized[1]) != line_length:
        raise ValueError(
            f"MRV lines must have equal length; got {[len(line) for line in normalized]}"
        )

    first, second = normalized
    if not first.startswith("V"):
        raise ValueError("MRV document class must start with V")
    format_name = "MRV-A" if line_length == 44 else "MRV-B"
    document_number = second[0:9].replace("<", "")
    document_valid = valid_check_digit(second[0:9], second[9])
    birth_valid = valid_check_digit(second[13:19], second[19])
    expiry_valid = valid_check_digit(second[21:27], second[27])
    surname_block, _, given_block = first[5:line_length].partition("<<")
    surname = " ".join(filter(None, surname_block.split("<")))
    given_names = " ".join(filter(None, given_block.split("<")))
    issuing_state = first[2:5].replace("<", "")
    optional_data = second[28:line_length].replace("<", "")
    birth_date = parse_date(second[13:19], "birth")
    expiry_date = parse_date(second[21:27], "expiry")
    fields = [
        field("Document Number", document_number, document_valid),
        field("Date of Birth", birth_date, birth_valid),
        field("Date of Expiry", expiry_date, expiry_valid),
        field("Sex", second[20].replace("<", "")),
        field("Nationality", second[10:13].replace("<", "")),
        field("Surname", surname),
        field("Given Names", given_names),
    ]
    if optional_data:
        fields.append(field("Optional Data", optional_data))
    checks = {
        "documentNumber": document_valid,
        "dateOfBirth": birth_valid,
        "dateOfExpiry": expiry_valid,
    }
    return {
        "errorCode": 0,
        "DocumentName": "Visa",
        "dCountryName": COUNTRY_NAMES.get(issuing_state, issuing_state),
        "documentClassCode": first[0:2].replace("<", ""),
        "issuingStateCode": issuing_state,
        "documentNumber": document_number,
        "dateOfBirth": birth_date,
        "dateOfExpiry": expiry_date,
        "sex": second[20].replace("<", ""),
        "nationalityCode": second[10:13].replace("<", ""),
        "surname": surname,
        "givenNames": given_names,
        "name": " ".join(filter(None, (surname, given_names))),
        "surnameAndGivenNames": " ".join(filter(None, (surname, given_names))),
        "optionalData": optional_data,
        "mrzStrings": normalized,
        "mrzCode": "\n".join(normalized),
        "availableSourceList": ["MRZ"],
        "source": "MRZ",
        "validityStatus": 1 if all(checks.values()) else 0,
        "checks": checks,
        "fieldList": fields,
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "lineConfidence": confidences,
            "mrzLines": [
                {"index": index + 1, "value": value, "confidence": confidence}
                for index, (value, confidence) in enumerate(zip(normalized, confidences))
            ],
        },
        "recognitionProfile": f"ICAO-{format_name}",
        "recognitionProfileStatus": "selected_by_valid_mrz_checks",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }


def normalize_blr_personal_number(value: str) -> str:
    if len(value) != 14:
        return value
    digit_map = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1"})
    alpha_map = str.maketrans({"8": "B", "0": "O", "1": "I"})
    return "".join(
        (
            value[:7].translate(digit_map),
            value[7].translate(alpha_map),
            value[8:11],
            value[11:13].translate(alpha_map),
            value[13].translate(digit_map),
        )
    )


def enrich_blr_passport(result: dict, resource: Path, image: Image.Image) -> dict:
    width, height = image.size
    crops = {
        "authority": (0.500, 0.580, 0.583, 0.619),
        "placeOfBirth": (0.500, 0.651, 0.583, 0.688),
        "dateOfIssue": (0.305, 0.654, 0.475, 0.693),
    }
    values, confidences = {}, {}
    for name, (left, top, right, bottom) in crops.items():
        crop = image.crop(
            (round(width * left), round(height * top), round(width * right), round(height * bottom))
        )
        output = recognize_line(resource, crop, False, "minus-one-one")
        values[name] = output["text"].strip()
        confidences[name] = output["confidence"]

    issue_digits = "".join(character for character in values["dateOfIssue"] if character.isdigit())
    if len(issue_digits) == 8:
        try:
            values["dateOfIssue"] = date(
                int(issue_digits[4:8]), int(issue_digits[2:4]), int(issue_digits[:2])
            ).isoformat()
        except ValueError:
            pass

    status_crop = image.crop(
        (round(width * 0.913), round(height * 0.283), round(width * 0.983), round(height * 0.654))
    ).rotate(270, expand=True)
    status_image = ImageOps.autocontrast(ImageOps.grayscale(status_crop)).point(
        lambda pixel: 255 if pixel > 160 else 0
    )
    status_output = recognize_line(resource, status_image, False, "minus-one-one")
    status_letters = "".join(character for character in status_output["text"].upper() if character.isalpha())
    document_status = "SPECIMEN" if status_letters.startswith("SAMI") else ""

    result.update(
        {
            "DocumentName": "Belarus - ePassport (2021)",
            "dCountryName": "Belarus",
            "nationality": "Belarus",
            "authority": values["authority"],
            "placeOfBirth": values["placeOfBirth"],
            "dateOfIssue": values["dateOfIssue"],
            "documentStatus": document_status,
            "validState": 0 if document_status == "SPECIMEN" else result["validityStatus"],
            "nation": {
                "authority": BLR_AUTHORITY_CODES.get(values["authority"], ""),
            },
        }
    )
    if document_status == "SPECIMEN":
        result["validityStatus"] = 0
    for name in ("authority", "placeOfBirth", "dateOfIssue", "documentStatus"):
        value = result[name]
        if value:
            item = field(name, value)
            item["source"] = "VISUAL"
            item["valueList"][0]["source"] = "VISUAL"
            result["fieldList"].append(item)
    result["recognition"]["visualSupplementConfidence"] = {
        **confidences,
        "documentStatus": status_output["confidence"],
    }
    return result


def parse_visual_date(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) != 8:
        return ""
    try:
        return date(int(digits[4:]), int(digits[2:4]), int(digits[:2])).isoformat()
    except ValueError:
        return ""


def visual_name_word(word: str, mrz_name_zone: str) -> str:
    normalized = word.upper().replace("0", "O")
    if not normalized or normalized in mrz_name_zone:
        return normalized
    candidates = {
        mrz_name_zone[index : index + len(normalized)]
        for index in range(max(0, len(mrz_name_zone) - len(normalized) + 1))
    }
    nearest = min(
        candidates,
        key=lambda candidate: sum(
            left != right for left, right in zip(normalized, candidate)
        ),
        default=normalized,
    )
    distance = sum(left != right for left, right in zip(normalized, nearest))
    return nearest if distance == 1 else normalized


def normalize_mex_personal_number(value: str) -> str:
    compact = "".join(character for character in value.upper() if character.isalnum())
    if len(compact) != 18:
        return compact
    alpha_map = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})
    digit_map = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5"})
    return "".join(
        (
            compact[:4].translate(alpha_map),
            compact[4:10].translate(digit_map),
            compact[10].translate(alpha_map),
            compact[11:16].translate(alpha_map),
            compact[16:18],
        )
    )


def enrich_mex_passport(result: dict, resource: Path, image: Image.Image) -> dict:
    width, height = image.size
    def recognize_region(bounds: tuple[float, float, float, float]) -> dict:
        left, top, right, bottom = bounds
        crop = image.crop(
            (
                round(width * left),
                round(height * top),
                round(width * right),
                round(height * bottom),
            )
        )
        crop = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
        return recognize_line(resource, crop, False, "minus-one-one")

    profile_candidates = []
    for profile_index, profile in enumerate(MEX_PASSPORT_VISUAL_PROFILES):
        output = recognize_region(profile["nationality"])
        normalized = output["text"].strip().upper().replace("0", "O")
        profile_candidates.append(
            (
                normalized == "MEXICANA",
                output["confidence"],
                profile_index,
                profile,
                output,
            )
        )

    _, _, selected_profile_index, selected_profile, nationality_output = max(
        profile_candidates, key=lambda item: item[:2]
    )
    values = {"nationality": nationality_output["text"].strip()}
    confidences = {"nationality": nationality_output["confidence"]}
    for name, bounds in selected_profile.items():
        if name == "nationality":
            continue
        output = recognize_region(bounds)
        values[name] = output["text"].strip()
        confidences[name] = output["confidence"]
    mrz_name_zone = (
        result["mrzStrings"][0][5:]
        .replace("<", "")
        .translate(str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"}))
    )
    surname = " ".join(
        visual_name_word(word, mrz_name_zone) for word in values["surname"].split()
    )
    given_names = " ".join(
        visual_name_word(word, mrz_name_zone)
        for word in values["givenNames"].split()
    )
    nationality = values["nationality"].upper().replace("0", "O")
    place_of_birth = values["placeOfBirth"].upper().replace("0", "O")
    personal_number = normalize_mex_personal_number(values["personalNumber"])
    dates = {
        name: parse_visual_date(values[name])
        for name in ("dateOfBirth", "dateOfIssue", "dateOfExpiry")
    }

    if confidences["surname"] >= 0.85 and confidences["givenNames"] >= 0.85:
        name_zone = (
            surname.replace(" ", "<")
            + "<<"
            + given_names.replace(" ", "<")
        )
        first_line = (
            result["mrzStrings"][0][:5] + name_zone
        )[:44].ljust(44, "<")
        result["mrzStrings"][0] = first_line
        result["mrzCode"] = "\n".join(result["mrzStrings"])
        result["recognition"]["mrzLines"][0]["value"] = first_line

    result.update(
        {
            "DocumentName": "Mexico - ePassport (2021)",
            "surname": surname,
            "givenNames": given_names,
            "name": " ".join(filter(None, (surname, given_names))),
            "surnameAndGivenNames": " ".join(
                filter(None, (surname, given_names))
            ),
            "nationality": nationality,
            "personalNumber": personal_number,
            "placeOfBirth": place_of_birth,
            "dateOfIssue": dates["dateOfIssue"],
            "folioNumber": values["folioNumber"]
            if values["folioNumber"].isdigit()
            else "",
        }
    )
    result["visualRegions"] = deepcopy(
        MEX_PASSPORT_VISUAL_REGIONS[selected_profile_index]
    )
    for name in ("dateOfBirth", "dateOfExpiry"):
        if dates[name] and dates[name] == result[name]:
            result[name] = dates[name]
    result["recognition"]["visualSupplementConfidence"] = confidences
    return result


def repair_td3(lines: list[str], confidences: list[float]) -> list[str]:
    normalized = [line.strip().upper().replace(" ", "<") for line in lines]
    if len(normalized) != 2:
        return normalized

    first, second = normalized
    if first.startswith(("P<", "P")) and len(first) < 44:
        first = first.ljust(44, "<")
    second_candidates = [second]
    if 30 <= len(second) < 44 and len(second) >= 2 and second[-2:].isdigit():
        second_candidates.append(second[:-2] + "<" * (44 - len(second)) + second[-2:])
    candidates = [[first, candidate] for candidate in second_candidates]
    valid_candidates = []
    for candidate in candidates:
        if any(len(line) != 44 for line in candidate):
            continue
        try:
            checks = parse_td3(candidate, confidences)["checks"]
        except (ValueError, IndexError):
            continue
        valid_candidates.append((sum(checks.values()), candidate))
    if valid_candidates:
        return max(valid_candidates, key=lambda item: item[0])[1]
    return [first, second]


def td3_candidate_score(candidate: tuple[list[str], list[float]]) -> tuple[int, float]:
    lines, confidences = candidate
    if len(lines) != 2 or any(len(line) != 44 for line in normalize_td3_lines(lines)):
        return 0, sum(confidences)
    try:
        return sum(parse_td3(lines, confidences)["checks"].values()), sum(confidences)
    except (ValueError, IndexError):
        return 0, sum(confidences)


def mrv_candidate_score(candidate: tuple[list[str], list[float]]) -> tuple[int, float]:
    lines, confidences = candidate
    normalized = normalize_td3_lines(lines)
    if (
        len(normalized) != 2
        or len(normalized[0]) not in {36, 44}
        or len(normalized[1]) != len(normalized[0])
        or not normalized[0].startswith("V")
    ):
        return 0, sum(confidences)
    try:
        return sum(parse_mrv(lines, confidences)["checks"].values()), sum(confidences)
    except (ValueError, IndexError):
        return 0, sum(confidences)


def td2_candidate_score(candidate: tuple[list[str], list[float]]) -> tuple[int, float]:
    lines, confidences = candidate
    normalized = normalize_td3_lines(lines)
    if (
        len(normalized) != 2
        or any(len(line) != 36 for line in normalized)
        or normalized[0][0] not in {"A", "C", "I"}
    ):
        return 0, sum(confidences)
    try:
        return sum(parse_td2(lines, confidences)["checks"].values()), sum(confidences)
    except (ValueError, IndexError):
        return 0, sum(confidences)


def two_line_mrz_candidate_score(
    candidate: tuple[list[str], list[float]],
) -> tuple[int, float]:
    return max(
        td3_candidate_score(candidate),
        td2_candidate_score(candidate),
        mrv_candidate_score(candidate),
    )


def fully_valid_two_line_mrz(candidate: tuple[list[str], list[float]]) -> bool:
    normalized = normalize_td3_lines(candidate[0])
    if len(normalized) != 2 or len(normalized[0]) != len(normalized[1]):
        return False
    if normalized[0].startswith("V"):
        return mrv_candidate_score(candidate)[0] == 3
    if len(normalized[0]) == 36 and normalized[0][0] in {"A", "C", "I"}:
        return td2_candidate_score(candidate)[0] == 4
    if len(normalized[0]) == 44 and normalized[0].startswith("P"):
        return td3_candidate_score(candidate)[0] == 5
    return False


def unsupported_document_result() -> dict:
    return {
        "errorCode": 0,
        "DocumentName": "Unsupported Document",
        "availableSourceList": [],
        "source": "NOT_AVAILABLE",
        "validityStatus": -1,
        "fieldList": [],
        "Images": {},
        "recognition": {
            "engine": None,
            "reason": "No supported MRZ, barcode, or explicit visual profile matched.",
        },
        "recognitionProfile": "UNSUPPORTED",
        "recognitionProfileStatus": "no_supported_profile_matched",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }


def classifier_layout_hint(classification: dict | None) -> str | None:
    """Map catalog candidates to implemented visual layouts using aggregate confidence."""
    scores: dict[str, float] = {}
    for candidate in (classification or {}).get("candidates", []):
        document = candidate.get("document") or {}
        iso_codes = set(document.get("isoCodes") or [])
        document_type = (document.get("documentType") or {}).get("name")
        caption = document.get("caption", "")
        hint = None
        if "MEX" in iso_codes and document_type == "VotingCard":
            hint = "mex_ine"
        elif (
            "SWE" in iso_codes
            and document_type == "IdentityCard"
            and "2021" in caption
        ):
            hint = "swe_id_2021"
        if hint:
            scores[hint] = scores.get(hint, 0.0) + float(candidate.get("confidence", 0.0))
    if not scores:
        return None
    hint, confidence = max(scores.items(), key=lambda item: item[1])
    return hint if confidence >= 0.45 else None


def orient_mexican_voter_card(
    resource: Path, image: Image.Image
) -> tuple[Image.Image, int, dict]:
    candidates = []
    for degrees in (0, 90, 180, 270):
        candidate = image
        if degrees:
            candidate = image.rotate(degrees, expand=True).resize(
                image.size, Image.Resampling.BICUBIC
            )
        classification = classify_document(resource, candidate, top_k=5)
        score = sum(
            float(item.get("confidence", 0.0))
            for item in classification.get("candidates", [])
            if "MEX" in set((item.get("document") or {}).get("isoCodes") or [])
            and ((item.get("document") or {}).get("documentType") or {}).get("name")
            == "VotingCard"
        )
        candidates.append((score, -degrees, candidate, degrees, classification))
    _, _, selected, degrees, classification = max(
        candidates, key=lambda item: item[:2]
    )
    return selected, degrees, classification


def mexican_voter_card_candidate(classification: dict | None) -> dict | None:
    candidates = [
        candidate
        for candidate in (classification or {}).get("candidates", [])
        if "MEX" in set((candidate.get("document") or {}).get("isoCodes") or [])
        and (
            ((candidate.get("document") or {}).get("documentType") or {}).get("name")
            == "VotingCard"
        )
    ]
    if not candidates:
        return None
    candidate = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
    return candidate if float(candidate.get("confidence", 0.0)) >= 0.2 else None


def classifier_visual_layout_candidate(classification: dict | None) -> dict | None:
    candidates = [
        candidate
        for candidate in (classification or {}).get("candidates", [])
        if candidate.get("document")
    ]
    if not candidates:
        return None
    top = candidates[0]
    runner_up = float(candidates[1].get("confidence", 0.0)) if len(candidates) > 1 else 0.0
    confidence = float(top.get("confidence", 0.0))
    return top if confidence >= 0.65 and confidence - runner_up >= 0.15 else None


def recognized_document_visual_layout_candidate(
    result: dict, classification: dict | None
) -> dict | None:
    candidates = (classification or {}).get("candidates", [])
    if not candidates:
        return None
    candidate = candidates[0]
    if float(candidate.get("confidence", 0.0)) < 0.2:
        return None
    document = candidate.get("document") or {}
    iso_codes = set(document.get("isoCodes") or [])
    issuer = result.get("issuingStateCode")
    if issuer and issuer not in iso_codes:
        return None
    country = result.get("dCountryName")
    document_name = result.get("DocumentName", "")
    document_type = (document.get("documentType") or {}).get("name", "")
    document_format = (document.get("documentFormat") or {}).get("name")
    if (
        country == "Mexico"
        and document_name == "Voter Credential"
        and "MEX" in iso_codes
        and document_type == "VotingCard"
    ):
        return candidate
    if (
        country == "Sweden"
        and "Id Card" in document_name
        and "SWE" in iso_codes
        and document_type == "IdentityCard"
    ):
        return candidate
    if document_name == "Passport" and document_format == "ID3" and (
        "Passport" in document_type or document_type in {
            "TravelDocument",
            "RefugeeTravelDocument",
            "LaissezPasser",
        }
    ):
        return candidate
    if document_name == "Visa" and document_type == "Visa" and document_format in {
        "ID2",
        "ID3",
    }:
        return candidate
    if document_name == "Identity Card" and document_format == "ID1":
        return candidate
    if document_name == "Identity Document" and document_format == "ID2":
        return candidate
    return None


def classifier_mrz_hint(classification: dict | None) -> str | None:
    candidate = classifier_visual_layout_candidate(classification)
    if not candidate:
        return None
    document = candidate["document"]
    mrz = document.get("mrz") or {}
    if mrz.get("ignored"):
        return None
    if mrz.get("expectedProfile") in {"td1", "td2", "td3", "mrv"}:
        return mrz["expectedProfile"]
    document_format = (document.get("documentFormat") or {}).get("name")
    document_type = (document.get("documentType") or {}).get("name", "")
    caption = document.get("caption", "")
    if document_type == "Visa" and document_format in {"ID2", "ID3"}:
        return "mrv"
    passport_family = (
        "Passport" in document_type
        or document_type
        in {
            "TravelDocument",
            "RefugeeTravelDocument",
            "LaissezPasser",
            "AliensPassport",
        }
    )
    if (
        document_format == "ID3"
        and passport_family
        and document_type not in {"PassportPage", "DomesticPassport", "VehiclePassport"}
        and "not MRZ" not in caption
    ):
        return "td3"
    if not mrz.get("present"):
        return None
    if document_format == "ID1":
        return "td1"
    if document_format == "ID2":
        return "td2"
    if document_format == "ID3":
        return "mrv" if "Visa" in document_type else "td3"
    return None


def should_scan_pdf417(
    profile: str, layout_hint: str | None, mrz_hint: str | None
) -> bool:
    if profile == "aamva_pdf417":
        return True
    if profile != "auto_research":
        return False
    return layout_hint not in {"mex_ine", "swe_id_2021"} and mrz_hint not in {
        "td1",
        "td2",
        "td3",
        "mrv",
    }


def catalog_visual_regions(layout: dict) -> dict:
    regions = {}
    for region in graphic_regions(
        layout, {"portrait", "ghostPortrait", "signature", "portraitOfChild"}
    ):
        regions.setdefault(
            region["name"],
            {
                "box": region["bounds"],
                "faceExpected": region["faceExpected"],
                "coordinateSpace": "processed_document",
            },
        )
    return regions


def catalog_barcode_regions(layout: dict) -> list[dict]:
    regions = barcode_regions(layout)
    return regions or graphic_regions(layout, {"barCode"})


def corroborated_mexican_birth_date(
    visual_value: str | None,
    curp: str | None,
    voter_key: str | None,
) -> str | None:
    date_digits = "".join(
        character for character in (visual_value or "") if character.isdigit()
    )
    curp_birth = curp[4:10] if len(curp or "") == 18 else None
    voter_birth = voter_key[6:12] if len(voter_key or "") == 18 else None
    visual_birth = (
        date_digits[6:8] + date_digits[2:4] + date_digits[:2]
        if len(date_digits) == 8
        else None
    )
    if (
        not curp_birth
        or curp_birth != voter_birth
        or visual_birth is None
        or sum(
            left != right
            for left, right in zip(curp_birth, visual_birth)
        )
        > 1
    ):
        return None
    try:
        parsed_birth = date(
            int(date_digits[4:6] + curp_birth[:2]),
            int(curp_birth[2:4]),
            int(curp_birth[4:6]),
        )
    except ValueError:
        return None
    return parsed_birth.strftime("%d/%m/%Y")


def valid_mexican_curp(value: str | None) -> bool:
    normalized = (value or "").upper()
    if not re.fullmatch(
        r"[A-Z][AEIOUX][A-Z]{2}\d{6}[HM][A-Z]{5}[A-Z0-9]\d",
        normalized,
    ):
        return False
    alphabet = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
    checksum = sum(
        alphabet.index(character) * (18 - index)
        for index, character in enumerate(normalized[:17])
    )
    return str((10 - checksum % 10) % 10) == normalized[17]


def recognize_mexican_curp_grid(
    resource: Path,
    image: Image.Image,
    voter_key: str | None,
) -> tuple[str | None, float]:
    candidates = {}
    for left in (0.31, 0.33, 0.35):
        for right in (0.62, 0.68, 0.74):
            for top in (0.79, 0.805, 0.82):
                for bottom in (0.855, 0.875, 0.895):
                    crop = image.crop(
                        (
                            int(left * image.width),
                            int(top * image.height),
                            int(right * image.width),
                            int(bottom * image.height),
                        )
                    )
                    variants = (
                        crop,
                        ImageOps.autocontrast(
                            ImageOps.grayscale(crop)
                        ).convert("RGB"),
                    )
                    for recognition_crop in variants:
                        output = recognize_line(
                            resource,
                            recognition_crop,
                            False,
                            "minus-one-one",
                            locale=2058,
                        )
                        value = "".join(
                            character
                            for character in output["text"].upper()
                            if character.isalnum()
                        )
                        confidence = float(output["confidence"])
                        if (
                            confidence >= 0.9
                            and valid_mexican_curp(value)
                            and (
                                len(voter_key or "") != 18
                                or value[4:10] == voter_key[6:12]
                            )
                        ):
                            candidates[value] = max(
                                candidates.get(value, 0.0), confidence
                            )
    if not candidates:
        return None, 0.0
    return max(candidates.items(), key=lambda item: item[1])


def recognize_mexican_voter_name_fallback(
    resource: Path, image: Image.Image
) -> tuple[list[str], list[float]]:
    bands = (
        [0.34, 0.345, 0.58, 0.397],
        [0.34, 0.388, 0.58, 0.438],
        [0.34, 0.425, 0.62, 0.48],
    )
    values = []
    confidences = []
    for bounds in bands:
        crop = image.crop(
            (
                int(bounds[0] * image.width),
                int(bounds[1] * image.height),
                int(bounds[2] * image.width),
                int(bounds[3] * image.height),
            )
        )
        recognition_crop = ImageOps.autocontrast(
            ImageOps.grayscale(crop)
        ).convert("RGB")
        output = recognize_line(
            resource,
            recognition_crop,
            False,
            "minus-one-one",
            locale=2058,
        )
        value = " ".join(output["text"].upper().split())
        confidence = float(output["confidence"])
        if (
            confidence < 0.9
            or not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ ]{2,40}", value)
        ):
            return [], []
        values.append(value)
        confidences.append(confidence)
    return values, confidences


def recognize_mexican_voter_name_grid(
    resource: Path,
    image: Image.Image,
    personal_number: str,
) -> tuple[list[str], list[float]]:
    bands = (
        [0.34, 0.345, 0.58, 0.397],
        [0.34, 0.388, 0.58, 0.438],
        [0.34, 0.425, 0.62, 0.48],
    )
    expected_initials = [personal_number[index] for index in (0, 2, 3)]
    values = []
    confidences = []
    for bounds, expected_initial in zip(bands, expected_initials):
        candidates = {}
        for left_offset in (-0.03, -0.015, 0.0, 0.015):
            for right_offset in (0.0, 0.025, 0.05):
                for vertical_offset in (-0.012, 0.0, 0.012):
                    crop = image.crop(
                        (
                            int((bounds[0] + left_offset) * image.width),
                            int((bounds[1] + vertical_offset) * image.height),
                            int((bounds[2] + right_offset) * image.width),
                            int((bounds[3] + vertical_offset) * image.height),
                        )
                    )
                    recognition_crop = ImageOps.autocontrast(
                        ImageOps.grayscale(crop)
                    ).convert("RGB")
                    output = recognize_line(
                        resource,
                        recognition_crop,
                        False,
                        "minus-one-one",
                        locale=2058,
                    )
                    value = " ".join(output["text"].upper().split())
                    confidence = float(output["confidence"])
                    if (
                        confidence >= 0.9
                        and value.startswith(expected_initial)
                        and re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ ]{2,40}", value)
                    ):
                        candidates[value] = max(
                            candidates.get(value, 0.0), confidence
                        )
        if candidates:
            value, confidence = max(
                candidates.items(),
                key=lambda item: (
                    len(item[0].replace(" ", "")),
                    item[1],
                ),
            )
        else:
            value, confidence = "", 0.0
        values.append(value)
        confidences.append(confidence)
    return values, confidences


def merge_mexican_voter_names(
    original_names: list[str | None],
    fallback_names: list[str],
    personal_number: str | None,
) -> list[str]:
    shifted_name_lines = bool(
        (original_names[0] or "").upper() == "NOMBRE"
        and " ".join((original_names[1] or "").upper().split())
        == " ".join(fallback_names[0].upper().split())
    )
    expected_initials = (
        list(personal_number[index] for index in (0, 2, 3))
        if len(personal_number or "") == 18
        else [None, None, None]
    )
    selected_names = []
    for original_name, fallback_name, expected_initial in zip(
        original_names, fallback_names, expected_initials
    ):
        if shifted_name_lines:
            selected_names.append(fallback_name)
            continue
        candidates = [
            value
            for value in (original_name, fallback_name)
            if value
            and (
                expected_initial is None
                or value.upper().startswith(expected_initial)
            )
        ]
        selected_names.append(
            max(candidates, key=len)
            if candidates
            else original_name or fallback_name
        )
    return selected_names


def catalog_visual_result(
    resource: Path,
    image: Image.Image,
    candidate: dict,
    layout: dict,
    field_names: set[str] | None = None,
) -> dict:
    fields = recognize_visual_layout(resource, image, layout, field_names)
    available = [item for item in fields if item["value"]]
    values = {}
    confidences = {}
    for item in sorted(available, key=lambda value: value["confidence"]):
        values[item["name"]] = item["value"]
        confidences[item["name"]] = item["confidence"]
    combined_name = values.get("surnameAndGivenNames", "")
    surname = values.get("surname", "")
    given_names = values.get("givenNames", "")
    address_parts = [
        values.get(name, "")
        for name in (
            "addressStreet",
            "addressHouse",
            "addressBuilding",
            "addressFlat",
            "addressCity",
            "addressState",
            "addressPostalCode",
        )
        if values.get(name)
    ]
    address_lines = (
        [line.strip() for line in values.get("address", "").splitlines() if line.strip()]
        or address_parts
    )
    address = ", ".join(address_lines)
    document = candidate["document"]
    identifier = int(layout["identifier"])
    profile_identifier = f"N{abs(identifier)}" if identifier < 0 else f"P{identifier}"
    identity_evidence = {
        "documentNumber",
        "surnameAndGivenNames",
        "surname",
        "givenNames",
        "personalNumber",
        "dateOfBirth",
    }
    evidence_names = field_names or identity_evidence
    if not any(
        item["name"] in evidence_names and item["confidence"] >= 0.35
        for item in available
    ):
        return unsupported_document_result()
    is_mexican_voter = (
        "MEX" in (document.get("isoCodes") or [])
        and (document.get("documentType") or {}).get("name") == "VotingCard"
    )
    name_lines = [line.strip() for line in combined_name.splitlines() if line.strip()]
    if is_mexican_voter and len(name_lines) >= 3:
        first_surname = name_lines[0]
        second_surname = name_lines[1]
        surname = " ".join(name_lines[:2])
        given_names = " ".join(name_lines[2:])
    else:
        first_surname = values.get("firstSurname")
        second_surname = values.get("secondSurname")
    name_personal_number = "".join(
        character
        for character in values.get("personalNumber", "").upper()
        if character.isalnum()
    )
    if is_mexican_voter:
        fallback_names, fallback_confidences = recognize_mexican_voter_name_fallback(
            resource, image
        )
        if fallback_names:
            original_names = [first_surname, second_surname, given_names]
            selected_names = merge_mexican_voter_names(
                original_names,
                fallback_names,
                name_personal_number,
            )
            first_surname, second_surname, given_names = selected_names
            surname = " ".join(selected_names[:2])
            combined_name = "\n".join(selected_names)
            confidences["surnameAndGivenNames"] = float(
                np.mean(fallback_confidences)
            )
    if is_mexican_voter and len(name_personal_number) == 18:
        grid_names, grid_confidences = recognize_mexican_voter_name_grid(
            resource,
            image,
            name_personal_number,
        )
        selected_names = merge_mexican_voter_names(
            [first_surname, second_surname, given_names],
            grid_names,
            name_personal_number,
        )
        first_surname, second_surname, given_names = selected_names
        surname = " ".join(selected_names[:2])
        combined_name = "\n".join(selected_names)
        available_grid_confidences = [
            confidence for confidence in grid_confidences if confidence
        ]
        if available_grid_confidences:
            confidences["surnameAndGivenNames"] = max(
                confidences.get("surnameAndGivenNames", 0.0),
                float(np.mean(available_grid_confidences)),
            )
    if is_mexican_voter:
        address_lines = [re.sub(r"\bSN\b", "S/N", line) for line in address_lines]
        address = ", ".join(address_lines)
    personal_number = values.get("personalNumber")
    voter_key = values.get("voterKey")
    if is_mexican_voter:
        personal_number = "".join(
            character for character in (personal_number or "").upper() if character.isalnum()
        )
        voter_key = "".join(
            character for character in (voter_key or "").upper() if character.isalnum()
        )
        if not valid_mexican_curp(personal_number):
            grid_curp, grid_confidence = recognize_mexican_curp_grid(
                resource,
                image,
                voter_key,
            )
            if grid_curp:
                personal_number = grid_curp
                confidences["personalNumber"] = max(
                    confidences.get("personalNumber", 0.0),
                    grid_confidence,
                )
    date_of_birth = values.get("dateOfBirth")
    field_validation = {}
    if is_mexican_voter:
        date_of_birth = corroborated_mexican_birth_date(
            date_of_birth,
            personal_number,
            voter_key,
        )
        birth_verified = date_of_birth is not None
        field_validation = {
            "holderName": {
                "decision": "review",
                "reason": "single_visual_ocr_source",
            },
            "dateOfBirth": {
                "decision": "pass" if birth_verified else "review",
                "reason": (
                    "visual_curp_elector_key_consensus"
                    if birth_verified
                    else "insufficient_cross_field_agreement"
                ),
            },
            "curp": {
                "decision": "review",
                "reason": "structure_does_not_verify_all_characters",
            },
            "electorKey": {
                "decision": "review",
                "reason": "single_visual_ocr_source",
            },
        }
    registration_year = values.get("dateOfRegistration")
    section = values.get("section")
    if is_mexican_voter:
        year_match = re.search(r"(?:19|20)\d{2}", registration_year or "")
        registration_year = year_match.group(0) if year_match else None
        section_digits = "".join(
            character for character in (section or "") if character.isdigit()
        )
        section = section_digits if len(section_digits) == 4 else None
    recognized_fields = {
        item["name"]: {
            "value": item["value"],
            "confidence": item["confidence"],
            "available": bool(item["value"]),
            "source": "VISUAL",
            "fieldType": item["type"],
            "cropNormalized": item["bounds"],
            "derived": item.get("derived", False),
            "derivedFrom": item.get("derivedFrom"),
            "orientationCorrection": item.get("orientationCorrection"),
            "preprocessing": item.get("preprocessing"),
            "colorType": item.get("colorType"),
            "fontLayer": item.get("fontLayer"),
            "layer": item.get("layer"),
            "comparisonMode": item.get("comparisonMode"),
        }
        for item in available
    }
    field_list = [field(item["name"], item["value"]) for item in available]
    for item in field_list:
        item["source"] = "VISUAL"
        item["valueList"][0]["source"] = "VISUAL"
    regions = catalog_visual_regions(layout)
    return {
        "errorCode": 0,
        "DocumentName": document.get("caption"),
        "dCountryName": document.get("country"),
        "documentClassCode": values.get("documentClassCode"),
        "documentNumber": values.get("documentNumber"),
        "documentStatus": values.get("documentStatus"),
        "issuingStateCode": values.get("issuingStateCode"),
        "name": " ".join(combined_name.splitlines())
        or " ".join(filter(None, (surname, given_names))),
        "surnameAndGivenNames": combined_name,
        "firstSurname": first_surname,
        "secondSurname": second_surname,
        "surname": surname,
        "givenNames": given_names,
        "sex": values.get("sex"),
        "nationality": values.get("nationality"),
        "nationalityCode": values.get("nationalityCode"),
        "height": values.get("height"),
        "dateOfBirth": date_of_birth,
        "dateOfIssue": values.get("dateOfIssue"),
        "dateOfExpiry": values.get("dateOfExpiry"),
        "placeOfBirth": values.get("placeOfBirth"),
        "personalNumber": personal_number,
        "electorKey": voter_key if is_mexican_voter else None,
        "curp": personal_number if is_mexican_voter else None,
        "registrationYear": registration_year if is_mexican_voter else None,
        "section": section if is_mexican_voter else None,
        "authority": values.get("authority"),
        "address": address,
        "addressLines": address_lines,
        "city": values.get("addressCity"),
        "state": values.get("addressState"),
        "municipality": values.get("addressMunicipality"),
        "locality": values.get("addressLocation"),
        "postalCode": values.get("addressPostalCode"),
        "availableSourceList": ["VISUAL"],
        "source": "VISUAL",
        "validityStatus": -1,
        "fieldList": field_list,
        "recognizedFields": recognized_fields,
        "fieldValidation": field_validation,
        "visualRegions": regions,
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "fieldConfidence": confidences,
            "template": str(layout["identifier"]),
        },
        "recognitionProfile": f"CATALOG-{profile_identifier}",
        "recognitionProfileStatus": "selected_by_classifier_and_declarative_layout",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }


def repair_td1(lines: list[str], confidences: list[float]) -> list[str]:
    normalized = normalize_td1_lines(lines)
    if len(normalized) != 3:
        return lines
    candidates = [normalized]
    if len(normalized[0]) == 30 and len(normalized[1]) == 28 and len(normalized[2]) == 30:
        candidates.extend(
            [normalized[0], normalized[1] + "<" + digit, normalized[2]]
            for digit in "0123456789"
        )
    if len(normalized[2]) == 31 and normalized[2].endswith("<"):
        candidates.append([normalized[0], normalized[1], normalized[2][:-1]])
    for line_index in (0, 1):
        if len(normalized[line_index]) != 31:
            continue
        for character_index in range(31):
            candidate = normalized.copy()
            candidate[line_index] = (
                candidate[line_index][:character_index]
                + candidate[line_index][character_index + 1 :]
            )
            candidates.append(candidate)
    for candidate in candidates:
        if any(len(line) != 30 for line in candidate):
            continue
        try:
            if all(parse_td1(candidate, confidences)["checks"].values()):
                return candidate
        except (ValueError, IndexError):
            pass
    return normalized


def process_td1_back_first(image_path: Path, models_path: Path) -> dict | None:
    if not image_path.is_file():
        return None
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        original = load_image(image_path, temporary)
        images = [original]
        if original.width / original.height < 1.55:
            rectified, _ = rectify_document(models_path, original)
            images.insert(0, rectified)

        candidates = []
        for image_index, image in enumerate(images):
            for rotation_index, degrees in enumerate((90, 270, 0, 180)):
                oriented = image.rotate(degrees, expand=True) if degrees else image
                lines, confidences = recognize_td1(
                    models_path, oriented, localize=True
                )
                normalized = normalize_td1_lines(repair_td1(lines, confidences))
                if len(normalized) != 3 or any(len(line) != 30 for line in normalized):
                    continue
                try:
                    parsed = parse_td1(normalized, confidences)
                except (ValueError, IndexError):
                    continue
                valid_checks = sum(parsed["checks"].values())
                candidates.append(
                    (
                        valid_checks,
                        sum(confidences),
                        -image_index,
                        -rotation_index,
                        parsed,
                        degrees,
                    )
                )
                if valid_checks == 4:
                    break
            if candidates and max(candidates, key=lambda item: item[:4])[0] == 4:
                break

        if not candidates:
            return None
        valid_checks, _, _, _, result, rotation = max(
            candidates, key=lambda item: item[:4]
        )
        if valid_checks != 4 or result.get("issuingStateCode") != "MEX":
            return None

        machine_barcodes = decode_machine_barcodes(original)
        ine_barcode = ine_qr_evidence(machine_barcodes)
        result["requestedProfile"] = "auto_research"
        result["recognitionProfile"] = "ICAO-TD1"
        result["recognitionProfileStatus"] = "mrz_first_four_checks_valid"
        result["recognition"]["rotationApplied"] = rotation
        result["qualitySignals"] = analyze_quality(models_path, original)
        result["metadataIntegrity"] = check_metadata_integrity(image_path, models_path)
        if machine_barcodes:
            result["machineBarcodes"] = machine_barcodes
        if ine_barcode:
            result["ineQr"] = ine_barcode
        result["ContainerList"] = [{"OneCandidate": result.copy()}]
        result["Count"] = 1
        return result


def convert_image(source: Path, destination: Path) -> None:
    subprocess.run(
        ["magick", f"{source}[0]", "-auto-orient", str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def load_image(image_path: Path, temporary: Path) -> Image.Image:
    try:
        image = Image.open(image_path)
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        converted = temporary / "document.jpg"
        convert_image(image_path, converted)
        return Image.open(converted).convert("RGB")


def smoothed_projection(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), values.size))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def mrz_region_candidates(
    image: Image.Image,
    line_count: int,
    maximum: int = 2,
    geometry: dict | None = None,
) -> list[list[float]]:
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(image))
    scale = min(1.0, 1200 / max(grayscale.width, 1))
    if scale < 1.0:
        grayscale = grayscale.resize(
            (1200, max(1, round(grayscale.height * scale))),
            Image.Resampling.BILINEAR,
        )
    pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    if pixels.shape[0] < 20 or pixels.shape[1] < 80:
        return []
    horizontal_edges = np.zeros_like(pixels)
    horizontal_edges[:, 1:-1] = np.abs(pixels[:, 2:] - pixels[:, :-2])
    darkness = np.clip(0.82 - pixels, 0.0, 0.82) / 0.82
    signal = horizontal_edges + darkness * 0.18
    height, width = signal.shape
    row_signal = smoothed_projection(signal.mean(axis=1), max(3, height // 180))
    ratios = mrz_search_ratios(line_count, geometry)
    minimum_coverage = 0.55
    if geometry:
        minimum_coverage = max(
            0.35, min(0.75, float(geometry["widthRatio"]) * 0.70)
        )
    proposals = []
    for ratio in ratios:
        region_height = max(line_count * 8, round(height * ratio))
        if region_height >= height:
            continue
        step = max(2, height // 90)
        for top in range(0, height - region_height + 1, step):
            bottom = top + region_height
            region = signal[top:bottom]
            column_signal = region.mean(axis=0)
            threshold = max(float(np.percentile(column_signal, 55)), 0.018)
            active = np.flatnonzero(column_signal >= threshold)
            if active.size < width * minimum_coverage * 0.65:
                continue
            left = int(np.percentile(active, 1))
            right = int(np.percentile(active, 99)) + 1
            coverage = (right - left) / width
            if coverage < minimum_coverage:
                continue
            density = float(row_signal[top:bottom].mean())
            lower_bias = 0.9 + 0.1 * ((top + bottom) / (2 * height))
            geometry_bonus = 1.0
            if geometry:
                expected_height = float(geometry["heightRatio"]) * 1.10
                expected_width = float(geometry["widthRatio"])
                height_match = np.exp(
                    -abs(region_height / height - expected_height)
                    / max(expected_height, 0.01)
                )
                width_match = np.exp(
                    -abs(coverage - expected_width) / max(expected_width, 0.01)
                )
                geometry_bonus += 0.35 * float(height_match * width_match)
            proposals.append(
                (
                    density * coverage * lower_bias * geometry_bonus,
                    [
                        max(0.0, left / width - 0.015),
                        max(0.0, top / height - 0.015),
                        min(1.0, right / width + 0.015),
                        min(1.0, bottom / height + 0.015),
                    ],
                )
            )
    proposals.sort(reverse=True, key=lambda item: item[0])
    selected = []
    for _, bounds in proposals:
        center = (bounds[1] + bounds[3]) / 2
        if any(abs(center - (item[1] + item[3]) / 2) < 0.08 for item in selected):
            continue
        selected.append(bounds)
        if len(selected) == maximum:
            break
    return selected


def mrz_search_ratios(line_count: int, geometry: dict | None = None) -> tuple[float, ...]:
    defaults = (
        (0.13, 0.17, 0.21, 0.25)
        if line_count == 2
        else (0.18, 0.23, 0.28, 0.33)
    )
    if not geometry:
        return defaults
    expected = float(geometry["heightRatio"])
    guided = tuple(
        ratio
        for ratio in (expected * 0.95, expected * 1.10, expected * 1.25)
        if 0.08 <= ratio <= 0.40
    )
    return tuple(dict.fromkeys((*guided, *defaults)))


def mrz_line_bounds(region: list[float], line_count: int) -> list[list[float]]:
    left, top, right, bottom = region
    line_height = (bottom - top) / line_count
    overlap = line_height * 0.12
    return [
        [
            left,
            max(0.0, top + index * line_height - overlap),
            right,
            min(1.0, top + (index + 1) * line_height + overlap),
        ]
        for index in range(line_count)
    ]


def recognize_mrz_regions(
    resource: Path,
    image: Image.Image,
    line_count: int,
    geometry: dict | None = None,
) -> list[tuple[list[str], list[float]]]:
    candidates = []
    width, height = image.size
    for region in mrz_region_candidates(image, line_count, geometry=geometry):
        lines, confidences = [], []
        for left, top, right, bottom in mrz_line_bounds(region, line_count):
            crop = image.crop(
                (
                    round(width * left),
                    round(height * top),
                    round(width * right),
                    round(height * bottom),
                )
            )
            result = recognize_line(resource, crop, False, "minus-one-one")
            lines.append(result["text"])
            confidences.append(result["confidence"])
        candidates.append((lines, confidences))
    return candidates


def recognize_td1(
    resource: Path,
    image_source: Path | Image.Image,
    localize: bool = False,
    geometry: dict | None = None,
) -> tuple[list[str], list[float]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        image = load_image(image_source, temporary) if isinstance(image_source, Path) else image_source

        width, height = image.size
        crop_profiles = (
            (
                (0.031, 0.714, 0.974, 0.779),
                (0.031, 0.797, 0.974, 0.862),
                (0.031, 0.879, 0.974, 0.944),
            ),
            (
                (0.038, 0.666, 0.961, 0.723),
                (0.038, 0.737, 0.961, 0.794),
                (0.038, 0.807, 0.961, 0.864),
            ),
            (
                (0.006, 0.714, 0.994, 0.792),
                (0.006, 0.804, 0.994, 0.882),
                (0.006, 0.893, 0.994, 0.971),
            ),
        )
        candidates = []
        for profile_index, crop_specs in enumerate(crop_profiles):
            lines, confidences = [], []
            for line_index, bounds in enumerate(crop_specs, start=1):
                left, top, right, bottom = bounds
                crop = image.crop((round(width * left), round(height * top), round(width * right), round(height * bottom)))
                result = recognize_line(resource, crop, False, "minus-one-one")
                lines.append(result["text"])
                confidences.append(result["confidence"])
            lines = repair_td1(lines, confidences)
            exact_length = all(len(line) == 30 for line in normalize_td1_lines(lines))
            valid_checks = 0
            if exact_length:
                try:
                    valid_checks = sum(parse_td1(lines, confidences)["checks"].values())
                except (ValueError, IndexError):
                    pass
            candidates.append((valid_checks, exact_length, sum(confidences), lines, confidences))
        if localize and max(candidates, key=lambda candidate: candidate[:3])[0] < 4:
            for lines, confidences in recognize_mrz_regions(
                resource, image, 3, geometry
            ):
                lines = repair_td1(lines, confidences)
                exact_length = all(
                    len(line) == 30 for line in normalize_td1_lines(lines)
                )
                valid_checks = 0
                if exact_length:
                    try:
                        valid_checks = sum(parse_td1(lines, confidences)["checks"].values())
                    except (ValueError, IndexError):
                        pass
                candidates.append(
                    (valid_checks, exact_length, sum(confidences), lines, confidences)
                )
        _, _, _, best_lines, best_confidences = max(candidates, key=lambda candidate: candidate[:3])
    return best_lines, best_confidences


def recognize_td3(
    resource: Path,
    image_source: Path | Image.Image,
    localize: bool = False,
    geometry: dict | None = None,
) -> tuple[list[str], list[float]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        image = load_image(image_source, temporary) if isinstance(image_source, Path) else image_source
        width, height = image.size
        crop_profiles = (
            ((0.030, 0.810, 0.970, 0.863), (0.030, 0.880, 0.970, 0.935)),
            ((0.010, 0.800, 0.990, 0.860), (0.010, 0.870, 0.990, 0.930)),
            ((0.010, 0.790, 0.990, 0.850), (0.010, 0.860, 0.990, 0.920)),
            ((0.010, 0.775, 0.990, 0.855), (0.010, 0.865, 0.990, 0.945)),
            ((0.000, 0.735, 1.000, 0.825), (0.000, 0.835, 1.000, 0.925)),
        )
        candidates = []
        for crop_specs in crop_profiles:
            lines, confidences = [], []
            for bounds in crop_specs:
                left, top, right, bottom = bounds
                crop = image.crop(
                    (round(width * left), round(height * top), round(width * right), round(height * bottom))
                )
                result = recognize_line(resource, crop, False, "minus-one-one")
                lines.append(result["text"])
                confidences.append(result["confidence"])
            lines = repair_td3(lines, confidences)
            normalized = normalize_td3_lines(lines)
            exact_length = (
                len(normalized) == 2
                and len(normalized[0]) in {36, 44}
                and len(normalized[1]) == len(normalized[0])
            )
            valid_checks = two_line_mrz_candidate_score((lines, confidences))[0]
            candidates.append((valid_checks, exact_length, sum(confidences), lines, confidences))
        best_fixed = max(candidates, key=lambda candidate: candidate[:3])
        if localize and not fully_valid_two_line_mrz((best_fixed[3], best_fixed[4])):
            for lines, confidences in recognize_mrz_regions(
                resource, image, 2, geometry
            ):
                lines = repair_td3(lines, confidences)
                normalized = normalize_td3_lines(lines)
                exact_length = (
                    len(normalized) == 2
                    and len(normalized[0]) in {36, 44}
                    and len(normalized[1]) == len(normalized[0])
                )
                valid_checks = two_line_mrz_candidate_score((lines, confidences))[0]
                candidates.append(
                    (valid_checks, exact_length, sum(confidences), lines, confidences)
                )
        best_localized = max(candidates, key=lambda candidate: candidate[:3])
        if localize and not fully_valid_two_line_mrz(
            (best_localized[3], best_localized[4])
        ):
            for lines, confidences in recognize_adaptive_td3_bands(resource, image):
                lines = repair_td3(lines, confidences)
                normalized = normalize_td3_lines(lines)
                exact_length = (
                    len(normalized) == 2
                    and all(len(line) == 44 for line in normalized)
                )
                valid_checks = td3_candidate_score((lines, confidences))[0]
                candidates.append(
                    (valid_checks, exact_length, sum(confidences), lines, confidences)
                )
        _, _, _, best_lines, best_confidences = max(candidates, key=lambda candidate: candidate[:3])
    return best_lines, best_confidences


def recognize_adaptive_td3_bands(
    resource: Path,
    image: Image.Image,
) -> list[tuple[list[str], list[float]]]:
    width, height = image.size
    first_candidates = []
    for top in (0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90):
        for band_height in (0.045, 0.055, 0.065):
            crop = image.crop(
                (
                    round(width * 0.01),
                    round(height * top),
                    round(width * 0.99),
                    round(height * min(top + band_height, 0.97)),
                )
            )
            crop = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
            result = recognize_line(resource, crop, False, "minus-one-one")
            normalized = normalize_td3_lines([result["text"]])[0]
            plausibility = (
                1 if normalized.startswith("P") else 0,
                -abs(len(normalized) - 44),
                normalized.count("<"),
                result["confidence"],
            )
            first_candidates.append((plausibility, top, band_height, result))

    results = []
    for _, first_top, first_height, first_result in sorted(
        first_candidates, reverse=True
    )[:4]:
        for offset in (0.055, 0.065, 0.075, 0.085):
            second_top = first_top + offset
            for band_height in (first_height, 0.055, 0.065):
                if second_top + band_height > 0.995:
                    continue
                crop = image.crop(
                    (
                        round(width * 0.01),
                        round(height * second_top),
                        round(width * 0.99),
                        round(height * (second_top + band_height)),
                    )
                )
                crop = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
                second_result = recognize_line(
                    resource, crop, False, "minus-one-one"
                )
                results.append(
                    (
                        [first_result["text"], second_result["text"]],
                        [first_result["confidence"], second_result["confidence"]],
                    )
                )
    return results


def recognize_best_two_line_mrz(
    resource: Path,
    original: Image.Image,
    effective_image: Image.Image,
    localize: bool = False,
    geometry: dict | None = None,
) -> tuple[list[str], list[float]]:
    candidates = [recognize_td3(resource, original, localize, geometry)]
    if effective_image is not original:
        candidates.append(
            recognize_td3(resource, effective_image, localize, geometry)
        )
    else:
        rectified, _ = rectify_document(resource, original)
        candidates.append(recognize_td3(resource, rectified, localize, geometry))
    return max(candidates, key=two_line_mrz_candidate_score)


def recognize_front(resource: Path, image_source: Path | Image.Image) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        image = load_image(image_source, temporary) if isinstance(image_source, Path) else image_source
        width, height = image.size
        profiles = {
            "INE-front-legacy": {
                "surname1": (0.318, 0.295, 0.637, 0.338),
                "surname2": (0.318, 0.338, 0.637, 0.381),
                "givenNames": (0.318, 0.385, 0.728, 0.432),
                "dateOfBirth": (0.773, 0.295, 0.937, 0.345),
                "address1": (0.310, 0.480, 0.650, 0.525),
                "address2": (0.310, 0.535, 0.680, 0.580),
                "address3": (0.310, 0.590, 0.570, 0.635),
                "electorKey": (0.485, 0.645, 0.765, 0.700),
                "curp": (0.355, 0.700, 0.660, 0.750),
                "registrationYear": (0.835, 0.700, 0.960, 0.750),
                "state": (0.390, 0.775, 0.455, 0.825),
                "municipality": (0.600, 0.775, 0.680, 0.825),
                "section": (0.755, 0.775, 0.850, 0.825),
                "locality": (0.415, 0.830, 0.500, 0.885),
                "issueYear": (0.600, 0.830, 0.690, 0.885),
                "validity": (0.750, 0.830, 0.850, 0.885),
            },
            "INE-front-modern": {
                "surname1": (0.3125, 0.332, 0.625, 0.382),
                "surname2": (0.3125, 0.377, 0.625, 0.427),
                "givenNames": (0.3125, 0.422, 0.688, 0.472),
                "address1": (0.315, 0.570, 0.745, 0.620),
                "address2": (0.305, 0.615, 0.620, 0.660),
                "address3": (0.315, 0.660, 0.625, 0.700),
                "electorKey": (0.516, 0.729, 0.828, 0.779),
                "curp": (0.3125, 0.818, 0.688, 0.868),
                "dateOfBirth": (0.305, 0.910, 0.445, 0.965),
                "registrationYear": (0.688, 0.818, 0.850, 0.868),
                "section": (0.5625, 0.913, 0.675, 0.963),
                "validity": (0.688, 0.913, 0.850, 0.963),
                "sex": (0.894, 0.298, 0.956, 0.348),
            },
        }
        candidates = []
        for profile_name, crops in profiles.items():
            values, confidences = {}, {}
            for field_name, bounds in crops.items():
                left, top, right, bottom = bounds
                crop = image.crop((round(width * left), round(height * top), round(width * right), round(height * bottom)))
                recognition_crop = crop
                if profile_name == "INE-front-modern" and field_name == "address1":
                    recognition_crop = ImageOps.autocontrast(ImageOps.grayscale(crop))
                result = recognize_line(resource, recognition_crop, False, "minus-one-one")
                values[field_name] = result["text"].strip(" <")
                confidences[field_name] = result["confidence"]
            key_fields = ("surname1", "surname2", "givenNames", "electorKey", "curp")
            score = sum(confidences.get(name, 0.0) for name in key_fields if values.get(name))
            candidates.append((score, profile_name, values, confidences))
        _, profile_name, values, confidences = max(candidates, key=lambda candidate: candidate[0])

        values["electorKey"] = "".join(
            character for character in values.get("electorKey", "").upper() if character.isalnum()
        )
        values["curp"] = "".join(
            character for character in values.get("curp", "").upper() if character.isalnum()
        )
        if len(values["curp"]) == 18 and values["curp"][10] in {"H", "M"}:
            values["sex"] = values["curp"][10]
        registration_parts = values.get("registrationYear", "").split()
        if registration_parts and len(registration_parts[0]) == 4 and registration_parts[0].isdigit():
            values["registrationYear"] = registration_parts[0]
        if profile_name == "INE-front-legacy":
            for address_key in ("address1", "address2", "address3"):
                values[address_key] = normalize_ine_address_line(values.get(address_key, ""))
        else:
            values["address1"] = " ".join(values.get("address1", "").split()).upper()
            values["address2"] = " ".join(values.get("address2", "").split()).title()
            values["address3"] = (
                " ".join(values.get("address3", "").split()).title().replace(" De ", " de ")
            )
        birth_digits = "".join(
            character for character in values.get("dateOfBirth", "") if character.isdigit()
        )
        if len(birth_digits) == 8:
            values["dateOfBirth"] = (
                f"{birth_digits[:2]}/{birth_digits[2:4]}/{birth_digits[4:]}"
            )

        quality = analyze_quality(resource, image)

    surname = " ".join(filter(None, (values["surname1"], values["surname2"])))
    address_lines = [
        values[field_name]
        for field_name in ("address1", "address2", "address3")
        if values[field_name]
    ]
    address = ", ".join(address_lines)
    visual_fields = [
        field("First Surname", values["surname1"]),
        field("Second Surname", values["surname2"]),
        field("Surname", surname),
        field("Given Names", values["givenNames"]),
        field("Date of Birth", values["dateOfBirth"]),
        field("Address", address),
    ]
    visual_fields.extend(
        field(f"Address Line {index}", values[f"address{index}"])
        for index in range(1, 4)
        if values.get(f"address{index}")
    )
    optional_fields = (
        ("Elector Key", "electorKey"),
        ("CURP", "curp"),
        ("Sex", "sex"),
        ("Registration Year", "registrationYear"),
        ("State", "state"),
        ("Municipality", "municipality"),
        ("Section", "section"),
        ("Locality", "locality"),
        ("Issue Year", "issueYear"),
        ("Validity", "validity"),
    )
    visual_fields.extend(
        field(display_name, values[key])
        for display_name, key in optional_fields
        if values.get(key)
    )
    for item in visual_fields:
        item["source"] = "VISUAL"
        item["valueList"][0]["source"] = "VISUAL"
    return {
        "errorCode": 0,
        "DocumentName": "Voter Credential",
        "dCountryName": "Mexico",
        "firstSurname": values["surname1"],
        "secondSurname": values["surname2"],
        "surname": surname,
        "givenNames": values["givenNames"],
        "surnameAndGivenNames": " ".join(filter(None, (surname, values["givenNames"]))),
        "dateOfBirth": values["dateOfBirth"],
        "address": address,
        "addressLines": address_lines,
        "addressLine1": values.get("address1", ""),
        "addressLine2": values.get("address2", ""),
        "addressLine3": values.get("address3", ""),
        "electorKey": values.get("electorKey", ""),
        "curp": values.get("curp", ""),
        "registrationYear": values.get("registrationYear", ""),
        "state": values.get("state", ""),
        "municipality": values.get("municipality", ""),
        "section": values.get("section", ""),
        "locality": values.get("locality", ""),
        "issueYear": values.get("issueYear", ""),
        "validity": values.get("validity", ""),
        "sex": values.get("sex", ""),
        "availableSourceList": ["VISUAL"],
        "source": "VISUAL",
        "validityStatus": -1,
        "fieldList": visual_fields,
        "recognizedFields": map_visual_fields(
            values, confidences, profiles[profile_name]
        ),
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "fieldConfidence": confidences,
            "template": profile_name,
        },
        "recognitionProfile": profile_name.replace("INE-front", "MEX-INE-front"),
        "recognitionProfileStatus": "selected_by_known_template_ocr_score",
        "sdkCompatibility": SDK_COMPATIBILITY,
        "qualitySignals": quality,
    }


def normalize_ine_address_line(value: str) -> str:
    normalized = " ".join(value.split()).title()
    if normalized.startswith("Col "):
        normalized = "COL " + normalized[4:]
    words = []
    for word in normalized.split():
        stripped = word.rstrip(".,")
        suffix = word[len(stripped):]
        if suffix and 2 <= len(stripped) <= 4 and stripped.isalpha():
            word = stripped.upper() + suffix
        words.append(word)
    return " ".join(words)


def matches_mex_ine(result: dict) -> bool:
    curp = "".join(character for character in result.get("curp", "").upper() if character.isalnum())
    elector_key = "".join(
        character for character in result.get("electorKey", "").upper() if character.isalnum()
    )
    curp_like = len(curp) == 18 and curp[:4].isalpha() and curp[4:10].isdigit()
    elector_like = len(elector_key) >= 16
    supporting_fields = sum(
        bool(result.get(name))
        for name in ("registrationYear", "section", "validity", "dateOfBirth")
    )
    return curp_like or (elector_like and supporting_fields >= 1)


def parse_swedish_visual_date(value: str) -> str:
    match = re.fullmatch(r"(\d{2})\s+([A-Z]{3})(?:/[A-Z]{3})?\s+(\d{2})", value.strip().upper())
    if not match:
        return value
    day, month_name, year = match.groups()
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = months.get(month_name)
    if not month:
        return value
    numeric_year = int(year)
    full_year = 1900 + numeric_year if numeric_year > date.today().year % 100 else 2000 + numeric_year
    return date(full_year, month, int(day)).isoformat()


def recognize_swe_id_2021(resource: Path, image: Image.Image) -> tuple[dict, int]:
    width, height = image.size
    crops = {
        "surname": (0.0417, 0.1518, 0.2917, 0.2147),
        "givenNames": (0.0417, 0.2251, 0.2500, 0.2906),
        "sexRaw": (0.2833, 0.3246, 0.3917, 0.3848),
        "nationalityRaw": (0.4083, 0.3246, 0.6417, 0.3848),
        "dateOfBirth": (0.2833, 0.4084, 0.5500, 0.4738),
        "personalNumber": (0.2833, 0.4895, 0.5750, 0.5576),
        "documentNumber": (0.0417, 0.5812, 0.2583, 0.6492),
        "height": (0.2833, 0.5812, 0.4500, 0.6492),
        "dateOfIssue": (0.0417, 0.6806, 0.3417, 0.7513),
        "dateOfExpiry": (0.0417, 0.7618, 0.3417, 0.8325),
        "authority": (0.0333, 0.8534, 0.3667, 0.9058),
        "documentStatus": (0.3250, 0.1257, 0.5833, 0.2042),
    }
    values, confidences = {}, {}
    for name, (left, top, right, bottom) in crops.items():
        crop = image.crop(
            (round(width * left), round(height * top), round(width * right), round(height * bottom))
        )
        result = recognize_line(resource, crop, False, "minus-one-one")
        values[name] = result["text"].strip(" <")
        confidences[name] = result["confidence"]

    nationality_code = values["nationalityRaw"].split("/")[-1].strip()
    sex = {"K/F": "F", "F": "F", "M/M": "M", "M": "M"}.get(values["sexRaw"], "")
    for name in ("dateOfBirth", "dateOfIssue", "dateOfExpiry"):
        values[name] = parse_swedish_visual_date(values[name])
    match_score = sum(
        (
            2 if nationality_code == "SWE" else 0,
            2 if "POLISMYNDIGHETEN" in values["authority"] else 0,
            1 if sex else 0,
            1 if values["documentNumber"].isdigit() and len(values["documentNumber"]) == 8 else 0,
            1 if values["personalNumber"].replace("-", "").isdigit() else 0,
        )
    )
    normalized = {
        "surname": values["surname"],
        "givenNames": values["givenNames"],
        "dateOfBirth": values["dateOfBirth"],
        "dateOfIssue": values["dateOfIssue"],
        "dateOfExpiry": values["dateOfExpiry"],
        "documentNumber": values["documentNumber"],
        "personalNumber": values["personalNumber"].replace("-", ""),
        "sex": sex,
        "height": values["height"].lower(),
        "authority": values["authority"],
        "documentStatus": values["documentStatus"],
        "nationality": "Sweden" if nationality_code == "SWE" else values["nationalityRaw"],
        "nationalityCode": nationality_code,
        "issuingStateCode": "SWE",
    }
    recognized_fields = {
        name: {
            "value": value,
            "confidence": confidences.get(name if name in confidences else f"{name}Raw"),
            "available": bool(value),
            "source": "VISUAL",
            "cropNormalized": list(crops.get(name, crops.get(f"{name}Raw", ()))) or None,
        }
        for name, value in normalized.items()
        if name not in {"issuingStateCode", "nationalityCode"}
    }
    field_list = [field(name, value) for name, value in normalized.items() if value]
    for item in field_list:
        item["source"] = "VISUAL"
        item["valueList"][0]["source"] = "VISUAL"
    result = {
        "errorCode": 0,
        "DocumentName": "Sweden - Id Card (2021)",
        "dCountryName": "Sweden",
        **normalized,
        "name": " ".join(filter(None, (normalized["surname"], normalized["givenNames"]))),
        "surnameAndGivenNames": " ".join(
            filter(None, (normalized["surname"], normalized["givenNames"]))
        ),
        "availableSourceList": ["VISUAL"],
        "source": "VISUAL",
        "validityStatus": 0 if normalized["documentStatus"] == "SPECIMEN" else -1,
        "fieldList": field_list,
        "recognizedFields": recognized_fields,
        "Images": {},
        "recognition": {
            "engine": LINE_RECOGNITION_ENGINE,
            "fieldConfidence": confidences,
            "template": "SWE-ID-2021-front",
        },
        "recognitionProfile": "SWE-ID-2021-front",
        "recognitionProfileStatus": "selected_by_visual_structure",
        "sdkCompatibility": SDK_COMPATIBILITY,
    }
    return result, match_score


SUPPORTED_REQUEST_PROFILES = {
    "mex_ine",
    "mex_passport",
    "icao_td1",
    "icao_td2",
    "icao_td3",
    "icao_mrv",
    "aamva_pdf417",
    "swe_id_2021",
    "auto_research",
}


def should_try_td1(profile: str) -> bool:
    return profile not in {
        "mex_passport",
        "icao_td2",
        "icao_td3",
        "icao_mrv",
        "swe_id_2021",
    }

def process_document(
    image_path: Path,
    models_path: Path,
    profile: str = "auto_research",
    document_identifier: int | None = None,
    field_names: set[str] | None = None,
) -> dict:
    """Process one image using an explicitly supported recognition profile."""
    image_path = Path(image_path).resolve()
    models_path = Path(models_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not models_path.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {models_path}")
    if profile not in SUPPORTED_REQUEST_PROFILES:
        raise ValueError(
            f"Unsupported profile {profile!r}; choose one of {sorted(SUPPORTED_REQUEST_PROFILES)}"
        )
    if field_names is not None and document_identifier is None:
        raise ValueError("field_names requires an explicit documentIdentifier")
    validate_assets(models_path)
    explicit_layout = (
        visual_layout(models_path, document_identifier)
        if document_identifier is not None
        else None
    )
    if document_identifier is not None and explicit_layout is None:
        raise ValueError(f"Unknown documentIdentifier: {document_identifier}")

    rectification = None
    with tempfile.TemporaryDirectory() as processing_directory:
        original = load_image(image_path, Path(processing_directory))
        classification = None
        layout_hint = None
        if explicit_layout is not None:
            document = document_catalog_entry(models_path, document_identifier)
            if document is None:
                document = deepcopy(explicit_layout)
                recognition_hints = document.get("catalogHints", {}).get(
                    "recognition", {}
                )
                document["mrz"] = {
                    "present": bool(recognition_hints.get("dMRZ")),
                    "ignored": False,
                }
            classification = {
                "candidates": [
                    {
                        "documentIdentifier": document_identifier,
                        "confidence": 1.0,
                        "document": document,
                    }
                ],
            }
            layout_hint = classifier_layout_hint(classification)
        elif classifier_available(models_path):
            try:
                classification = classify_document(models_path, original)
            except (OSError, ValueError, KeyError):
                classification = None
            if classification is not None and profile == "auto_research":
                layout_hint = classifier_layout_hint(classification)
        mrz_hint = classifier_mrz_hint(classification)
        explicit_barcode_regions = (
            catalog_barcode_regions(explicit_layout)
            if explicit_layout and field_names is None
            else []
        )
        machine_barcodes = (
            decode_machine_barcodes(original)
            if profile in {"auto_research", "mex_ine"}
            and (explicit_layout is None or explicit_barcode_regions)
            else []
        )
        ine_barcode = ine_qr_evidence(machine_barcodes)
        barcode = (
            decode_pdf417(original)
            if should_scan_pdf417(profile, layout_hint, mrz_hint)
            and ine_barcode is None
            and (explicit_layout is None or explicit_barcode_regions)
            else None
        )
        if barcode:
            result = aamva_result(barcode)
            result["requestedProfile"] = profile
            result["qualitySignals"] = analyze_quality(models_path, original)
            result["metadataIntegrity"] = check_metadata_integrity(image_path, models_path)
            result["sdkCompatibility"] = SDK_COMPATIBILITY
            result["ContainerList"] = [{"OneCandidate": result.copy()}]
            result["Count"] = 1
            return result
        if profile == "aamva_pdf417":
            raise ValueError("No valid AAMVA PDF417 barcode was recognized")
        effective_image = original
        if original.width / original.height < 1.45:
            rectified, rectification = rectify_document(models_path, original)
            effective_image = rectified
        elif (
            original.width / original.height < 1.55
            and (profile == "mex_ine" or layout_hint == "mex_ine")
        ):
            rectified, candidate_rectification = rectify_document(models_path, original)
            corners = candidate_rectification["sourceCorners"]
            detected_width = max(point[0] for point in corners) - min(
                point[0] for point in corners
            )
            detected_height = max(point[1] for point in corners) - min(
                point[1] for point in corners
            )
            detected_coverage = (
                detected_width
                * detected_height
                / max(1, original.width * original.height)
            )
            if detected_coverage < 0.82:
                effective_image = rectified
                rectification = candidate_rectification
        if rectification and (
            profile == "mex_ine" or layout_hint == "mex_ine"
        ):
            effective_image, rotation, canonical_classification = orient_mexican_voter_card(
                models_path, effective_image
            )
            rectification["rotationApplied"] = rotation
            classification = canonical_classification
            layout_hint = classifier_layout_hint(classification) or layout_hint
            mrz_hint = classifier_mrz_hint(classification)
        catalog_candidate = (
            classifier_visual_layout_candidate(classification)
            if classification and profile == "auto_research"
            else mexican_voter_card_candidate(classification)
            if classification and profile == "mex_ine"
            else None
        )
        catalog_layout = (
            explicit_layout
            or (
                visual_layout(models_path, catalog_candidate["documentIdentifier"])
                if catalog_candidate
                else None
            )
        )
        if explicit_layout is not None:
            catalog_candidate = classification["candidates"][0]
        guided_barcode_regions = (
            catalog_barcode_regions(catalog_layout)
            if catalog_layout and field_names is None
            else []
        )
        if guided_barcode_regions and not machine_barcodes:
            machine_barcodes = decode_machine_barcodes_in_regions(
                effective_image, guided_barcode_regions
            )
            ine_barcode = ine_qr_evidence(machine_barcodes)
        if guided_barcode_regions and not barcode:
            barcode = decode_pdf417_in_regions(effective_image, guided_barcode_regions)
            if barcode:
                result = aamva_result(barcode)
                result["requestedProfile"] = profile
                result["qualitySignals"] = analyze_quality(models_path, original)
                result["metadataIntegrity"] = check_metadata_integrity(
                    image_path, models_path
                )
                if explicit_layout is None:
                    result["documentClassification"] = classification
                result["sdkCompatibility"] = SDK_COMPATIBILITY
                result["ContainerList"] = [{"OneCandidate": result.copy()}]
                result["Count"] = 1
                return result
        mrz_geometry = (
            mrz_physical_geometry(catalog_layout) if catalog_layout else None
        )
        two_line_candidate = None
        if explicit_layout is not None and field_names is not None:
            result = catalog_visual_result(
                models_path,
                effective_image,
                catalog_candidate,
                catalog_layout,
                field_names,
            )
        elif (
            explicit_layout is not None
            and mrz_hint is None
            and layout_hint == "mex_ine"
        ):
            result = recognize_front(models_path, effective_image)
            if not matches_mex_ine(result):
                result = catalog_visual_result(
                    models_path, effective_image, catalog_candidate, catalog_layout
                )
        elif (
            explicit_layout is not None
            and mrz_hint is None
            and layout_hint == "swe_id_2021"
        ):
            result, match_score = recognize_swe_id_2021(
                models_path, effective_image
            )
            if match_score < 5:
                result = catalog_visual_result(
                    models_path, effective_image, catalog_candidate, catalog_layout
                )
        elif explicit_layout is not None and mrz_hint is None:
            result = catalog_visual_result(
                models_path, effective_image, catalog_candidate, catalog_layout
            )
        else:
            result = None
        if (
            result is None
            and profile == "auto_research"
            and mrz_hint in {"td2", "td3", "mrv"}
        ):
            two_line_candidate = recognize_best_two_line_mrz(
                models_path,
                original,
                effective_image,
                localize=True,
                geometry=mrz_geometry,
            )
            guided_lines, guided_confidences = two_line_candidate
            guided_valid = {
                "td2": td2_candidate_score(two_line_candidate)[0] == 4,
                "td3": td3_candidate_score(two_line_candidate)[0] == 5,
                "mrv": mrv_candidate_score(two_line_candidate)[0] == 3,
            }[mrz_hint]
            if guided_valid:
                parser = {"td2": parse_td2, "td3": parse_td3, "mrv": parse_mrv}[
                    mrz_hint
                ]
                result = parser(guided_lines, guided_confidences)
                result["recognition"]["catalogMrzHint"] = mrz_hint
                if mrz_hint == "td3" and result["issuingStateCode"] == "BLR":
                    result = enrich_blr_passport(result, models_path, original)

        if result is None:
            if not should_try_td1(profile):
                lines, confidences = [], []
            else:
                lines, confidences = recognize_td1(
                    models_path,
                    effective_image,
                    localize=(
                        profile == "icao_td1"
                        or (profile == "auto_research" and mrz_hint == "td1")
                    ),
                    geometry=mrz_geometry,
                )
            has_td1 = len(lines) == 3 and all(
                len(line) == 30 for line in normalize_td1_lines(lines)
            )
            accept_td1 = profile in {"mex_ine", "icao_td1", "auto_research"}
            if has_td1 and accept_td1:
                result = parse_td1(lines, confidences)
                if mrz_hint == "td1":
                    result["recognition"]["catalogMrzHint"] = mrz_hint
            elif profile == "icao_td1":
                raise ValueError("No valid ICAO TD1 MRZ was recognized")
            elif profile == "mex_ine":
                result = (
                    catalog_visual_result(
                        models_path, effective_image, catalog_candidate, catalog_layout
                    )
                    if catalog_candidate and catalog_layout
                    else recognize_front(models_path, effective_image)
                )
            elif profile == "swe_id_2021":
                swedish_result, match_score = recognize_swe_id_2021(
                    models_path, effective_image
                )
                if match_score < 5:
                    raise ValueError(
                        "Image does not match the Sweden ID Card 2021 profile"
                    )
                result = swedish_result
            else:
                if two_line_candidate is None:
                    two_line_candidate = recognize_best_two_line_mrz(
                        models_path,
                        original,
                        effective_image,
                        localize=profile
                        in {
                            "mex_passport",
                            "icao_td2",
                            "icao_td3",
                            "icao_mrv",
                            "auto_research",
                        },
                        geometry=mrz_geometry,
                    )
                td3_lines, td3_confidences = two_line_candidate
                has_mrv = mrv_candidate_score(two_line_candidate)[0] == 3
                has_td2 = td2_candidate_score(two_line_candidate)[0] == 4
                has_td3 = (
                    len(td3_lines) == 2
                    and all(
                        len(line) == 44 for line in normalize_td3_lines(td3_lines)
                    )
                    and td3_candidate_score(two_line_candidate)[0] == 5
                )
                if has_mrv and profile in {"icao_mrv", "auto_research"}:
                    result = parse_mrv(td3_lines, td3_confidences)
                elif has_td2 and profile in {"icao_td2", "auto_research"}:
                    result = parse_td2(td3_lines, td3_confidences)
                elif has_td3 and profile in {
                    "mex_passport",
                    "icao_td3",
                    "auto_research",
                }:
                    result = parse_td3(td3_lines, td3_confidences)
                    if (
                        profile == "mex_passport"
                        and result["issuingStateCode"] != "MEX"
                    ):
                        raise ValueError(
                            "The recognized TD3 passport was not issued by Mexico"
                        )
                    if profile == "mex_passport":
                        result["recognitionProfile"] = "MEX-PASSPORT-TD3"
                    if result["issuingStateCode"] == "MEX":
                        result = enrich_mex_passport(
                            result, models_path, effective_image
                        )
                    if result["issuingStateCode"] == "BLR":
                        result = enrich_blr_passport(result, models_path, original)
                elif profile in {
                    "mex_passport",
                    "icao_td2",
                    "icao_td3",
                    "icao_mrv",
                }:
                    expected = {
                        "mex_passport": "Mexican TD3 passport",
                        "icao_td2": "TD2",
                        "icao_td3": "TD3",
                        "icao_mrv": "MRV-A or MRV-B",
                    }[profile]
                    raise ValueError(f"No valid ICAO {expected} MRZ was recognized")
                elif profile == "auto_research":
                    if layout_hint == "mex_ine":
                        ine_result = recognize_front(models_path, effective_image)
                        if matches_mex_ine(ine_result):
                            result = ine_result
                        else:
                            swedish_result, match_score = recognize_swe_id_2021(
                                models_path, effective_image
                            )
                            result = (
                                swedish_result
                                if match_score >= 5
                                else unsupported_document_result()
                            )
                    else:
                        swedish_result, match_score = recognize_swe_id_2021(models_path, effective_image)
                        if match_score >= 5:
                            result = swedish_result
                        else:
                            ine_result = recognize_front(models_path, effective_image)
                            result = ine_result if matches_mex_ine(ine_result) else unsupported_document_result()
                else:
                    result = recognize_front(models_path, effective_image)
        if result.get("recognitionProfile") == "UNSUPPORTED" and ine_barcode:
            result = ine_qr_result(ine_barcode)
        if (
            result.get("recognitionProfile") == "UNSUPPORTED"
            and classification
            and explicit_layout is None
        ):
            candidate = catalog_candidate
            layout = catalog_layout
            if candidate and layout:
                result = catalog_visual_result(
                    models_path, effective_image, candidate, layout
                )
        result["requestedProfile"] = profile
        if machine_barcodes:
            result["machineBarcodes"] = machine_barcodes
        if ine_barcode:
            result["ineQr"] = ine_barcode
        if classification and explicit_layout is None:
            result["documentClassification"] = classification
            if not result.get("visualRegions"):
                graphic_candidate = classifier_visual_layout_candidate(
                    classification
                ) or recognized_document_visual_layout_candidate(result, classification)
                graphic_layout = (
                    visual_layout(models_path, graphic_candidate["documentIdentifier"])
                    if graphic_candidate
                    else None
                )
                if graphic_layout:
                    result["visualRegions"] = catalog_visual_regions(graphic_layout)
        if explicit_layout is not None and not result.get("visualRegions"):
            result["visualRegions"] = catalog_visual_regions(explicit_layout)
        if "qualitySignals" not in result:
            result["qualitySignals"] = analyze_quality(models_path, original)
        if explicit_layout is not None:
            result["requestedDocumentIdentifier"] = document_identifier
            if field_names is not None:
                result["requestedFields"] = sorted(field_names)
            result["recognitionProfileStatus"] = (
                "explicit_layout_not_recognized"
                if result.get("recognitionProfile") == "UNSUPPORTED"
                else "selected_by_explicit_layout"
            )
        role_identifier = effective_layout_identifier(result)
        if role_identifier is not None:
            result["pageRole"] = layout_page_role(models_path, role_identifier)
        result["metadataIntegrity"] = check_metadata_integrity(
            image_path, models_path
        )
        if rectification:
            result["DocumentPosition"] = rectification
            for region in result.get("visualRegions", {}).values():
                region["processedDocumentBox"] = region["box"]
                region["box"] = map_rectified_bounds_to_source(
                    region["box"],
                    rectification["sourceCorners"],
                    original.size,
                    rectification.get("rotationApplied", 0),
                )
                region["coordinateSpace"] = "original_image"
        else:
            for region in result.get("visualRegions", {}).values():
                region["coordinateSpace"] = "original_image"
    result["ContainerList"] = [{"OneCandidate": result.copy()}]
    result["Count"] = 1
    return result


def normalized_identity_text(value: str | None) -> str:
    return identity_key(value)


def normalized_identity_date(value: str | None) -> str:
    digits = "".join(character for character in (value or "") if character.isdigit())
    if len(digits) != 8:
        return digits
    if value and value[:4].isdigit():
        return digits
    return digits[4:] + digits[2:4] + digits[:2]


def compatible_identity_text(
    first: str | None, second: str | None, models_path: Path | None = None
) -> bool | None:
    left_variants = identity_variants(first, models_path)
    right_variants = identity_variants(second, models_path)
    if not left_variants or not right_variants:
        return None
    return any(
        left == right or left.startswith(right) or right.startswith(left)
        for left in left_variants
        for right in right_variants
    )


def side_summary(result: dict, side: str) -> dict:
    classification = next(
        (
            candidate
            for candidate in result.get("documentClassification", {}).get("candidates", [])
            if candidate.get("document")
        ),
        None,
    )
    if classification and float(classification.get("confidence", 0.0)) < 0.65:
        classification = None
    return {
        "side": side,
        "recognized": result.get("recognitionProfile") != "UNSUPPORTED",
        "profile": result.get("recognitionProfile"),
        "layoutIdentifier": result.get("requestedDocumentIdentifier"),
        "source": result.get("source"),
        "classification": {
            "identifier": classification.get("documentIdentifier"),
            "name": classification["document"].get("caption"),
            "confidence": classification.get("confidence"),
        }
        if classification
        else None,
    }


def catalog_side_relation(front: dict, models_path: Path) -> dict | None:
    candidates = [
        candidate
        for candidate in front.get("documentClassification", {}).get("candidates", [])
        if candidate.get("document")
    ]
    requested_identifier = front.get("requestedDocumentIdentifier")
    if requested_identifier is not None:
        requested_layout = visual_layout(models_path, requested_identifier)
        candidates = (
            [
                {
                    "documentIdentifier": requested_identifier,
                    "confidence": None,
                    "document": requested_layout,
                }
            ]
            if requested_layout
            else []
        )
    if front.get("recognitionProfile", "").startswith("MEX-INE"):
        candidates = [
            candidate
            for candidate in candidates
            if "MEX" in candidate["document"].get("isoCodes", [])
            and (candidate["document"].get("documentType") or {}).get("name")
            == "VotingCard"
        ]
    for candidate in candidates:
        layout = visual_layout(models_path, candidate["documentIdentifier"])
        if not layout or not (
            layout.get("pairedPages") or layout.get("childDocuments")
        ):
            continue
        related = []
        relation_identifiers = layout.get("pairedPages") or layout["childDocuments"]
        for identifier in relation_identifiers:
            child = visual_layout(models_path, identifier)
            if child:
                related.append({"identifier": identifier, "name": child.get("caption")})
        if related:
            return {
                "frontIdentifier": candidate["documentIdentifier"],
                "frontConfidence": candidate.get("confidence"),
                "relatedDocuments": related,
                "relationType": (
                    "paired_page" if layout.get("pairedPages") else "related_document"
                ),
                "status": "catalog_relation_available",
            }
    return None


def aggregate_document_capture_decisions(decisions: list[str | None]) -> str:
    available = [
        decision
        for decision in decisions
        if decision not in {None, "not_available"}
    ]
    if any(decision in {"review", "spoof"} for decision in available):
        return "review"
    if available and all(decision in {"pass", "real"} for decision in available):
        return "pass"
    return "not_available"


def effective_layout_identifier(result: dict) -> int | None:
    requested_identifier = result.get("requestedDocumentIdentifier")
    if requested_identifier is not None:
        return int(requested_identifier)
    candidate = next(
        (
            item
            for item in result.get("documentClassification", {}).get(
                "candidates", []
            )
            if item.get("documentIdentifier") is not None
            and item.get("document")
            and float(item.get("confidence", 0.0)) >= 0.65
        ),
        None,
    )
    return int(candidate["documentIdentifier"]) if candidate else None


def declared_page_ordinal(layout: dict) -> int | None:
    match = re.search(
        r"\b(?:page|side)\s+([0-9]+|[A-Z])\b",
        layout.get("caption", ""),
        re.IGNORECASE,
    )
    if not match:
        return 1
    token = match.group(1).upper()
    if token.isdigit():
        ordinal = int(token)
        return ordinal if ordinal > 0 else None
    return ord(token) - ord("A") + 1


def semantic_page_order(results: list[dict], models_path: Path) -> dict:
    identifiers = [effective_layout_identifier(result) for result in results]
    layouts = [
        visual_layout(models_path, identifier) if identifier is not None else None
        for identifier in identifiers
    ]
    ordinals = [
        declared_page_ordinal(layout) if layout is not None else None
        for layout in layouts
    ]
    precedence: set[tuple[int, int]] = set()
    for left_index, left in enumerate(layouts):
        if left is None or ordinals[left_index] is None:
            continue
        left_relations = {int(value) for value in left.get("pairedPages") or []}
        for right_index in range(left_index + 1, len(layouts)):
            right = layouts[right_index]
            if right is None or ordinals[right_index] is None:
                continue
            right_relations = {int(value) for value in right.get("pairedPages") or []}
            related = (
                identifiers[right_index] in left_relations
                or identifiers[left_index] in right_relations
            )
            if not related or ordinals[left_index] == ordinals[right_index]:
                continue
            if ordinals[left_index] < ordinals[right_index]:
                precedence.add((left_index, right_index))
            else:
                precedence.add((right_index, left_index))

    ordered: list[int] = []
    remaining = set(range(len(results)))
    while remaining:
        available = [
            index
            for index in sorted(remaining)
            if not any(
                predecessor in remaining
                for predecessor, successor in precedence
                if successor == index
            )
        ]
        if not available:
            return {
                "indices": list(range(len(results))),
                "decision": "preserved",
                "method": "caller_order",
            }
        selected = available[0]
        ordered.append(selected)
        remaining.remove(selected)

    changed = ordered != list(range(len(results)))
    return {
        "indices": ordered,
        "decision": "reordered" if changed else "preserved",
        "method": (
            "catalog_relation_and_declared_page_ordinal"
            if precedence
            else "caller_order"
        ),
    }


def paired_quality(front: dict, back: dict) -> dict:
    decisions = [
        result.get("qualitySignals", {}).get("spoofingDecision")
        for result in (front, back)
    ]
    decision = aggregate_document_capture_decisions(decisions)
    return {
        "spoofingDecision": decision,
        "livenessDecision": "not_available",
        "sideDecisions": {"front": decisions[0], "back": decisions[1]},
    }


def related_side_barcodes(
    front: dict, back_path: Path, models_path: Path
) -> tuple[list[dict], dict | None]:
    relation = catalog_side_relation(front, models_path)
    if not relation:
        return [], None
    regions = []
    layout_identifiers = []
    for related in relation["relatedDocuments"]:
        layout = visual_layout(models_path, related["identifier"])
        if not layout:
            continue
        regions_for_barcodes = catalog_barcode_regions(layout)
        if regions_for_barcodes:
            layout_identifiers.append(related["identifier"])
            regions.extend(regions_for_barcodes)
    if not regions:
        return [], relation
    with tempfile.TemporaryDirectory() as directory:
        image = load_image(Path(back_path), Path(directory))
        decoded = decode_machine_barcodes_in_regions(image, regions)
    for barcode in decoded:
        barcode["guidedByRelatedLayouts"] = layout_identifiers
    return decoded, relation


def fuse_document_sides(front: dict, back: dict, models_path: Path) -> dict:
    front_name = front.get("name") or front.get("surnameAndGivenNames") or " ".join(
        filter(None, (front.get("surname"), front.get("givenNames")))
    )
    back_name = back.get("name") or back.get("surnameAndGivenNames") or " ".join(
        filter(None, (back.get("surname"), back.get("givenNames")))
    )
    checks = {
        "country": compatible_identity_text(
            front.get("dCountryName"), back.get("dCountryName"), models_path
        ),
        "name": compatible_identity_text(front_name, back_name, models_path),
        "surname": compatible_identity_text(
            front.get("surname"), back.get("surname"), models_path
        ),
        "dateOfBirth": (
            normalized_identity_date(front.get("dateOfBirth"))
            == normalized_identity_date(back.get("dateOfBirth"))
            if front.get("dateOfBirth") and back.get("dateOfBirth")
            else None
        ),
        "sex": compatible_identity_text(front.get("sex"), back.get("sex"), models_path),
    }
    comparable = [value for value in checks.values() if value is not None]
    if any(value is False for value in comparable):
        decision = "mismatch"
    elif sum(value is True for value in comparable) >= 2:
        decision = "matched"
    else:
        decision = "review"
    result = deepcopy(front)
    if decision == "matched":
        for key in (
            "documentNumber",
            "documentClassCode",
            "issuingStateCode",
            "nationality",
            "nationalityCode",
            "dateOfExpiry",
            "personalNumber",
            "optionalData",
            "mrzCode",
            "mrzStrings",
            "checks",
            "machineBarcodes",
            "ineQr",
        ):
            if back.get(key) not in (None, "", [], {}):
                result[key] = deepcopy(back[key])
        result["validityStatus"] = max(
            front.get("validityStatus", -1), back.get("validityStatus", -1)
        )
        result["source"] = "MULTI_SOURCE"
        result["availableSourceList"] = list(
            dict.fromkeys(
                (front.get("availableSourceList") or [])
                + (back.get("availableSourceList") or [])
            )
        )
        result["recognitionProfile"] = "PAIRED-DOCUMENT"
        result["recognitionProfileStatus"] = "front_back_identity_checks_passed"
        if back.get("mrzCode"):
            result["machineReadableProfile"] = back.get("recognitionProfile")
    for region in result.get("visualRegions", {}).values():
        region["side"] = "front"
    result["qualitySignals"] = paired_quality(front, back)
    result["pairing"] = {
        "decision": decision,
        "checks": checks,
        "catalogRelation": catalog_side_relation(front, models_path),
        "relatedSideBarcodeCount": back.get("relatedSideDecoding", {}).get(
            "decodedBarcodeCount"
        ),
        "sides": [side_summary(front, "front"), side_summary(back, "back")],
    }
    result["sideResults"] = {"front": front, "back": back}
    result["ContainerList"] = [{"OneCandidate": result.copy()}]
    result["Count"] = 1
    return result


def process_document_pair(
    front_path: Path,
    back_path: Path,
    models_path: Path,
    profile: str = "auto_research",
    front_document_identifier: int | None = None,
    back_document_identifier: int | None = None,
) -> dict:
    def process_back() -> dict:
        if back_document_identifier is None:
            td1_result = process_td1_back_first(
                Path(back_path).resolve(), Path(models_path).resolve()
            )
            if td1_result is not None:
                return td1_result
        return process_document(
            back_path, models_path, "auto_research", back_document_identifier
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        front_future = executor.submit(
            process_document,
            front_path,
            models_path,
            profile,
            front_document_identifier,
        )
        back_future = executor.submit(process_back)
        front = front_future.result()
        back = back_future.result()
    resolved_models_path = Path(models_path).resolve()
    if back.get("ineQr"):
        guided_barcodes = []
        relation = catalog_side_relation(front, resolved_models_path)
    else:
        guided_barcodes, relation = related_side_barcodes(
            front, Path(back_path), resolved_models_path
        )
    if guided_barcodes:
        merged = {}
        for barcode in (back.get("machineBarcodes") or []) + guided_barcodes:
            merged[(barcode["format"], barcode["text"])] = barcode
        back["machineBarcodes"] = list(merged.values())
        evidence = ine_qr_evidence(back["machineBarcodes"])
        if evidence:
            back["ineQr"] = evidence
    if relation:
        back["relatedSideDecoding"] = {
            "catalogRelation": relation,
            "decodedBarcodeCount": len(guided_barcodes),
        }
    return fuse_document_sides(front, back, resolved_models_path)


def process_document_pages(
    image_paths: list[Path],
    models_path: Path,
    profile: str = "auto_research",
    document_identifiers: list[int | None] | None = None,
) -> dict:
    if not 2 <= len(image_paths) <= 10:
        raise ValueError("Document page processing requires between 2 and 10 images")
    models_path = Path(models_path).resolve()
    paths = [Path(path).resolve() for path in image_paths]
    identifiers = (
        [None] * len(paths)
        if document_identifiers is None
        else list(document_identifiers)
    )
    if len(identifiers) != len(paths):
        raise ValueError("Document identifiers must align with image paths")
    with ThreadPoolExecutor(max_workers=min(4, len(paths))) as executor:
        results = list(
            executor.map(
                lambda item: process_document(
                    item[0],
                    models_path,
                    profile,
                    item[1],
                ),
                zip(paths, identifiers),
            )
        )
    page_order = semantic_page_order(results, models_path)
    input_positions = list(range(1, len(results) + 1))
    results = [results[index] for index in page_order["indices"]]
    paths = [paths[index] for index in page_order["indices"]]
    identifiers = [identifiers[index] for index in page_order["indices"]]
    input_positions = [input_positions[index] for index in page_order["indices"]]

    aggregate = deepcopy(results[0])
    comparisons = []
    matched_pages = 1
    guided_barcode_count = 0
    for page_index, (page, path) in enumerate(zip(results[1:], paths[1:]), start=2):
        if page.get("ineQr"):
            guided_barcodes = []
            relation = catalog_side_relation(aggregate, models_path)
        else:
            guided_barcodes, relation = related_side_barcodes(
                aggregate, path, models_path
            )
        if guided_barcodes:
            merged = {
                (barcode["format"], barcode["text"]): barcode
                for barcode in (page.get("machineBarcodes") or []) + guided_barcodes
            }
            page["machineBarcodes"] = list(merged.values())
            evidence = ine_qr_evidence(page["machineBarcodes"])
            if evidence:
                page["ineQr"] = evidence
        if relation:
            page["relatedSideDecoding"] = {
                "catalogRelation": relation,
                "decodedBarcodeCount": len(guided_barcodes),
            }
        guided_barcode_count += len(guided_barcodes)

        base = deepcopy(aggregate)
        for internal_key in ("pairing", "sideResults", "pageProcessing", "pageResults"):
            base.pop(internal_key, None)
        candidate = fuse_document_sides(base, page, models_path)
        pairing = candidate["pairing"]
        comparisons.append(
            {
                "page": page_index,
                "decision": pairing["decision"],
                "checks": pairing["checks"],
            }
        )
        if pairing["decision"] == "matched":
            aggregate = candidate
            matched_pages += 1

    decisions = [comparison["decision"] for comparison in comparisons]
    if "mismatch" in decisions:
        overall_decision = "mismatch"
    elif matched_pages > 1 and all(decision == "matched" for decision in decisions):
        overall_decision = "matched"
    else:
        overall_decision = "review"
    aggregate.pop("pairing", None)
    aggregate.pop("sideResults", None)
    if matched_pages > 1:
        aggregate["recognitionProfile"] = "MULTI-PAGE-DOCUMENT"
        aggregate["recognitionProfileStatus"] = "validated_pages_fused"
        aggregate["source"] = "MULTI_SOURCE"
    for region in aggregate.get("visualRegions", {}).values():
        region["side"] = "page_1"
    page_decisions = [
        result.get("qualitySignals", {}).get("spoofingDecision") for result in results
    ]
    available_quality = [
        decision
        for decision in page_decisions
        if decision not in {None, "not_available"}
    ]
    aggregate["qualitySignals"] = {
        "spoofingDecision": aggregate_document_capture_decisions(
            available_quality
        ),
        "livenessDecision": "not_available",
        "pageDecisions": page_decisions,
    }
    aggregate["pageProcessing"] = {
        "decision": overall_decision,
        "pageCount": len(results),
        "matchedPageCount": matched_pages,
        "relatedSideBarcodeCount": guided_barcode_count,
        "ordering": {
            "decision": page_order["decision"],
            "method": page_order["method"],
            "inputOrder": input_positions,
        },
        "comparisons": comparisons,
        "pages": [
            {
                **side_summary(result, f"page_{index}"),
                "inputPage": input_positions[index - 1],
            }
            for index, result in enumerate(results, 1)
        ],
    }
    aggregate["pageResults"] = results
    aggregate["ContainerList"] = [{"OneCandidate": aggregate.copy()}]
    aggregate["Count"] = 1
    return aggregate


def warm_up(models_path: Path) -> dict[str, bool]:
    """Load the ONNX sessions that are installed, without processing anything.

    Every model family is optional: the runtime is a "bring your own models"
    host (see docs/models.md), so a family that is missing or unreadable is
    reported as unavailable instead of aborting start-up.
    """

    models_path = Path(models_path).resolve()
    stages = {
        "assets": lambda: validate_assets(models_path),
        "lineRecognition": lambda: load_runtime(models_path),
        "documentRectification": lambda: rectification_session(str(models_path)),
        "captureQuality": lambda: warm_up_quality(models_path),
        "documentClassification": lambda: warm_up_document_classifier(models_path),
        "layoutCatalog": lambda: warm_up_visual_layouts(models_path),
    }
    loaded = {}
    for name, load in stages.items():
        try:
            load()
        except Exception:  # noqa: BLE001 - any missing family degrades to unavailable
            loaded[name] = False
        else:
            loaded[name] = True
    return loaded
