"""Portable PDF417 decoding and AAMVA field normalization."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

from PIL import Image, ImageOps


AAMVA_FIELDS = {
    "DCS": "familyName",
    "DAC": "firstName",
    "DAD": "middleName",
    "DBB": "dateOfBirth",
    "DBA": "dateOfExpiry",
    "DBD": "dateOfIssue",
    "DBC": "sexCode",
    "DAY": "eyeColor",
    "DAU": "height",
    "DAG": "addressLine1",
    "DAH": "addressLine2",
    "DAI": "city",
    "DAJ": "jurisdictionCode",
    "DAK": "postalCode",
    "DAQ": "documentNumber",
    "DCF": "documentDiscriminator",
    "DCG": "countryCode",
    "DAZ": "hairColor",
}

CATALOG_BARCODE_FORMATS = {
    1: "Code128",
    2: "Code39",
    3: "EAN8",
    4: "ITF",
    5: "PDF417",
    9: "Codabar",
    10: "UPCA",
    11: "Code93",
    12: "UPCE",
    13: "EAN13",
    14: "QRCode",
    15: "Aztec",
    16: "DataMatrix",
}


def catalog_barcode_format(code_type: int | None) -> str | None:
    return CATALOG_BARCODE_FORMATS.get(code_type)


def _parse_date(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    for pattern in ("%m%d%Y", "%Y%m%d"):
        try:
            return datetime.strptime(digits, pattern).date().isoformat()
        except ValueError:
            pass
    return value


def parse_aamva(payload: bytes) -> dict:
    text = payload.decode("latin1", errors="replace")
    raw_fields = {}
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.strip("\x00\x1e ")
        if line.startswith("DL") and len(line) >= 5:
            line = line[2:]
        if line[:3] not in AAMVA_FIELDS:
            embedded = [
                (line.rfind(designator), designator)
                for designator in AAMVA_FIELDS
                if line.rfind(designator) >= 0
            ]
            if embedded:
                position, _ = max(embedded)
                line = line[position:]
        if len(line) >= 4 and line[:3] in AAMVA_FIELDS:
            raw_fields[line[:3]] = line[3:].strip()

    normalized = {
        output_name: raw_fields.get(designator, "")
        for designator, output_name in AAMVA_FIELDS.items()
    }
    for date_field in ("dateOfBirth", "dateOfExpiry", "dateOfIssue"):
        if normalized[date_field]:
            normalized[date_field] = _parse_date(normalized[date_field])
    normalized["sex"] = {"1": "M", "2": "F", "9": "X"}.get(
        normalized.pop("sexCode"), ""
    )
    normalized["addressLines"] = [
        value
        for value in (normalized["addressLine1"], normalized["addressLine2"])
        if value
    ]
    normalized["address"] = ", ".join(
        value
        for value in (
            *normalized["addressLines"],
            normalized["city"],
            normalized["jurisdictionCode"],
            normalized["postalCode"],
        )
        if value
    )
    return {
        "headerValid": text.startswith("@") and "ANSI " in text[:20],
        "fields": normalized,
        "availableDesignators": sorted(raw_fields),
    }


def decode_pdf417(image: Image.Image) -> dict | None:
    try:
        import zxingcpp
    except ImportError:
        return None

    for scale in (1, 2, 3):
        candidate = image
        if scale > 1:
            candidate = image.resize(
                (image.width * scale, image.height * scale), Image.Resampling.LANCZOS
            )
        variants = (candidate, ImageOps.autocontrast(ImageOps.grayscale(candidate)))
        for variant in variants:
            barcodes = zxingcpp.read_barcodes(
                variant,
                formats=zxingcpp.BarcodeFormat.PDF417,
                try_rotate=True,
                try_downscale=False,
            )
            for barcode in barcodes:
                if barcode.valid:
                    parsed = parse_aamva(bytes(barcode.bytes))
                    if not parsed["headerValid"] or not parsed["availableDesignators"]:
                        continue
                    parsed["format"] = "PDF417"
                    parsed["decoder"] = "zxing-cpp"
                    return parsed
    return None


def decode_pdf417_in_regions(image: Image.Image, regions: list[dict]) -> dict | None:
    width, height = image.size
    for region in regions:
        left, top, right, bottom = region["bounds"]
        bounds = (
            max(0, min(width, round(left * width))),
            max(0, min(height, round(top * height))),
            max(0, min(width, round(right * width))),
            max(0, min(height, round(bottom * height))),
        )
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        parsed = decode_pdf417(image.crop(bounds))
        if parsed:
            parsed["guidedRegion"] = {"box": region["bounds"]}
            return parsed
    return None


def decode_machine_barcodes(
    image: Image.Image, expected_format: str | None = None
) -> list[dict]:
    try:
        import zxingcpp
    except ImportError:
        return []

    decoded = []
    seen = set()
    options = {"try_rotate": True}
    if expected_format:
        barcode_format = getattr(zxingcpp.BarcodeFormat, expected_format, None)
        if barcode_format is not None:
            options["formats"] = barcode_format
    for barcode in zxingcpp.read_barcodes(image.convert("RGB"), **options):
        text = barcode.text.strip()
        if not barcode.valid or not text:
            continue
        if any(token in text for token in ("<NUL>", "<U+", "<SOH>", "<STX>")):
            continue
        if len(text) > 512:
            continue
        printable_ratio = sum(character.isprintable() for character in text) / len(text)
        if printable_ratio < 0.9:
            continue
        format_name = str(barcode.format).removeprefix("BarcodeFormat.")
        if format_name == "PDF417":
            parsed = parse_aamva(bytes(barcode.bytes))
            if not parsed["headerValid"] or not parsed["availableDesignators"]:
                continue
        key = (format_name, text)
        if key in seen:
            continue
        seen.add(key)
        decoded.append({"format": format_name, "text": text, "decoder": "zxing-cpp"})
    return decoded


def decode_machine_barcodes_in_regions(
    image: Image.Image, regions: list[dict]
) -> list[dict]:
    decoded = []
    seen = set()
    width, height = image.size
    for region_index, region in enumerate(regions):
        left, top, right, bottom = region["bounds"]
        bounds = (
            max(0, min(width, round(left * width))),
            max(0, min(height, round(top * height))),
            max(0, min(width, round(right * width))),
            max(0, min(height, round(bottom * height))),
        )
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        crop = image.crop(bounds)
        expected_format = catalog_barcode_format(region.get("codeType"))
        for scale in (1, 2, 3):
            candidate = crop
            if scale > 1:
                candidate = crop.resize(
                    (crop.width * scale, crop.height * scale),
                    Image.Resampling.LANCZOS,
                )
            variants = (
                candidate.convert("RGB"),
                ImageOps.autocontrast(ImageOps.grayscale(candidate)),
            )
            for variant in variants:
                for barcode in decode_machine_barcodes(variant, expected_format):
                    key = (barcode["format"], barcode["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    decoded.append(
                        {
                            **barcode,
                            "guidedRegion": {
                                "index": region_index,
                                "box": region["bounds"],
                                "scale": scale,
                                "expectedFormat": expected_format,
                            },
                        }
                    )
    return decoded


def parse_ine_qr(value: str) -> dict | None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "qr.ine.mx":
        return None
    segments = [segment for segment in parsed.path.split("/") if segment]
    if (
        len(segments) != 4
        or not segments[0].isdigit()
        or len(segments[1]) != 8
        or not segments[1].isdigit()
        or len(segments[2]) != 1
        or not segments[3].isdigit()
    ):
        return None
    try:
        issue_date = datetime.strptime(segments[1], "%Y%m%d").date().isoformat()
    except ValueError:
        return None
    return {
        "credentialIdentifier": segments[0],
        "issueDate": issue_date,
        "credentialType": segments[2],
        "queryIdentifier": segments[3],
        "verificationUrl": value,
    }


def ine_qr_evidence(barcodes: list[dict]) -> dict | None:
    qr = next(
        (
            (barcode, parsed)
            for barcode in barcodes
            if barcode["format"] == "QRCode"
            if (parsed := parse_ine_qr(barcode["text"]))
        ),
        None,
    )
    if qr is None:
        return None
    barcode, parsed = qr
    code128 = next(
        (
            candidate["text"]
            for candidate in barcodes
            if candidate["format"] == "Code128" and candidate["text"].isdigit()
        ),
        "",
    )
    return {
        **parsed,
        "barcodeNumber": code128,
        "format": barcode["format"],
        "decoder": barcode["decoder"],
    }


def ine_qr_result(evidence: dict) -> dict:
    return {
        "errorCode": 0,
        "DocumentName": "Voter Credential",
        "dCountryName": "Mexico",
        "documentNumber": evidence["credentialIdentifier"],
        "dateOfIssue": evidence["issueDate"],
        "credentialType": evidence["credentialType"],
        "queryIdentifier": evidence["queryIdentifier"],
        "barcodeNumber": evidence["barcodeNumber"],
        "availableSourceList": ["BARCODE"],
        "source": "BARCODE",
        "validityStatus": 1,
        "fieldList": [],
        "Images": {},
        "recognition": {
            "engine": evidence["decoder"],
            "barcodeFormat": evidence["format"],
        },
        "recognitionProfile": "MEX-INE-QR",
        "recognitionProfileStatus": "selected_by_public_verification_qr_structure",
    }


def aamva_result(decoded: dict) -> dict:
    values = decoded["fields"]
    field_list = [
        {
            "fieldName": name,
            "lcidName": "",
            "value": value,
            "valueList": [{"value": value, "source": "BARCODE"}],
            "source": "BARCODE",
        }
        for name, value in values.items()
        if isinstance(value, str) and value
    ]
    return {
        "errorCode": 0,
        "DocumentName": "Driver License",
        "dCountryName": values.get("countryCode", "USA"),
        **values,
        "surname": values.get("familyName", ""),
        "givenNames": " ".join(
            filter(None, (values.get("firstName", ""), values.get("middleName", "")))
        ),
        "surnameAndGivenNames": " ".join(
            filter(
                None,
                (
                    values.get("familyName", ""),
                    values.get("firstName", ""),
                    values.get("middleName", ""),
                ),
            )
        ),
        "availableSourceList": ["BARCODE"],
        "source": "BARCODE",
        "validityStatus": 1 if decoded["headerValid"] else -1,
        "fieldList": field_list,
        "Images": {},
        "recognition": {
            "engine": decoded["decoder"],
            "barcodeFormat": decoded["format"],
            "availableDesignators": decoded["availableDesignators"],
        },
        "recognitionProfile": "AAMVA-PDF417",
        "recognitionProfileStatus": "selected_by_valid_barcode",
    }
