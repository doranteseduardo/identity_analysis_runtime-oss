from PIL import Image

import identity_analysis.barcodes as barcodes
from identity_analysis.barcodes import (
    catalog_barcode_format,
    decode_machine_barcodes_in_regions,
    ine_qr_evidence,
    parse_aamva,
    parse_ine_qr,
)
from identity_analysis.visual_layouts import barcode_regions, visual_layout

from conftest import BACK_LAYOUT, FIXTURE_CATALOG, ID_BACK


# Fabricated verification payloads: an all-zero credential number and a
# far-future issue date, rendered onto examples/samples/synthetic_id_back.jpg.
FAKE_QR_URL = "http://qr.ine.mx/000000000000000000000000/20990101/X/000000"
FAKE_BARCODE_NUMBER = "000000000"


def test_parses_aamva_fields() -> None:
    parsed = parse_aamva(
        b"@\n\x1e\rANSI 636000000002DL00410288ZA03290015DLDAQ123456789\n"
        b"DCSDOE\nDACJANE\nDADANN\nDBB10161986\nDBA10162028\nDBC2\n"
        b"DAG123 MAIN STREET\nDAIANTOWN\nDAJAZ\nDAK12345\nDCGUSA\n"
    )

    assert parsed["headerValid"] is True
    assert parsed["fields"]["documentNumber"] == "123456789"
    assert parsed["fields"]["familyName"] == "DOE"
    assert parsed["fields"]["firstName"] == "JANE"
    assert parsed["fields"]["dateOfBirth"] == "1986-10-16"
    assert parsed["fields"]["sex"] == "F"
    assert parsed["fields"]["address"] == "123 MAIN STREET, ANTOWN, AZ, 12345"


def test_parses_public_ine_verification_qr() -> None:
    parsed = parse_ine_qr(FAKE_QR_URL)

    assert parsed == {
        "credentialIdentifier": "000000000000000000000000",
        "issueDate": "2099-01-01",
        "credentialType": "X",
        "queryIdentifier": "000000",
        "verificationUrl": FAKE_QR_URL,
    }


def test_correlates_ine_qr_and_code128() -> None:
    evidence = ine_qr_evidence(
        [
            {
                "format": "Code128",
                "text": FAKE_BARCODE_NUMBER,
                "decoder": "zxing-cpp",
            },
            {
                "format": "QRCode",
                "text": FAKE_QR_URL,
                "decoder": "zxing-cpp",
            },
        ]
    )

    assert evidence["barcodeNumber"] == FAKE_BARCODE_NUMBER
    assert evidence["credentialIdentifier"].endswith(evidence["barcodeNumber"])


def test_declared_back_layout_guides_code128_decoding() -> None:
    image = Image.open(ID_BACK)
    layout = visual_layout(FIXTURE_CATALOG, BACK_LAYOUT)
    regions = barcode_regions(layout)

    decoded = decode_machine_barcodes_in_regions(image, regions)

    code128 = next(barcode for barcode in decoded if barcode["format"] == "Code128")
    assert code128["text"] == FAKE_BARCODE_NUMBER
    assert code128["guidedRegion"]["box"] in [region["bounds"] for region in regions]
    assert code128["guidedRegion"]["expectedFormat"] == "Code128"

    qr = next(barcode for barcode in decoded if barcode["format"] == "QRCode")
    assert qr["text"] == FAKE_QR_URL


def test_catalog_barcode_regions_preserve_decoder_hints() -> None:
    layout = visual_layout(FIXTURE_CATALOG, BACK_LAYOUT)

    regions = barcode_regions(layout)

    assert len(regions) == 3
    assert {region["codeType"] for region in regions} == {1, 14, 99}
    assert any(region["name"] == "documentNumber" for region in regions)


def test_catalog_barcode_format_maps_only_public_exact_values() -> None:
    assert catalog_barcode_format(1) == "Code128"
    assert catalog_barcode_format(2) == "Code39"
    assert catalog_barcode_format(3) == "EAN8"
    assert catalog_barcode_format(11) == "Code93"
    assert catalog_barcode_format(16) == "DataMatrix"
    assert catalog_barcode_format(18) is None
    assert catalog_barcode_format(99) is None


def test_region_decoder_routes_exact_catalog_format(monkeypatch) -> None:
    calls = []

    def decode_expected(image, expected_format=None):
        calls.append(expected_format)
        if expected_format == "Code128":
            return [{"format": "Code128", "text": "1234", "decoder": "test"}]
        return []

    monkeypatch.setattr(barcodes, "decode_machine_barcodes", decode_expected)
    decoded = decode_machine_barcodes_in_regions(
        Image.new("RGB", (100, 50)),
        [
            {
                "bounds": [0.0, 0.0, 0.5, 1.0],
                "codeType": 1,
            },
            {
                "bounds": [0.5, 0.0, 1.0, 1.0],
                "codeType": 99,
            },
        ],
    )

    assert "Code128" in calls
    assert None in calls
    assert decoded[0]["guidedRegion"]["expectedFormat"] == "Code128"
