"""End-to-end pipeline behaviour on the synthetic sample documents.

Every expectation here was produced by running the pipeline against the
fixtures rendered by ``tools/generate_synthetic_samples.py``; the identity data
is fabricated (issuing state ``ZZT``, holder ``SPECIMEN, ALEX TAYLOR``).
"""

from pathlib import Path

import identity_analysis.pipeline as pipeline
import pytest

from identity_analysis import process_document, process_document_pages, process_document_pair
from identity_analysis.pipeline import (
    aggregate_document_capture_decisions,
    declared_page_ordinal,
    fuse_document_sides,
    semantic_page_order,
)

from conftest import (
    BACK_LAYOUT,
    FIXTURE_CATALOG,
    FRONT_LAYOUT,
    ID_BACK,
    ID_FRONT,
    PASSPORT_1,
    PASSPORT_2,
    requires_assets,
)


BOOKLET_PAGE_1 = 900000010
BOOKLET_PAGE_2 = 900000011


@requires_assets
@pytest.mark.parametrize("sample", [PASSPORT_1, PASSPORT_2])
def test_auto_recognizes_uncropped_synthetic_passports(sample: Path, assets) -> None:
    result = process_document(sample, assets, "auto_research")

    assert result["recognitionProfile"] == "ICAO-TD3"
    assert result["DocumentName"] == "Passport"
    assert result["issuingStateCode"] == "ZZT"
    assert result["validityStatus"] == 1
    assert all(result["checks"].values())


@requires_assets
def test_explicit_td3_profile_reads_every_declared_field(assets) -> None:
    result = process_document(PASSPORT_1, assets, "icao_td3")

    assert result["recognitionProfile"] == "ICAO-TD3"
    assert result["issuingStateCode"] == "ZZT"
    assert result["nationalityCode"] == "ZZT"
    assert result["validityStatus"] == 1
    assert result["surname"] == "SPECIMEN"
    assert result["givenNames"] == "ALEX TAYLOR"
    assert result["documentNumber"] == "AB1234567"
    assert result["dateOfBirth"] == "1990-01-01"
    assert result["dateOfExpiry"] == "2035-01-01"
    assert result["personalNumber"] == "0000000000000"
    assert result["mrzStrings"] == [
        "P<ZZTSPECIMEN<<ALEX<TAYLOR<<<<<<<<<<<<<<<<<<",
        "AB12345671ZZT9001011M35010140000000000000<04",
    ]


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        (["pass", "pass"], "pass"),
        (["real", "pass"], "pass"),
        (["pass", "review"], "review"),
        (["pass", "spoof"], "review"),
        (["not_available", None], "not_available"),
    ],
)
def test_document_capture_decision_aggregation(decisions, expected) -> None:
    assert aggregate_document_capture_decisions(decisions) == expected


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("ePassport (2018)", 1),
        ("ePassport (2018) Page 3", 3),
        ("Residence Permit Page B", 2),
        ("Passport Side A", 1),
    ],
)
def test_declared_page_ordinal(caption, expected) -> None:
    assert declared_page_ordinal({"caption": caption}) == expected


def test_semantic_page_order_uses_related_declared_ordinals() -> None:
    results = [
        {"requestedDocumentIdentifier": BOOKLET_PAGE_2},
        {"requestedDocumentIdentifier": BOOKLET_PAGE_1},
    ]

    order = semantic_page_order(results, FIXTURE_CATALOG)

    assert order == {
        "indices": [1, 0],
        "decision": "reordered",
        "method": "catalog_relation_and_declared_page_ordinal",
    }


def test_semantic_page_order_preserves_unrelated_documents() -> None:
    results = [
        {"requestedDocumentIdentifier": BOOKLET_PAGE_1},
        {"requestedDocumentIdentifier": BACK_LAYOUT},
    ]

    order = semantic_page_order(results, FIXTURE_CATALOG)

    assert order == {
        "indices": [0, 1],
        "decision": "preserved",
        "method": "caller_order",
    }


@requires_assets
def test_synthetic_card_back_reads_mrz_and_verification_barcodes(assets) -> None:
    result = process_document(ID_BACK, assets)

    assert result["errorCode"] == 0
    assert result["source"] == "MRZ"
    assert result["recognitionProfile"] == "ICAO-TD1"
    assert result["fieldList"]
    assert result["sdkCompatibility"]["coverage"] == "partial"
    assert len(result["recognition"]["mrzLines"]) == 3
    assert result["recognition"]["mrzLines"][0]["value"] == result["mrzStrings"][0]
    assert result["mrzStrings"] == [
        "IDZZTZZ76543216<<<<<<<<<<<<<<<",
        "9001011M3501014ZZT<<<<<<<<<<<4",
        "SPECIMEN<<ALEX<TAYLOR<<<<<<<<<",
    ]
    assert result["documentNumber"] == "ZZ7654321"
    assert result["validityStatus"] == 1
    assert result["ineQr"]["credentialIdentifier"] == "0" * 24
    assert result["ineQr"]["issueDate"] == "2099-01-01"
    assert result["ineQr"]["barcodeNumber"] == "0" * 9
    assert [item["format"] for item in result["machineBarcodes"]] == [
        "Code128",
        "QRCode",
    ]


def test_synthetic_card_front_reads_declared_layout_fields(catalog_assets) -> None:
    result = process_document(ID_FRONT, catalog_assets, "auto_research", FRONT_LAYOUT)

    assert result["recognitionProfile"] == f"CATALOG-P{FRONT_LAYOUT}"
    assert result["recognitionProfileStatus"] == "selected_by_explicit_layout"
    assert result["source"] == "VISUAL"
    assert result["DocumentName"] == "Specimen Identity Card (2024) Front"
    assert result["surname"] == "SPECIMEN"
    assert result["givenNames"] == "ALEX TAYLOR"
    assert result["surnameAndGivenNames"] == "SPECIMEN ALEX TAYLOR"
    assert result["dateOfBirth"] == "01/01/1990"
    assert result["dateOfExpiry"] == "01/01/2035"
    assert result["documentNumber"] == "ZZ7654321"
    assert result["sex"] == "M"
    assert result["nationality"] == "ZZT"
    assert set(result["visualRegions"]) == {"portrait", "ghostPortrait", "signature"}
    assert all(
        region["coordinateSpace"] == "original_image"
        for region in result["visualRegions"].values()
    )
    assert result["visualRegions"]["portrait"]["box"] == [0.055, 0.3, 0.3, 0.7]


def test_explicit_layout_skips_two_line_mrz(monkeypatch, catalog_assets) -> None:
    def unexpected_two_line_mrz(*args, **kwargs):
        raise AssertionError("an explicit card front must not run two-line MRZ OCR")

    monkeypatch.setattr(pipeline, "recognize_best_two_line_mrz", unexpected_two_line_mrz)

    result = process_document(ID_FRONT, catalog_assets, "auto_research", FRONT_LAYOUT)

    assert result["recognitionProfile"] == f"CATALOG-P{FRONT_LAYOUT}"


def test_front_back_pair_fuses_visual_and_mrz_sources(catalog_assets) -> None:
    result = process_document_pair(
        ID_FRONT, ID_BACK, catalog_assets, "auto_research", FRONT_LAYOUT, BACK_LAYOUT
    )

    assert result["pairing"]["decision"] == "matched"
    assert all(result["pairing"]["checks"].values())
    assert result["recognitionProfile"] == "PAIRED-DOCUMENT"
    assert result["source"] == "MULTI_SOURCE"
    assert result["givenNames"] == "ALEX TAYLOR"
    assert result["documentNumber"] == "ZZ7654321"
    assert result["dateOfBirth"] == "01/01/1990"
    assert result["validityStatus"] == 1
    assert result["qualitySignals"]["sideDecisions"] == {
        "front": "review",
        "back": "review",
    }
    assert result["pairing"]["catalogRelation"]["relationType"] == "related_document"
    assert result["pairing"]["catalogRelation"]["relatedDocuments"] == [
        {
            "identifier": BACK_LAYOUT,
            "name": "Specimen Identity Card (2024) Back",
        }
    ]


def test_pair_skips_guided_barcode_retry_when_back_already_has_ine_qr(
    monkeypatch,
) -> None:
    calls = []

    def fake_document(path, *_args, **_kwargs):
        calls.append((Path(path).name, _args[1]))
        result = {
            "errorCode": 0,
            "DocumentName": "Specimen Identity Card",
            "dCountryName": "ZZT",
            "name": "EXAMPLE PERSON",
            "surname": "EXAMPLE",
            "givenNames": "PERSON",
            "dateOfBirth": "1990-01-01",
            "sex": "M",
            "validityStatus": 1,
            "availableSourceList": ["VISUAL"],
            "source": "VISUAL",
            "recognitionProfile": "TEST",
            "qualitySignals": {"spoofingDecision": "pass"},
            "visualRegions": {},
        }
        if Path(path).name == "back.jpg":
            result["ineQr"] = {
                "credentialIdentifier": "123",
                "verificationUrl": "https://qr.ine.mx/example",
            }
        return result

    def unexpected_guided_retry(*_args, **_kwargs):
        raise AssertionError(
            "guided barcode decoding must not repeat valid verification QR evidence"
        )

    monkeypatch.setattr(pipeline, "process_document", fake_document)
    monkeypatch.setattr(pipeline, "related_side_barcodes", unexpected_guided_retry)

    result = process_document_pair(
        Path("front.jpg"),
        Path("back.jpg"),
        FIXTURE_CATALOG,
        "mex_ine",
    )

    assert result["pairing"]["decision"] == "matched"
    assert result["ineQr"]["credentialIdentifier"] == "123"
    assert calls == [
        ("front.jpg", "mex_ine"),
        ("back.jpg", "auto_research"),
    ]


@requires_assets
def test_valid_verification_qr_skips_incompatible_pdf417_scan(
    monkeypatch, assets
) -> None:
    monkeypatch.setattr(
        pipeline,
        "decode_machine_barcodes",
        lambda _image: [
            {
                "format": "QRCode",
                "text": "https://qr.ine.mx/123/20240101/A/456",
                "decoder": "test",
            }
        ],
    )

    def unexpected_pdf417_scan(*_args, **_kwargs):
        raise AssertionError("valid QR evidence must suppress the PDF417 route")

    monkeypatch.setattr(pipeline, "decode_pdf417", unexpected_pdf417_scan)

    result = process_document(ID_BACK, assets, "auto_research")

    assert result["recognitionProfile"] == "ICAO-TD1"
    assert result["ineQr"]["credentialIdentifier"] == "123"


@requires_assets
def test_explicit_profile_uses_fast_machine_barcode_scan(monkeypatch, assets) -> None:
    calls = 0

    def machine_barcodes(_image):
        nonlocal calls
        calls += 1
        return [
            {
                "format": "QRCode",
                "text": "https://qr.ine.mx/123/20240101/A/456",
                "decoder": "test",
            }
        ]

    monkeypatch.setattr(pipeline, "decode_machine_barcodes", machine_barcodes)

    result = process_document(ID_BACK, assets, "mex_ine")

    assert calls == 1
    assert result["recognitionProfile"] == "ICAO-TD1"
    assert result["ineQr"]["credentialIdentifier"] == "123"


@requires_assets
def test_catalog_td3_hint_skips_td1_when_guided_candidate_is_valid(
    monkeypatch, assets
) -> None:
    classification = {
        "candidates": [
            {
                "documentIdentifier": 999999,
                "confidence": 0.95,
                "document": {
                    "caption": "Example Passport",
                    "country": "Specimen State",
                    "isoCodes": ["ZZT"],
                    "documentType": {"name": "Passport"},
                    "documentFormat": {"name": "ID3"},
                    "mrz": {
                        "present": False,
                        "ignored": False,
                        "expectedProfile": "td3",
                    },
                },
            },
            {
                "documentIdentifier": 999998,
                "confidence": 0.01,
                "document": {"caption": "Other"},
            },
        ]
    }
    td3 = (
        [
            "P<ZZTSPECIMEN<<ALEX<TAYLOR<<<<<<<<<<<<<<<<<<",
            "AB12345671ZZT9001011M35010140000000000000<04",
        ],
        [0.99, 0.98],
    )
    monkeypatch.setattr(pipeline, "decode_machine_barcodes", lambda image: [])
    monkeypatch.setattr(pipeline, "decode_pdf417", lambda image: None)
    monkeypatch.setattr(pipeline, "classifier_available", lambda models: True)
    monkeypatch.setattr(pipeline, "classify_document", lambda models, image: classification)
    monkeypatch.setattr(
        pipeline,
        "recognize_best_two_line_mrz",
        lambda models, original, effective, localize=False, geometry=None: td3,
    )
    monkeypatch.setattr(
        pipeline,
        "recognize_td1",
        lambda *args: (_ for _ in ()).throw(AssertionError("TD1 must not run")),
    )
    monkeypatch.setattr(pipeline, "analyze_quality", lambda models, image: {})
    monkeypatch.setattr(pipeline, "check_metadata_integrity", lambda image, models: {})

    result = pipeline.process_document(ID_FRONT, assets)

    assert result["recognitionProfile"] == "ICAO-TD3"
    assert result["recognition"]["catalogMrzHint"] == "td3"


def test_front_back_pair_does_not_fuse_mismatched_identity() -> None:
    front = {
        "recognitionProfile": "FRONT",
        "source": "VISUAL",
        "dCountryName": "ZZT",
        "surname": "SPECIMEN",
        "givenNames": "ALEX",
        "dateOfBirth": "01/01/1990",
        "documentNumber": "FRONT123",
    }
    back = {
        "recognitionProfile": "BACK",
        "source": "MRZ",
        "dCountryName": "ZZT",
        "surname": "OTHER",
        "givenNames": "PERSON",
        "dateOfBirth": "2000-01-01",
        "documentNumber": "BACK999",
    }

    result = fuse_document_sides(front, back, FIXTURE_CATALOG)

    assert result["pairing"]["decision"] == "mismatch"
    assert result["recognitionProfile"] == "FRONT"
    assert result["documentNumber"] == "FRONT123"


def test_td1_repair_removes_trailing_name_filler() -> None:
    lines = [
        "IDZZT1745648762<<1564072273290",
        "8704142M2812313ZZT<03<<07412<9",
        "EXAMPLE<SAMPLE<<CASEY<JORDAN<<<",
    ]

    repaired = pipeline.repair_td1(lines, [0.99, 0.99, 0.99])

    assert repaired == [
        "IDZZT1745648762<<1564072273290",
        "8704142M2812313ZZT<03<<07412<9",
        "EXAMPLE<SAMPLE<<CASEY<JORDAN<<",
    ]


def test_multi_page_pipeline_fuses_only_validated_pages(catalog_assets) -> None:
    result = process_document_pages(
        [ID_FRONT, ID_BACK, ID_BACK],
        catalog_assets,
        "auto_research",
        [FRONT_LAYOUT, BACK_LAYOUT, BACK_LAYOUT],
    )

    assert result["recognitionProfile"] == "MULTI-PAGE-DOCUMENT"
    assert result["pageProcessing"]["decision"] == "matched"
    assert result["pageProcessing"]["pageCount"] == 3
    assert result["pageProcessing"]["matchedPageCount"] == 3
    assert result["pageProcessing"]["relatedSideBarcodeCount"] == 0
    assert [item["decision"] for item in result["pageProcessing"]["comparisons"]] == [
        "matched",
        "matched",
    ]
    assert result["documentNumber"] == "ZZ7654321"
    assert result["visualRegions"]["portrait"]["side"] == "page_1"
    assert result["qualitySignals"]["pageDecisions"] == [
        "review",
        "review",
        "review",
    ]


def test_exact_layout_pair_skips_classification_and_preserves_match(
    monkeypatch, catalog_assets
) -> None:
    def unexpected_classification(*args, **kwargs):
        raise AssertionError("classifier must not run for exact side layouts")

    monkeypatch.setattr(pipeline, "classify_document", unexpected_classification)
    result = process_document_pair(
        ID_FRONT,
        ID_BACK,
        catalog_assets,
        "auto_research",
        FRONT_LAYOUT,
        BACK_LAYOUT,
    )

    assert result["recognitionProfile"] == "PAIRED-DOCUMENT"
    assert result["pairing"]["decision"] == "matched"
    assert result["documentNumber"] == "ZZ7654321"
    assert result["pairing"]["catalogRelation"]["frontIdentifier"] == FRONT_LAYOUT
    assert result["pairing"]["catalogRelation"]["frontConfidence"] is None
    assert [side["layoutIdentifier"] for side in result["pairing"]["sides"]] == [
        FRONT_LAYOUT,
        BACK_LAYOUT,
    ]
    assert all(side["classification"] is None for side in result["pairing"]["sides"])


def test_exact_layout_pages_skip_classification_and_preserve_matches(
    monkeypatch, catalog_assets
) -> None:
    def unexpected_classification(*args, **kwargs):
        raise AssertionError("classifier must not run for exact page layouts")

    monkeypatch.setattr(pipeline, "classify_document", unexpected_classification)
    result = process_document_pages(
        [ID_FRONT, ID_BACK, ID_BACK],
        catalog_assets,
        "auto_research",
        [FRONT_LAYOUT, BACK_LAYOUT, BACK_LAYOUT],
    )

    assert result["recognitionProfile"] == "MULTI-PAGE-DOCUMENT"
    assert result["pageProcessing"]["decision"] == "matched"
    assert result["pageProcessing"]["matchedPageCount"] == 3
    assert result["documentNumber"] == "ZZ7654321"
    assert [
        page["layoutIdentifier"] for page in result["pageProcessing"]["pages"]
    ] == [FRONT_LAYOUT, BACK_LAYOUT, BACK_LAYOUT]
    assert all(
        page["classification"] is None
        for page in result["pageProcessing"]["pages"]
    )


def test_page_identifiers_must_align_with_paths() -> None:
    with pytest.raises(ValueError, match="align"):
        process_document_pages(
            [ID_FRONT, ID_BACK],
            FIXTURE_CATALOG,
            document_identifiers=[],
        )
