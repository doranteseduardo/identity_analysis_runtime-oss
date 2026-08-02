from identity_analysis.responses import (
    comparison_response,
    document_response,
    face_analysis_response,
    face_template_response,
    liveness_response,
)


def test_document_response_removes_internal_duplicates() -> None:
    response = document_response(
        {
            "DocumentName": "Passport",
            "dCountryName": "Belarus",
            "documentNumber": "AB123",
            "surname": "DOE",
            "givenNames": "JANE",
            "dateOfBirth": "1990-01-01",
            "source": "MRZ",
            "recognitionProfile": "ICAO-TD3",
            "pageRole": {
                "role": "primary_page",
                "ordinal": 1,
                "method": "related_numbered_page",
                "confidence": "inferred",
            },
            "validityStatus": 1,
            "mrzCode": "LINE1\nLINE2",
            "checks": {"documentNumber": True},
            "ContainerList": [{"OneCandidate": {}}],
            "fieldList": [{"fieldName": "Surname", "value": "DOE"}],
            "recognizedFields": {"surname": {"value": "DOE"}},
            "sdkCompatibility": {"coverage": "partial"},
            "metadataIntegrity": {"available": True},
            "recognition": {"engine": "internal-model.onnx"},
            "qualitySignals": {"spoofingDecision": "real"},
        }
    )

    assert response["status"] == "ok"
    assert response["data"]["document"]["number"] == "AB123"
    assert response["data"]["document"]["pageRole"]["role"] == "primary_page"
    assert response["data"]["holder"] == {
        "name": "DOE JANE",
        "surname": "DOE",
        "givenNames": "JANE",
    }
    assert "address" not in response["data"]
    assert "identifiers" not in response["data"]
    assert "details" not in response["data"]
    serialized = str(response)
    for internal_key in (
        "ContainerList",
        "fieldList",
        "recognizedFields",
        "sdkCompatibility",
        "metadataIntegrity",
        "internal-model.onnx",
    ):
        assert internal_key not in serialized


def test_document_response_exposes_field_validation_decisions() -> None:
    response = document_response(
        {
            "recognitionProfile": "CATALOG-P1",
            "dateOfBirth": "01/01/1990",
            "fieldValidation": {
                "dateOfBirth": {
                    "decision": "pass",
                    "reason": "visual_curp_elector_key_consensus",
                },
                "curp": {
                    "decision": "review",
                    "reason": "structure_does_not_verify_all_characters",
                },
            },
        }
    )

    assert response["data"]["validation"]["fields"]["dateOfBirth"]["decision"] == "pass"
    assert response["data"]["validation"]["fields"]["curp"]["decision"] == "review"


def test_document_response_exposes_only_top_classification() -> None:
    response = document_response(
        {
            "recognitionProfile": "UNSUPPORTED",
            "documentClassification": {
                "engine": "internal",
                "candidates": [
                    {
                        "confidence": 0.87,
                        "documentIdentifier": 123,
                        "document": {
                            "caption": "Example Identity Card (2025)",
                            "country": "Exampleland",
                            "isoCodes": ["EXP"],
                            "documentType": {"name": "IdentityCard", "value": 1},
                            "documentFormat": {"name": "ID1", "value": 0},
                            "year": "2025",
                            "series": "EXP_ID_2025",
                            "stateCodes": ["EX"],
                            "issuedFrom": "2025-01-01",
                            "issuedTo": "2029-12-31",
                            "deprecated": True,
                            "sourceMember": "123.json",
                        },
                    },
                    {"confidence": 0.1, "document": {"caption": "Noise"}},
                ],
            },
        }
    )

    assert response["data"]["document"]["classification"] == {
        "name": "Example Identity Card (2025)",
        "country": "Exampleland",
        "countryCode": "EXP",
        "type": "IdentityCard",
        "format": "ID1",
        "edition": "2025",
        "series": "EXP_ID_2025",
        "jurisdictionCodes": ["EX"],
        "issuedFrom": "2025-01-01",
        "issuedTo": "2029-12-31",
        "deprecated": True,
        "confidence": 0.87,
    }


def test_document_response_hides_classification_conflicting_with_valid_mrz() -> None:
    response = document_response(
        {
            "recognitionProfile": "ICAO-TD3",
            "source": "MRZ",
            "validityStatus": 1,
            "DocumentName": "Passport",
            "dCountryName": "Mexico",
            "documentClassification": {
                "candidates": [
                    {
                        "confidence": 0.34,
                        "document": {
                            "caption": "Id Card #1",
                            "country": "Nigeria",
                            "documentType": {"name": "IdentityCard"},
                        },
                    }
                ]
            },
        }
    )

    assert "classification" not in response["data"]["document"]
    serialized = str(response)
    assert "candidates" not in serialized
    assert "documentIdentifier" not in serialized


def test_document_response_fuses_ine_qr_without_duplicate_url() -> None:
    url = "http://qr.ine.mx/000000000000000000000000/20990101/X/000000"
    response = document_response(
        {
            "recognitionProfile": "ICAO-TD1",
            "source": "MRZ",
            "mrzCode": "LINE1\nLINE2\nLINE3",
            "checks": {"composite": True},
            "machineBarcodes": [
                {"format": "Code128", "text": "000000000"},
                {"format": "QRCode", "text": url},
            ],
            "ineQr": {
                "credentialIdentifier": "000000000000000000000000",
                "issueDate": "2099-01-01",
                "credentialType": "X",
                "queryIdentifier": "000000",
                "barcodeNumber": "000000000",
                "verificationUrl": url,
            },
        }
    )

    machine = response["data"]["machineReadable"]
    assert machine["barcodes"] == [
        {"format": "Code128", "value": "000000000"}
    ]
    assert machine["verification"]["verificationUrl"] == url
    assert str(response).count(url) == 1


def test_document_response_exposes_useful_visual_regions_only() -> None:
    response = document_response(
        {
            "recognitionProfile": "CATALOG-P1",
            "source": "VISUAL",
            "visualRegions": {
                "portrait": {
                    "box": [0.1, 0.2, 0.3, 0.8],
                    "coordinateSpace": "original_image",
                    "faceExpected": True,
                    "facePresence": {
                        "expected": True,
                        "detected": True,
                        "count": 1,
                        "threshold": 0.2,
                        "confidence": 0.98,
                        "status": "pass",
                    },
                    "processedDocumentBox": [0.05, 0.1, 0.35, 0.9],
                },
                "colorDynamic": {"box": [0, 0, 1, 1]},
            },
        }
    )

    assert response["data"]["regions"] == {
        "portrait": {
            "box": [0.1, 0.2, 0.3, 0.8],
            "coordinateSpace": "original_image",
            "faceExpected": True,
            "facePresence": {
                "expected": True,
                "detected": True,
                "count": 1,
                "threshold": 0.2,
                "confidence": 0.98,
                "status": "pass",
            },
        }
    }
    assert "processedDocumentBox" not in str(response)


def test_document_response_filters_extracted_image_fields() -> None:
    response = document_response(
        {
            "recognitionProfile": "CATALOG-P1",
            "source": "VISUAL",
            "extractedImages": [
                {
                    "type": "portrait",
                    "side": "front",
                    "mediaType": "image/jpeg",
                    "imageBase64": "YWJj",
                    "width": 40,
                    "height": 50,
                    "internalBounds": [0, 0, 1, 1],
                }
            ],
        }
    )

    assert response["data"]["images"] == [
        {
            "type": "portrait",
            "side": "front",
            "mediaType": "image/jpeg",
            "imageBase64": "YWJj",
            "width": 40,
            "height": 50,
        }
    ]
    assert "internalBounds" not in str(response)


def test_document_response_compacts_multi_page_summary() -> None:
    response = document_response(
        {
            "recognitionProfile": "MULTI-PAGE-DOCUMENT",
            "source": "MULTI_SOURCE",
            "pageProcessing": {
                "decision": "matched",
                "pageCount": 3,
                "matchedPageCount": 2,
                "relatedSideBarcodeCount": 1,
                "ordering": {
                    "decision": "reordered",
                    "method": "catalog_relation_and_declared_page_ordinal",
                    "inputOrder": [2, 1, 3],
                },
                "comparisons": [
                    {"page": 2, "decision": "matched", "checks": {"country": True}},
                    {"page": 3, "decision": "review", "checks": {"country": None}},
                ],
                "pages": [
                    {"side": "page_1", "inputPage": 2, "recognized": True, "profile": "FRONT", "source": "VISUAL"},
                    {"side": "page_2", "inputPage": 1, "recognized": True, "profile": "BACK", "source": "MRZ"},
                ],
            },
            "pageResults": [{"internal": True}],
        }
    )

    assert response["data"]["pages"]["pageCount"] == 3
    assert response["data"]["pages"]["ordering"]["inputOrder"] == [2, 1, 3]
    assert response["data"]["pages"]["items"][0]["page"] == "page_1"
    assert response["data"]["pages"]["items"][0]["inputPage"] == 2
    assert "pageResults" not in str(response)


def test_liveness_response_keeps_only_product_fields() -> None:
    response = liveness_response(
        {
            "decision": "real",
            "score": 0.8,
            "threshold": 0.5,
            "linearScore": 123.0,
            "modelLogits": [1.0] * 7,
            "ensemble": "internal",
        },
        [0.1, 0.2, 0.3, 0.4],
    )

    assert set(response["data"]) == {"decision", "score", "threshold", "face"}


def test_comparison_response_removes_embedding_diagnostics() -> None:
    response = comparison_response(
        {
            "decision": "same_person",
            "score": 0.9,
            "threshold": 0.67,
            "cosineSimilarity": 0.8,
            "embeddingLength": 512,
            "templateVectors": [[1.0] * 512, [2.0] * 512],
            "faceBoxes": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
        }
    )

    assert set(response["data"]) == {"decision", "score", "threshold", "faces"}


def test_face_template_response_keeps_only_portable_contract() -> None:
    response = face_template_response(
        {
            "templateBase64": "AAAA",
            "length": 512,
            "byteLength": 2048,
            "faceBox": [0.1, 0.2, 0.3, 0.4],
            "vector": [1.0] * 512,
        }
    )

    assert response["data"] == {
        "templateBase64": "AAAA",
        "format": "float32-le",
        "length": 512,
        "byteLength": 2048,
        "face": {"box": [0.1, 0.2, 0.3, 0.4]},
    }


def test_face_analysis_uses_one_coordinate_system() -> None:
    response = face_analysis_response(
        {
            "faceCount": 1,
            "faces": [
                {
                    "confidence": 0.99,
                    "normalizedBox": [0.1, 0.2, 0.3, 0.4],
                    "pixelBox": [10, 20, 30, 40],
                    "landmarksAndQuality": {
                        "qualityScore": 99.0,
                        "landmarks": [
                            {
                                "imageNormalized": [0.2, 0.3],
                                "cropNormalized": [0.4, 0.5],
                                "pixel": [20, 30],
                            }
                        ],
                    },
                }
            ],
        }
    )

    assert response["data"]["faces"] == [
        {"confidence": 0.99, "box": [0.1, 0.2, 0.3, 0.4], "landmarks": [[0.2, 0.3]]}
    ]
