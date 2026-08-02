"""Stable, minimal REST response serializers."""

from __future__ import annotations

from typing import Any


def compact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            compacted = compact(item)
            if compacted is not None and compacted != "" and compacted != [] and compacted != {}:
                result[key] = compacted
        return result
    if isinstance(value, list):
        return [compacted for item in value if (compacted := compact(item)) not in (None, "", [], {})]
    return value


def success(data: dict, omit_empty: bool = True) -> dict:
    return {"status": "ok", "data": compact(data) if omit_empty else data}


def head_pose_response(value: dict | None) -> dict | None:
    if not value:
        return None
    return {axis: value.get(axis) for axis in ("yaw", "pitch", "roll")}


def face_quality_response(value: dict | None) -> dict | None:
    if not value:
        return None
    return {"status": value.get("status"), "warnings": value.get("warnings", [])}


def document_images_response(values: list[dict] | None) -> list[dict] | None:
    if not values:
        return None
    return [
        {
            key: image.get(key)
            for key in (
                "type",
                "side",
                "mediaType",
                "imageBase64",
                "width",
                "height",
            )
        }
        for image in values
    ]


def document_response(result: dict) -> dict:
    profile = result.get("recognitionProfile", "UNSUPPORTED")
    source = result.get("source", "NOT_AVAILABLE")
    validity = result.get("validityStatus", -1)
    structural_status = {1: "valid", 0: "invalid"}.get(validity, "not_available")
    quality = result.get("qualitySignals", {})
    classification_candidate = next(
        (
            candidate
            for candidate in result.get("documentClassification", {}).get("candidates", [])
            if candidate.get("document")
        ),
        None,
    )
    classification = None
    if classification_candidate:
        classified_document = classification_candidate["document"]
        classified_type = (classified_document.get("documentType") or {}).get("name")
        structural_conflict = source == "MRZ" and (
            (
                result.get("DocumentName") == "Passport"
                and classified_type not in {"Passport", "PassportPage"}
            )
            or (
                result.get("dCountryName")
                and classified_document.get("country")
                and result["dCountryName"] != classified_document["country"]
            )
        )
        if not structural_conflict:
            classification = {
                "name": classified_document.get("caption"),
                "country": classified_document.get("country"),
                "countryCode": (classified_document.get("isoCodes") or [None])[0],
                "type": classified_type,
                "format": (classified_document.get("documentFormat") or {}).get("name"),
                "edition": classified_document.get("year"),
                "series": classified_document.get("series"),
                "jurisdictionCodes": classified_document.get("stateCodes"),
                "issuedFrom": classified_document.get("issuedFrom"),
                "issuedTo": classified_document.get("issuedTo"),
                "deprecated": classified_document.get("deprecated"),
                "confidence": classification_candidate.get("confidence"),
            }

    document = {
        "recognized": profile != "UNSUPPORTED",
        "name": result.get("DocumentName"),
        "country": result.get("dCountryName"),
        "classCode": result.get("documentClassCode"),
        "number": result.get("documentNumber"),
        "status": result.get("documentStatus"),
        "issuingStateCode": result.get("issuingStateCode"),
        "source": source,
        "profile": profile,
        "layoutIdentifier": result.get("requestedDocumentIdentifier"),
        "pageRole": result.get("pageRole"),
        "requestedFields": result.get("requestedFields"),
        "classification": classification,
    }
    holder_name = result.get("name") or result.get("surnameAndGivenNames") or " ".join(
        filter(None, (result.get("surname"), result.get("givenNames")))
    )
    holder = {
        "name": holder_name,
        "firstSurname": result.get("firstSurname"),
        "secondSurname": result.get("secondSurname"),
        "surname": result.get("surname"),
        "givenNames": result.get("givenNames"),
        "sex": result.get("sex"),
        "nationality": result.get("nationality"),
        "nationalityCode": result.get("nationalityCode"),
        "height": result.get("height"),
    }
    dates = {
        "birth": result.get("dateOfBirth"),
        "issue": result.get("dateOfIssue") or result.get("issueYear"),
        "expiry": result.get("dateOfExpiry") or result.get("validity"),
    }
    address = {
        "full": result.get("address"),
        "lines": result.get("addressLines"),
        "city": result.get("city"),
        "state": result.get("state") or result.get("jurisdictionCode"),
        "municipality": result.get("municipality"),
        "locality": result.get("locality"),
        "postalCode": result.get("postalCode"),
    }
    identifiers = {
        "personalNumber": result.get("personalNumber"),
        "folioNumber": result.get("folioNumber"),
        "electorKey": result.get("electorKey"),
        "curp": result.get("curp"),
        "documentDiscriminator": result.get("documentDiscriminator"),
        "registrationYear": result.get("registrationYear"),
        "section": result.get("section"),
        "optionalData": result.get("optionalData"),
        "credentialIdentifier": result.get("credentialIdentifier")
        or result.get("ineQr", {}).get("credentialIdentifier"),
        "queryIdentifier": result.get("queryIdentifier")
        or result.get("ineQr", {}).get("queryIdentifier"),
        "barcodeNumber": result.get("barcodeNumber")
        or result.get("ineQr", {}).get("barcodeNumber"),
    }
    details = {
        "authority": result.get("authority"),
        "placeOfBirth": result.get("placeOfBirth"),
        "observations": result.get("observations"),
    }
    machine_readable = None
    if result.get("mrzCode"):
        machine_readable = {
            "type": result.get("machineReadableProfile", profile).replace("ICAO-", ""),
            "code": result["mrzCode"],
            "checks": result.get("checks"),
        }
    elif source == "BARCODE":
        machine_readable = {
            "type": result.get("recognition", {}).get("barcodeFormat", "PDF417")
        }
    barcode_summaries = [
        {"format": barcode.get("format"), "value": barcode.get("text")}
        for barcode in result.get("machineBarcodes", [])
        if barcode.get("text") != result.get("ineQr", {}).get("verificationUrl")
    ]
    if barcode_summaries:
        if machine_readable is None:
            machine_readable = {"type": "BARCODE"}
        machine_readable["barcodes"] = barcode_summaries
    if result.get("ineQr"):
        if machine_readable is None:
            machine_readable = {"type": "BARCODE"}
        machine_readable["verification"] = {
            key: result["ineQr"].get(key)
            for key in (
                "issueDate",
                "credentialType",
                "queryIdentifier",
                "verificationUrl",
            )
        }

    validation = {
        "structural": structural_status,
        "spoofingDecision": quality.get("spoofingDecision"),
        "livenessDecision": quality.get("livenessDecision"),
        "fields": result.get("fieldValidation"),
    }
    regions = {
        name: {
            "box": region.get("box"),
            "coordinateSpace": region.get("coordinateSpace"),
            "faceExpected": region.get("faceExpected"),
            "facePresence": region.get("facePresence"),
            "side": region.get("side"),
        }
        for name, region in result.get("visualRegions", {}).items()
        if name in {"portrait", "ghostPortrait", "signature", "portraitOfChild"}
    }
    pairing_source = result.get("pairing") or {}
    catalog_relation = pairing_source.get("catalogRelation") or {}
    pairing = {
        "decision": pairing_source.get("decision"),
        "checks": pairing_source.get("checks"),
        "relationType": catalog_relation.get("relationType"),
        "expectedRelatedDocuments": [
            {"name": document.get("name")}
            for document in catalog_relation.get("relatedDocuments", [])
        ],
        "relatedSideBarcodeCount": pairing_source.get("relatedSideBarcodeCount"),
        "sides": [
            {
                "side": side.get("side"),
                "recognized": side.get("recognized"),
                "profile": side.get("profile"),
                "layoutIdentifier": side.get("layoutIdentifier"),
                "source": side.get("source"),
                "classification": (
                    {
                        "name": side["classification"].get("name"),
                        "confidence": side["classification"].get("confidence"),
                    }
                    if side.get("classification")
                    else None
                ),
            }
            for side in pairing_source.get("sides", [])
        ],
    }
    page_source = result.get("pageProcessing") or {}
    pages = {
        "decision": page_source.get("decision"),
        "pageCount": page_source.get("pageCount"),
        "matchedPageCount": page_source.get("matchedPageCount"),
        "relatedSideBarcodeCount": page_source.get("relatedSideBarcodeCount"),
        "ordering": page_source.get("ordering"),
        "comparisons": page_source.get("comparisons"),
        "items": [
            {
                "page": page.get("side"),
                "inputPage": page.get("inputPage"),
                "recognized": page.get("recognized"),
                "profile": page.get("profile"),
                "layoutIdentifier": page.get("layoutIdentifier"),
                "source": page.get("source"),
                "classification": (
                    {
                        "name": page["classification"].get("name"),
                        "confidence": page["classification"].get("confidence"),
                    }
                    if page.get("classification")
                    else None
                ),
            }
            for page in page_source.get("pages", [])
        ],
    }
    return success(
        {
            "document": document,
            "holder": holder,
            "dates": dates,
            "address": address,
            "identifiers": identifiers,
            "details": details,
            "machineReadable": machine_readable,
            "regions": regions,
            "images": document_images_response(result.get("extractedImages")),
            "pairing": pairing,
            "pages": pages,
            "validation": validation,
        }
    )


def face_analysis_response(result: dict) -> dict:
    faces = []
    for face in result.get("faces", []):
        serialized = {
            "confidence": face.get("confidence"),
            "box": face.get("normalizedBox"),
        }
        landmarks = face.get("landmarksAndQuality", {}).get("landmarks", [])
        if landmarks:
            serialized["landmarks"] = [point.get("imageNormalized") for point in landmarks]
        if face.get("headPose"):
            serialized["headPose"] = head_pose_response(face["headPose"])
        if face.get("quality"):
            serialized["quality"] = face_quality_response(face["quality"])
        faces.append(serialized)
    return success({"faceCount": result.get("faceCount", 0), "faces": faces})


def liveness_response(result: dict, face_box: list[float]) -> dict:
    return success(
        {
            "decision": result.get("decision"),
            "score": result.get("score"),
            "threshold": result.get("threshold"),
            "spoofThreshold": result.get("spoofThreshold"),
            "face": {"box": face_box},
            "headPose": head_pose_response(result.get("headPose")),
            "quality": face_quality_response(result.get("quality")),
        }
    )


def comparison_response(result: dict) -> dict:
    return success(
        {
            "decision": result.get("decision"),
            "score": result.get("score"),
            "threshold": result.get("threshold"),
            "faces": [{"box": box} for box in result.get("faceBoxes", [])],
            "templates": result.get("templates"),
        }
    )


def document_portrait_comparison_response(result: dict) -> dict:
    liveness = result.get("selfieLiveness") or {}
    return success(
        {
            "decision": result.get("decision"),
            "score": result.get("score"),
            "threshold": result.get("threshold"),
            "verification": result.get("verification"),
            "document": {
                "profile": result.get("documentProfile"),
                "layoutIdentifier": result.get("layoutIdentifier"),
                "portrait": {
                    "box": result.get("portraitBox"),
                    "faceBox": result.get("documentFaceBox"),
                    "detectionConfidence": result.get(
                        "documentDetectionConfidence"
                    ),
                    "detectionThreshold": result.get("documentDetectionThreshold"),
                },
            },
            "selfie": {
                "faceBox": result.get("selfieFaceBox"),
                "liveness": (
                    {
                        "decision": liveness.get("decision"),
                        "score": liveness.get("score"),
                        "threshold": liveness.get("threshold"),
                        "spoofThreshold": liveness.get("spoofThreshold"),
                        "headPose": head_pose_response(liveness.get("headPose")),
                        "quality": face_quality_response(liveness.get("quality")),
                    }
                    if liveness
                    else None
                ),
            },
        }
    )


def face_template_response(result: dict) -> dict:
    return success(
        {
            "templateBase64": result.get("templateBase64"),
            "format": "float32-le",
            "length": result.get("length"),
            "byteLength": result.get("byteLength"),
            "face": {"box": result.get("faceBox")},
        }
    )
