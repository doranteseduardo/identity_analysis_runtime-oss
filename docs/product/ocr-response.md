# REST Response Contract

All REST endpoints use the same top-level envelope. Successful responses contain
`status: ok` and a `data` object. Empty strings, empty arrays, empty objects, and
`null` values are omitted.

```json
{
  "status": "ok",
  "data": {}
}
```

The REST contract intentionally excludes model filenames, logits, embeddings,
duplicate field collections, crop diagnostics, asset metadata, and compatibility
catalogs. The CLI retains the full research-oriented result.

## Document OCR

`POST /v1/ocr` returns semantic groups instead of the pipeline's flat internal
object.

```json
{
  "status": "ok",
  "data": {
    "document": {
      "recognized": true,
      "name": "Passport",
      "country": "Belarus",
      "classCode": "P",
      "number": "AB1234567",
      "source": "MRZ",
      "profile": "ICAO-TD3"
    },
    "holder": {
      "name": "DOE JANE",
      "surname": "DOE",
      "givenNames": "JANE",
      "sex": "F",
      "nationality": "Belarus",
      "nationalityCode": "BLR"
    },
    "dates": {
      "birth": "1990-01-01",
      "issue": "2021-08-27",
      "expiry": "2031-08-27"
    },
    "identifiers": {
      "personalNumber": "12345678901234"
    },
    "machineReadable": {
      "type": "TD3",
      "code": "LINE1\nLINE2",
      "checks": {
        "documentNumber": true,
        "dateOfBirth": true,
        "dateOfExpiry": true,
        "personalNumber": true,
        "composite": true
      }
    },
    "validation": {
      "structural": "valid",
      "spoofingDecision": "pass",
      "livenessDecision": "not_available"
    }
  }
}
```

### Document Groups

| Group | Contents |
|---|---|
| `document` | Recognition state, document name, country, class, number, status, source, and profile |
| `holder` | Name components, sex, nationality, and height |
| `dates` | Birth, issue, and expiry dates in ISO format when available |
| `address` | Full address, ordered lines, city, state, municipality, locality, and postal code |
| `identifiers` | Personal number, elector key, CURP, discriminator, registration year, section, and optional visa MRZ data |
| `details` | Issuing authority and place of birth |
| `machineReadable` | Validated MRZ code and checks, barcode type, additional barcode evidence, and supported verification data |
| `regions` | Normalized portrait, child portrait, ghost portrait, and signature boxes when an exact visual layout supplies them |
| `images` | Optional base64 JPEG crops for supported regions when `includeImages` is enabled |
| `pairing` | Front/back match decision, relation type, cross-side checks, expected related document names, and concise side summaries |
| `pages` | Multi-page collection decision, ordering provenance, counts, per-page comparisons, and concise page summaries |
| `validation` | Structural state and available spoofing or liveness decisions |

`document.classification` describes the highest-ranked named template. Optional
metadata includes `format`, `edition`, `series`, `jurisdictionCodes`,
`issuedFrom`, `issuedTo`, and `deprecated`. The issuance window belongs to the
template edition and is not the holder credential's issue or expiry date.

`document.pageRole` contains:

| Field | Meaning |
|---|---|
| `role` | `front`, `back`, `numbered_page`, `primary_page`, `front_cover`, `back_cover`, or `unknown` |
| `ordinal` | One-based page number when available |
| `method` | `caption_marker`, `related_back_layout`, `related_numbered_page`, or `unavailable` |
| `confidence` | `declared`, `inferred`, or `none` |
| `relatedLayoutIdentifiers` | Catalog layouts that support an inferred role |

Role resolution occurs after layout identification. It does not represent an
independent image-side probability.

Region boxes use `[left, top, right, bottom]`. `coordinateSpace` is
`original_image` when a rectified document region has been projected back onto
the uploaded image; otherwise it identifies the processed document geometry.
Paired responses also include `side: front` for regions inherited from the
front image.

When `analyzePortraits` is enabled, eligible portrait regions also include
`facePresence`. `pass` means at least one face was detected inside the declared
crop; `review` means none was detected. The result is region evidence only and
is not a document-authenticity, liveness, or face-comparison decision.

`POST /v1/document/portrait/compare` returns a dedicated comparison contract:
`decision`, `score`, and `threshold`; document profile and layout identifier;
the declared portrait box and face box within that crop; and the selfie face
box. `same_person` is a one-to-one recognition result, not a document
authenticity or selfie-liveness decision. Optional `selfie.liveness` remains a
separate decision with its own score, threshold, head pose, and quality.
`verification` combines portrait identity, selfie liveness, and document
capture-risk signals:

- `pass`: `same_person`, `real`, and document capture `pass`;
- `fail`: `different_person` or selfie `spoof`;
- `review`: an available signal is inconclusive or document capture is `review`;
- `not_available`: liveness was not requested or unavailable.

This aggregate does not include document authenticity.

Image items contain:

| Field | Meaning |
|---|---|
| `type` | `portrait`, `portraitOfChild`, `ghostPortrait`, or `signature` |
| `side` | Source upload: `document`, `front`, `back`, or `page_N` |
| `mediaType` | Currently `image/jpeg` |
| `imageBase64` | Raw base64 without a Data URL prefix |
| `width`, `height` | Encoded crop dimensions in pixels |

`images` is omitted unless the request explicitly sets `includeImages: true`.

For paired processing, `pairing.decision` is `matched`, `mismatch`, or `review`.
Only `matched` responses use profile `PAIRED-DOCUMENT` and source
`MULTI_SOURCE`. Visual fields such as full names and address remain from the
front, while validated MRZ or barcode values can supply document number,
normalized dates, and structural validity.
`relatedSideBarcodeCount` indicates how many payloads were independently
decoded through expected reverse-side barcode regions.
Each side summary includes `layoutIdentifier` when that side was routed by an
exact caller-provided catalog layout.

Multi-page responses use profile `MULTI-PAGE-DOCUMENT` after at least one
additional page is matched and merged. `pages.matchedPageCount` includes the
primary page. Each comparison identifies the additional page, its decision,
and the cross-page checks. Internal page results and duplicate OCR collections
are never included in REST. Page summaries include `layoutIdentifier` for
caller-provided exact layouts.

`pages.ordering.decision` is `reordered` only when related catalog layouts have
distinct page markers or an unmarked primary-page form.
`pages.ordering.inputOrder` maps canonical output order back to one-based
request positions. Each `pages.items[]` entry also exposes its original
`inputPage`. When the evidence is ambiguous, `pages.ordering.method` is
`caller_order`.

An unsupported image is still a completed analysis:

```json
{
  "status": "ok",
  "data": {
    "document": {
      "recognized": false,
      "name": "Unsupported Document",
      "source": "NOT_AVAILABLE",
      "profile": "UNSUPPORTED"
    },
    "validation": {
      "structural": "not_available"
    }
  }
}
```

## Face Analysis

`POST /v1/face/analyze` returns face confidence, one normalized bounding box, and
68 image-normalized landmark pairs. It also returns yaw, pitch, and roll plus a
quality status. Pixel and crop-relative duplicates are not returned.

```json
{
  "status": "ok",
  "data": {
    "faceCount": 1,
    "faces": [
      {
        "confidence": 0.99,
        "box": [0.1, 0.2, 0.6, 0.8],
        "landmarks": [[0.2, 0.3]],
        "headPose": {"yaw": -3.0, "pitch": 2.3, "roll": -0.5},
        "quality": {"status": "pass"}
      }
    ]
  }
}
```

## Face Liveness

`POST /v1/face/liveness` returns the product decision, calibrated score, real
threshold, spoof threshold, detected face box, geometric head pose, and
face-quality status. Per-model logits and internal calibration values are
excluded.

When geometry reports a small, cutoff, or multiple face, the endpoint returns
`decision: review` and omits the PAD score rather than promoting an unreliable
crop to `real` or `spoof`. Excessive pose and covered-face warnings remain
reported as quality concerns, but do not suppress PAD when the selected crop
is otherwise geometrically usable.

```json
{
  "status": "ok",
  "data": {
    "decision": "real",
    "score": 0.81,
    "threshold": 0.37,
    "spoofThreshold": 0.25,
    "face": {"box": [0.1, 0.2, 0.6, 0.8]},
    "headPose": {"yaw": -3.0, "pitch": 2.3, "roll": -0.5},
    "quality": {"status": "pass"}
  }
}
```

Scores greater than `threshold` return `real`, scores less than or equal to
`spoofThreshold` return `spoof`, and scores between both boundaries return
`review`. Quality gating can independently return `review` without a score.

## Face Comparison

`POST /v1/face/compare` returns the decision, normalized score, threshold, and
the two detected face boxes. Cosine similarity and embedding size are internal
diagnostics and are not exposed.

```json
{
  "status": "ok",
  "data": {
    "decision": "same_person",
    "score": 0.91,
    "threshold": 0.67,
    "faces": [
      {"box": [0.1, 0.2, 0.6, 0.8]},
      {"box": [0.2, 0.1, 0.7, 0.9]}
    ]
  }
}
```

## Face Templates

`POST /v1/face/template` extracts the aligned 512-value recognition embedding
as exactly 2,048 little-endian `float32` bytes. The binary value is returned as
base64 so it can be persisted without converting the vector to JSON numbers.
The response also includes the detected face box, format, value count, and byte
length.

`POST /v1/face/template/compare` accepts two such templates and returns the same
decision, normalized score, and threshold contract as image comparison. Inputs
with an incorrect length, non-finite values, or zero norm are rejected rather
than scored.

`POST /v1/face/compare` omits templates by default. With
`includeTemplates: true`, it returns the two templates under `data.templates`
using the same binary contract. This is more efficient than calling template
extraction again because comparison already computed both embeddings.

Face templates are biometric data. Store and transmit them with access control,
encryption, retention limits, and consent appropriate to the deployment. Base64
is only a transport encoding and does not protect the template.

## Errors

HTTP status codes remain meaningful. Error bodies use one stable shape:

```json
{
  "status": "error",
  "error": {
    "code": "unprocessable_image",
    "message": "Image processing failed: expected one face, found 0"
  }
}
```

| HTTP status | Error code |
|---:|---|
| `400` | `invalid_request` |
| `413` | `payload_too_large` |
| `415` | `unsupported_media_type` |
| `422` | `unprocessable_image` |
| `503` | `service_unavailable` |
