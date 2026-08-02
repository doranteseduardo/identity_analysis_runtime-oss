# Bringing Your Own Models

This repository contains runtime code only. No model weights, no document
catalog, and no third-party SDK material are distributed with it. You supply an
asset directory and point the runtime at it:

```bash
export IDENTITY_ANALYSIS_ASSETS=/path/to/assets
identity-api
```

The CLIs take the same directory (`identity-document --assets ...`), and the
Docker image expects it mounted at `/app/assets`.

Everything below is optional. A family that is absent is reported as
unavailable by `GET /v1/capabilities`; it never prevents the service from
starting.

---

## Directory Layout

```
assets/
├── manifest.json                     # integrity manifest (see below)
├── models/
│   ├── ocr_latin.onnx                # default line-recognition model
│   ├── document_corners.onnx         # document rectification
│   ├── focus_device.onnx             # capture focus quality
│   ├── electronic_device.onnx        # document-capture PAD (screen replay)
│   ├── moire.onnx                    # document-capture PAD (moire)
│   └── ocr/<lcid>.onnx               # per-locale line recognition
├── charsets/
│   ├── latin.txt                     # charset for models/ocr_latin.onnx
│   └── ocr/<lcid>.txt                # charset per locale model
├── metadata/
│   ├── ocr_scripts.json              # locale -> model routing
│   ├── icao_transliteration.json     # ICAO 9303 transliteration table
│   └── doc_liveness.json             # document-capture PAD configuration
├── facial/
│   ├── detector/face_detector.onnx
│   ├── landmarks/landmarks_quality.onnx
│   ├── liveness/
│   │   ├── manifest.json
│   │   └── <model-name>/model.onnx
│   └── recognition/
│       ├── manifest.json
│       └── <model-name>/model.onnx
└── document_classifier/
    ├── manifest.json
    ├── model.onnx
    ├── info.json
    ├── catalog.json
    └── visual-layouts.json.gz
```

### Model Contracts

| File | Input | Output |
|---|---|---|
| `models/ocr_latin.onnx`, `models/ocr/<lcid>.onnx` | `float32[1, 1, 48, W]`, grayscale scaled to `[-1, 1]` | CTC logits `[1, T, len(charset) + 1]`, blank last |
| `models/document_corners.onnx` | `float32[1, 3, 512, 512]` | four document corners |
| `models/focus_device.onnx` | `float32[1, 256, 256, 1]`, values in `[0, 1]` | 2 logits, class 0 blurred, class 1 focused |
| `facial/detector/face_detector.onnx` | `float32[1, 3, 512, 512]`, scaled to `[-1, 1]` | SSD scores `[N, 2]` and box encodings `[N, 4]` over 5,118 anchors |
| `facial/landmarks/landmarks_quality.onnx` | `float32[1, 3, 224, 224]` | two 68-value vectors (x, y) plus one scalar coverage score |
| `facial/recognition/*/model.onnx` | aligned face crop | 512-value embedding |
| `document_classifier/model.onnx` | `float32[1, 3, 256, 256]`, per-image z-scored | one probability per catalog class |

A charset file is one integer code point per line; the CTC blank is the index
one past the last entry.

---

## `manifest.json`

`identity-assets manifest <dir>` walks a directory and writes the file for you;
`identity-assets validate <dir>` verifies it. The schema is:

```json
{
  "formatVersion": 2,
  "files": [
    {
      "path": "models/ocr_latin.onnx",
      "sha256": "2218bd12…",
      "size": 619988
    }
  ]
}
```

`path` is POSIX-relative to the asset root. Validation reads every listed file
and fails on the first hash mismatch. Files present on disk but absent from the
manifest are ignored, so the manifest defines exactly what is pinned.

---

## Document Classification Catalog

The catalog is the only asset family with a schema of its own. Its
`manifest.json` is separate from the top-level one:

```json
{
  "schemaVersion": 1,
  "engine": "your-engine-name",
  "model":         { "path": "model.onnx",             "sha256": "…" },
  "info":          { "path": "info.json",              "sha256": "…" },
  "catalog":       { "path": "catalog.json",           "sha256": "…",
                     "classCount": 3 },
  "visualLayouts": { "path": "visual-layouts.json.gz", "sha256": "…",
                     "layoutCount": 4 }
}
```

`engine` is echoed back in every classification response.

**`catalog.json`** maps classifier output indices to documents:

```json
{
  "schemaVersion": 2,
  "classCount": 3,
  "identifierMap": [
    {
      "outputIndex": 0,
      "documentIdentifier": 900000001,
      "document": {
        "identifier": 900000001,
        "caption": "Specimen Identity Card (2024) Front",
        "country": "ZZT",
        "isoCodes": ["ZZT"],
        "documentType": { "value": 12, "name": "IdentityCard" },
        "documentFormat": { "value": 0, "name": "ID1" },
        "mrz": { "present": false, "ignored": false, "expectedProfile": null },
        "deprecated": false,
        "year": "2024"
      }
    }
  ]
}
```

**`visual-layouts.json.gz`** is a gzipped JSON document describing where each
field sits on each document. All bounds are `[left, top, right, bottom]`
fractions of the document rectangle.

```json
{
  "schemaVersion": 2,
  "fieldTypes":        { "1": { "officialName": "SURNAME", "name": "surname" } },
  "graphicFieldTypes": { "101": { "officialName": "PORTRAIT", "name": "portrait" } },
  "layouts": {
    "900000001": {
      "identifier": 900000001,
      "caption": "Specimen Identity Card (2024) Front",
      "country": "ZZT",
      "isoCodes": ["ZZT"],
      "documentType": { "value": 12, "name": "IdentityCard" },
      "documentFormat": { "value": 0, "name": "ID1" },
      "year": "2024",
      "orientation": 0,
      "dimensionsMm": { "width": 86, "height": 54 },
      "twoSided": true,
      "mainDocument": true,
      "childDocuments": [900000002],
      "pairedPages": [],
      "fields": [
        {
          "number": 1, "type": 1, "name": "surname",
          "bounds": [0.33, 0.213, 0.62, 0.291],
          "lightType": 6, "lcid": 1033, "textHeight": 0.055,
          "mask": "{DAY_DD}/{MONTH_DD}/{YEAR}", "inComparison": 1
        }
      ],
      "graphics":  [{ "number": 1, "type": 101, "name": "portrait",
                      "bounds": [0.055, 0.3, 0.3, 0.7],
                      "lightType": 6, "checkFacePresent": true }],
      "barcodes":  [{ "number": 1, "type": 6, "name": "documentNumber",
                      "bounds": [0.025, 0.025, 0.275, 0.175],
                      "lightType": 6, "codeType": 1, "codeClass": 1 }],
      "securityRegions": [],
      "referencePatches": [],
      "assemblies": [],
      "catalogHints": {}
    }
  }
}
```

Notes:

- `lightType` must be `6` or `24` for a region to be treated as visible in
  ordinary illumination; anything else is skipped.
- `codeType` follows the mapping in `identity_analysis.barcodes`
  (`1` Code 128, `2` Code 39, `5` PDF417, `11` Code 93, `14` QR,
  `16` Data Matrix, …). Unmapped values fall back to multi-format decoding.
- `caption` markers such as `Front`, `Side B` or `Page 3` drive page-role
  resolution; `childDocuments` and `pairedPages` drive front/back and multi-page
  ordering.
- `assemblies` compose a derived field from other fields (for example a full
  name from surname plus given names).
- `catalogHints.recognition.dMRZ*` values, in tenths of a millimetre alongside
  `dimensionsMm`, guide MRZ localisation.

`tests/fixtures/catalog/document_classifier` is a complete, minimal working
example, and `tools/generate_test_catalog.py` is the script that produces it.

---

## Feature Availability

| Feature | Requires | Without it |
|---|---|---|
| MRZ reading (`/v1/ocr`, TD1/TD2/TD3/MRV) | `models/ocr_latin.onnx`, `charsets/latin.txt` | unavailable |
| Non-Latin field OCR | `models/ocr/<lcid>.onnx`, `charsets/ocr/<lcid>.txt`, `metadata/ocr_scripts.json` | falls back to the Latin model |
| Barcode decoding | nothing (zxing-cpp) | always available |
| Document rectification | `models/document_corners.onnx` | the original image is used |
| Capture focus quality | `models/focus_device.onnx` | quality signals omitted |
| Document-capture PAD | `models/electronic_device.onnx`, `models/moire.onnx`, `metadata/doc_liveness.json` | `spoofingDecision: not_available` |
| Cross-script name matching | `metadata/icao_transliteration.json` | exact/diacritic-folded matching only |
| Document classification (`/v1/document/classify`) | `document_classifier/{manifest,model,info,catalog}` | unavailable; explicit layouts still work |
| Exact-layout OCR, catalog search, layout evidence | `document_classifier/{manifest,catalog,visual-layouts}` | unavailable |
| Face detection, landmarks, pose, quality | `facial/detector`, `facial/landmarks` | unavailable |
| Passive face liveness | `facial/liveness` | unavailable |
| Face comparison and templates | `facial/recognition` | unavailable |

`GET /v1/capabilities` returns this as data:

```json
{
  "runtime": {
    "assetsPath": "/app/assets",
    "assetsPresent": true,
    "features": { "lineRecognition": true, "faceLiveness": false, "…": false },
    "documentation": "docs/models.md"
  }
}
```
