# Adding a Document

## Overview

Adding document support does not always require new recognition code. Choose the
smallest extension that produces reliable identity-bearing evidence:

| Situation | Recommended path |
|---|---|
| The document already exists in the packaged catalog | Validate and use its `documentIdentifier` |
| It follows a supported MRZ or barcode standard | Reuse the machine-readable parser |
| It has stable visual fields and can use the packaged line OCR models | Add a declarative layout |
| It needs document-specific crops, normalization, parsing, or validation | Add a dedicated profile |
| It introduces a new machine-readable payload | Add a parser and routing rules |

Classification confidence alone is not sufficient. A document is considered
recognized only after OCR, MRZ, barcode, or profile-specific structural evidence
is available.

## Before You Start

Collect representative, authorized samples:

- front and back when the document is two-sided;
- every edition or materially different visual layout;
- at least one sharp, well-lit baseline capture;
- realistic mobile captures with perspective, glare, blur, and background;
- expected values for every field you intend to expose;
- machine-readable payloads and their expected parsed values;
- portrait and signature regions when relevant.

Do not commit personal identity images unless repository policy and consent
explicitly allow it. Synthetic or specimen documents are preferable for
long-lived regression fixtures.

Create a small acceptance table before implementation:

| Sample | Side/page | Expected layout | Expected source | Required fields |
|---|---|---|---|---|
| `sample-front.jpg` | front | exact identifier or profile | `VISUAL` | name, birth, number |
| `sample-back.jpg` | back | exact identifier or profile | `MRZ` or `BARCODE` | number, expiry |

## Path 1: Enable an Existing Catalog Document

The packaged classifier and declarative-layout catalog already cover thousands
of document editions. Verify catalog support before writing code.

### 1. Search the Catalog

```bash
curl --get http://localhost:8000/v1/document/catalog \
  --data-urlencode 'q=passport' \
  --data-urlencode 'countryCode=SWE' \
  --data-urlencode 'limit=20'
```

Useful filters are `q`, `countryCode`, `documentType`, `documentFormat`, and
`includeDeprecated`.

### 2. Classify a Representative Image

```bash
curl -X POST http://localhost:8000/v1/document/classify \
  -F 'image=@sample-front.jpg' \
  -F 'topK=10'
```

Record the candidate identifier and confidence, but do not treat the
classification as a successful OCR result.

### 3. Inspect Layout Evidence

```bash
curl \
  http://localhost:8000/v1/document/layout/123456789/evidence
```

Confirm:

- document name, country, type, format, and edition;
- normalized text fields and masks;
- portrait, signature, and barcode regions;
- orientation and physical dimensions;
- compatible front/back or page relationships;
- MRZ and capture requirements.

### 4. Run Exact-Layout OCR

```bash
curl -X POST \
  http://localhost:8000/v1/document/layout/123456789/ocr \
  -F 'image=@sample-front.jpg'
```

Request only required fields while tuning:

```bash
curl -X POST \
  http://localhost:8000/v1/document/layout/123456789/ocr \
  -F 'image=@sample-front.jpg' \
  -F 'fields=surname' \
  -F 'fields=givenNames' \
  -F 'fields=documentNumber'
```

Exact-layout routing skips classification and is the preferred way to validate a
specific edition.

### 5. Validate Automatic Routing

After exact OCR succeeds, test the normal endpoint:

```bash
curl -X POST http://localhost:8000/v1/ocr \
  -F 'image=@sample-front.jpg'
```

Automatic routing must return the same identity-bearing values without forcing
an unrelated document type. If classification is ambiguous, callers may retain
the exact identifier in their own capture flow.

### 6. Validate Related Sides

```bash
curl -X POST http://localhost:8000/v1/ocr/pair \
  -F 'frontImage=@sample-front.jpg' \
  -F 'backImage=@sample-back.jpg' \
  -F 'frontDocumentIdentifier=123456789' \
  -F 'backDocumentIdentifier=987654321'
```

Check `pairing.decision`, cross-side identity checks, field precedence,
machine-readable fusion, page roles, and capture-risk aggregation.

## Path 2: Add a Declarative Visual Layout

Use a declarative layout when the packaged OCR models can read the document and
the main missing information is geometry or field metadata.

### Layout Record

Each layout requires a stable identifier and normalized metadata:

```json
{
  "identifier": 123456789,
  "caption": "Example Identity Card (2026)",
  "country": "Example",
  "isoCodes": ["XMP"],
  "documentType": {"name": "IdentityCard", "value": 12},
  "documentFormat": {"name": "ID1", "value": 0},
  "classifierLinked": false,
  "year": "2026",
  "orientation": 0,
  "dimensionsMm": {"width": 85.6, "height": 53.98},
  "twoSided": true,
  "mainDocument": true,
  "childDocuments": [987654321],
  "pairedPages": [],
  "fields": [],
  "graphics": [],
  "barcodes": [],
  "securityRegions": [],
  "referencePatches": [],
  "assemblies": [],
  "catalogHints": {}
}
```

Identifiers must be unique and stable. Never reuse an identifier for a different
edition.

### Text Fields

Text coordinates use normalized document geometry:

```text
[left, top, right, bottom]
```

Values range from zero to one, with the origin at the top left. A field should
declare, when known:

- semantic field name;
- normalized bounds;
- locale identifier;
- mask or expected structure;
- text orientation;
- expected line count or text height;
- low-contrast and background-removal hints;
- comparison mode;
- visible-light availability.

Use standard names already consumed by the response mapper, such as:

- `surname`, `givenNames`, `surnameAndGivenNames`;
- `dateOfBirth`, `dateOfIssue`, `dateOfExpiry`;
- `documentNumber`, `personalNumber`, `sex`, `nationality`;
- `address`, `addressStreet`, `addressCity`, `addressState`;
- `addressMunicipality`, `addressLocation`, `addressPostalCode`;
- `authority`, `placeOfBirth`, `height`.

If a new field name is required, add its internal mapping in
`src/identity_analysis/pipeline.py` and its public placement in
`src/identity_analysis/responses.py`. Do not expose duplicate internal forms.

### Graphic and Barcode Regions

Declare graphic regions for portraits, ghost portraits, child portraits, and
signatures. This enables:

- normalized `data.regions`;
- optional `includeImages`;
- optional portrait face-presence analysis;
- document-portrait to selfie comparison.

Declare barcode regions with an exact format only when the format is known.
Unknown or extended declarations should retain multiformat fallback.

### Field Assemblies

Use assembly rules when the printed document separates one public value into
multiple regions. Common examples are:

- surname plus given names;
- street plus city and postal code;
- document-number prefixes and suffixes.

Assemblies should preserve direct recognized values when available and derive a
combined value only from successfully recognized components.

### Build the Asset Bundle

Declarative layouts live in the generated classifier asset bundle. Extend the
source used by the catalog preparation workflow, rebuild:

- `assets/document_classifier/catalog.json`;
- `assets/document_classifier/visual-layouts.json.gz`;
- `assets/document_classifier/manifest.json`.

Do not edit compressed output without updating manifest hashes and counts.
Validate the finished bundle:

```bash
identity-assets validate assets
```

Then verify catalog lookup, layout evidence, and exact-layout OCR through REST.

## Path 3: Reuse a Machine-Readable Standard

No new visual profile is needed when the document follows an already supported
standard:

| Standard | Request profile | Main implementation |
|---|---|---|
| ICAO TD1 | `icao_td1` | `src/identity_analysis/pipeline.py` |
| ICAO TD2 | `icao_td2` | `src/identity_analysis/pipeline.py` |
| ICAO TD3 | `icao_td3` | `src/identity_analysis/pipeline.py` |
| ICAO MRV-A/MRV-B | `icao_mrv` | `src/identity_analysis/pipeline.py` |
| AAMVA PDF417 | `aamva_pdf417` | `src/identity_analysis/barcodes.py` |

Validate:

1. line or payload detection;
2. field parsing;
3. check digits or standard structural checks;
4. date normalization;
5. country and document-class mapping;
6. malformed-input rejection;
7. automatic routing;
8. compact REST serialization.

A document name or country label should come from parsed evidence or a validated
catalog candidate, not from an image filename.

## Path 4: Add a Dedicated Profile

Use a dedicated profile only when declarative OCR cannot express the required
behavior.

### 1. Implement Recognition

Add a focused recognizer in `src/identity_analysis/pipeline.py` or a separate
module. It should:

- accept a Pillow RGB image and the runtime asset root;
- use normalized geometry rather than source-pixel constants;
- preserve field confidence and source;
- normalize only documented formats;
- return `UNSUPPORTED` or an explicit failure when structural evidence is
  insufficient;
- avoid document-specific values in code.

Return the internal canonical keys consumed by `document_response()`, including
as applicable:

```text
DocumentName
dCountryName
surname
givenNames
dateOfBirth
documentNumber
address
addressLines
sex
nationality
authority
validityStatus
source
recognitionProfile
qualitySignals
```

### 2. Register the Profile

Update:

- `SUPPORTED_REQUEST_PROFILES` in `src/identity_analysis/pipeline.py`;
- routing in `process_document()`;
- `--profile` choices in `src/identity_analysis/cli.py`;
- implemented profiles in `src/identity_analysis/capabilities.py`;
- recognition-profile documentation in `docs/product/api.md`.

The REST API reads `SUPPORTED_REQUEST_PROFILES`, so no separate endpoint is
normally required.

### 3. Define Selection Evidence

Automatic selection must require more than OCR confidence. Use at least one
structural discriminator:

- valid MRZ checks;
- parsed barcode structure;
- unambiguous catalog identifier;
- country-specific fixed labels;
- field masks and expected lengths;
- a combination of required fields unique to the edition.

If the discriminator fails, return `UNSUPPORTED` rather than mapping the image
to the nearest known profile.

## Path 5: Add a New Barcode Payload

If decoding works but payload interpretation is missing:

1. add a parser in `src/identity_analysis/barcodes.py`;
2. validate signatures, lengths, versions, and checksums before mapping;
3. return normalized fields and retain the raw barcode only where the public
   contract allows it;
4. define source precedence against visual OCR and MRZ;
5. add malformed and truncated payload tests;
6. document the supported payload family.

Do not infer a country-specific schema from one example payload.

## Visual Region Tuning

For profile development, use the local band editor:

```bash
python3 -m http.server 8081 --directory tools
```

Open `http://localhost:8081/band_editor.html`, load the image, select or define
the layout, and inspect normalized regions. Original photographs are displayed
directly. For a rectified profile, select **Rectify from 4 corners** and click
top-left, top-right, bottom-right, and bottom-left. Text fields and
`documentFrontSide`, `portrait`, `ghostPortrait`, and `signature` regions can
all be moved, resized, imported, and exported. The editor is a development aid;
the committed layout metadata and regression tests remain authoritative.

## Tests to Add

### Unit Tests

Add focused tests for:

- field parsing and normalization;
- masks and invalid values;
- date and check-digit behavior;
- page-role or relation metadata;
- barcode payload validation;
- response-field mapping.

### Pipeline Regression

Add a consented or specimen fixture and assert:

```python
result = process_document(
    sample,
    assets,
    "auto_research",
    document_identifier,
)

assert result["recognitionProfile"] != "UNSUPPORTED"
assert result["DocumentName"] == "Expected Document"
assert result["documentNumber"] == "EXPECTED"
```

Also test a visually similar unrelated document and confirm it is not forced
into the new profile.

### REST Regression

Cover:

- multipart `image` input;
- raw base64 JSON;
- Data URL base64 JSON;
- exact identifier when applicable;
- compact response fields;
- `includeImages` and `analyzePortraits` when graphic regions exist;
- pair or page fusion for related layouts;
- invalid identifier and malformed-image errors.

### Performance Regression

Measure cold and warm requests. Compare automatic classification with exact
layout routing. Avoid adding unconditional OCR regions to every document.

## Documentation to Update

Every added document family or dedicated profile should update:

1. `README.md` only when product-level coverage changes;
2. `docs/product/api.md` for new profile or request behavior;
3. `docs/product/ocr-response.md` for new public fields;
4. `docs/product/architecture.md` for a new pipeline stage;
5. `src/identity_analysis/capabilities.py`;
7. this guide if the extension process changes.

## Definition of Done

A document is complete when:

- [ ] the edition has a stable identifier or explicit profile;
- [ ] exact-layout or explicit-profile recognition succeeds;
- [ ] automatic routing succeeds or the need for exact routing is documented;
- [ ] required fields match expected values;
- [ ] malformed and visually similar inputs do not produce false recognition;
- [ ] MRZ or barcode checks are enforced when available;
- [ ] front/back or page relationships are validated when applicable;
- [ ] portrait and signature regions are exposed when declared;
- [ ] REST file and base64 paths are tested;
- [ ] response fields remain compact and standardized;
- [ ] capability metadata is updated;
- [ ] product and research documentation is updated;
- [ ] the complete test suite passes;
- [ ] the Docker image builds and validates its assets.
