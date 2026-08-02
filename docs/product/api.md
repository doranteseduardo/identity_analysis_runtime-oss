# API Guide

## Base URL

Local and Docker examples use `http://localhost:8000`. Interactive OpenAPI
documentation is available at `/docs`.

There is no public hosted instance; run your own. With models installed under
`./assets` (see [`../models.md`](../models.md)):

```bash
IDENTITY_ANALYSIS_ASSETS=assets identity-api
# or
docker build -t identity-analysis .
docker run --rm -p 8000:8000 -v "$PWD/assets:/app/assets:ro" identity-analysis

export BASE_URL="http://localhost:8000"
```

Swagger UI is served at `$BASE_URL/docs` and the OpenAPI document at
`$BASE_URL/openapi.json`. The examples below use `BASE_URL`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Validate service and asset availability |
| `GET` | `/v1/capabilities` | Report engine coverage and limitations |
| `POST` | `/v1/ocr` | Process one identity-document image |
| `POST` | `/v1/ocr/pair` | Process and validate front/back document images |
| `POST` | `/v1/ocr/pages` | Process, order when declared, and validate a document-page collection |
| `GET` | `/v1/document/catalog` | Search supported document definitions |
| `GET` | `/v1/document/catalog/facets` | List catalog filter values and counts |
| `GET` | `/v1/document/catalog/{documentIdentifier}` | Read one catalog definition |
| `POST` | `/v1/document/classify` | Rank document templates without OCR |
| `POST` | `/v1/document/layout/{documentIdentifier}/ocr` | Recognize one exact document layout |
| `GET` | `/v1/document/layout/{documentIdentifier}/evidence` | Read security-region and reference-patch metadata for an exact layout |
| `POST` | `/v1/document/reference-metrics` | Compare visible regions with an exact catalog layout |
| `POST` | `/v1/face/analyze` | Detect faces and return landmarks |
| `POST` | `/v1/face/liveness` | Evaluate passive facial liveness |
| `POST` | `/v1/face/compare` | Compare faces from two images |
| `POST` | `/v1/document/portrait/compare` | Compare a declared document portrait with a selfie |
| `POST` | `/v1/face/template` | Extract a portable face template |
| `POST` | `/v1/face/template/compare` | Compare two portable face templates |

## Image Input

Single-image endpoints accept multipart form data with an `image` field or JSON
with `filename` and `imageBase64`. Base64 may be raw or use a Data URL prefix.
Comparison accepts `firstImage` and `secondImage`, or the JSON fields
`firstFilename`, `secondFilename`, `firstImageBase64`, and
`secondImageBase64`.

Multipart image names follow one semantic convention:

| Request shape | Multipart fields |
|---|---|
| Single image | `image` |
| Document front/back | `frontImage`, `backImage` |
| Document pages | repeated `images` |
| Face comparison | `firstImage`, `secondImage` |
| Document portrait comparison | `documentImage`, `selfieImage` |

Transport-oriented names such as `file`, `files`, `file1`, and `file2` are not
part of the contract. Requests using them return `400`.

Template extraction accepts the same single-image input as face analysis.
Template comparison accepts JSON fields `template1Base64` and
`template2Base64`, plus an optional `threshold`. Template base64 may be raw or
use a Data URL prefix.

Image comparison may set `includeTemplates` to `true` in multipart, JSON, or
the query string. The response then includes both templates already computed by
that comparison; no additional face detection or embedding inference runs.

Paired document processing accepts multipart fields `frontImage` and
`backImage`, plus an
optional `profile`. JSON requests use `frontImageBase64`, `backImageBase64`,
optional `frontFilename`, `backFilename`, and `profile`. Both base64 fields may
be raw or use Data URL prefixes. Callers may additionally provide independent
`frontDocumentIdentifier` and `backDocumentIdentifier` integers. Either side may
remain automatic. Per-side identifiers require the default automatic profile;
unknown identifiers return `404`.

Multi-page processing accepts repeated multipart fields named `images`, plus an
optional `profile`. JSON requests use an `images` array whose items contain
`imageBase64` and an optional `filename`. Requests require at least two pages;
the maximum defaults to ten and is configured with
`IDENTITY_ANALYSIS_MAX_DOCUMENT_PAGES`. Multipart may repeat
`documentIdentifiers` with exactly one value per file; an empty value leaves
that page automatic. JSON image items may include an optional integer
`documentIdentifier`. Exact page routing requires the default automatic profile,
and unknown identifiers return `404`.

OCR requests may set `includeImages` to `true`. Multipart requests use the text
field `includeImages`, JSON requests use a boolean property, and either format
may use the `?includeImages=true` query parameter. A body or form value takes
precedence over the query parameter. The default is
`false`. When enabled, the service crops available portrait, child portrait,
ghost portrait, and signature regions from their original uploaded page and
returns them under `data.images`.
Recognized visual documents can return catalog image regions when recognition
and classification independently agree on country and document family. Clients
do not need to select a profile to receive images. Candidate geometry must also
meet the service's confidence, issuer, family, and document-format checks.

OCR requests may independently set `analyzePortraits` to `true` as a multipart
field, JSON boolean, or query parameter. The option runs face detection only
inside returned portrait regions whose layout declares that a face is expected.
Each analyzed region reports `facePresence.expected`, `detected`, `count`,
`threshold`, `confidence`, and `status`. This structural presence check does not alter
authenticity, spoofing, liveness, or identity decisions.

Document portrait comparison accepts multipart fields `documentImage` and
`selfieImage`.
JSON requests use `documentImageBase64` and `selfieImageBase64`; base64 values
may include a Data URL prefix. Optional inputs are `profile`,
`documentIdentifier`, `threshold`, `analyzeLiveness`, `livenessThreshold`, and
`livenessSpoofThreshold`. An exact identifier requires the default automatic profile
and skips document classification. The response contains the comparison
decision and score, document profile and layout identifier, declared portrait
box, face detection inside that crop, and the selfie face box. When
`analyzeLiveness` is enabled, `selfie.liveness` independently reports its
decision, score, threshold, head pose, and quality. `verification.decision` is
`pass` only when portrait comparison is `same_person`, selfie liveness is
`real`, and document capture risk is `pass`; it is `fail` for
`different_person` or selfie `spoof`, `review` for an inconclusive signal or
document capture `review`, and `not_available` when liveness was not requested
or produced no decision.

Face liveness accepts optional `threshold` and `spoofThreshold` values.
Defaults are `0.37` and `0.25`, respectively, and callers must keep
`spoofThreshold <= threshold`. Scores above `threshold` are `real`, scores at
or below `spoofThreshold` are `spoof`, and intermediate scores are `review`.
When a caller lowers `threshold` below the default spoof boundary without
providing `spoofThreshold`, the spoof boundary is lowered to match it. This
preserves legacy binary requests such as `threshold: 0`.
Document portrait comparison uses the equivalent `livenessThreshold` and
`livenessSpoofThreshold` names.

Supported file types are JPEG, PNG, HEIC, and HEIF. The decoded request limit is
configured with `IDENTITY_ANALYSIS_MAX_UPLOAD_BYTES`.

Standalone classification accepts the standard single-image formats.
Multipart requests may include `topK`; JSON requests use the numeric `topK`
property. The default is 5 and the maximum is 25. The response returns ranked,
normalized catalog candidates with confidence and layout availability.
Classification performs no OCR, barcode decoding, validity check, or
authenticity decision.

Exact-layout recognition accepts the standard single-image request and may use
`includeImages`. It may also request up to 64 exact catalog field names.
Multipart requests repeat `fields` or provide comma-separated names; JSON uses
a string array. Unknown names return `422`. Assembly dependencies are resolved
internally, while the response contains only requested fields. The path
identifier selects the catalog layout directly, so no classification inference
or template-selection threshold runs. Known dedicated families use their
validated extractor when no field subset is requested; other layouts use their
declarative fields, barcode regions, and MRZ guidance. The identifier is a
routing declaration, not a validity assertion: the selected path must still
produce structural or OCR evidence. The response reports the selected
identifier and requested names under `data.document`.

Layouts declaring orientation `90`, `180`, or `270` rotate text crops before
OCR. Orthogonal layouts try the declared correction first and use the opposite
direction unless the primary result is both mask-compatible and recognized with
very high confidence. Region coordinates remain normalized in the uploaded
document space; orientation correction affects text recognition, not returned
boxes.

For Mexican voter cards, `validation.fields` reports whether identity fields
have independent corroboration. A birth date is returned only when the visual
date, CURP birth segment, and elector-key birth segment form a valid consensus.
CURP and elector key must agree exactly; the printed date may differ by at most
one digit and supplies only the century. Malformed four-digit section or
registration values are omitted. Holder names, CURP, and elector key remain
`review` when they come from a single visual OCR path; format validity alone is
not represented as identity verification.

Fields declared as low-contrast or background-removal candidates retain the
original crop and may add a grayscale autocontrast fallback. This guidance is
applied automatically by exact and classified declarative layout OCR.

Barcode regions with an exact standard format declaration restrict decoding to
that format. Extended or composite declarations remain multiformat so an
unknown numeric hint cannot suppress a valid symbol.

Reference metrics accept the same single-image formats plus a required integer
`documentIdentifier`. Multipart uses `file` and `documentIdentifier`; JSON uses
`imageBase64`, optional `filename`, and `documentIdentifier`. The identifier
must select an exact catalog layout. The response reports per-patch intensity,
gradient, and absolute-error similarity metrics. Its `decision` is always
`null`: these measurements are not a document-authenticity verdict and no
pass/fail threshold is defined.

Layout evidence accepts an exact integer `documentIdentifier` in the URL. It
returns the document descriptor, physical dimensions, compatible-page
relations, and normalized text, graphic, barcode, and security-region geometry.
Text regions include declared locale, mask, text height, color type, font and
text layers, contrast and background-removal flags, and the numeric comparison
mode. A non-zero mode is exposed as `usedForComparison`, but it is not converted
into a validation decision. The endpoint also returns reference-patch metadata,
capture-light requirements, OCR and MRZ tolerances, electronic-document flags,
source references, and authenticity-check parameters using stable response names.
Encoded reference images are intentionally omitted. Numeric masks and type
identifiers are preserved without assigning undocumented labels. Page relations
identify compatibility but do not declare capture order by themselves; the
multi-page pipeline combines them with explicit page markers only when that
produces an unambiguous order.
`authenticityDecision` is always `null`: the endpoint reports declared evidence
locations and parameters, not a pass/fail result.

Document catalog search accepts optional `q`, `countryCode`, `documentType`,
`documentFormat`, and `includeDeprecated` query parameters. Results are
alphabetically stable and use `offset` plus `limit` pagination; `limit` defaults
to 50 and cannot exceed 100. Each item includes normalized document metadata,
MRZ and barcode availability, exact-layout availability, page-role evidence,
and the corresponding layout-evidence path. Search reads metadata only and does
not invoke document classification or OCR.

Catalog facets return current document counts grouped by country code, document
type, and document format. Set `includeDeprecated=true` to include obsolete
definitions in both facet counts and regular search. Direct identifier lookup
returns the same item schema used by search and responds with `404` for an
unknown identifier.

## Recognition Profiles

Request profiles include `mex_ine`, `mex_passport`, `icao_td1`, `icao_td2`, `icao_td3`,
`aamva_pdf417`, `icao_mrv`, `swe_id_2021`, and `auto_research`. Automatic mode selects validated MRZ and
barcode profiles or uses portable document-template classification to prioritize
a visual profile with sufficient structural evidence. When no dedicated profile
matches, an unambiguous exact-template result may activate catalog-driven visual
field extraction. Classification never replaces profile validation. It
returns an `UNSUPPORTED` result instead of forcing an unrelated visual template
when no supported profile matches. `auto_research` is the default when the
request omits `profile`; `mex_ine` must be requested explicitly.

`mex_passport` performs strict TD3 recognition, requires all ICAO check digits
to pass, and rejects a structurally valid passport whose issuing state is not
`MEX`. For uncropped booklet photographs, document rectification runs before
an adaptive lower-page MRZ band search. `auto_research` uses the same structural
fallback when visual catalog classification is weak or incorrect.

After a Mexican TD3 passes all structural checks, the pipeline selects between
the supported 2021 visual alignments and supplements the MRZ with printed
nationality, personal number, place of birth, issue date, and folio number.
High-confidence printed names also normalize the unchecked TD3 name line;
the check-digit-protected second line remains unchanged.
When `includeImages=true`, this profile returns `documentFrontSide`,
`ghostPortrait`, `portrait`, and `signature` crops in the standard
`data.images` collection.

When the caller already knows a supported family, supplying its explicit
`profile` avoids incompatible recognition routes and reduces latency. This is a
routing hint, not a substitute for structural validation: the selected parser
must still validate the expected document format. Classification uses one model
call rather than iterating through every catalog document; layouts with many
requested text fields may still require several seconds of OCR work.

The `mex_ine` profile retains QR and Code128 decoding because those
machine-readable sources are part of the supported voter-card pipeline. It
skips incompatible document families while preserving the fast barcode path
used by automatic routing.

TD3 passport responses require two 44-character lines and valid document,
birth-date, expiry-date, personal-number, and composite checks. AAMVA responses
use `data.document.source: BARCODE` and map supported designators into the
standard holder, address, date, and identifier groups. Low-resolution or
blurred PDF417 symbols may remain undecodable even when visually identifiable.

TD2 identity-document and residence-permit responses require two 36-character
lines plus valid document-number, birth-date, expiry-date, and composite checks.

Automatic mode also reads printable QR and Code 128 evidence. Supported INE
verification QR values are returned under `data.machineReadable.verification`,
while the correlated Code 128 value appears under
`data.machineReadable.barcodes`. A valid MRZ remains the primary profile when
both sources are available.

Visa responses support ICAO MRV-A (two 44-character lines) and MRV-B (two
36-character lines). Both require valid document-number, birth-date, and
expiry-date checks. Optional MRZ data is returned under
`data.identifiers.optionalData` when present.

## Decisions

- Document `validityStatus` represents available structural validation only.
- Document `spoofingDecision` represents capture and recapture risk.
- Facial liveness returns `real` or `spoof` against a configurable threshold,
  or `review` when input geometry is unsuitable for a reliable PAD decision.
- Face comparison returns `same_person` or `different_person` against a
  configurable threshold.
- Facial analysis and liveness return yaw, pitch, and roll estimates plus
  quality warnings for small, cutoff, excessively turned, covered, or multiple
  faces.

Thresholds are application policy. Validate them with representative data and
monitor score distributions after deployment.

## Response Shape

Every endpoint returns `{"status": "ok", "data": {...}}` on success and
`{"status": "error", "error": {...}}` on failure. Product responses omit empty
fields and internal inference diagnostics. See
[`ocr-response.md`](ocr-response.md) for all endpoint contracts.

When automatic document classification produces a named candidate,
`data.document.classification` contains its name, country, country code, type,
format, edition, series, jurisdiction codes, template issuance window,
deprecation flag, and confidence when those values are available. This
identifies the likely catalog template; `recognized` remains false until a
machine-readable parser, dedicated profile, or declarative layout returns
identity-bearing evidence. Template issuance dates do not determine whether the
presented credential is currently valid.

`data.document.pageRole` reports `front`, `back`, `numbered_page`,
`primary_page`, `front_cover`, `back_cover`, or `unknown` when a recognized
layout provides enough catalog evidence. `confidence: declared` identifies an
explicit caption marker; `confidence: inferred` identifies a direct relation
to a declared back or numbered page. This field resolves the role of an
identified layout and is not a separate image-side classification score.

Catalog-driven results may include `data.regions.portrait`,
`data.regions.ghostPortrait`, `data.regions.portraitOfChild`, and
`data.regions.signature`. Each region contains a normalized
`[left, top, right, bottom]` box and its coordinate space. When document
rectification runs, REST boxes are mapped back to `original_image` coordinates.

When `includeImages` is enabled, every `data.images` item contains `type`,
`side`, `mediaType`, raw `imageBase64`, `width`, and `height`. `side` is
`document`, `front`, `back`, or `page_N`, depending on the endpoint and region
provenance. Returned base64 has no Data URL prefix; use `mediaType` when building
one. Crops are JPEG-encoded at quality 90. The option increases response size
and returns biometric content, so callers should enable it only when required
and apply appropriate retention and access controls.

`POST /v1/ocr/pair` processes both sides concurrently and fuses them only after
available country, name, surname, birth-date, and sex values are compared. A
single contradiction produces `pairing.decision: mismatch` and prevents fields
from the second image being merged into the first. Insufficient comparable
identity evidence produces `review`.

For Mexican voter-card reverses, pair processing prioritizes rotation-aware
TD1 MRZ recognition. A candidate contributes identity data only when the
issuing state is `MEX` and all four ICAO check groups pass. QR and Code 128 are
decoded afterward as secondary machine-readable evidence; QR presence alone
cannot produce `matched`.

Pair responses expose an optional `layoutIdentifier` for each side. Exact-side
routing skips classification independently while retaining catalog relation
lookup, guided barcode regions, cross-side checks, and the normal pair decision.

Multi-page item summaries likewise expose `layoutIdentifier` when supplied.
Exact and automatic pages may be mixed. When recognized layouts are directly
related and expose distinct page markers, or one is an unmarked primary-page
form, the runtime orders them canonically before bounded parallel-processing
results are fused. Ambiguous and unrelated collections preserve caller order.
Name comparisons include normalized diacritic and ICAO transliteration variants
when available. Returned holder values retain their recognized script; matching
does not replace native text with transliterated text.
When related reverse geometry yields additional validated barcode payloads,
`data.pairing.relatedSideBarcodeCount` reports their deduplicated count.
`data.pairing.relationType` is `paired_page` when the catalog explicitly marks
the expected page relationship, or `related_document` when only a broader
front/back or variant relationship is available.

`POST /v1/ocr/pages` treats the first canonically ordered image as the primary
page. Every additional page is independently compared with the current
aggregate. A `matched` page may contribute fields; `mismatch` and `review`
pages remain in the page summary but cannot modify the aggregate. The final
response reports `data.pages.pageCount`, `matchedPageCount`, comparisons, page
profiles, ordering provenance, and the overall collection decision. Each item
includes `inputPage`, its one-based position in the submitted collection.

## Document Name Fields

Supported visual document profiles return holder names under `data.holder`.

Address recognition is available under `data.address` as a combined value and
ordered lines, plus locality components when present.

Generic MRZ processing returns the surname block defined by the travel-document
format. It does not split that block into first and second surnames because the
MRZ does not encode that distinction reliably.

MRZ responses include the normalized code and check results under
`data.machineReadable`. OCR confidence and crop diagnostics remain CLI-only.

## Errors

| Status | Meaning |
|---:|---|
| `400` | Malformed image, base64 payload, or boolean option |
| `413` | Decoded payload exceeds the configured limit |
| `415` | Unsupported content type or image extension |
| `422` | Invalid request fields, profile, or threshold |
| `503` | Required runtime assets are unavailable |

Consult `/docs` for the generated schema and current request definitions.

## Curl Cookbook

These examples use `jq` for readable JSON output and
`--fail-with-body` so automation fails on non-2xx responses.

### Service and Catalog

```bash
curl --fail-with-body "$BASE_URL/health" | jq
curl --fail-with-body "$BASE_URL/v1/capabilities" | jq

curl --fail-with-body --get "$BASE_URL/v1/document/catalog" \
  --data-urlencode "q=passport" \
  --data-urlencode "countryCode=MEX" \
  --data-urlencode "documentType=Passport" \
  --data-urlencode "documentFormat=ID3" \
  --data-urlencode "includeDeprecated=false" \
  --data-urlencode "offset=0" \
  --data-urlencode "limit=25" | jq

curl --fail-with-body \
  "$BASE_URL/v1/document/catalog/facets?includeDeprecated=false" | jq

DOCUMENT_ID="-1084765647"
curl --fail-with-body \
  "$BASE_URL/v1/document/catalog/$DOCUMENT_ID" | jq
```

### Single Document OCR

```bash
curl --fail-with-body "$BASE_URL/v1/ocr" \
  -F "image=@document.jpg" \
  -F "profile=auto_research" \
  -F "includeImages=false" \
  -F "analyzePortraits=false" | jq
```

Mexican passport with all declared image crops:

```bash
curl --fail-with-body "$BASE_URL/v1/ocr" \
  -F "image=@passport.jpg" \
  -F "profile=mex_passport" \
  -F "includeImages=true" \
  -F "analyzePortraits=true" | jq
```

JSON base64:

```bash
IMAGE_BASE64="$(base64 < document.jpg | tr -d '\n')"

jq -n --arg image "$IMAGE_BASE64" '{
  filename: "document.jpg",
  imageBase64: $image,
  profile: "auto_research",
  includeImages: false,
  analyzePortraits: false
}' |
curl --fail-with-body "$BASE_URL/v1/ocr" \
  -H "Content-Type: application/json" \
  --data-binary @- | jq
```

### Pair and Pages

```bash
curl --fail-with-body "$BASE_URL/v1/ocr/pair" \
  -F "frontImage=@front.jpg" \
  -F "backImage=@back.jpg" \
  -F "profile=auto_research" \
  -F "includeImages=false" | jq

curl --fail-with-body "$BASE_URL/v1/ocr/pages" \
  -F "images=@page-1.jpg" \
  -F "images=@page-2.jpg" \
  -F "images=@page-3.jpg" \
  -F "profile=auto_research" \
  -F "includeImages=false" | jq
```

Exact identifiers can be supplied with `frontDocumentIdentifier` and
`backDocumentIdentifier` for pair requests, or one repeated
`documentIdentifiers` field per page. Exact identifiers require
`profile=auto_research`.

### Classification and Exact Layout

```bash
curl --fail-with-body "$BASE_URL/v1/document/classify" \
  -F "image=@document.jpg" \
  -F "topK=5" | jq

curl --fail-with-body \
  "$BASE_URL/v1/document/layout/$DOCUMENT_ID/evidence" | jq

curl --fail-with-body \
  "$BASE_URL/v1/document/layout/$DOCUMENT_ID/ocr" \
  -F "image=@document.jpg" \
  -F "fields=surname" \
  -F "fields=givenNames" \
  -F "fields=documentNumber" \
  -F "includeImages=true" | jq

curl --fail-with-body "$BASE_URL/v1/document/reference-metrics" \
  -F "image=@document.jpg" \
  -F "documentIdentifier=$DOCUMENT_ID" | jq
```

### Face Analysis and Liveness

```bash
curl --fail-with-body "$BASE_URL/v1/face/analyze" \
  -F "image=@selfie.jpg" | jq

curl --fail-with-body "$BASE_URL/v1/face/liveness" \
  -F "image=@selfie.jpg" \
  -F "threshold=0.37" \
  -F "spoofThreshold=0.25" | jq
```

### Face and Document Comparison

```bash
curl --fail-with-body "$BASE_URL/v1/face/compare" \
  -F "firstImage=@selfie-1.jpg" \
  -F "secondImage=@selfie-2.jpg" \
  -F "threshold=0.67" \
  -F "includeTemplates=true" | jq

curl --fail-with-body "$BASE_URL/v1/document/portrait/compare" \
  -F "documentImage=@document.jpg" \
  -F "selfieImage=@selfie.jpg" \
  -F "profile=auto_research" \
  -F "threshold=0.67" \
  -F "analyzeLiveness=true" \
  -F "livenessThreshold=0.37" \
  -F "livenessSpoofThreshold=0.25" | jq
```

### Face Templates

```bash
curl --fail-with-body "$BASE_URL/v1/face/template" \
  -F "image=@selfie.jpg" | tee template.json | jq

jq -r '.data.templateBase64' template.json > template.b64

TEMPLATE_1="$(cat template-1.b64)"
TEMPLATE_2="$(cat template-2.b64)"

jq -n --arg first "$TEMPLATE_1" --arg second "$TEMPLATE_2" '{
  template1Base64: $first,
  template2Base64: $second,
  threshold: 0.67
}' |
curl --fail-with-body "$BASE_URL/v1/face/template/compare" \
  -H "Content-Type: application/json" \
  --data-binary @- | jq
```

### Decode Returned Images

```bash
curl --fail-with-body "$BASE_URL/v1/ocr" \
  -F "image=@document.jpg" \
  -F "includeImages=true" |
jq -r '.data.images[0].imageBase64' |
base64 --decode > extracted-image.jpg
```

On macOS, `base64 -D` may be used instead of `base64 --decode`.
