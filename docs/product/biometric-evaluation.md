# Biometric Evaluation

## Purpose

`identity-face-eval` evaluates one-to-one face-comparison scores and selects
candidate operating thresholds. It reports:

- genuine and impostor score distributions;
- receiver operating characteristic curve;
- area under the ROC curve;
- equal error rate and its candidate threshold;
- operating points at requested false-accept-rate targets;
- confusion-matrix counts for every evaluated threshold.

The tool uses the same decision rule as the runtime:

```text
same_person when score > threshold
```

Evaluation is an offline deployment activity. It is not exposed as a REST
endpoint and does not modify the runtime threshold automatically.

## Input Formats

Input is UTF-8 CSV. Every row requires `samePerson`. Accepted true labels are
`true`, `1`, `yes`, `same`, `same_person`, and `genuine`; accepted false labels
include `false`, `0`, `no`, `different`, `different_person`, and `impostor`.

An optional `pairIdentifier` is accepted in all modes.

### Existing Scores

```csv
pairIdentifier,samePerson,score
genuine-001,true,0.84
impostor-001,false,0.43
```

Run:

```bash
identity-face-eval evaluation.csv \
  --target-far 0.001 \
  --target-far 0.01 \
  --output report.json
```

Scores must be finite values between zero and one.

### Persisted Templates

```csv
pairIdentifier,samePerson,template1Base64,template2Base64
genuine-001,true,BASE64_TEMPLATE_A,BASE64_TEMPLATE_B
```

Templates use the same 2,048-byte little-endian `float32` format returned by
`POST /v1/face/template`. The evaluator calculates the normal runtime
comparison score before building metrics.

### Image Pairs

```csv
pairIdentifier,samePerson,image1,image2
genuine-001,true,images/person-1-a.jpg,images/person-1-b.jpg
impostor-001,false,images/person-1-a.jpg,images/person-2-a.jpg
```

Relative paths resolve from the CSV directory. Image mode requires runtime
assets:

```bash
identity-face-eval evaluation.csv \
  --assets assets \
  --target-far 0.001 \
  --output report.json
```

Embeddings are cached by image path within one evaluation run, so repeated
images are inferred once.

## Report

The JSON report includes:

| Field | Meaning |
|---|---|
| `inputMode` | `score`, `template`, or `image` |
| `recordCount` | Total comparison pairs |
| `genuineCount` | Same-person comparisons |
| `impostorCount` | Different-person comparisons |
| `distributions` | Minimum, maximum, mean, and median by class |
| `rocAuc` | Trapezoidal area under the full ROC curve |
| `equalErrorRate` | Point where FAR and FRR are closest |
| `equalErrorThreshold` | Candidate threshold at that point |
| `operatingPoints` | Best observed TAR that does not exceed each requested FAR |
| `curve` | Threshold, rates, and confusion counts for every score boundary |

The evaluator checks every threshold interval that can change a decision. It
does not sample a fixed grid, so reported operating points are exact for the
provided scores.

## Selecting a Threshold

1. Define the maximum false-accept rate for the deployment.
2. Use a representative, consented dataset from the intended population and
   capture conditions.
3. Split threshold selection and final validation into independent datasets.
4. Select the reported operating point for the target FAR.
5. Confirm its false-reject rate and confidence bounds are acceptable.
6. Re-evaluate after camera, model, preprocessing, or population changes.

The target-FAR operating point maximizes observed true-accept rate while
remaining at or below the requested observed FAR. If no false accepts are
observed, the result does not prove the real-world FAR is zero.

## Statistical Limits

- AUC and EER summarize the provided pairs, not all deployment conditions.
- Repeated images or identities can make pair counts look larger than the
  independent sample size.
- Demographic, device, lighting, pose, age, and image-quality subgroups should
  be evaluated separately.
- Very low target FARs require enough independent impostor comparisons to
  measure rare errors.
- Threshold calibration does not evaluate presentation-attack detection.

