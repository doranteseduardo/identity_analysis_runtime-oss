"""Machine-readable coverage of the identity analysis runtime.

``SDK_COMPATIBILITY`` describes what this runtime implements, independently of
which models an operator has installed.  ``runtime_capabilities()`` reports what
is actually loadable from a given asset directory.
"""

from pathlib import Path

from .document_classifier import classifier_available, document_catalog_available
from .visual_layouts import layout_catalog_available


SDK_COMPATIBILITY = {
    "coverage": "partial",
    "documentScope": "multi_document",
    "modelSourcing": {
        "policy": "bring_your_own_onnx_models",
        "documentation": "docs/models.md",
        "bundledModelWeights": False
    },
    "engineLifecycle": {
        "documentOcr": {
            "status": "implemented",
            "endpoint": "/v1/ocr",
            "pairedEndpoint": "/v1/ocr/pair",
            "pagesEndpoint": "/v1/ocr/pages"
        },
        "faceLiveness": {
            "status": "implemented_with_validation_limitations",
            "endpoint": "/v1/face/liveness"
        },
        "faceRecognition": {
            "status": "implemented_with_validation_limitations",
            "endpoint": "/v1/face/compare",
            "documentPortraitEndpoint": "/v1/document/portrait/compare",
            "documentPortraitOptionalLiveness": True,
            "documentPortraitVerificationPolicy": {
                "pass": "same_person_and_real_and_document_capture_pass",
                "fail": "different_person_or_selfie_spoof",
                "review": "inconclusive_signal_or_document_capture_review",
                "notAvailable": "liveness_not_requested_or_unavailable",
                "authenticityDecision": False
            },
            "templateEndpoint": "/v1/face/template",
            "templateComparisonEndpoint": "/v1/face/template/compare"
        },
        "documentLiveness": {
            "status": "substrate_only_no_models_supplied",
            "endpoint": None,
            "implementedSubstrate": "src/identity_analysis/document_liveness.py"
        }
    },
    "implementedProfiles": [
        {
            "id": "MEX-INE-front-legacy",
            "documentFamily": "voter_credential",
            "country": "MEX",
            "source": "VISUAL"
        },
        {
            "id": "MEX-INE-front-modern",
            "documentFamily": "voter_credential",
            "country": "MEX",
            "source": "VISUAL"
        },
        {
            "id": "ICAO-TD1",
            "documentFamily": "identity_card",
            "country": "issuer_from_mrz",
            "source": "MRZ",
            "limitation": "Current crop profiles are calibrated with INE reverse samples."
        },
        {
            "id": "ICAO-TD2",
            "documentFamily": "identity_document_or_residence_permit",
            "country": "issuer_from_mrz",
            "source": "MRZ"
        },
        {
            "id": "ICAO-TD3",
            "documentFamily": "passport",
            "country": "issuer_from_mrz",
            "source": "MRZ"
        },
        {
            "id": "ICAO-MRV-A",
            "documentFamily": "visa",
            "country": "issuer_from_mrz",
            "source": "MRZ"
        },
        {
            "id": "ICAO-MRV-B",
            "documentFamily": "visa",
            "country": "issuer_from_mrz",
            "source": "MRZ"
        },
        {
            "id": "AAMVA-PDF417",
            "documentFamily": "driver_license",
            "country": "issuer_from_barcode",
            "source": "BARCODE"
        },
        {
            "id": "MEX-INE-QR",
            "documentFamily": "voter_credential",
            "country": "MEX",
            "source": "BARCODE",
            "additionalEvidence": "Code128"
        },
        {
            "id": "SWE-ID-2021-front",
            "documentFamily": "identity_card",
            "country": "SWE",
            "source": "VISUAL"
        }
    ],
    "observedFamiliesRequiringValidation": [
        "non_latin_visual_ocr_calibration",
        "country_specific_non_ine_barcode_payloads",
        "rfid_epassport"
    ],
    "visualClassification": {
        "mode": "onnx_document_classifier_with_structural_confirmation",
        "catalogHintGroups": [
            "capture",
            "recognition",
            "electronic_document",
            "source_references",
            "authenticity_configuration"
        ],
        "catalogHintSemantics": {
            "preserved": True,
            "endpoint": "/v1/document/layout/{documentIdentifier}/evidence",
            "restExposed": True,
            "outcomesInferred": False
        },
        "layoutMetadata": {
            "status": "implemented",
            "endpoint": "/v1/document/layout/{documentIdentifier}/evidence",
            "implemented": [
                "document_descriptor",
                "visible_text_regions",
                "visible_graphic_regions",
                "visible_barcode_regions",
                "compatible_page_relations",
                "declared_capture_and_recognition_requirements",
                "conservative_page_role_resolution"
            ],
            "limitation": "Page roles use explicit caption markers or directly related layouts; layouts without either marker resolve to unknown."
        },
        "pageRoleResolution": {
            "status": "implemented_with_unknown_fallback",
            "restExposed": True,
            "limitation": "This is catalog-role resolution after layout identification, not an independent image-side classifier."
        },
        "documentCatalog": {
            "status": "implemented",
            "endpoint": "/v1/document/catalog",
            "detailEndpoint": "/v1/document/catalog/{documentIdentifier}",
            "facetsEndpoint": "/v1/document/catalog/facets",
            "filters": [
                "free_text",
                "country_code",
                "document_type",
                "document_format",
                "deprecated"
            ],
            "pagination": True
        },
        "standaloneClassification": {
            "status": "implemented",
            "endpoint": "/v1/document/classify",
            "maximumCandidates": 25,
            "structuralDecision": False,
            "limitation": "Ranked template confidence does not establish document validity or authenticity."
        },
        "explicitLayoutRecognition": {
            "status": "implemented",
            "endpoint": "/v1/document/layout/{documentIdentifier}/ocr",
            "classifierInference": False,
            "specializedProfileRouting": True,
            "declarativeFallback": True,
            "selectiveFields": True,
            "maximumRequestedFields": 64,
            "limitation": "The caller-provided layout still requires OCR, MRZ, barcode, or profile-specific structural evidence."
        },
        "securityMetadataSemantics": {
            "preserved": [
                "normalized_geometry",
                "illumination_type",
                "element_class_and_type",
                "criticality",
                "threshold_and_filter_parameters",
                "reference_image_payload_and_hash"
            ],
            "endpoint": "/v1/document/layout/{documentIdentifier}/evidence",
            "restExposed": True,
            "decisionImplemented": False,
            "limitation": "Metadata, reference evidence, and experimental comparison are portable; calibration and authenticity scoring remain unimplemented."
        },
        "referencePatchMatching": {
            "status": "experimental_metric_only",
            "endpoint": "/v1/document/reference-metrics",
            "defaultLightTypes": [
                6,
                24
            ],
            "candidateCountPerPatch": 75,
            "scales": [
                0.9,
                1.0,
                1.1
            ],
            "offsetsInPatchWidths": [
                -0.2,
                -0.1,
                0.0,
                0.1,
                0.2
            ],
            "metrics": [
                "intensity_correlation",
                "gradient_magnitude_correlation",
                "mean_absolute_error_similarity"
            ],
            "calibratedThreshold": None,
            "restExposed": True
        },
        "assembledFieldSemantics": [
            "typed_source_references",
            "optional_locale_matching",
            "literal_separators",
            "missing_component_separator_collapse",
            "direct_value_precedence",
            "derived_field_provenance"
        ],
        "mrzGuidance": {
            "implemented": [
                "high_confidence_template_routing",
                "passport_td3_priority",
                "visa_mrv_a_or_b_priority",
                "explicit_ignore_flag",
                "numpy_projection_region_localization",
                "physical_width_and_height_candidate_priority",
                "complete_default_ratio_fallback",
                "conditional_localized_line_crops",
                "full_fallback_on_structural_failure"
            ],
            "limitation": "Declared physical geometry prioritizes additional projection candidates when a layout supplies it; all default candidates and ICAO checks remain mandatory."
        },
        "maskSemantics": {
            "implemented": [
                "fixed_alphanumeric_length",
                "fixed_digit_positions",
                "numeric_digit_tokens",
                "calendar_valid_numeric_dates",
                "numeric_date_component_widths_and_ranges",
                "calendar_valid_text_dates_across_18_languages",
                "four_digit_years",
                "sex_mf",
                "english_sex_words",
                "explicit_two_marker_sex_tokens",
                "three_letter_country_codes",
                "portable_document_country_code_membership",
                "usa_canada_australia_jurisdiction_codes",
                "aamva_eye_and_hair_color_codes",
                "usa_zip_and_zip_plus_four",
                "canadian_postal_codes",
                "feet_and_inches_height_ranges",
                "declared_imperial_and_metric_measures",
                "two_character_color_classes",
                "abo_rh_blood_group_shape",
                "unicode_word_and_name_shapes",
                "unicode_alphanumeric_word_shapes"
            ],
            "limitation": "External city, address, sex-word, and private lexical transforms remain uninterpreted."
        },
        "adaptiveFieldSearch": {
            "status": "implemented_with_exhaustive_fallback",
            "centerFirst": True,
            "singleLineConfidenceThreshold": 0.995,
            "paddingStopConfidenceThreshold": 0.995,
            "minimumPaddingVariants": 3,
            "validatedInferenceReductionMinimum": 0.15,
            "limitation": "Low-confidence and multiline fields retain all candidate windows; representative validation remains document-specific."
        },
        "orientedFieldRecognition": {
            "status": "implemented",
            "supportedOrientations": [
                0,
                90,
                180,
                270
            ],
            "orthogonalDirectionFallback": True,
            "highConfidencePrimaryStop": 0.995,
            "limitation": "Orientation-aware OCR requires representative validation for each rotated document edition."
        },
        "declaredFieldPreprocessing": {
            "status": "implemented_with_original_fallback",
            "adaptiveContrastPaddingRatio": 0.04,
            "originalCandidatePreserved": True,
            "limitation": "Portable flags guide conservative contrast normalization; they do not define a proprietary background-removal algorithm."
        },
        "barcodeFormatRouting": {
            "status": "implemented_with_multiformat_fallback",
            "exactFormats": [
                "Code128",
                "Code39",
                "EAN8",
                "Code93",
                "DataMatrix"
            ],
            "limitation": "Extended and composite code-type values remain numeric and use multiformat decoding."
        },
        "fieldMetadataSemantics": {
            "status": "preserved_without_guessed_decisions",
            "comparisonModes": [
                0,
                1,
                2,
                3
            ],
            "preserved": [
                "color_type",
                "font_layer",
                "text_layer",
                "comparison_mode",
                "low_contrast_flag",
                "background_removal_flag"
            ],
            "comparisonDecisionImplemented": False,
            "limitation": "Numeric comparison modes are preserved as evidence because no public declaration establishes their decision semantics."
        },
        "portraitFacePresence": {
            "status": "implemented_opt_in",
            "requestOption": "analyzePortraits",
            "endpointCoverage": [
                "/v1/ocr",
                "/v1/ocr/pair",
                "/v1/ocr/pages",
                "/v1/document/layout/{documentIdentifier}/ocr"
            ],
            "regionTypes": [
                "portrait",
                "ghostPortrait"
            ],
            "detectorThreshold": 0.2,
            "authenticityDecision": False,
            "livenessDecision": False,
            "limitation": "Face presence confirms detector evidence inside a declared region; it does not establish portrait identity, authenticity, or liveness."
        },
        "limitation": "Catalog-driven visual extraction requires an unambiguous classifier result and remains sample-validated per document edition."
    },
    "multiscriptOcr": {
        "status": "implemented_with_locale_routing",
        "canonicalModelLocales": [
            1032,
            1034,
            1037,
            1049,
            1050,
            1055,
            1061,
            1066,
            1067,
            1079
        ],
        "scripts": [
            "latin",
            "greek",
            "cyrillic",
            "hebrew",
            "armenian",
            "georgian"
        ],
        "implemented": [
            "official_lcid_group_routing",
            "lazy_session_loading",
            "ctc_charset_validation",
            "latin0_fallback",
            "regional_latin_ascii_comparison",
            "person_name_digit_penalty",
            "icao_579_entry_transliteration",
            "cross_script_pairing_comparison",
            "diacritic_insensitive_comparison"
        ],
        "limitation": "Locale coverage does not replace representative per-language document validation."
    },
    "documentPairing": {
        "status": "implemented_with_identity_validation",
        "endpoint": "/v1/ocr/pair",
        "implemented": [
            "parallel_side_processing",
            "cross_side_country_name_surname_birth_and_sex_checks",
            "catalog_parent_child_relations",
            "catalog_semantic_page_relations",
            "related_side_barcode_region_guidance",
            "visual_and_machine_readable_field_fusion",
            "combined_capture_risk_decision",
            "independent_exact_layout_routing"
        ],
        "limitation": "Related-side geometry is available broadly, but representative paired-image validation remains document-edition specific."
    },
    "multiPageDocuments": {
        "status": "implemented_with_identity_validation",
        "endpoint": "/v1/ocr/pages",
        "minimumPages": 2,
        "defaultMaximumPages": 10,
        "implemented": [
            "bounded_parallel_page_processing",
            "incremental_identity_validated_fusion",
            "mismatch_isolation",
            "per_page_profile_summary",
            "related_page_barcode_guidance",
            "conservative_page_quality_fusion",
            "independent_exact_layout_routing",
            "conservative_semantic_page_ordering",
            "original_input_position_provenance"
        ],
        "limitation": "Canonical ordering requires related layouts with distinct page markers or an unmarked primary-page form; ambiguous and unrelated collections preserve caller order."
    },
    "documentLivenessEngine": {
        "status": "substrate_only_no_models_supplied",
        "implementedSubstrate": [
            "config_catalog",
            "bbox_alignment",
            "resize_interpolation",
            "orientation_normalization",
            "pixel_normalization",
            "ghost_portrait_channel_stack"
        ],
        "missing": [
            "detection_models_and_anchors",
            "spoof_and_quality_models",
            "pipeline_fusion_and_calibration_configs",
            "reference_outputs_for_validation"
        ],
        "boundary": "Document-capture PAD signals are reported on their own terms and are not mapped onto any third-party product's output."
    },
    "facialAnalysis": {
        "endpoints": [
            "/v1/face/analyze",
            "/v1/face/liveness",
            "/v1/face/compare",
            "/v1/document/portrait/compare",
            "/v1/face/template",
            "/v1/face/template/compare"
        ],
        "implemented": [
            "face_detection",
            "68_point_landmarks",
            "six_point_geometric_head_pose",
            "portable_face_geometry_quality_codes",
            "seven_model_passive_liveness",
            "512_value_face_embedding",
            "face_template_comparison",
            "declared_document_portrait_to_selfie_comparison",
            "shared_selfie_detection_for_identity_and_liveness",
            "conservative_identity_verification_fusion",
            "document_capture_risk_fusion",
            "2048_byte_face_template_export",
            "persisted_face_template_comparison",
            "opt_in_templates_from_image_comparison"
        ],
        "unverifiedOutputs": [
            "native_quality_thresholds",
            "deepfake_sentinel"
        ],
        "limitations": [
            "Head pose is geometrically estimated and is not asserted to reproduce proprietary angle calibration.",
            "Small-face, cutoff-face, excessive-pose, covered-face, and multiple-face warnings use documented semantics with portable policy thresholds.",
            "Deepfake sentinel equivalence remains unavailable; passive PAD exposes its own spoof decision.",
            "Passive liveness has an exploratory real/spoof operating point but no independent calibration and holdout validation.",
            "The configured recognition threshold requires population-specific validation before deployment."
        ]
    },
    "biometricEvaluation": {
        "status": "implemented_offline_cli",
        "command": "identity-face-eval",
        "inputModes": [
            "score",
            "template",
            "image"
        ],
        "metrics": [
            "score_distributions",
            "roc_curve",
            "roc_auc",
            "equal_error_rate",
            "target_false_accept_rate_operating_points",
            "confusion_matrix_counts"
        ],
        "decisionRule": "same_person_when_score_greater_than_threshold",
        "automaticRuntimeThresholdChange": False,
        "limitation": "Reported metrics characterize only the supplied evaluation pairs and do not replace independent deployment validation."
    }
}


ASSET_FAMILIES = {
    "lineRecognition": ("models/ocr_latin.onnx", "charsets/latin.txt"),
    "documentRectification": ("models/document_corners.onnx",),
    "captureQuality": ("models/focus_device.onnx",),
    "documentCapturePad": ("models/electronic_device.onnx", "models/moire.onnx"),
    "faceDetection": ("facial/detector/face_detector.onnx",),
    "faceLandmarks": ("facial/landmarks/landmarks_quality.onnx",),
    "faceRecognition": ("facial/recognition/00_R_L_CF_V1_16GPUs/model.onnx",),
    "faceLiveness": ("facial/liveness/manifest.json",),
}


def runtime_capabilities(assets_path: Path) -> dict:
    """Report which optional model families are installed under ``assets_path``.

    Nothing here raises: an absent directory simply reports every family as
    unavailable, which is how a fresh checkout without models behaves.
    """

    root = Path(assets_path)
    features = {
        name: all((root / relative).exists() for relative in relatives)
        for name, relatives in ASSET_FAMILIES.items()
    }
    features["documentClassification"] = classifier_available(root)
    features["layoutCatalog"] = layout_catalog_available(root)
    features["documentCatalog"] = document_catalog_available(root)
    return {
        "assetsPath": str(root),
        "assetsPresent": root.is_dir(),
        "features": dict(sorted(features.items())),
        "documentation": "docs/models.md",
    }
