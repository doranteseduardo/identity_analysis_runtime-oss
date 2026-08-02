"""Shared test configuration.

This repository ships no model weights.  Tests that need real ONNX inference
are skipped unless an asset directory is available, either at ``./assets`` or
wherever ``IDENTITY_ANALYSIS_ASSETS`` points; see ``docs/models.md``.  Every
other test — pure logic, response shaping, and everything driven by the
synthetic catalog in ``tests/fixtures/catalog`` — runs anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Keep API start-up from eagerly loading models the test host may not have.
os.environ.setdefault("IDENTITY_ANALYSIS_WARMUP", "false")

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "samples"
FIXTURE_CATALOG = ROOT / "tests" / "fixtures" / "catalog"

SELFIE = SAMPLES / "synthetic_selfie.jpg"
ID_FRONT = SAMPLES / "synthetic_id_front.jpg"
ID_BACK = SAMPLES / "synthetic_id_back.jpg"
PASSPORT_1 = SAMPLES / "synthetic_passport_1.jpg"
PASSPORT_2 = SAMPLES / "synthetic_passport_2.jpg"

FRONT_LAYOUT = 900000001
BACK_LAYOUT = 900000002

ASSETS = Path(os.environ.get("IDENTITY_ANALYSIS_ASSETS") or ROOT / "assets")
ASSETS_AVAILABLE = (ASSETS / "manifest.json").is_file()

SKIP_REASON = (
    "requires a local ONNX model directory; point IDENTITY_ANALYSIS_ASSETS at one "
    "(see docs/models.md)"
)
requires_assets = pytest.mark.skipif(not ASSETS_AVAILABLE, reason=SKIP_REASON)


@pytest.fixture(scope="session")
def assets() -> Path:
    """The operator-supplied model directory."""

    if not ASSETS_AVAILABLE:
        pytest.skip(SKIP_REASON)
    return ASSETS


@pytest.fixture(scope="session")
def catalog_assets(tmp_path_factory) -> Path:
    """Operator models combined with this repository's synthetic catalog.

    The recognition models come from the operator's directory; the document
    catalog and visual layouts come from ``tests/fixtures/catalog``, so the
    catalog-driven routes are exercised against data this project authored.
    """

    if not ASSETS_AVAILABLE:
        pytest.skip(SKIP_REASON)
    root = tmp_path_factory.mktemp("catalog-assets")
    for entry in ASSETS.iterdir():
        if entry.name == "document_classifier":
            continue
        (root / entry.name).symlink_to(entry)
    (root / "document_classifier").symlink_to(
        FIXTURE_CATALOG / "document_classifier", target_is_directory=True
    )
    return root


@pytest.fixture
def catalog_api(monkeypatch, catalog_assets: Path) -> Path:
    """Point the HTTP application at the combined model/catalog directory."""

    import identity_analysis.api as api_module

    monkeypatch.setattr(api_module, "ASSETS_PATH", catalog_assets)
    return catalog_assets


@pytest.fixture
def catalog_only_api(monkeypatch) -> Path:
    """Point the HTTP application at the synthetic catalog alone.

    Catalog metadata routes need no ONNX model at all, so these stay runnable
    on a machine with no assets installed.
    """

    import identity_analysis.api as api_module

    monkeypatch.setattr(api_module, "ASSETS_PATH", FIXTURE_CATALOG)
    return FIXTURE_CATALOG
