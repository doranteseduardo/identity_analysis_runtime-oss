from PIL import Image

import identity_analysis.visual_layouts as visual_layouts
from identity_analysis.ocr import (
    load_runtime,
    ocr_locale_catalog,
    resolve_ocr_locale,
    run,
)


from conftest import ASSETS, requires_assets


pytestmark = requires_assets


def test_official_locale_groups_select_portable_models() -> None:
    assert resolve_ocr_locale(ASSETS, 1032) == 1032
    assert resolve_ocr_locale(ASSETS, 2058) == 1034
    assert resolve_ocr_locale(ASSETS, 1058) == 1049
    assert resolve_ocr_locale(ASSETS, 1045) == 1050
    assert resolve_ocr_locale(ASSETS, 1068) == 1055
    assert resolve_ocr_locale(ASSETS, 1062) == 1061
    assert resolve_ocr_locale(ASSETS, 1046) == 1066
    assert resolve_ocr_locale(ASSETS, 1067) == 1067
    assert resolve_ocr_locale(ASSETS, 1079) == 1079
    assert resolve_ocr_locale(ASSETS, 1033) == 0
    assert resolve_ocr_locale(ASSETS, None) == 0


def test_every_script_model_matches_its_charset_classes() -> None:
    catalog = ocr_locale_catalog(ASSETS)

    assert len(catalog["models"]) == 10
    for locale in sorted(catalog["models"]):
        session, charset = load_runtime(ASSETS, locale)
        assert session.get_outputs()[0].shape[-1] == len(charset) + 1


def test_runtime_reports_selected_and_fallback_locale() -> None:
    image = Image.new("L", (160, 48), "white")

    cyrillic = run(ASSETS, image, False, "minus-one-one", locale=1058)
    fallback = run(ASSETS, image, False, "minus-one-one", locale=1033)

    assert cyrillic["requestedLocale"] == 1058
    assert cyrillic["modelLocale"] == 1049
    assert fallback["requestedLocale"] == 1033
    assert fallback["modelLocale"] == 0


def test_declarative_field_passes_lcid_to_line_recognizer(monkeypatch) -> None:
    observed = []

    def recognize(resource, image, invert, normalization, locale=None):
        observed.append(locale)
        return {"text": "ТЕСТ", "confidence": 0.9}

    monkeypatch.setattr(visual_layouts, "recognize_line", recognize)
    definition = {
        "bounds": [0.1, 0.1, 0.9, 0.2],
        "textHeight": 0.08,
        "mask": "{TEXT}",
        "lcid": 1049,
    }

    result = visual_layouts.recognize_field(
        ASSETS, Image.new("RGB", (800, 500), "white"), definition
    )

    assert result["text"] == "ТЕСТ"
    assert observed and set(observed) == {1049}
