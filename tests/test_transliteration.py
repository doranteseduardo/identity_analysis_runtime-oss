from identity_analysis.pipeline import compatible_identity_text, fuse_document_sides
from identity_analysis.transliteration import (
    icao_table,
    icao_transliterate,
    identity_variants,
)


from conftest import ASSETS, requires_assets


pytestmark = requires_assets


def test_icao_table_transliterates_supported_scripts() -> None:
    assert len(icao_table(ASSETS)) == 579
    assert icao_transliterate("ИВАНОВ", ASSETS) == "IVANOV"
    assert icao_transliterate("ΠΑΠΑΔΟΠΟΥΛΟΣ", ASSETS) == "PAPADOPOULOS"


def test_identity_variants_include_icao_and_diacritic_forms() -> None:
    assert "IVANOV" in identity_variants("ИВАНОВ", ASSETS)
    assert "PAPADOPOULOS" in identity_variants("ΠΑΠΑΔΟΠΟΥΛΟΣ", ASSETS)
    assert "MULLER" in identity_variants("MÜLLER", ASSETS)


def test_cross_script_identity_comparison_uses_portable_table() -> None:
    assert compatible_identity_text("ИВАНОВ", "IVANOV", ASSETS) is True
    assert compatible_identity_text("ΠΑΠΑΔΟΠΟΥΛΟΣ", "PAPADOPOULOS", ASSETS) is True
    assert compatible_identity_text("MÜLLER", "MULLER", ASSETS) is True
    assert compatible_identity_text("ИВАНОВ", "PETROV", ASSETS) is False


def test_pairing_fuses_native_and_transliterated_names() -> None:
    front = {
        "recognitionProfile": "VISUAL",
        "source": "VISUAL",
        "surname": "ИВАНОВ",
        "givenNames": "ИВАН",
        "name": "ИВАНОВ ИВАН",
        "dateOfBirth": "1990-01-01",
        "fieldList": [],
    }
    back = {
        "recognitionProfile": "ICAO-TD3",
        "source": "MRZ",
        "surname": "IVANOV",
        "givenNames": "IVAN",
        "name": "IVANOV IVAN",
        "dateOfBirth": "1990-01-01",
        "documentNumber": "A1234567",
        "fieldList": [],
    }

    result = fuse_document_sides(front, back, ASSETS)

    assert result["pairing"]["decision"] == "matched"
    assert result["pairing"]["checks"]["name"] is True
    assert result["pairing"]["checks"]["surname"] is True
    assert result["surname"] == "ИВАНОВ"
    assert result["documentNumber"] == "A1234567"
