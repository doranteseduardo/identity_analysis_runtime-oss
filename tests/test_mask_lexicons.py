from identity_analysis.mask_lexicons import (
    AAMVA_COLOR_CODES,
    ALPHANUMERIC_WORD_TOKENS,
    JURISDICTION_CODES,
    LETTER_WORD_TOKENS,
    NUMERIC_DATE_COMPONENTS,
    SEX_WORDS,
    TEXT_MONTHS,
    explicit_sex_values,
    named_token_valid,
    numeric_date_component_valid,
    structural_mask_results,
    structural_mask_kinds,
    text_month_number,
)



def test_every_public_lexicon_entry_is_accepted() -> None:
    for token, values in {
        **TEXT_MONTHS,
        **JURISDICTION_CODES,
        **AAMVA_COLOR_CODES,
        **SEX_WORDS,
    }.items():
        assert values
        assert all(named_token_valid(token, value) for value in values)


def test_every_text_month_resolves_to_a_calendar_month() -> None:
    for token, values in TEXT_MONTHS.items():
        assert all(text_month_number(token, value) in range(1, 13) for value in values)


def test_unknown_tokens_remain_uninterpreted() -> None:
    assert named_token_valid("CITY_USA", "SEATTLE") is None
    assert named_token_valid("Sex_CHN", "女") is None


def test_invalid_public_codes_are_rejected() -> None:
    assert named_token_valid("STATE_CODE_USA", "ZZ") is False
    assert named_token_valid("EYE_COLOR_D20", "ORANGE") is False


def test_explicit_sex_tokens_validate_their_declared_markers() -> None:
    assert named_token_valid("Sex_WORD", "female") is True
    assert named_token_valid("Sex_WORD", "unknown") is False
    assert named_token_valid("Sex_KM", "K") is True
    assert named_token_valid("Sex_KM", "F") is False
    assert named_token_valid("Sex_МЖ", "Ж") is True
    assert named_token_valid("Sex_ЖЧ", "Ч") is True
    assert named_token_valid("Sex_АЭ", "Э") is True
    assert named_token_valid("Sex_CHN", "女") is None


def test_compact_color_and_blood_group_tokens_validate_shape() -> None:
    assert named_token_valid("EYE_COLOR_2C", "BL") is True
    assert named_token_valid("HAIR_COLOR_2C", "B1") is False
    assert named_token_valid("Blood_group", "AB+") is True
    assert named_token_valid("Blood_group", "C+") is False


def test_text_shape_tokens_accept_unicode_without_name_dictionaries() -> None:
    assert named_token_valid("WORD", "O’CONNOR") is True
    assert named_token_valid("Surname_ENG", "GARCÍA-LÓPEZ") is True
    assert named_token_valid("Given_Name_ENG", "ANNA MARIA") is True
    assert named_token_valid("WORD", "A12") is False
    assert named_token_valid("WORD_D", "A12") is True
    assert named_token_valid("WORD_D", "A@12") is False


def test_numeric_date_components_validate_width_and_range() -> None:
    assert numeric_date_component_valid("DAY", "7") is True
    assert numeric_date_component_valid("DAY_DD", "07") is True
    assert numeric_date_component_valid("DAY_DD", "7") is False
    assert numeric_date_component_valid("DAY", "32") is False
    assert numeric_date_component_valid("MONTH_DD", "12") is True
    assert numeric_date_component_valid("MONTH_DD", "13") is False
    assert numeric_date_component_valid("YEAR_DD", "00") is True
    assert numeric_date_component_valid("YEAR_DD", "2000") is False


def test_textual_months_resolve_to_calendar_numbers() -> None:
    assert text_month_number("MONTH_SPA_WORD", "diciembre") == 12
    assert text_month_number("MONTH_FRA_Full", "août") == 8
    assert text_month_number("CITY_USA", "MAY") is None


def test_postal_and_height_structures_are_validated() -> None:
    usa = '{CITY_USA}, {STATE_CODE_USA} {POSTAL_CODE_USA_9}'
    canada = '{CITY_CAN} {STATE_CODE_CAN}^{POSTAL_CODE_CAN_P1} {POSTAL_CODE_CAN_P2}'
    height = "{FOOTS}-{INCHES}"

    assert structural_mask_results(usa, "Seattle, WA 98101") == [True]
    assert structural_mask_results(usa, "Seattle, WA ABCDE") == [False]
    assert structural_mask_results(canada, "Toronto ON M5V 3A8") == [True]
    assert structural_mask_results(canada, "Toronto ON D5V 3A8") == [False]
    assert structural_mask_results(height, "5-11") == [True]
    assert structural_mask_results(height, "5' 12\"") == [False]


def test_declared_measurement_structures_are_validated() -> None:
    assert structural_mask_results("{POUNDS}", "180 lbs") == [True]
    assert structural_mask_results("{POUNDS}", "heavy") == [False]
    assert structural_mask_results("{KGS}!{MEASURE}", "82 KG") == [True]
    assert structural_mask_results("{METRE},{CMS} {MEASURE}", "1,82") == [True]
    assert structural_mask_results("{METRE},{CMS} {MEASURE}", "1,125") == [False]
    assert structural_mask_results("{INCHES} {MEASURE}", "070 IN") == [True]
    assert structural_mask_results("{FOOTS} {MEASURE}", "6 FT") == [True]
