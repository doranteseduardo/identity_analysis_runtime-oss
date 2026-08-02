"""Public-standard lexicons used by declarative OCR masks."""

import re
import unicodedata


def normalized_word(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.upper())
    return "".join(character for character in folded if character.isalpha())


TEXT_MONTHS = {
    "Month_Ccc": "JAN FEB MAR APR MAY JUN JUL AUG SEP SEPT OCT NOV DEC",
    "MONTH_ENG_Cc": "JAN FEB MAR APR MAY JUN JUL AUG SEP SEPT OCT NOV DEC",
    "MONTH_ENG_F": "JANUARY FEBRUARY MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER OCTOBER NOVEMBER DECEMBER",
    "MONTH_Full": "JANUARY FEBRUARY MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER OCTOBER NOVEMBER DECEMBER",
    "MONTH_SPA": "ENE FEB MAR ABR MAY JUN JUL AGO SEP SEPT OCT NOV DIC",
    "MONTH_SPA_Ccc": "ENE FEB MAR ABR MAY JUN JUL AGO SEP SEPT OCT NOV DIC",
    "MONTH_SPA_WORD": "ENERO FEBRERO MARZO ABRIL MAYO JUNIO JULIO AGOSTO SEPTIEMBRE OCTUBRE NOVIEMBRE DICIEMBRE",
    "MONTH_FRA": "JAN JANV FEV FEVR MAR MARS AVR MAI JUN JUIN JUL JUIL AOU AOUT SEP SEPT OCT NOV DEC",
    "MONTH_FRA_CCC": "JAN JANV FEV FEVR MAR MARS AVR MAI JUN JUIN JUL JUIL AOU AOUT SEP SEPT OCT NOV DEC",
    "MONTH_FRA_Full": "JANVIER FEVRIER MARS AVRIL MAI JUIN JUILLET AOUT SEPTEMBRE OCTOBRE NOVEMBRE DECEMBRE",
    "MONTH_FRA_CCCC_cccc": "JAN JANV FEV FEVR MAR MARS AVR MAI JUN JUIN JUL JUIL AOU AOUT SEP SEPT OCT NOV DEC",
    "MONTH_FRA_Cccc": "JAN JANV FEV FEVR MAR MARS AVR MAI JUN JUIN JUL JUIL AOU AOUT SEP SEPT OCT NOV DEC",
    "MONTH_NED": "JAN FEB MRT APR MEI JUN JUL AUG SEP OKT NOV DEC",
    "MONTH_NED_Ccc": "JAN FEB MRT APR MEI JUN JUL AUG SEP OKT NOV DEC",
    "MONTH_NED_CCCC_cccc": "JAN FEB MRT APR MEI JUN JUL AUG SEP OKT NOV DEC",
    "MONTH_NED_Full": "JANUARI FEBRUARI MAART APRIL MEI JUNI JULI AUGUSTUS SEPTEMBER OKTOBER NOVEMBER DECEMBER",
    "MONTH_POR": "JAN FEV MAR ABR MAI JUN JUL AGO SET OUT NOV DEZ",
    "MONTH_POR_WORD": "JANEIRO FEVEREIRO MARCO ABRIL MAIO JUNHO JULHO AGOSTO SETEMBRO OUTUBRO NOVEMBRO DEZEMBRO",
    "MONTH_ROM": "IAN FEB MAR APR MAI IUN IUL AUG SEP SEPT OCT NOI DEC",
    "MONTH_IDN_CCC_ccc": "JAN FEB MAR APR MEI JUN JUL AGU SEP OKT NOV DES",
    "MONTH_IDN_Full": "JANUARI FEBRUARI MARET APRIL MEI JUNI JULI AGUSTUS SEPTEMBER OKTOBER NOVEMBER DESEMBER",
    "MONTH_HUN": "JAN FEB MAR APR MAJ JUN JUL AUG SZE OKT NOV DEC",
    "MONTH_HUN_Full": "JANUAR FEBRUAR MARCIUS APRILIS MAJUS JUNIUS JULIUS AUGUSZTUS SZEPTEMBER OKTOBER NOVEMBER DECEMBER",
    "MONTH_ITA": "GEN FEB MAR APR MAG GIU LUG AGO SET OTT NOV DIC",
    "MONTH_ITA_WORD": "GENNAIO FEBBRAIO MARZO APRILE MAGGIO GIUGNO LUGLIO AGOSTO SETTEMBRE OTTOBRE NOVEMBRE DICEMBRE",
    "MONTH_ITA_CCC_Ccc": "GEN FEB MAR APR MAG GIU LUG AGO SET OTT NOV DIC",
    "MONTH_NOR": "JAN FEB MAR APR MAI JUN JUL AUG SEP OKT NOV DES",
    "MONTH_SWE": "JAN FEB MAR APR MAJ JUN JUL AUG SEP OKT NOV DEC",
    "MONTH_ISL": "JAN FEB MAR APR MAI JUN JUL AGU SEP OKT NOV DES",
    "MONTH_ISL_Full": "JANUAR FEBRUAR MARS APRIL MAI JUNI JULI AGUST SEPTEMBER OKTOBER NOVEMBER DESEMBER",
    "MONTH_DEU": "JAN FEB MAR APR MAI JUN JUL AUG SEP OKT NOV DEZ",
    "MONTH_DEU_Full": "JANUAR FEBRUAR MARZ APRIL MAI JUNI JULI AUGUST SEPTEMBER OKTOBER NOVEMBER DEZEMBER",
    "MONTH_DNK_Full": "JANUAR FEBRUAR MARTS APRIL MAJ JUNI JULI AUGUST SEPTEMBER OKTOBER NOVEMBER DECEMBER",
    "MONTH_TUR": "OCA SUB MAR NIS MAY HAZ TEM AGU EYL EKI KAS ARA",
    "MONTH_RUS_Ccc": "ЯНВ ФЕВ МАР АПР МАЙ ИЮН ИЮЛ АВГ СЕН ОКТ НОЯ ДЕК",
    "MONTH_UKR": "СІЧ ЛЮТ БЕР КВІ ТРА ЧЕР ЛИП СЕР ВЕР ЖОВ ЛИС ГРУ",
    "MONTH_UKR_Word": "СІЧНЯ ЛЮТОГО БЕРЕЗНЯ КВІТНЯ ТРАВНЯ ЧЕРВНЯ ЛИПНЯ СЕРПНЯ ВЕРЕСНЯ ЖОВТНЯ ЛИСТОПАДА ГРУДНЯ",
    "MONTH_GRC": "ΙΑΝ ΦΕΒ ΜΑΡ ΑΠΡ ΜΑΙ ΙΟΥΝ ΙΟΥΛ ΑΥΓ ΣΕΠ ΟΚΤ ΝΟΕ ΔΕΚ",
    "MONTH_GRC_Ccc": "ΙΑΝ ΦΕΒ ΜΑΡ ΑΠΡ ΜΑΙ ΙΟΥΝ ΙΟΥΛ ΑΥΓ ΣΕΠ ΟΚΤ ΝΟΕ ΔΕΚ",
}
TEXT_MONTHS = {
    name: {normalized_word(value) for value in values.split()}
    for name, values in TEXT_MONTHS.items()
}

JURISDICTION_CODES = {
    "STATE_CODE_USA": set(
        "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
    ),
    "STATE_CODE_CAN": set("AB BC MB NB NL NS NT NU ON PE QC SK YT".split()),
    "STATE_CODE_AUS": set("ACT NSW NT QLD SA TAS VIC WA".split()),
}

AAMVA_COLOR_CODES = {
    "EYE_COLOR_D20": set("BLK BLU BRO GRY GRN HAZ MAR PNK DIC UNK".split()),
    "HAIR_COLOR_D20": set("BAL BLK BLN BRO GRY RED SDY WHI UNK".split()),
}

SEX_WORDS = {"Sex_WORD": {"MALE", "FEMALE"}}
LETTER_WORD_TOKENS = {"WORD", "_WORD_", "Surname_ENG", "Given_Name_ENG"}
ALPHANUMERIC_WORD_TOKENS = {"WORD_D"}
NUMERIC_DATE_COMPONENTS = {
    "DAY": (1, 31, {1, 2}),
    "DAY_D": (1, 9, {1}),
    "DAY_DD": (1, 31, {2}),
    "DAY_D_DD": (1, 31, {1, 2}),
    "DAYS_LZ": (1, 31, {2}),
    "MONTH_DD": (1, 12, {2}),
    "MONTH_D_DD": (1, 12, {1, 2}),
    "MONTH_LZ": (1, 12, {2}),
    "YEAR_DD": (0, 99, {2}),
}

MONTH_NUMBERS = {
    **dict.fromkeys(("JAN", "JANV", "JANVIER", "JANUARY", "ENE", "ENERO"), 1),
    **dict.fromkeys(("FEB", "FEV", "FEVR", "FEVRIER", "FEBRUARY", "FEBRERO"), 2),
    **dict.fromkeys(("MAR", "MARS", "MARCH", "MARZO"), 3),
    **dict.fromkeys(("APR", "APRIL", "AVR", "AVRIL", "ABR", "ABRIL"), 4),
    **dict.fromkeys(("MAY", "MAI", "MAYO"), 5),
    **dict.fromkeys(("JUN", "JUNE", "JUIN", "JUNIO"), 6),
    **dict.fromkeys(("JUL", "JULY", "JUIL", "JUILLET", "JULIO"), 7),
    **dict.fromkeys(("AUG", "AUGUST", "AOU", "AOUT", "AGO", "AGOSTO"), 8),
    **dict.fromkeys(("SEP", "SEPT", "SEPTEMBER", "SEPTEMBRE", "SEPTIEMBRE"), 9),
    **dict.fromkeys(("OCT", "OCTOBER", "OCTOBRE", "OCTUBRE"), 10),
    **dict.fromkeys(("NOV", "NOVEMBER", "NOVEMBRE", "NOVIEMBRE"), 11),
    **dict.fromkeys(("DEC", "DECEMBER", "DECEMBRE", "DIC", "DICIEMBRE"), 12),
}

ADDITIONAL_MONTH_ALIASES = (
    "JAN JANV JANVIER JANUARY ENE ENERO IAN IANUAR IANUARIE JANEIRO JANUARI JANUAR GEN GENNAIO OCA ЯНВ СІЧ СІЧНЯ ΙΑΝ",
    "FEB FEV FEVR FEVRIER FEBRUARY FEBRERO FEVEREIRO FEBRUARI FEBRUAR FEBBRAIO SUB ЛЮТ ЛЮТОГО ФЕВ ΦΕΒ",
    "MAR MARS MARCH MARZO MARCO MAART MARET MARCIUS MARZ MRT МАР БЕР БЕРЕЗНЯ ΜΑΡ",
    "APR APRIL AVR AVRIL ABR ABRIL APRILE APRILIS NIS MARTS АПР КВІ КВІТНЯ ΑΠΡ",
    "MAY MAI MAYO MAIO MEI MAJ MAJUS MAG MAGGIO MAY ТРА ТРАВНЯ МАЙ ΜΑΙ",
    "JUN JUNE JUIN JUNIO JUNHO JUNI JUNIUS IUN GIU GIUGNO HAZ ИЮН ЧЕР ЧЕРВНЯ ΙΟΥΝ",
    "JUL JULY JUIL JUILLET JULIO JULHO JULI JULIUS IUL LUG LUGLIO TEM ИЮЛ ЛИП ЛИПНЯ ΙΟΥΛ",
    "AUG AUGUST AOU AOUT AGO AGOSTO AGOSTUS AGU AGUST AGUSTUS AUGUSTUS AUGUSZTUS EYL АВГ СЕР СЕРПНЯ ΑΥΓ",
    "SEP SEPT SEPTEMBER SEPTEMBRE SEPTIEMBRE SET SETEMBRO SETTEMBRE SZE SZEPTEMBER SEN СЕН ВЕР ВЕРЕСНЯ ΣΕΠ",
    "OCT OCTOBER OCTOBRE OCTUBRE OKT OKTOBER OUT OUTUBRO OTT OTTOBRE EKI ОКТ ЖОВ ЖОВТНЯ ΟΚΤ",
    "NOV NOVEMBER NOVEMBRE NOVIEMBRE NOVEMBRO NOI KAS НОЯ ЛИС ЛИСТОПАДА ΝΟΕ",
    "DEC DECEMBER DECEMBRE DIC DICIEMBRE DICEMBRE DEZ DEZEMBRO DES DESEMBER DEZEMBER ARA ДЕК ГРУ ГРУДНЯ ΔΕΚ",
)
for month_number, aliases in enumerate(ADDITIONAL_MONTH_ALIASES, 1):
    MONTH_NUMBERS.update(
        {normalized_word(alias): month_number for alias in aliases.split()}
    )


def named_token_valid(token: str, value: str) -> bool | None:
    normalized = normalized_word(value)
    lexicon = TEXT_MONTHS.get(token)
    if lexicon is None:
        lexicon = JURISDICTION_CODES.get(token)
    if lexicon is None:
        lexicon = AAMVA_COLOR_CODES.get(token)
    if lexicon is None:
        lexicon = SEX_WORDS.get(token)
    if lexicon is None:
        lexicon = explicit_sex_values(token)
    if lexicon is None and token in {"EYE_COLOR_2C", "HAIR_COLOR_2C"}:
        return len(normalized) == 2 and normalized.isalpha()
    if lexicon is None and token == "Blood_group":
        return re.fullmatch(r"(?i)(?:A|B|AB|O)[+-]?", value.strip()) is not None
    if lexicon is None and token in LETTER_WORD_TOKENS:
        return text_shape_valid(value, allow_digits=False)
    if lexicon is None and token in ALPHANUMERIC_WORD_TOKENS:
        return text_shape_valid(value, allow_digits=True)
    if lexicon is None and token in NUMERIC_DATE_COMPONENTS:
        return numeric_date_component_valid(token, value)
    if lexicon is None:
        return None
    return normalized in lexicon


def explicit_sex_values(token: str) -> set[str] | None:
    if not token.upper().startswith("SEX_"):
        return None
    suffix = token[4:]
    if len(suffix) != 2 or not suffix.isalpha() or token == "Sex_MF":
        return None
    return {normalized_word(character) for character in suffix}


def text_shape_valid(value: str, allow_digits: bool) -> bool:
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized or not any(character.isalnum() for character in normalized):
        return False
    return all(
        character.isalpha()
        or (allow_digits and character.isdigit())
        or character in " -'’."
        for character in normalized
    )


def numeric_date_component_valid(token: str, value: str) -> bool:
    minimum, maximum, widths = NUMERIC_DATE_COMPONENTS[token]
    stripped = value.strip()
    return (
        stripped.isascii()
        and stripped.isdigit()
        and len(stripped) in widths
        and minimum <= int(stripped) <= maximum
    )


def text_month_number(token: str, value: str) -> int | None:
    if token not in TEXT_MONTHS:
        return None
    return MONTH_NUMBERS.get(normalized_word(value))


def structural_mask_results(mask: str, value: str) -> list[bool]:
    primary = mask.split("|", 1)[0]
    results = []
    kinds = structural_mask_kinds(primary)
    if "usa_postal_code" in kinds:
        results.append(
            re.search(r"(?<!\d)\d{5}(?:[- ]?\d{4})?(?!\d)", value) is not None
        )
    if "canadian_postal_code" in kinds:
        results.append(
            re.search(
                r"(?i)(?<![A-Z0-9])[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d(?![A-Z0-9])",
                value,
            )
            is not None
        )
    if "feet_and_inches" in kinds:
        match = re.fullmatch(r"\s*(\d)\s*(?:'|-|FT)\s*(\d{1,2})\s*(?:\"|IN)?\s*", value, re.I)
        results.append(
            match is not None
            and 3 <= int(match.group(1)) <= 8
            and 0 <= int(match.group(2)) <= 11
        )
    if "pounds" in kinds:
        match = re.fullmatch(r"\s*(\d{1,3})\s*(?:LB|LBS|POUND|POUNDS)?\s*", value, re.I)
        results.append(match is not None and 20 <= int(match.group(1)) <= 999)
    if "kilograms" in kinds:
        match = re.fullmatch(r"\s*(\d{1,3})\s*(?:KG|KGS)?\s*", value, re.I)
        results.append(match is not None and 10 <= int(match.group(1)) <= 500)
    if "metres_and_centimetres" in kinds:
        match = re.fullmatch(
            r"\s*([0-3])\s*[,.: -]\s*(\d{1,2})\s*(?:M|METRE|METRES)?\s*",
            value,
            re.I,
        )
        results.append(
            match is not None
            and 0 <= int(match.group(2)) <= 99
            and 50 <= int(match.group(1)) * 100 + int(match.group(2)) <= 300
        )
    if "total_inches" in kinds:
        match = re.fullmatch(r"\s*(\d{2,3})\s*(?:IN|INS|INCH|INCHES|\")?\s*", value, re.I)
        results.append(match is not None and 20 <= int(match.group(1)) <= 120)
    if "feet" in kinds:
        match = re.fullmatch(r"\s*(\d)\s*(?:FT|FOOT|FEET|')?\s*", value, re.I)
        results.append(match is not None and 1 <= int(match.group(1)) <= 9)
    return results


def structural_mask_kinds(mask: str) -> set[str]:
    primary = mask.split("|", 1)[0]
    kinds = set()
    if "{POSTAL_CODE_USA_9" in primary:
        kinds.add("usa_postal_code")
    if "{POSTAL_CODE_CAN_P1" in primary and "{POSTAL_CODE_CAN_P2" in primary:
        kinds.add("canadian_postal_code")
    if any(token in primary for token in ("{FOOTS}", "{Foot}")) and "{INCHES}" in primary:
        kinds.add("feet_and_inches")
    elif "{INCHES}" in primary:
        kinds.add("total_inches")
    elif any(token in primary for token in ("{FOOTS}", "{Foot}")):
        kinds.add("feet")
    if "{POUNDS}" in primary:
        kinds.add("pounds")
    if "{KGS}" in primary:
        kinds.add("kilograms")
    if "{METRE}" in primary and "{CMS}" in primary:
        kinds.add("metres_and_centimetres")
    return kinds
