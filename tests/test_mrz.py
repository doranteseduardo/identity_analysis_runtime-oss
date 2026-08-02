import pytest
from PIL import Image, ImageDraw

from identity_analysis.pipeline import (
    normalize_td1_lines,
    normalize_blr_personal_number,
    parse_mrv,
    parse_td2,
    parse_swedish_visual_date,
    parse_td3,
    repair_td3,
    td3_candidate_score,
    mrv_candidate_score,
    mrz_line_bounds,
    mrz_region_candidates,
    two_line_mrz_candidate_score,
    valid_check_digit,
)


def test_icao_check_digits() -> None:
    assert valid_check_digit("L898902C3", "6")
    assert valid_check_digit("740812", "2")
    assert valid_check_digit("120415", "9")


def test_line_three_spaces_become_fillers() -> None:
    lines = normalize_td1_lines(
        [
            "I<UTOERIKSSON<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA MARIA<<<<<<<<<",
        ]
    )
    assert "ANNA<MARIA" in lines[2]


def test_parses_icao_td3_passport() -> None:
    result = parse_td3(
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        ],
        [0.99, 0.98],
    )

    assert result["DocumentName"] == "Passport"
    assert result["documentNumber"] == "L898902C3"
    assert result["surname"] == "ERIKSSON"
    assert result["givenNames"] == "ANNA MARIA"
    assert result["checks"] == {
        "documentNumber": True,
        "dateOfBirth": True,
        "dateOfExpiry": True,
        "personalNumber": True,
        "composite": True,
    }


def test_two_line_candidate_scores_return_structural_counts() -> None:
    td3 = (
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        ],
        [0.99, 0.98],
    )
    mrv = (
        [
            "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<<<",
        ],
        [0.99, 0.98],
    )

    assert td3_candidate_score(td3) == (5, 1.97)
    assert mrv_candidate_score(mrv) == (3, 1.97)
    assert two_line_mrz_candidate_score(td3) == (5, 1.97)
    assert two_line_mrz_candidate_score(mrv) == (3, 1.97)


def test_localizes_dense_two_line_mrz_region_without_opencv() -> None:
    image = Image.new("RGB", (1000, 650), "white")
    drawing = ImageDraw.Draw(image)
    for top in (500, 555):
        for left in range(60, 950, 20):
            drawing.rectangle((left, top, left + 10, top + 28), fill="black")

    regions = mrz_region_candidates(image, line_count=2)

    assert regions
    left, top, right, bottom = regions[0]
    assert left < 0.08
    assert right > 0.94
    assert 0.72 < top < 0.79
    assert 0.88 < bottom < 0.94
    lines = mrz_line_bounds(regions[0], 2)
    assert len(lines) == 2
    assert lines[0][1] < lines[0][3] <= lines[1][3]


def test_mrz_localizer_rejects_small_images() -> None:
    assert mrz_region_candidates(Image.new("RGB", (60, 19), "white"), 2) == []


def test_parses_icao_td2_identity_document() -> None:
    result = parse_td2(
        [
            "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<",
            "D231458907UTO7408122F1204159<<<<<<<6",
        ],
        [0.99, 0.98],
    )

    assert result["recognitionProfile"] == "ICAO-TD2"
    assert result["documentNumber"] == "D23145890"
    assert result["dateOfBirth"] == "1974-08-12"
    assert result["dateOfExpiry"] == "2012-04-15"
    assert result["surname"] == "ERIKSSON"
    assert result["givenNames"] == "ANNA MARIA"
    assert result["checks"] == {
        "documentNumber": True,
        "dateOfBirth": True,
        "dateOfExpiry": True,
        "composite": True,
    }


def test_parses_icao_mrv_a_visa() -> None:
    result = parse_mrv(
        [
            "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<<<",
        ],
        [0.99, 0.98],
    )

    assert result["DocumentName"] == "Visa"
    assert result["recognitionProfile"] == "ICAO-MRV-A"
    assert result["documentNumber"] == "L898902C3"
    assert result["optionalData"] == "ZE184226B"
    assert result["surname"] == "ERIKSSON"
    assert result["givenNames"] == "ANNA MARIA"
    assert all(result["checks"].values())


def test_parses_icao_mrv_b_visa() -> None:
    result = parse_mrv(
        [
            "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE18422<",
        ],
        [0.99, 0.98],
    )

    assert result["recognitionProfile"] == "ICAO-MRV-B"
    assert result["optionalData"] == "ZE18422"
    assert result["validityStatus"] == 1


def test_td3_parser_rejects_mrv_a_even_at_same_line_length() -> None:
    with pytest.raises(ValueError, match="class must start with P"):
        parse_td3(
            [
                "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                "L898902C36UTO7408122F1204159ZE184226B<<<<<<<",
            ],
            [0.99, 0.98],
        )


def test_repairs_collapsed_td3_fillers() -> None:
    repaired = repair_td3(
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<10",
        ],
        [0.9, 0.9],
    )

    assert repaired == [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]


def test_parses_swedish_visual_dates() -> None:
    assert parse_swedish_visual_date("21 AUG/AUG 82") == "1982-08-21"
    assert parse_swedish_visual_date("09 JUL/JUL 21") == "2021-07-09"


def test_normalizes_belarus_personal_number_positions() -> None:
    assert normalize_blr_personal_number("40101908000P80") == "4010190B000PB0"
