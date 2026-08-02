#!/usr/bin/env python3
"""Render the synthetic sample images shipped in ``examples/samples``.

Every fixture used by this repository's tests and documentation is generated
here from code: a procedurally drawn face, two identity-card sides and two
passport data pages.  No photograph of a real person, and no real credential
number, appears anywhere in this repository.

All identity data is fabricated:

* the issuing state is ``ZZT``, which lies in the ISO 3166-1 user-assignable
  ``ZZ*`` range and therefore can never denote a real country;
* the MRZ check digits are computed with the real ICAO 7-3-1 algorithm so the
  fixtures exercise the genuine parsing path;
* the barcodes on the card back encode an all-zero credential number and a
  far-future issue date so they can never collide with a real credential.

Usage::

    python tools/generate_synthetic_samples.py [output_directory]

The default output directory is ``examples/samples``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:  # pragma: no cover - only needed when regenerating fixtures
    import zxingcpp
except ImportError:  # pragma: no cover
    zxingcpp = None


# --------------------------------------------------------------------------
# Fabricated identity data
# --------------------------------------------------------------------------

# ``ZZT`` sits in the ISO 3166-1 user-assignable ``ZZ*`` range, so it can never
# collide with a real issuing state.  (ICAO's own specimen code ``UTO`` was the
# first choice, but the letter ``O`` between digits is indistinguishable from a
# zero for the line-recognition model, which corrupted the MRZ nationality.)
ISSUING_STATE = "ZZT"
ISSUING_STATE_NAME = "SPECIMEN STATE"
SURNAME = "SPECIMEN"
GIVEN_NAMES = "ALEX TAYLOR"
DATE_OF_BIRTH = "900101"  # YYMMDD
DATE_OF_EXPIRY = "350101"
SEX = "M"
CARD_NUMBER = "ZZ7654321"
PASSPORT_NUMBER = "AB1234567"
PERSONAL_NUMBER = "0000000000000"

# Deliberately impossible verification payloads for the card back.
FAKE_CREDENTIAL_IDENTIFIER = "0" * 24
FAKE_QR_URL = f"http://qr.ine.mx/{FAKE_CREDENTIAL_IDENTIFIER}/20990101/X/000000"
FAKE_CODE128_NUMBER = "0" * 9

MONOSPACE_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/PTMono.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)
SANS_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    return _font(MONOSPACE_CANDIDATES, size)


def sans(size: int) -> ImageFont.FreeTypeFont:
    return _font(SANS_CANDIDATES, size)


# --------------------------------------------------------------------------
# MRZ construction (ICAO 9303 7-3-1 weighting)
# --------------------------------------------------------------------------

WEIGHTS = (7, 3, 1)


def _character_value(character: str) -> int:
    if character == "<":
        return 0
    if character.isdigit():
        return int(character)
    return ord(character) - ord("A") + 10


def check_digit(value: str) -> str:
    total = sum(
        _character_value(character) * WEIGHTS[index % 3]
        for index, character in enumerate(value)
    )
    return str(total % 10)


def name_zone(surname: str, given_names: str, length: int) -> str:
    zone = surname.replace(" ", "<") + "<<" + given_names.replace(" ", "<")
    if len(zone) > length:
        raise ValueError("name zone overflows the MRZ")
    return zone.ljust(length, "<")


def td1_lines() -> list[str]:
    document = CARD_NUMBER[:9].ljust(9, "<")
    first = f"ID{ISSUING_STATE}{document}{check_digit(document)}" + "<" * 15
    optional = "<" * 11
    second_core = (
        f"{DATE_OF_BIRTH}{check_digit(DATE_OF_BIRTH)}{SEX}"
        f"{DATE_OF_EXPIRY}{check_digit(DATE_OF_EXPIRY)}{ISSUING_STATE}{optional}"
    )
    composite = first[5:30] + second_core[0:7] + second_core[8:15] + second_core[18:29]
    second = second_core + check_digit(composite)
    third = name_zone(SURNAME, GIVEN_NAMES, 30)
    lines = [first, second, third]
    if any(len(line) != 30 for line in lines):
        raise ValueError(f"TD1 line lengths: {[len(line) for line in lines]}")
    return lines


def td3_lines() -> list[str]:
    first = f"P<{ISSUING_STATE}" + name_zone(SURNAME, GIVEN_NAMES, 39)
    document = PASSPORT_NUMBER[:9].ljust(9, "<")
    personal = PERSONAL_NUMBER[:14].ljust(14, "<")
    second_core = (
        f"{document}{check_digit(document)}{ISSUING_STATE}"
        f"{DATE_OF_BIRTH}{check_digit(DATE_OF_BIRTH)}{SEX}"
        f"{DATE_OF_EXPIRY}{check_digit(DATE_OF_EXPIRY)}{personal}"
        f"{check_digit(personal)}"
    )
    composite = second_core[0:10] + second_core[13:20] + second_core[21:43]
    second = second_core + check_digit(composite)
    lines = [first, second]
    if any(len(line) != 44 for line in lines):
        raise ValueError(f"TD3 line lengths: {[len(line) for line in lines]}")
    return lines


# --------------------------------------------------------------------------
# Procedural face
# --------------------------------------------------------------------------


def render_face(
    size: tuple[int, int] = (720, 900),
    skin: tuple[int, int, int] = (222, 178, 148),
    hair: tuple[int, int, int] = (58, 42, 34),
    background: tuple[int, int, int] = (206, 212, 216),
    seed: int = 7,
) -> Image.Image:
    """Draw a frontal, neutral, clearly synthetic face.

    The proportions are deliberately canonical so a face detector finds exactly
    one region with a near-zero head pose.
    """

    width, height = size
    supersample = 3
    canvas_width, canvas_height = width * supersample, height * supersample
    image = Image.new("RGB", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(image)

    cx = canvas_width / 2
    face_w = canvas_width * 0.46
    face_h = face_w * 1.32
    cy = canvas_height * 0.44
    top, bottom = cy - face_h / 2, cy + face_h / 2
    left, right = cx - face_w / 2, cx + face_w / 2

    neck_w = face_w * 0.46
    draw.rounded_rectangle(
        [cx - neck_w / 2, cy + face_h * 0.28, cx + neck_w / 2, canvas_height],
        radius=int(neck_w * 0.3),
        fill=tuple(max(0, value - 22) for value in skin),
    )
    shoulder_top = cy + face_h * 0.72
    draw.ellipse(
        [
            cx - canvas_width * 0.55,
            shoulder_top,
            cx + canvas_width * 0.55,
            shoulder_top + canvas_height,
        ],
        fill=(66, 78, 96),
    )

    draw.ellipse(
        [
            left - face_w * 0.09,
            top - face_h * 0.11,
            right + face_w * 0.09,
            top + face_h * 0.72,
        ],
        fill=hair,
    )
    draw.ellipse([left, top, right, bottom], fill=skin)
    draw.polygon(
        [
            (left - 2, cy + face_h * 0.10),
            (right + 2, cy + face_h * 0.10),
            (right + 2, bottom + 4),
            (left - 2, bottom + 4),
        ],
        fill=background,
    )
    jaw_w = face_w * 0.86
    draw.ellipse([cx - jaw_w / 2, cy - face_h * 0.16, cx + jaw_w / 2, bottom], fill=skin)
    draw.ellipse([left, top, right, cy + face_h * 0.12], fill=skin)

    ear_h = face_h * 0.18
    for sign in (-1, 1):
        ear_x = cx + sign * face_w * 0.49
        draw.ellipse(
            [
                ear_x - face_w * 0.055,
                cy - ear_h / 2,
                ear_x + face_w * 0.055,
                cy + ear_h / 2,
            ],
            fill=tuple(max(0, value - 10) for value in skin),
        )

    draw.chord(
        [
            left - face_w * 0.09,
            top - face_h * 0.11,
            right + face_w * 0.09,
            top + face_h * 0.62,
        ],
        180,
        360,
        fill=hair,
    )
    draw.ellipse(
        [
            cx - face_w * 0.30,
            top + face_h * 0.10,
            cx + face_w * 0.30,
            top + face_h * 0.30,
        ],
        fill=skin,
    )

    eye_y = cy - face_h * 0.08
    eye_dx = face_w * 0.205
    eye_w = face_w * 0.175
    eye_h = eye_w * 0.46

    for sign in (-1, 1):
        eye_x = cx + sign * eye_dx
        draw.arc(
            [
                eye_x - eye_w * 0.72,
                eye_y - eye_h * 3.1,
                eye_x + eye_w * 0.72,
                eye_y - eye_h * 0.35,
            ],
            200,
            340,
            fill=tuple(max(0, value - 6) for value in hair),
            width=int(eye_h * 0.55),
        )
        draw.ellipse(
            [
                eye_x - eye_w * 0.85,
                eye_y - eye_h * 1.5,
                eye_x + eye_w * 0.85,
                eye_y + eye_h * 1.5,
            ],
            fill=tuple(max(0, value - 16) for value in skin),
        )

    for sign in (-1, 1):
        eye_x = cx + sign * eye_dx
        draw.ellipse(
            [eye_x - eye_w, eye_y - eye_h, eye_x + eye_w, eye_y + eye_h],
            fill=(246, 243, 240),
        )
        iris = eye_h * 0.94
        draw.ellipse(
            [eye_x - iris, eye_y - iris, eye_x + iris, eye_y + iris], fill=(92, 66, 46)
        )
        pupil = iris * 0.45
        draw.ellipse(
            [eye_x - pupil, eye_y - pupil, eye_x + pupil, eye_y + pupil],
            fill=(18, 14, 12),
        )
        highlight = iris * 0.24
        draw.ellipse(
            [
                eye_x - iris * 0.45 - highlight,
                eye_y - iris * 0.45 - highlight,
                eye_x - iris * 0.45 + highlight,
                eye_y - iris * 0.45 + highlight,
            ],
            fill=(252, 252, 250),
        )
        draw.arc(
            [eye_x - eye_w, eye_y - eye_h * 1.15, eye_x + eye_w, eye_y + eye_h * 0.95],
            185,
            355,
            fill=(70, 52, 44),
            width=int(eye_h * 0.34),
        )

    nose_top = eye_y + eye_h * 1.4
    nose_bottom = cy + face_h * 0.14
    nose_w = face_w * 0.115
    draw.polygon(
        [(cx, nose_top), (cx + nose_w, nose_bottom), (cx - nose_w, nose_bottom)],
        fill=tuple(max(0, value - 12) for value in skin),
    )
    draw.ellipse(
        [
            cx - nose_w * 1.05,
            nose_bottom - nose_w * 0.75,
            cx + nose_w * 1.05,
            nose_bottom + nose_w * 0.35,
        ],
        fill=tuple(min(255, value + 6) for value in skin),
    )
    for sign in (-1, 1):
        draw.ellipse(
            [
                cx + sign * nose_w * 0.62 - nose_w * 0.24,
                nose_bottom - nose_w * 0.22,
                cx + sign * nose_w * 0.62 + nose_w * 0.24,
                nose_bottom + nose_w * 0.14,
            ],
            fill=(120, 84, 70),
        )

    mouth_y = cy + face_h * 0.27
    mouth_w = face_w * 0.20
    draw.polygon(
        [
            (cx - mouth_w, mouth_y),
            (cx - mouth_w * 0.35, mouth_y - mouth_w * 0.30),
            (cx, mouth_y - mouth_w * 0.16),
            (cx + mouth_w * 0.35, mouth_y - mouth_w * 0.30),
            (cx + mouth_w, mouth_y),
            (cx + mouth_w * 0.5, mouth_y + mouth_w * 0.42),
            (cx - mouth_w * 0.5, mouth_y + mouth_w * 0.42),
        ],
        fill=(176, 104, 96),
    )
    draw.line(
        [
            (cx - mouth_w, mouth_y),
            (cx, mouth_y - mouth_w * 0.06),
            (cx + mouth_w, mouth_y),
        ],
        fill=(126, 68, 62),
        width=int(mouth_w * 0.09),
    )

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.GaussianBlur(0.8))

    generator = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.float32)
    rows, columns = np.mgrid[0:height, 0:width]
    light = 1.0 + 0.13 * (
        1.0
        - np.hypot((columns - width * 0.35) / width, (rows - height * 0.28) / height)
        * 1.7
    )
    pixels *= light[..., None]
    pixels += generator.normal(0.0, 2.6, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------
# Shared document helpers
# --------------------------------------------------------------------------


def _paper(size: tuple[int, int], tint: tuple[int, int, int], seed: int) -> Image.Image:
    width, height = size
    generator = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:] = tint
    rows, columns = np.mgrid[0:height, 0:width]
    base *= (
        1.0
        + 0.035 * np.sin(columns / max(width, 1) * 6.0)
        + 0.02 * np.cos(rows / max(height, 1) * 5.0)
    )[..., None]
    base += generator.normal(0.0, 2.2, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _guilloche(draw: ImageDraw.ImageDraw, size: tuple[int, int], color) -> None:
    width, height = size
    for index in range(0, width, 26):
        draw.line(
            [(index, 0), (index + height // 3, height)],
            fill=color,
            width=1,
        )


MRZ_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
# Fraction of the character cell covered by ink, and the share of the crop
# region kept clear at each end.  Both were tuned by recognising the rendered
# zones with the runtime's own line recogniser until every line round-tripped.
MRZ_GLYPH_RATIO = 0.90
MRZ_EDGE_INSET = 0.03


def _draw_mrz(
    image: Image.Image,
    lines: list[str],
    bands: tuple[tuple[float, float], ...],
    left_ratio: float,
    right_ratio: float,
) -> None:
    """Render MRZ lines at a fixed character pitch inside their declared band.

    Characters are drawn one cell at a time rather than as a single string.  A
    real OCR-B machine-readable zone leaves visible whitespace between glyphs,
    and the CTC decoder in :mod:`identity_analysis.ocr` needs that gap: without
    it, runs of identical filler characters collapse into a single symbol.
    """

    width, height = image.size
    draw = ImageDraw.Draw(image)
    # Inset the text from the region the recognizer crops, so no glyph is
    # clipped by the crop edge.
    inset = (right_ratio - left_ratio) * MRZ_EDGE_INSET
    left_ratio, right_ratio = left_ratio + inset, right_ratio - inset
    target_width = (right_ratio - left_ratio) * width
    for line, (top_ratio, bottom_ratio) in zip(lines, bands):
        band_height = (bottom_ratio - top_ratio) * height
        cell = target_width / len(line)
        size = max(8, int(band_height * 0.95))
        while size > 8:
            font = mono(size)
            widest = max(
                draw.textlength(character, font=font) for character in MRZ_ALPHABET
            )
            if widest <= cell * MRZ_GLYPH_RATIO:
                break
            size -= 1
        font = mono(size)
        ascent, descent = font.getmetrics()
        baseline = top_ratio * height + (band_height + ascent - descent) / 2
        for index, character in enumerate(line):
            advance = draw.textlength(character, font=font)
            x = left_ratio * width + cell * (index + 0.5) - advance / 2
            draw.text((x, baseline), character, font=font, fill=(16, 16, 18), anchor="ls")


def _barcode_image(text: str, fmt: str, module_size: int) -> Image.Image:
    if zxingcpp is None:  # pragma: no cover
        raise RuntimeError("zxing-cpp is required to generate the barcode fixtures")
    barcode_format = getattr(zxingcpp.BarcodeFormat, fmt)
    if hasattr(zxingcpp, "create_barcode"):
        created = zxingcpp.create_barcode(text, barcode_format)
        rendered = zxingcpp.write_barcode_to_image(created)
    else:  # pragma: no cover - older zxing-cpp
        rendered = zxingcpp.write_barcode(barcode_format, text)
    image = Image.fromarray(np.asarray(rendered)).convert("L")
    return image.resize(
        (image.width * module_size, image.height * module_size),
        Image.Resampling.NEAREST,
    )


def _finish(image: Image.Image, seed: int) -> Image.Image:
    generator = np.random.default_rng(seed)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixels += generator.normal(0.0, 1.4, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------
# Identity card (ID-1)
# --------------------------------------------------------------------------

CARD_SIZE = (2200, 1388)  # 85.6 x 54 mm at ~650 dpi


def render_card_front(face: Image.Image) -> Image.Image:
    width, height = CARD_SIZE
    image = _paper(CARD_SIZE, (236, 238, 232), seed=11)
    draw = ImageDraw.Draw(image)
    _guilloche(draw, CARD_SIZE, (214, 220, 226))
    draw.rectangle([0, 0, width - 1, int(height * 0.115)], fill=(34, 62, 96))
    draw.text(
        (int(width * 0.035), int(height * 0.028)),
        f"{ISSUING_STATE_NAME}  -  SPECIMEN IDENTITY CARD",
        font=sans(int(height * 0.056)),
        fill=(244, 246, 250),
    )

    portrait_box = (
        int(width * 0.055),
        int(height * 0.300),
        int(width * 0.300),
        int(height * 0.700),
    )
    portrait = face.resize(
        (portrait_box[2] - portrait_box[0], portrait_box[3] - portrait_box[1]),
        Image.Resampling.LANCZOS,
    )
    image.paste(portrait, portrait_box[:2])
    draw.rectangle(portrait_box, outline=(120, 128, 140), width=3)

    ghost_box = (
        int(width * 0.825),
        int(height * 0.300),
        int(width * 0.955),
        int(height * 0.455),
    )
    ghost = face.convert("L").convert("RGB").resize(
        (ghost_box[2] - ghost_box[0], ghost_box[3] - ghost_box[1]),
        Image.Resampling.LANCZOS,
    )
    image.paste(Image.blend(image.crop(ghost_box), ghost, 0.55), ghost_box[:2])

    label_font = sans(int(height * 0.033))
    value_font = sans(int(height * 0.050))
    rows = (
        ("SURNAME", SURNAME),
        ("GIVEN NAMES", GIVEN_NAMES),
        ("DATE OF BIRTH", "01/01/1990"),
        ("SEX", SEX),
        ("NATIONALITY", ISSUING_STATE),
        ("DOCUMENT No.", CARD_NUMBER),
        ("DATE OF EXPIRY", "01/01/2035"),
    )
    y = int(height * 0.185)
    for label, value in rows:
        draw.text((int(width * 0.335), y), label, font=label_font, fill=(96, 104, 118))
        draw.text(
            (int(width * 0.335), y + int(height * 0.036)),
            value,
            font=value_font,
            fill=(24, 26, 32),
        )
        y += int(height * 0.098)

    signature_box = (
        int(width * 0.625),
        int(height * 0.640),
        int(width * 0.810),
        int(height * 0.755),
    )
    draw.line(
        [
            (signature_box[0] + 10, signature_box[3] - 18),
            (signature_box[0] + 70, signature_box[1] + 26),
            (signature_box[0] + 130, signature_box[3] - 26),
            (signature_box[2] - 40, signature_box[1] + 34),
            (signature_box[2] - 8, signature_box[3] - 20),
        ],
        fill=(30, 44, 96),
        width=5,
        joint="curve",
    )
    draw.text(
        (int(width * 0.035), int(height * 0.905)),
        "SPECIMEN - NOT A REAL DOCUMENT - SYNTHETIC TEST FIXTURE",
        font=sans(int(height * 0.038)),
        fill=(150, 60, 60),
    )
    return _finish(image, seed=12)


def render_card_back() -> Image.Image:
    width, height = CARD_SIZE
    image = _paper(CARD_SIZE, (240, 240, 236), seed=21)
    draw = ImageDraw.Draw(image)
    _guilloche(draw, CARD_SIZE, (220, 224, 228))

    code128 = _barcode_image(FAKE_CODE128_NUMBER, "Code128", module_size=3)
    code128_box = (int(width * 0.030), int(height * 0.030))
    code128 = code128.resize(
        (int(width * 0.240), int(height * 0.140)), Image.Resampling.NEAREST
    )
    image.paste(code128.convert("RGB"), code128_box)

    qr = _barcode_image(FAKE_QR_URL, "QRCode", module_size=6)
    qr_side = int(height * 0.310)
    qr = qr.resize((qr_side, qr_side), Image.Resampling.NEAREST)
    image.paste(qr.convert("RGB"), (int(width * 0.775), int(height * 0.075)))

    draw.text(
        (int(width * 0.030), int(height * 0.215)),
        "SPECIMEN VERIFICATION DATA - SYNTHETIC TEST FIXTURE",
        font=sans(int(height * 0.040)),
        fill=(96, 104, 118),
    )
    draw.text(
        (int(width * 0.030), int(height * 0.285)),
        f"CREDENTIAL {FAKE_CREDENTIAL_IDENTIFIER}",
        font=sans(int(height * 0.038)),
        fill=(120, 126, 138),
    )
    draw.text(
        (int(width * 0.030), int(height * 0.355)),
        f"ISSUED BY THE {ISSUING_STATE_NAME} - NOT A REAL DOCUMENT",
        font=sans(int(height * 0.038)),
        fill=(150, 60, 60),
    )

    draw.rectangle(
        [0, int(height * 0.690), width, height],
        fill=(250, 250, 248),
    )
    _draw_mrz(
        image,
        td1_lines(),
        ((0.714, 0.779), (0.797, 0.862), (0.879, 0.944)),
        0.031,
        0.974,
    )
    return _finish(image, seed=22)


# --------------------------------------------------------------------------
# Passport data page (TD3)
# --------------------------------------------------------------------------

PASSPORT_SIZE = (2000, 1420)  # 125 x 88 mm data page


def render_passport(face: Image.Image, variant: int) -> Image.Image:
    width, height = PASSPORT_SIZE
    tint = (238, 234, 226) if variant == 1 else (234, 236, 232)
    image = _paper(PASSPORT_SIZE, tint, seed=30 + variant)
    draw = ImageDraw.Draw(image)
    _guilloche(draw, PASSPORT_SIZE, (218, 214, 206))

    draw.text(
        (int(width * 0.035), int(height * 0.030)),
        f"PASSPORT   {ISSUING_STATE_NAME}",
        font=sans(int(height * 0.048)),
        fill=(40, 52, 84),
    )
    draw.line(
        [(int(width * 0.035), int(height * 0.098)), (int(width * 0.965), int(height * 0.098))],
        fill=(120, 130, 150),
        width=3,
    )

    portrait_box = (
        int(width * 0.045),
        int(height * 0.140),
        int(width * 0.300),
        int(height * 0.640),
    )
    portrait = face.resize(
        (portrait_box[2] - portrait_box[0], portrait_box[3] - portrait_box[1]),
        Image.Resampling.LANCZOS,
    )
    image.paste(portrait, portrait_box[:2])
    draw.rectangle(portrait_box, outline=(140, 140, 148), width=3)

    label_font = sans(int(height * 0.030))
    value_font = sans(int(height * 0.046))
    rows = (
        ("TYPE / TIPO", "P"),
        ("CODE / CODIGO", ISSUING_STATE),
        ("PASSPORT No.", PASSPORT_NUMBER),
        ("SURNAME / APELLIDOS", SURNAME),
        ("GIVEN NAMES / NOMBRES", GIVEN_NAMES),
        ("NATIONALITY", ISSUING_STATE),
        ("DATE OF BIRTH", "01/01/1990"),
        ("SEX", SEX),
        ("DATE OF EXPIRY", "01/01/2035"),
    )
    y = int(height * 0.145)
    for label, value in rows:
        draw.text((int(width * 0.345), y), label, font=label_font, fill=(104, 110, 124))
        draw.text(
            (int(width * 0.345), y + int(height * 0.030)),
            value,
            font=value_font,
            fill=(26, 28, 34),
        )
        y += int(height * 0.075)

    draw.text(
        (int(width * 0.045), int(height * 0.680)),
        "SPECIMEN - SYNTHETIC TEST FIXTURE - NOT A REAL DOCUMENT",
        font=sans(int(height * 0.036)),
        fill=(150, 60, 60),
    )

    draw.rectangle([0, int(height * 0.770), width, height], fill=(250, 249, 246))
    _draw_mrz(
        image,
        td3_lines(),
        ((0.810, 0.863), (0.880, 0.935)),
        0.030,
        0.970,
    )
    return _finish(image, seed=40 + variant)


# --------------------------------------------------------------------------


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "examples" / "samples"
    output.mkdir(parents=True, exist_ok=True)

    face = render_face()
    face.save(output / "synthetic_selfie.jpg", quality=92)
    render_card_front(face).save(output / "synthetic_id_front.jpg", quality=94)
    render_card_back().save(output / "synthetic_id_back.jpg", quality=94)
    render_passport(face, 1).save(output / "synthetic_passport_1.jpg", quality=94)
    render_passport(face, 2).save(output / "synthetic_passport_2.jpg", quality=94)

    print("TD1:", *td1_lines(), sep="\n  ")
    print("TD3:", *td3_lines(), sep="\n  ")
    print(f"wrote synthetic samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
