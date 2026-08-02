"""Portable identity-text normalization and ICAO transliteration."""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def icao_table(root: Path) -> dict[int, str]:
    path = root.resolve() / "metadata" / "icao_transliteration.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(codepoint): replacement for codepoint, replacement in payload.items()}


def identity_key(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").upper()
    return "".join(character for character in normalized if character.isalnum())


def remove_diacritics(value: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


def icao_transliterate(value: str | None, root: Path) -> str:
    table = icao_table(root)
    return "".join(table.get(ord(character), character) for character in (value or ""))


def identity_variants(value: str | None, root: Path | None = None) -> set[str]:
    variants = {
        identity_key(value),
        identity_key(remove_diacritics(value)),
    }
    if root is not None:
        variants.add(identity_key(icao_transliterate(value, root)))
        variants.add(identity_key(icao_transliterate((value or "").upper(), root)))
    return {variant for variant in variants if variant}
