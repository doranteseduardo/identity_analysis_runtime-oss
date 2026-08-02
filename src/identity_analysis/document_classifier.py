"""Portable ONNX document-template classification."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .onnx_runtime import create_session


DEFAULT_CLASSIFIER_ENGINE = "onnx-document-classifier"


def preprocess_document_classifier(
    image: Image.Image, width: int = 256, height: int = 256
) -> np.ndarray:
    rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32)
    denominator = max(float(pixels.std()), 1.0 / math.sqrt(pixels.size))
    normalized = (pixels - float(pixels.mean())) / denominator
    return np.transpose(normalized, (2, 0, 1))[None, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DocumentClassifier:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        for key in ("model", "info", "catalog"):
            entry = manifest.get(key)
            if entry is None:
                raise ValueError(
                    f"Document classifier manifest is missing {key!r}: {self.root}"
                )
            path = self.root / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Document classifier asset hash mismatch: {entry['path']}")
        catalog = json.loads(
            (self.root / manifest["catalog"]["path"]).read_text(encoding="utf-8")
        )
        self.engine = manifest.get("engine") or DEFAULT_CLASSIFIER_ENGINE
        self.identifier_map = catalog["identifierMap"]
        self.session = create_session(str(self.root / manifest["model"]["path"]))
        self.input = self.session.get_inputs()[0]
        self.height, self.width = map(int, self.input.shape[-2:])

    def classify(self, image: Image.Image, top_k: int = 10) -> dict:
        tensor = preprocess_document_classifier(image, self.width, self.height)
        probabilities = np.asarray(
            self.session.run(None, {self.input.name: tensor})[0]
        ).reshape(-1)
        limit = max(1, min(top_k, probabilities.size))
        indices = np.argpartition(probabilities, -limit)[-limit:]
        indices = indices[np.argsort(probabilities[indices])[::-1]]
        candidates = []
        for index in indices:
            mapping = self.identifier_map[int(index)]
            document = mapping.get("document")
            candidates.append(
                {
                    "documentIdentifier": mapping["documentIdentifier"],
                    "confidence": float(probabilities[index]),
                    "document": document,
                }
            )
        return {
            "engine": self.engine,
            "classCount": len(self.identifier_map),
            "candidates": candidates,
        }


@lru_cache(maxsize=2)
def classifier_for_root(root: str) -> DocumentClassifier:
    return DocumentClassifier(Path(root) / "document_classifier")


def classify_document(root: Path, image: Image.Image, top_k: int = 10) -> dict:
    return classifier_for_root(str(Path(root).resolve())).classify(image, top_k)


def warm_up_document_classifier(root: Path) -> None:
    classifier_for_root(str(Path(root).resolve()))


def classifier_available(root: Path) -> bool:
    root = Path(root).resolve() / "document_classifier"
    if not (root / "manifest.json").is_file():
        return False
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    entry = manifest.get("model")
    return bool(entry) and (root / entry["path"]).is_file()


def document_catalog_available(root: Path) -> bool:
    return bool(_optional_document_catalog(root))


def _optional_document_catalog(root: Path) -> tuple[dict, ...]:
    """Documents declared by the catalog, or an empty tuple when absent."""

    try:
        return document_catalog_for_root(str(Path(root).resolve()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return ()


@lru_cache(maxsize=2)
def document_catalog_for_root(root: str) -> tuple[dict, ...]:
    catalog_root = Path(root).resolve() / "document_classifier"
    manifest = json.loads(
        (catalog_root / "manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["catalog"]
    path = catalog_root / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Document catalog asset hash mismatch: {entry['path']}")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        mapping["document"]
        for mapping in catalog["identifierMap"]
        if mapping.get("document")
    )


def document_catalog_index_for_root(root: str) -> dict[int, dict]:
    return {
        document["identifier"]: document
        for document in _optional_document_catalog(Path(root))
    }


def document_catalog_entry(root: Path, identifier: int) -> dict | None:
    return document_catalog_index_for_root(str(Path(root).resolve())).get(identifier)


def search_document_catalog(
    root: Path,
    *,
    query: str | None = None,
    country_code: str | None = None,
    document_type: str | None = None,
    document_format: str | None = None,
    include_deprecated: bool = False,
) -> list[dict]:
    documents = _optional_document_catalog(root)
    query_value = (query or "").strip().casefold()
    country_value = (country_code or "").strip().upper()
    type_value = (document_type or "").strip().casefold()
    format_value = (document_format or "").strip().casefold()
    results = []
    for document in documents:
        if document.get("deprecated") and not include_deprecated:
            continue
        if country_value and country_value not in document.get("isoCodes", []):
            continue
        if type_value and type_value != (
            document.get("documentType") or {}
        ).get("name", "").casefold():
            continue
        if format_value and format_value != (
            document.get("documentFormat") or {}
        ).get("name", "").casefold():
            continue
        if query_value:
            searchable = " ".join(
                str(value)
                for value in (
                    document.get("caption"),
                    document.get("country"),
                    " ".join(document.get("isoCodes") or []),
                    (document.get("documentType") or {}).get("name"),
                    (document.get("documentFormat") or {}).get("name"),
                    document.get("year"),
                    document.get("series"),
                    " ".join(document.get("stateCodes") or []),
                )
                if value
            ).casefold()
            if query_value not in searchable:
                continue
        results.append(document)
    return sorted(
        results,
        key=lambda document: (
            document.get("country") or "",
            document.get("caption") or "",
            document.get("year") or "",
            document["identifier"],
        ),
    )


def document_catalog_facets(
    root: Path, *, include_deprecated: bool = False
) -> dict:
    return deepcopy(
        document_catalog_facets_for_root(
            str(Path(root).resolve()), include_deprecated
        )
    )


@lru_cache(maxsize=4)
def document_catalog_facets_for_root(
    root: str, include_deprecated: bool
) -> dict:
    documents = search_document_catalog(
        Path(root), include_deprecated=include_deprecated
    )
    countries: dict[str, dict] = {}
    document_types: dict[str, int] = {}
    document_formats: dict[str, int] = {}
    for document in documents:
        for code in document.get("isoCodes") or []:
            country = countries.setdefault(
                code,
                {
                    "code": code,
                    "name": document.get("country"),
                    "count": 0,
                },
            )
            country["count"] += 1
        type_name = (document.get("documentType") or {}).get("name")
        if type_name:
            document_types[type_name] = document_types.get(type_name, 0) + 1
        format_name = (document.get("documentFormat") or {}).get("name")
        if format_name:
            document_formats[format_name] = document_formats.get(format_name, 0) + 1
    return {
        "documentCount": len(documents),
        "countries": sorted(
            countries.values(),
            key=lambda item: ((item["name"] or "").casefold(), item["code"]),
        ),
        "documentTypes": [
            {"name": name, "count": count}
            for name, count in sorted(
                document_types.items(), key=lambda item: item[0].casefold()
            )
        ],
        "documentFormats": [
            {"name": name, "count": count}
            for name, count in sorted(
                document_formats.items(), key=lambda item: item[0].casefold()
            )
        ],
    }
