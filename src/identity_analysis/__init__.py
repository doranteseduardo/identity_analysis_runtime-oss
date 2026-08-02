"""Portable document and facial identity analysis pipelines."""

from .capabilities import SDK_COMPATIBILITY
from .pipeline import (
    parse_td1,
    process_document,
    process_document_pair,
    process_document_pages,
    recognize_front,
    recognize_td1,
)

__all__ = [
    "SDK_COMPATIBILITY",
    "parse_td1",
    "process_document",
    "process_document_pair",
    "process_document_pages",
    "recognize_front",
    "recognize_td1",
]
__version__ = "0.1.0"
