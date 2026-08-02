"""Document-catalog search, facets and detail routes.

These run against ``tests/fixtures/catalog``, a synthetic three-document
catalog authored by this project, so they need no model weights.
"""

from fastapi.testclient import TestClient

from identity_analysis.api import app
from identity_analysis.document_classifier import (
    document_catalog_entry,
    document_catalog_facets,
    search_document_catalog,
)

from conftest import BACK_LAYOUT, FIXTURE_CATALOG, FRONT_LAYOUT


def test_catalog_search_filters_country_type_and_query() -> None:
    results = search_document_catalog(
        FIXTURE_CATALOG,
        query="specimen",
        country_code="zzt",
        document_type="IdentityCard",
    )

    assert results
    assert all("ZZT" in document["isoCodes"] for document in results)
    assert all(
        document["documentType"]["name"] == "IdentityCard" for document in results
    )
    assert all("specimen" in document["caption"].casefold() for document in results)
    assert all(not document["deprecated"] for document in results)


def test_catalog_search_includes_deprecated_only_when_requested() -> None:
    current = search_document_catalog(FIXTURE_CATALOG)
    complete = search_document_catalog(FIXTURE_CATALOG, include_deprecated=True)

    assert all(not document["deprecated"] for document in current)
    assert len(complete) > len(current)
    assert any(document["deprecated"] for document in complete)


def test_catalog_search_on_an_absent_catalog_returns_nothing(tmp_path) -> None:
    assert search_document_catalog(tmp_path) == []
    assert document_catalog_entry(tmp_path, FRONT_LAYOUT) is None
    assert document_catalog_facets(tmp_path)["documentCount"] == 0


def test_catalog_endpoint_returns_stable_paginated_items(catalog_only_api) -> None:
    with TestClient(app) as client:
        response = client.get(
            "/v1/document/catalog",
            params={
                "q": "Specimen",
                "countryCode": "ZZT",
                "documentType": "IdentityCard",
                "limit": 1,
            },
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] > 1
    assert data["offset"] == 0
    assert data["limit"] == 1
    assert len(data["items"]) == 1
    assert all(item["countryCodes"] == ["ZZT"] for item in data["items"])
    assert all(item["type"] == "IdentityCard" for item in data["items"])
    assert all(item["layoutEvidencePath"].endswith("/evidence") for item in data["items"])
    assert all("sourceMember" not in item for item in data["items"])


def test_catalog_endpoint_paginates_deterministically(catalog_only_api) -> None:
    with TestClient(app) as client:
        first = client.get(
            "/v1/document/catalog",
            params={"countryCode": "ZZT", "offset": 0, "limit": 1},
        ).json()["data"]
        second = client.get(
            "/v1/document/catalog",
            params={"countryCode": "ZZT", "offset": 1, "limit": 1},
        ).json()["data"]

    assert first["total"] == second["total"]
    assert first["items"][0]["identifier"] != second["items"][0]["identifier"]


def test_catalog_entry_and_detail_endpoint_share_identity(catalog_only_api) -> None:
    entry = document_catalog_entry(FIXTURE_CATALOG, FRONT_LAYOUT)

    assert entry["caption"] == "Specimen Identity Card (2024) Front"
    with TestClient(app) as client:
        response = client.get(f"/v1/document/catalog/{FRONT_LAYOUT}")
        missing = client.get("/v1/document/catalog/999999999")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["identifier"] == FRONT_LAYOUT
    assert data["name"] == "Specimen Identity Card (2024) Front"
    assert data["countryCodes"] == ["ZZT"]
    assert data["layoutAvailable"] is True
    assert data["pageRole"]["role"] == "front"
    assert data["pageRole"]["method"] == "caption_marker"
    assert missing.status_code == 404


def test_catalog_back_layout_declares_the_back_page_role(catalog_only_api) -> None:
    with TestClient(app) as client:
        response = client.get(f"/v1/document/catalog/{BACK_LAYOUT}")

    assert response.status_code == 200
    assert response.json()["data"]["pageRole"]["role"] == "back"


def test_catalog_facets_expose_filter_values_and_counts(catalog_only_api) -> None:
    facets = document_catalog_facets(FIXTURE_CATALOG)
    complete = document_catalog_facets(FIXTURE_CATALOG, include_deprecated=True)

    assert complete["documentCount"] > facets["documentCount"]
    assert any(
        country["code"] == "ZZT" and country["count"] > 0
        for country in facets["countries"]
    )
    assert any(
        item["name"] == "IdentityCard" and item["count"] > 0
        for item in facets["documentTypes"]
    )
    assert any(
        item["name"] == "ID1" and item["count"] > 0
        for item in facets["documentFormats"]
    )
    facets["countries"][0]["count"] = -1
    assert document_catalog_facets(FIXTURE_CATALOG)["countries"][0]["count"] >= 0

    with TestClient(app) as client:
        response = client.get("/v1/document/catalog/facets")

    assert response.status_code == 200
    assert response.json()["data"]["documentCount"] == facets["documentCount"]


def test_catalog_endpoint_validates_pagination(catalog_only_api) -> None:
    with TestClient(app) as client:
        negative_offset = client.get(
            "/v1/document/catalog", params={"offset": -1}
        )
        oversized_limit = client.get(
            "/v1/document/catalog", params={"limit": 101}
        )

    assert negative_offset.status_code == 422
    assert oversized_limit.status_code == 422
