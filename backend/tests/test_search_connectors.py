from app.services.retrieval import search_connectors


def test_provider_results_for_query_filters_excluded_domains_for_seed_similar(monkeypatch):
    monkeypatch.setattr(
        search_connectors,
        "_search_exa_similar",
        lambda seed_url, cap: [
            {"url": "https://4tpm.fr/", "title": "4TPM"},
            {"url": "https://vendor.example.com/", "title": "Vendor"},
        ],
    )

    rows = search_connectors.provider_results_for_query(
        "exa",
        query="ignored",
        cap=5,
        exclude_domains=["4tpm.fr"],
        seed_url="https://seed.example.com",
    )

    assert [row["url"] for row in rows] == ["https://vendor.example.com/"]


def test_provider_results_for_query_filters_excluded_subdomains(monkeypatch):
    monkeypatch.setattr(
        search_connectors,
        "_search_exa_similar",
        lambda seed_url, cap: [
            {"url": "https://wwwuat.slib.com/fr", "title": "SLIB UAT"},
            {"url": "https://vendor.example.com/", "title": "Vendor"},
        ],
    )

    rows = search_connectors.provider_results_for_query(
        "exa",
        query="ignored",
        cap=5,
        exclude_domains=["slib.com"],
        seed_url="https://seed.example.com",
    )

    assert [row["url"] for row in rows] == ["https://vendor.example.com/"]
