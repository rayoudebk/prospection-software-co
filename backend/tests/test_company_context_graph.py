from app.models.workspace import CompanyProfile
from app.models.company_context import CompanyContextPack
from app.services.company_context_graph import (
    _build_secondary_queries,
    _canonicalize_graph,
    build_company_context_graph,
    build_company_context_payload,
    sync_company_context_pack_graph,
)
from app.services.company_context import build_company_context_artifacts


def _build_profile() -> CompanyProfile:
    return CompanyProfile(
        workspace_id=42,
        buyer_company_url="https://4tpm.fr",
        comparator_seed_urls=["https://wealth-dynamix.com"],
        supporting_evidence_urls=["https://4tpm.fr/platform/front-office"],
        comparator_seed_summaries={},
        geo_scope={"region": "EU+UK", "include_countries": ["FR"], "exclude_countries": []},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr",
                    "company_name": "4TPM",
                    "summary": "Wealth management platform for front-to-back office operations.",
                    "signals": [
                        {"type": "capability", "value": "PMS/OMS", "source_url": "https://4tpm.fr/platform/front-office"},
                        {"type": "workflow", "value": "Front office titres", "source_url": "https://4tpm.fr/platform/front-office"},
                        {"type": "integration", "value": "REST API", "source_url": "https://4tpm.fr/technology-services/documentation-api"},
                    ],
                    "customer_evidence": [
                        {"name": "BNP Paribas", "source_url": "https://4tpm.fr/customers"},
                    ],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "Front office",
                            "page_type": "platform",
                            "blocks": [
                                {"type": "heading", "content": "Pré-trade, trading et post-trade"},
                                {"type": "heading", "content": "Connectivité et routage multi-marchés"},
                                {"type": "heading", "content": "Banques privées"},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": (
                                "Pré-trade, trading et post-trade. Connectivité et routage multi-marchés. "
                                "Banques privées, asset managers, and online brokers use the platform."
                            ),
                        },
                        {
                            "url": "https://4tpm.fr/technology-services/documentation-api",
                            "title": "API Documentation",
                            "page_type": "docs",
                            "blocks": [{"type": "heading", "content": "REST API"}],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "REST API documentation for integration and infrastructure services.",
                        }
                    ],
                }
            ]
        },
        product_pages_found=8,
    )


def test_company_context_graph_preserves_primary_provenance(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {"results": [], "provider_mix": {}, "errors": []},
    )
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    profile = _build_profile()
    graph = build_company_context_graph(profile, payload=build_company_context_artifacts(profile))

    labels = {node["label"] for node in graph["nodes"]}
    assert "Company" in labels
    assert "CustomerEntity" in labels
    assert "Capability" in labels
    assert "Workflow" in labels
    assert "CustomerArchetype" in labels
    assert "DeliveryIntegration" in labels
    assert "SourceDocument" in labels
    assert graph["sourcing_brief"]["named_customer_proof"]
    assert any(edge["type"] == "SUPPORTED_BY" and edge["source_type"] == "primary" for edge in graph["edges"])
    assert any(node["label"] == "Capability" and node.get("source_document_id") for node in graph["nodes"])
    assert all("source_document_id" in node for node in graph["nodes"])
    assert len([node for node in graph["nodes"] if node["label"] == "Capability"]) <= 12


def test_company_context_graph_adds_secondary_evidence_proof(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {
            "results": [
                {
                    "url": "https://www.thewealthmosaic.com/needs/portfolio-wealth-management-systems/",
                    "title": "Portfolio wealth management systems | The Wealth Mosaic",
                    "snippet": "4TPM is listed among portfolio wealth management systems.",
                },
                {
                    "url": "https://partner.example.com/news/4tpm-integration",
                    "title": "Partner announces 4TPM integration",
                    "snippet": "The partner announces an integration with 4TPM and BNP Paribas.",
                },
                {
                    "url": "https://www.dossierc.com/companies/4tpm.fr",
                    "title": "Email @4tpm.fr",
                    "snippet": "Password Should contain atleast 6 characters",
                },
                {
                    "url": "https://www.rcsb.org/structure/4TPM",
                    "title": "RCSB PDB - 4TPM: Crystal structure of a ligand",
                    "snippet": "Protein Data Bank entry for 4TPM inhibitor complex",
                },
            ],
            "provider_mix": {"exa": 3},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.fetch_page_fast",
        lambda url: {
            "url": url,
            "content": (
                "Password Should contain atleast 6 characters"
                if "dossierc.com" in url
                else (
                "4TPM integration with BNP Paribas and REST API support."
                if "partner.example.com" in url
                else (
                    "Protein Data Bank entry for 4TPM inhibitor complex."
                    if "rcsb.org" in url
                    else "Portfolio wealth management systems category page for 4TPM."
                )
                )
            ),
            "provider": "jina_reader",
            "error": None,
        },
    )
    profile = _build_profile()
    payload = build_company_context_payload(build_company_context_artifacts(profile), profile)

    assert payload["company_context_graph"]["secondary_evidence_proof"]
    assert any(
        node["label"] == "Category" for node in payload["company_context_graph"]["nodes"]
    )
    assert any(
        edge["type"] in {"LISTED_IN_CATEGORY", "ANNOUNCED_BY_CUSTOMER", "INTEGRATES_WITH"}
        for edge in payload["company_context_graph"]["edges"]
    )
    assert all(
        "dossierc.com" not in str(item.get("url") or "")
        for item in payload["company_context_graph"]["secondary_evidence_proof"]
    )
    assert all(
        "rcsb.org" not in str(item.get("url") or "")
        for item in payload["company_context_graph"]["secondary_evidence_proof"]
    )
    assert payload["sourcing_brief"]["customer_partner_corroboration"]
    assert payload["sourcing_brief"]["directory_category_context"]
    assert all(
        item.get("query_type") in {"directory_category", "customer_corroboration", "partner_corroboration", "market_context"}
        for item in payload["company_context_graph"]["secondary_evidence_proof"]
    )


def test_company_context_payload_preserves_expansion_inputs(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {"results": [], "provider_mix": {}, "errors": []},
    )
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    profile = _build_profile()
    profile.context_pack_json["sites"].append(
        {
            "url": "https://wealth-dynamix.com/",
            "company_name": "CRM, CLM & Onboarding solutions for private banks & wealth management firms",
            "summary": "Client lifecycle management platform for private banks and wealth managers.",
            "signals": [
                {"type": "capability", "value": "Client lifecycle management", "source_url": "https://wealth-dynamix.com/"},
                {"type": "workflow", "value": "Client onboarding", "source_url": "https://wealth-dynamix.com/"},
            ],
            "customer_evidence": [],
            "pages": [
                {
                    "url": "https://wealth-dynamix.com/",
                    "title": "Wealth Dynamix",
                    "page_type": "homepage",
                    "blocks": [{"type": "heading", "content": "Client lifecycle management"}],
                    "signals": [],
                    "customer_evidence": [],
                    "raw_content": "Client lifecycle management and onboarding for private banks and wealth managers.",
                }
            ],
        }
    )
    payload = build_company_context_payload(build_company_context_artifacts(profile), profile)

    assert payload["deep_research_handoff"]["source_company_truth"]["source_company"]["name"]
    assert len(payload["expansion_inputs"]) == 1
    assert payload["expansion_inputs"][0]["name"] == "Wealth Dynamix"


def test_secondary_queries_anchor_on_exact_company_identity(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    profile = _build_profile()
    primary_graph = build_company_context_graph(profile, payload=build_company_context_artifacts(profile))

    queries = _build_secondary_queries(primary_graph, profile.comparator_seed_urls or [])

    assert queries
    assert any(
        query["query_type"] == "directory_category"
        and '"4TPM"' in str(query.get("query_text") or "")
        and "thewealthmosaic.com" in (query.get("domain_allowlist") or [])
        for query in queries
    )
    assert any(
        query["query_type"] == "customer_corroboration"
        and '"4TPM"' in str(query.get("query_text") or "")
        and '"BNP Paribas"' in str(query.get("query_text") or "")
        for query in queries
    )


def test_secondary_queries_include_one_customer_query_per_named_customer(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    profile = _build_profile()
    profile.context_pack_json["sites"][0]["customer_evidence"] = [
        {"name": "BNP Paribas", "source_url": "https://4tpm.fr/customers"},
        {"name": "Allianz Bank", "source_url": "https://4tpm.fr/customers"},
        {"name": "SwissLife", "source_url": "https://4tpm.fr/customers"},
    ]
    primary_graph = build_company_context_graph(profile, payload=build_company_context_artifacts(profile))

    queries = _build_secondary_queries(primary_graph, profile.comparator_seed_urls or [])

    customer_queries = [query for query in queries if query.get("query_type") == "customer_corroboration"]
    customer_query_texts = {str(query.get("query_text") or "") for query in customer_queries}
    assert '"4TPM" "BNP Paribas"' in customer_query_texts
    assert '"4TPM" "Allianz Bank"' in customer_query_texts
    assert '"4TPM" "SwissLife"' in customer_query_texts
    assert all("crunchbase.com" in (query.get("domain_blocklist") or []) for query in customer_queries)
    partner_queries = [query for query in queries if query.get("query_type") == "partner_corroboration"]
    assert all("REST API" not in str(query.get("query_text") or "") for query in partner_queries)


def test_secondary_company_graph_uses_secondary_specific_provider_order(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )

    captured: dict[str, object] = {}

    def _fake_search(
        queries,
        *,
        provider_order,
        per_query_cap,
        total_cap,
        per_domain_cap,
        max_seconds=None,
        cache=None,
    ):
        captured["provider_order"] = provider_order
        captured["per_query_cap"] = per_query_cap
        captured["total_cap"] = total_cap
        captured["per_domain_cap"] = per_domain_cap
        captured["max_seconds"] = max_seconds
        return {"results": [], "provider_mix": {}, "errors": []}

    monkeypatch.setattr("app.services.company_context_graph.run_external_search_queries", _fake_search)
    profile = _build_profile()

    build_company_context_graph(profile, payload=build_company_context_artifacts(profile))

    assert captured["provider_order"] == ["serper", "brave"]
    assert captured["per_query_cap"] == 4
    assert captured["max_seconds"] == 30


def test_company_profile_secondary_result_is_kept_as_other_context(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {
            "results": [
                {
                    "url": "https://www.crunchbase.com/organization/4tpm",
                    "title": "4TPM - Crunchbase Company Profile & Funding",
                    "snippet": "Crunchbase profile for 4TPM.",
                    "query_type": "market_context",
                }
            ],
            "provider_mix": {"exa": 1},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.fetch_page_fast",
        lambda url: {
            "url": url,
            "content": "4TPM is a wealth management technology company serving private banks and asset managers.",
            "provider": "jina_reader",
            "error": None,
        },
    )
    profile = _build_profile()

    payload = build_company_context_payload(build_company_context_artifacts(profile), profile)

    assert payload["sourcing_brief"]["secondary_evidence_proof"]
    assert any(
        "crunchbase.com" in str(item.get("url") or "")
        for item in payload["sourcing_brief"]["other_secondary_context"]
    )


def test_company_profile_result_is_not_treated_as_customer_corroboration(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {
            "results": [
                {
                    "url": "https://www.crunchbase.com/organization/4tpm",
                    "title": "4TPM - Crunchbase Company Profile & Funding",
                    "snippet": "Crunchbase profile for 4TPM mentioning BNP Paribas.",
                    "query_type": "customer_corroboration",
                }
            ],
            "provider_mix": {"serpapi": 1},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.fetch_page_fast",
        lambda url: {
            "url": url,
            "content": "4TPM company profile mentioning BNP Paribas.",
            "provider": "jina_reader",
            "error": None,
        },
    )
    profile = _build_profile()

    payload = build_company_context_payload(build_company_context_artifacts(profile), profile)

    assert payload["sourcing_brief"]["customer_partner_corroboration"] == []
    assert payload["sourcing_brief"]["secondary_evidence_proof"] == []


def test_low_signal_host_result_is_not_treated_as_customer_corroboration(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {
            "results": [
                {
                    "url": "https://www.pappers.fr/dirigeant/willy_van%20stappen_1956-01",
                    "title": "Willy Van Stappen - Pappers",
                    "snippet": "Corporate officer profile mentioning BNP Paribas.",
                    "query_type": "customer_corroboration",
                }
            ],
            "provider_mix": {"serper": 1},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "app.services.company_context_graph.fetch_page_fast",
        lambda url: {
            "url": url,
            "content": "Corporate officer profile mentioning BNP Paribas.",
            "provider": "jina_reader",
            "error": None,
        },
    )
    profile = _build_profile()

    payload = build_company_context_payload(build_company_context_artifacts(profile), profile)

    assert payload["sourcing_brief"]["customer_partner_corroboration"] == []
    assert payload["sourcing_brief"]["secondary_evidence_proof"] == []


def test_sync_company_context_pack_graph_persists_deep_research_handoff(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context_graph.Neo4jCompanyContextGraphStore.sync_graph",
        lambda self, graph: {"status": "success", "error": None, "graph_ref": graph.get("graph_ref")},
    )
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    profile = _build_profile()
    pack = CompanyContextPack(workspace_id=42)

    payload = sync_company_context_pack_graph(
        pack,
        profile,
        payload_override=build_company_context_artifacts(profile),
    )

    assert payload["deep_research_handoff"]["source_company_truth"]["source_company"]["name"] == "4TPM"
    assert pack.company_context_graph_cache_json["deep_research_handoff"]["source_company_truth"]["source_company"]["name"] == "4TPM"


def test_primary_graph_excludes_comparator_seed_documents(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context_graph.run_external_search_queries",
        lambda *args, **kwargs: {"results": [], "provider_mix": {}, "errors": []},
    )
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    profile = _build_profile()
    graph = build_company_context_graph(profile, payload=build_company_context_artifacts(profile))

    source_urls = {item.get("url") for item in graph.get("source_documents") or []}
    assert "https://4tpm.fr/" in source_urls
    assert "https://wealth-dynamix.com/" not in source_urls


def test_canonicalize_graph_dedupes_duplicate_node_ids():
    graph = {
        "nodes": [
            {"id": "source_document_x", "label": "SourceDocument", "url": "https://example.com/a", "name": "A"},
            {"id": "source_document_x", "label": "SourceDocument", "url": "https://example.com/a", "name": "A duplicate"},
            {"id": "company_x", "label": "Company", "name": "Acme"},
        ],
        "edges": [
            {"id": "edge_1", "type": "SUPPORTED_BY", "from_id": "company_x", "to_id": "source_document_x"},
            {"id": "edge_1", "type": "SUPPORTED_BY", "from_id": "company_x", "to_id": "source_document_x"},
            {"id": "edge_2", "type": "SUPPORTED_BY", "from_id": "company_x", "to_id": "missing_node"},
        ],
        "source_documents": [
            {"id": "source_document_x", "label": "SourceDocument", "url": "https://example.com/a", "name": "A"},
            {"id": "source_document_x", "label": "SourceDocument", "url": "https://example.com/a", "name": "A duplicate"},
        ],
        "graph_stats": {},
    }

    canonical = _canonicalize_graph(graph)

    assert len(canonical["nodes"]) == 2
    assert len(canonical["edges"]) == 1
    assert len(canonical["source_documents"]) == 1
