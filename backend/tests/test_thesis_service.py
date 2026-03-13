from app.models.workspace import CompanyProfile
from app.services.thesis import (
    apply_thesis_adjustment_operations,
    bootstrap_thesis_payload,
    build_context_pack_v2,
    derive_search_lane_payloads,
)


def _build_profile() -> CompanyProfile:
    return CompanyProfile(
        workspace_id=1,
        buyer_company_url="https://acme.example.com",
        generated_context_summary=(
            "Acme is a SaaS portfolio analytics platform for private equity and fund operations teams. "
            "The company also offers implementation services and API-based integrations."
        ),
        reference_company_urls=["https://comp-one.example.com"],
        reference_evidence_urls=["https://acme.example.com/customers"],
        reference_summaries={"https://comp-one.example.com": "Comparable analytics vendor."},
        geo_scope={"region": "US", "include_countries": ["US"], "exclude_countries": ["RU"]},
        context_pack_json={
            "sites": [
                {
                    "url": "https://acme.example.com",
                    "company_name": "Acme",
                    "summary": "Portfolio analytics and reporting for fund operations teams.",
                    "signals": [
                        {
                            "type": "capability",
                            "value": "Portfolio analytics",
                            "source_url": "https://acme.example.com/platform",
                        },
                        {
                            "type": "service",
                            "value": "Implementation services",
                            "source_url": "https://acme.example.com/services",
                        },
                        {
                            "type": "integration",
                            "value": "API integrations",
                            "source_url": "https://acme.example.com/integrations",
                        },
                    ],
                    "customer_evidence": [
                        {
                            "name": "Northwind Capital",
                            "source_url": "https://acme.example.com/customers",
                        }
                    ],
                    "pages": [
                        {
                            "url": "https://acme.example.com/platform",
                            "title": "Platform",
                            "signals": [],
                            "customer_evidence": [],
                        }
                    ],
                }
            ]
        },
        product_pages_found=5,
    )


def test_bootstrap_thesis_payload_generates_claims_and_source_pills():
    payload = bootstrap_thesis_payload(_build_profile())

    assert payload["summary"].startswith("Acme is a SaaS portfolio analytics platform")
    assert payload["source_pills"]
    assert payload["buyer_evidence"]["status"] == "sufficient"
    assert payload["buyer_evidence"]["used_for_inference"] is True
    sections = {claim["section"] for claim in payload["claims"]}
    assert "core_capability" in sections
    assert "adjacent_capability" in sections
    assert "business_model" in sections
    assert "customer_profile" in sections
    assert "geography" in sections
    renderings = {claim["rendering"] for claim in payload["claims"]}
    assert "fact" in renderings
    assert "hypothesis" in renderings
    assert payload["context_pack_v2"]["version"] == "v2"
    assert "taxonomy_nodes" in payload
    assert "market_map_brief" in payload
    assert payload["market_map_brief"]["named_customer_proof"]
    assert payload["lens_seeds"]


def test_derive_search_lane_payloads_prefers_core_and_adjacent_claims():
    profile = _build_profile()
    thesis_payload = bootstrap_thesis_payload(profile)

    lanes = derive_search_lane_payloads(thesis_payload, profile)

    assert len(lanes) == 2
    by_type = {lane["lane_type"]: lane for lane in lanes}
    assert "Portfolio analytics" in by_type["core"]["capabilities"]
    assert "Implementation services" in by_type["adjacent"]["capabilities"]
    assert by_type["core"]["status"] == "draft"
    assert "SaaS / subscription software" in by_type["core"]["must_include_terms"]


def test_apply_thesis_adjustment_operations_updates_claims_and_questions():
    thesis_payload = bootstrap_thesis_payload(_build_profile())
    first_claim = thesis_payload["claims"][0]

    adjusted = apply_thesis_adjustment_operations(
        thesis_payload,
        [
            {"op": "confirm_claim", "claim_id": first_claim["id"], "user_status": "confirmed"},
            {"op": "add_claim", "section": "adjacent_capability", "value": "Voting rights workflow"},
            {"op": "remove_claim", "claim_id": first_claim["id"]},
            {"op": "add_open_question", "value": "What company-size window matters most?"},
        ],
    )

    by_id = {claim["id"]: claim for claim in adjusted["claims"]}
    assert by_id[first_claim["id"]]["user_status"] == "removed"
    assert any(claim["value"] == "Voting rights workflow" for claim in adjusted["claims"])
    assert "What company-size window matters most?" in adjusted["open_questions"]
    assert adjusted["applied_operations"]


def test_bootstrap_thesis_payload_supports_investment_thesis_only():
    profile = CompanyProfile(
        workspace_id=2,
        buyer_company_url=None,
        manual_brief_text=(
            "I want to invest in companies in Europe that provide software to healthcare actors "
            "(hospitals, doctors, etc) as licences with no SaaS, under $10M in revenue, and with 30-50 employees."
        ),
        reference_company_urls=[],
        reference_evidence_urls=[],
        reference_summaries={},
        geo_scope={},
        context_pack_json={},
        product_pages_found=0,
    )

    payload = bootstrap_thesis_payload(profile)
    claims_by_section = {}
    for claim in payload["claims"]:
        claims_by_section.setdefault(claim["section"], []).append(claim["value"])

    assert "Software for healthcare actors" in claims_by_section["core_capability"]
    assert "License-based software" in claims_by_section["business_model"]
    assert "Exclude SaaS-first software companies" in claims_by_section["exclude_constraint"]
    assert "Prefer companies under $10M revenue" in claims_by_section["include_constraint"]
    assert "Employee estimate: 30-50 employees" in claims_by_section["size_signal"]
    assert "Primary sourcing region: Europe" in claims_by_section["geography"]
    assert "market_map_brief" in payload
    assert isinstance(payload["market_map_brief"]["open_questions"], list)

    lanes = derive_search_lane_payloads(payload, profile)
    core_lane = next(lane for lane in lanes if lane["lane_type"] == "core")
    assert "Software for healthcare actors" in core_lane["capabilities"]
    assert core_lane["title"].startswith("Core: ")


def test_bootstrap_thesis_payload_ignores_comparator_context_for_empty_buyer_site():
    profile = CompanyProfile(
        workspace_id=3,
        buyer_company_url="https://4tpm.fr/",
        generated_context_summary=(
            "CWAN targets asset managers and wealth managers with a cloud-native SaaS platform "
            "and managed services."
        ),
        reference_company_urls=["https://cwan.com/"],
        reference_evidence_urls=[],
        reference_summaries={
            "https://cwan.com/": (
                "CWAN serves institutional investors with cloud deployment, integrations, "
                "and implementation services."
            )
        },
        geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
        context_pack_markdown="# 4TPM\n\n**Website:** https://4tpm.fr\n\n---\n\n# CWAN\n\nCloud-native SaaS for asset managers.",
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/",
                            "title": "4TPM - Plateforme Wealth Management",
                            "blocks": [],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "",
                        }
                    ],
                },
                {
                    "url": "https://cwan.com/",
                    "company_name": "CWAN",
                    "summary": "Cloud-native SaaS for institutional investors.",
                    "signals": [
                        {
                            "type": "capability",
                            "value": "Portfolio analytics",
                            "source_url": "https://cwan.com/platform",
                        }
                    ],
                    "customer_evidence": [
                        {
                            "name": "Asset managers",
                            "source_url": "https://cwan.com/customers",
                        }
                    ],
                    "pages": [],
                },
            ]
        },
        product_pages_found=0,
    )

    payload = bootstrap_thesis_payload(profile)

    assert payload["buyer_evidence"]["status"] == "insufficient"
    assert payload["buyer_evidence"]["used_for_inference"] is False
    assert payload["summary"].startswith("Buyer website crawled, but no first-party product or customer evidence was extracted yet.")
    assert not any(claim["section"] == "customer_profile" for claim in payload["claims"])
    assert not any(claim["section"] == "business_model" for claim in payload["claims"])
    assert not any(claim["section"] == "deployment_model" for claim in payload["claims"])
    assert not any(claim["section"] == "core_capability" for claim in payload["claims"])
    assert any(
        claim["section"] == "geography" and claim["value"] == "Primary sourcing region: EU+UK"
        for claim in payload["claims"]
    )


def test_build_context_pack_v2_keeps_high_signal_job_pages():
    context_pack = build_context_pack_v2(
        {
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://hublo.com",
                    "company_name": "Hublo",
                    "pages": [
                        {
                            "url": "https://careers.hublo.com/jobs/7331974-head-of-data-ai-f-h-n",
                            "title": "Head of Data & AI",
                            "page_type": "careers",
                            "blocks": [{"type": "heading", "content": "Data platform for healthcare staffing"}],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Work closely with product, customers, operations, AI, and scheduling workflows.",
                        },
                        {
                            "url": "https://careers.hublo.com/jobs/talent-partner",
                            "title": "Talent Partner",
                            "page_type": "careers",
                            "blocks": [],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "General recruiting process and interviews.",
                        },
                    ],
                }
            ],
        }
    )

    selected_pages = context_pack["sites"][0]["selected_pages"]
    selected_urls = {page["url"] for page in selected_pages}
    assert "https://careers.hublo.com/jobs/7331974-head-of-data-ai-f-h-n" in selected_urls
    assert "https://careers.hublo.com/jobs/talent-partner" not in selected_urls


def test_build_context_pack_v2_filters_noisy_customer_evidence_and_uses_raw_text_for_phrases():
    context_pack = build_context_pack_v2(
        {
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://wealth.example.com",
                    "company_name": "WealthCo",
                    "summary": "Client lifecycle management software for private banks and wealth managers.",
                    "customer_evidence": [
                        {
                            "name": "Why first impressions matter: Rethinking onboarding in wealth management",
                            "source_url": "https://wealth.example.com/insights",
                            "evidence_type": "logo_alt",
                            "context": "Recent insights",
                        },
                        {
                            "name": "BNP Paribas",
                            "source_url": "https://wealth.example.com/customers",
                            "evidence_type": "logo_alt",
                            "context": "Trusted by leading private banks",
                        },
                    ],
                    "pages": [
                        {
                            "url": "https://wealth.example.com/solutions/clm",
                            "title": "Client lifecycle management for private banks",
                            "page_type": "solutions",
                            "blocks": [],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": (
                                "Our client lifecycle management platform helps private banks streamline "
                                "onboarding and portfolio reporting workflows."
                            ),
                        }
                    ],
                }
            ],
        }
    )

    site = context_pack["sites"][0]
    assert [item["name"] for item in site["named_customers"]] == ["BNP Paribas"]
    assert "private bank" in [phrase.lower() for phrase in context_pack["extracted_raw_phrases"]]
    assert "portfolio reporting" in [phrase.lower() for phrase in context_pack["extracted_raw_phrases"]]


def test_bootstrap_thesis_payload_builds_taxonomy_from_spa_style_phrases():
    profile = CompanyProfile(
        workspace_id=4,
        buyer_company_url="https://4tpm.fr/",
        generated_context_summary="4TPM supports wealth management workflows and API integrations.",
        reference_company_urls=[],
        reference_evidence_urls=[],
        reference_summaries={},
        geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
        context_pack_json={
            "version": "v2",
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://4tpm.fr/",
                    "company_name": "4TPM",
                    "website": "https://4tpm.fr/",
                    "selected_pages": [],
                }
            ],
                "evidence_items": [
                    {"id": "e1", "kind": "page_signal:customer_archetype", "text": "Banques privées"},
                    {"id": "e1b", "kind": "page_signal:workflow", "text": "Front office"},
                    {"id": "e2", "kind": "page_signal:workflow", "text": "Front office titres"},
                    {"id": "e2b", "kind": "page_signal:workflow", "text": "Operations"},
                    {"id": "e2c", "kind": "page_signal:workflow", "text": "Infrastructure"},
                    {"id": "e2d", "kind": "page_signal:workflow", "text": "4TPM PATIO"},
                    {"id": "e3", "kind": "page_signal:service", "text": "Documentation API"},
                    {"id": "e3b", "kind": "page_signal:service", "text": "APIs REST"},
                    {"id": "e3c", "kind": "page_signal:integration", "text": "BPCE"},
                    {"id": "e4", "kind": "page_customer:bundle_logo_manifest", "text": "Procapital"},
                ],
            "named_customers": [
                {"name": "Allianz Bank", "evidence_id": "cust1", "context": "Trusted by leading banques privées"}
            ],
            "integrations": [],
            "partners": [],
                "extracted_raw_phrases": [
                    "Banques privées",
                    "Bourse en ligne",
                    "Front office titres",
                    "Front office",
                    "Operations",
                    "Infrastructure",
                    "4TPM PATIO",
                    "Back office titres",
                    "Documentation API",
                    "APIs REST",
                    "4TPM - Plateforme Wealth Management",
                    "BPCE",
                    "Procapital",
                ],
                "crawl_coverage": {"total_sites": 1, "total_pages": 4},
            },
        product_pages_found=4,
    )

    payload = bootstrap_thesis_payload(profile)
    taxonomy_by_layer = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])

    assert "Private bank" in taxonomy_by_layer["customer_archetype"]
    assert "Online brokerage" in taxonomy_by_layer["customer_archetype"]
    assert "Front office titres" in taxonomy_by_layer["workflow"]
    assert "Front office" not in taxonomy_by_layer["workflow"]
    assert "Operations" not in taxonomy_by_layer["workflow"]
    assert "Infrastructure" not in taxonomy_by_layer["workflow"]
    assert "4TPM PATIO" not in taxonomy_by_layer["workflow"]
    assert "API documentation" in taxonomy_by_layer["delivery_or_integration"]
    assert "REST API" in taxonomy_by_layer["delivery_or_integration"]
    assert "Infrastructure" in taxonomy_by_layer["delivery_or_integration"]
    assert "BPCE" not in taxonomy_by_layer["delivery_or_integration"]
    assert "Wealth management platform" in taxonomy_by_layer["capability"]
    assert "Procapital" not in taxonomy_by_layer.get("capability", [])
    assert "Bank" not in taxonomy_by_layer["customer_archetype"]


def test_bootstrap_thesis_payload_scopes_market_map_to_buyer_site():
    profile = CompanyProfile(
        workspace_id=9,
        buyer_company_url="https://4tpm.fr/",
        generated_context_summary="",
        reference_company_urls=["https://wealth-dynamix.com/"],
        reference_evidence_urls=[],
        reference_summaries={},
        geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
        context_pack_json={
            "version": "v2",
            "generated_at": "2026-03-12T00:00:00Z",
            "sites": [
                {
                    "url": "https://4tpm.fr/",
                    "company_name": "4TPM",
                    "website": "https://4tpm.fr/",
                    "selected_pages": [],
                    "evidence_items": [
                        {"id": "s1", "kind": "page_signal:customer_archetype", "text": "Banques privées"},
                        {"id": "s2", "kind": "page_signal:workflow", "text": "Front office titres"},
                        {"id": "s3", "kind": "page_signal:service", "text": "Documentation API"},
                    ],
                    "named_customers": [
                        {"name": "Allianz Bank", "evidence_id": "cust1", "context": "Trusted by leading banques privées"}
                    ],
                    "integrations": [{"name": "BPCE", "evidence_id": "int1"}],
                    "partners": [{"name": "BPCE", "evidence_id": "int1"}],
                    "extracted_raw_phrases": [
                        "Banques privées",
                        "Front office titres",
                        "Documentation API",
                        "4TPM - Plateforme Wealth Management",
                    ],
                    "crawl_coverage": {"total_pages": 4, "page_type_counts": {"other": 4}, "pages_with_signals": 4, "pages_with_customer_evidence": 1, "career_pages_selected": 0},
                },
                {
                    "url": "https://wealth-dynamix.com/",
                    "company_name": "Wealth Dynamix",
                    "website": "https://wealth-dynamix.com/",
                    "selected_pages": [],
                    "evidence_items": [
                        {"id": "c1", "kind": "page_signal:workflow", "text": "Portfolio management"},
                        {"id": "c2", "kind": "page_signal:capability", "text": "Integrate data from all sources with a unified data model"},
                    ],
                    "named_customers": [{"name": "Salesforce", "evidence_id": "cust2"}],
                    "integrations": [],
                    "partners": [],
                    "extracted_raw_phrases": [
                        "Portfolio management",
                        "Integrate data from all sources with a unified data model",
                    ],
                    "crawl_coverage": {"total_pages": 10, "page_type_counts": {"product": 10}, "pages_with_signals": 10, "pages_with_customer_evidence": 1, "career_pages_selected": 0},
                },
            ],
            "evidence_items": [],
            "named_customers": [],
            "integrations": [],
            "partners": [],
            "extracted_raw_phrases": [],
            "crawl_coverage": {"total_sites": 2, "total_pages": 14},
        },
        product_pages_found=4,
    )

    payload = bootstrap_thesis_payload(profile)
    taxonomy_by_layer = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])

    assert "Front office titres" in taxonomy_by_layer["workflow"]
    assert "Wealth management platform" in taxonomy_by_layer["capability"]
    assert "Portfolio management" not in taxonomy_by_layer["workflow"]
    assert "Integrate data from all sources with" not in taxonomy_by_layer["capability"]


def test_bootstrap_thesis_payload_promotes_rendered_product_features_into_cleaner_capabilities():
    profile = CompanyProfile(
        workspace_id=10,
        buyer_company_url="https://4tpm.fr/platform/front-office",
        generated_context_summary="",
        reference_company_urls=[],
        reference_evidence_urls=[],
        reference_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "paragraph", "content": "Plateforme / Front Office titres"},
                                {"type": "heading", "content": "Front office titres pour gérants et desks de trading", "level": 1},
                                {"type": "heading", "content": "Capacités clés", "level": 2},
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                                {"type": "heading", "content": "Génération d'ordres blocs et routage full STP", "level": 3},
                                {"type": "heading", "content": "Architecture & Intégration", "level": 2},
                                {
                                    "type": "paragraph",
                                    "content": "FIX, fichiers, MQSeries, web services, APIs REST pour se connecter aux brokers et systèmes tiers.",
                                },
                                {"type": "heading", "content": "Fonctionnalités détaillées", "level": 2},
                                {
                                    "type": "list",
                                    "content": (
                                        "- Classification des clients par listes, contraintes et périmètre de gestion\n"
                                        "- Modèles de répartition d'actifs adaptés aux profils d'investissement\n"
                                        "- Le client alimente son compte espèces via un compte de dépôt appartenant à la banque"
                                    ),
                                },
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS REST API",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    payload = bootstrap_thesis_payload(profile)
    taxonomy_by_layer: dict[str, list[str]] = {}
    for node in payload["taxonomy_nodes"]:
        taxonomy_by_layer.setdefault(node["layer"], []).append(node["phrase"])
    surfaced_capabilities = [
        node["phrase"] for node in (payload["market_map_brief"].get("capability_nodes") or [])
    ]

    assert "Préparation et modélisation des portefeuilles" in taxonomy_by_layer["capability"]
    assert "Génération d'ordres blocs et routage full STP" in taxonomy_by_layer["capability"]
    assert "Le client alimente son compte espèces" not in taxonomy_by_layer["capability"]
    assert "REST API" in taxonomy_by_layer["delivery_or_integration"]
    assert "Préparation et modélisation des portefeuilles" in surfaced_capabilities
    assert "Génération d'ordres blocs et routage full STP" in surfaced_capabilities
    assert "Le client alimente son compte espèces" not in surfaced_capabilities


def test_bootstrap_thesis_payload_uses_market_map_reasoning_when_available(monkeypatch):
    profile = CompanyProfile(
        workspace_id=11,
        buyer_company_url="https://4tpm.fr/platform/front-office",
        generated_context_summary="",
        reference_company_urls=[],
        reference_evidence_urls=[],
        reference_summaries={},
        geo_scope={},
        context_pack_json={
            "sites": [
                {
                    "url": "https://4tpm.fr/platform/front-office",
                    "company_name": "4TPM",
                    "summary": "",
                    "signals": [],
                    "customer_evidence": [],
                    "pages": [
                        {
                            "url": "https://4tpm.fr/platform/front-office",
                            "title": "4TPM - Plateforme Wealth Management",
                            "page_type": "product",
                            "blocks": [
                                {"type": "heading", "content": "Front office titres pour gérants et desks de trading", "level": 1},
                                {"type": "heading", "content": "Préparation et modélisation des portefeuilles", "level": 3},
                                {"type": "heading", "content": "Génération d'ordres blocs et routage full STP", "level": 3},
                                {"type": "heading", "content": "Architecture & Intégration", "level": 2},
                                {"type": "paragraph", "content": "APIs REST et documentation API."},
                            ],
                            "signals": [],
                            "customer_evidence": [],
                            "raw_content": "Front office titres PMS OMS REST API",
                        }
                    ],
                }
            ]
        },
        product_pages_found=1,
    )

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

    class _FakeOrchestrator:
        def run_stage(self, request):
            import json

            assert request.stage.value == "market_map_reasoning"
            prompt = request.prompt
            input_idx = prompt.index("Input:\n") + len("Input:\n")
            payload = json.loads(prompt[input_idx:])
            node_by_phrase = {
                node["phrase"]: node["id"]
                for node in payload["taxonomy_nodes"]
            }
            return _FakeResponse(
                json.dumps(
                    {
                        "source_summary": "4TPM appears to sell front-office wealth management capabilities to private-bank and brokerage buyers.",
                        "customer_node_ids": [],
                        "workflow_node_ids": [],
                        "capability_node_ids": [
                            node_by_phrase["Préparation et modélisation des portefeuilles"],
                            node_by_phrase["Génération d'ordres blocs et routage full STP"],
                        ],
                        "delivery_or_integration_node_ids": [
                            node_by_phrase["REST API"],
                        ],
                        "active_lens_ids": [],
                        "adjacency_hypotheses": [
                            {
                                "text": "Private-bank buyers using front-office tooling may evaluate adjacent compliance and routing capabilities.",
                                "supporting_node_ids": [
                                    node_by_phrase["Préparation et modélisation des portefeuilles"],
                                    node_by_phrase["Génération d'ordres blocs et routage full STP"],
                                ],
                                "confidence": 0.77,
                            }
                        ],
                        "confidence_gaps": ["Named customer proof is still thin."],
                        "open_questions": ["Which buyer segment should the first adjacency map prioritize?"],
                    }
                )
            )

    monkeypatch.setattr("app.services.thesis.LLMOrchestrator", _FakeOrchestrator)

    payload = bootstrap_thesis_payload(profile)

    assert payload["market_map_brief"]["source_summary"].startswith("4TPM appears to sell front-office wealth management capabilities")
    assert [
        node["phrase"] for node in payload["market_map_brief"]["capability_nodes"]
    ] == [
        "Préparation et modélisation des portefeuilles",
        "Génération d'ordres blocs et routage full STP",
    ]
    assert [
        node["phrase"] for node in payload["market_map_brief"]["delivery_or_integration_nodes"]
    ] == ["REST API"]
    assert payload["market_map_brief"]["open_questions"][0] == (
        "Which buyer segment should the first adjacency map prioritize?"
    )
