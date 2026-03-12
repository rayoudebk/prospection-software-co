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
