from app.models.workspace import BrickTaxonomy, CompanyProfile
from app.services.thesis import (
    apply_thesis_adjustment_operations,
    bootstrap_thesis_payload,
    derive_search_lane_payloads,
)


def _build_profile() -> CompanyProfile:
    return CompanyProfile(
        workspace_id=1,
        buyer_company_url="https://acme.example.com",
        buyer_context_summary=(
            "Acme is a SaaS portfolio analytics platform for private equity and fund operations teams. "
            "The company also offers implementation services and API-based integrations."
        ),
        reference_vendor_urls=["https://comp-one.example.com"],
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


def _build_taxonomy() -> BrickTaxonomy:
    return BrickTaxonomy(
        workspace_id=1,
        bricks=[
            {"id": "brick-1", "name": "Portfolio analytics"},
            {"id": "brick-2", "name": "Reporting workflow"},
        ],
        priority_brick_ids=["brick-1"],
        vertical_focus=["private_equity"],
        confirmed=False,
    )


def test_bootstrap_thesis_payload_generates_claims_and_source_pills():
    payload = bootstrap_thesis_payload(_build_profile(), _build_taxonomy())

    assert payload["summary"]
    assert payload["source_pills"]
    sections = {claim["section"] for claim in payload["claims"]}
    assert "core_capability" in sections
    assert "adjacent_capability" in sections
    assert "business_model" in sections
    assert "customer_profile" in sections
    assert "geography" in sections
    renderings = {claim["rendering"] for claim in payload["claims"]}
    assert "fact" in renderings
    assert "hypothesis" in renderings


def test_derive_search_lane_payloads_prefers_core_and_adjacent_claims():
    profile = _build_profile()
    taxonomy = _build_taxonomy()
    thesis_payload = bootstrap_thesis_payload(profile, taxonomy)

    lanes = derive_search_lane_payloads(thesis_payload, profile, taxonomy)

    assert len(lanes) == 2
    by_type = {lane["lane_type"]: lane for lane in lanes}
    assert "Portfolio analytics" in by_type["core"]["capabilities"]
    assert "Implementation services" in by_type["adjacent"]["capabilities"]
    assert by_type["core"]["status"] == "draft"
    assert "Prioritize vendors serving private_equity" in by_type["core"]["must_include_terms"]


def test_apply_thesis_adjustment_operations_updates_claims_and_questions():
    thesis_payload = bootstrap_thesis_payload(_build_profile(), _build_taxonomy())
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
