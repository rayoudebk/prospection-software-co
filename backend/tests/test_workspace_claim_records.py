from app.workers.workspace_tasks import _build_claim_records, _select_top_claim


def test_build_claim_records_links_to_source_evidence_ids():
    records = _build_claim_records(
        workspace_id=1,
        vendor_id=2,
        screening_id=3,
        candidate={"website": "https://acme.com"},
        trusted_reasons=[
            {
                "dimension": "company_profile",
                "text": "Employees: 120",
                "citation_url": "https://acme.com/about",
            }
        ],
        matched_mentions=[],
        source_evidence_ids={"https://acme.com/about": 999},
    )
    assert len(records) >= 1
    # We expect at least one claim directly linked to the trusted evidence mapping.
    assert any(record.get("source_evidence_id") == 999 for record in records)


def test_build_claim_records_marks_numeric_conflicts():
    records = _build_claim_records(
        workspace_id=1,
        vendor_id=2,
        screening_id=3,
        candidate={"website": "https://acme.com"},
        trusted_reasons=[
            {
                "dimension": "company_profile",
                "text": "Employees: 10",
                "citation_url": "https://acme.com/about",
            },
            {
                "dimension": "company_profile",
                "text": "Employees: 20",
                "citation_url": "https://acme.com/about",
            },
        ],
        matched_mentions=[],
        source_evidence_ids={},
    )
    assert len(records) >= 2
    assert any(record.get("is_conflicting") for record in records)
    assert any(record.get("claim_status") == "contradicted" for record in records)


def test_select_top_claim_prefers_higher_tier_with_source_metadata():
    top_claim = _select_top_claim(
        [
            {
                "claim_text": "Directory summary",
                "dimension": "directory_context",
                "source_url": "https://www.thewealthmosaic.com/vendors/acme",
                "source_tier": "tier4_discovery",
                "source_type": "directory_comparator",
                "claim_group": "product_depth",
            },
            {
                "claim_text": "Official product module depth",
                "dimension": "product",
                "source_url": "https://acme.com/products",
                "source_tier": "tier1_vendor",
                "source_type": "first_party_website",
                "claim_group": "product_depth",
            },
        ]
    )
    assert top_claim["source_tier"] == "tier1_vendor"
    assert top_claim["source_url"] == "https://acme.com/products"
    assert top_claim.get("text")
