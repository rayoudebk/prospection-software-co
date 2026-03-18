import app.workers.workspace_tasks as workspace_tasks
from types import SimpleNamespace


def test_closed_world_candidate_validation_accepts_retrieved_url():
    retrieval_results = [
        {
            "url": "https://vendor.example.com",
            "normalized_url": "https://vendor.example.com",
            "provider": "exa",
            "query_id": "precision_1",
            "query_type": "precision",
            "rank": 1,
            "snippet": "Vendor provides portfolio management software.",
        }
    ]
    candidates = [
        {
            "name": "Vendor",
            "website": "https://vendor.example.com",
            "why_relevant": [
                {
                    "text": "Portfolio management platform.",
                    "citation_url": "https://vendor.example.com",
                }
            ],
        }
    ]

    validated, stats = workspace_tasks._validate_closed_world_candidates(candidates, retrieval_results)
    assert len(validated) == 1
    assert stats["dropped_missing_url"] == 0
    assert stats["dropped_missing_evidence"] == 0


def test_closed_world_candidate_validation_drops_unknown_url():
    retrieval_results = [
        {
            "url": "https://known.example.com",
            "normalized_url": "https://known.example.com",
            "provider": "exa",
            "query_id": "precision_1",
            "query_type": "precision",
            "rank": 1,
        }
    ]
    candidates = [
        {"name": "Unknown", "website": "https://unknown.example.com"},
    ]

    validated, stats = workspace_tasks._validate_closed_world_candidates(candidates, retrieval_results)
    assert validated == []
    assert stats["dropped_missing_url"] == 1


def test_scope_hints_driven_query_plan_overrides_legacy_brick_hints():
    scope_hints = {
        "source_capabilities": ["Portfolio analytics", "Fund reporting"],
        "source_customer_segments": ["private equity"],
        "adjacency_boxes": [
            {
                "label": "Voting rights workflow",
                "adjacency_kind": "adjacent_workflow",
                "status": "corroborated_expansion",
                "priority_tier": "meaningful_adjacent",
                "confidence": 0.74,
                "likely_customer_segments": ["fund ops"],
                "likely_workflows": ["proxy voting"],
                "retrieval_query_seeds": ["proxy voting workflow software"],
            }
        ],
        "adjacent_lanes": ["Voting rights workflow"],
        "named_account_anchors": ["Northwind Capital"],
        "comparator_seed_urls": ["https://comp-one.example.com"],
        "confirmed": True,
    }

    plan = workspace_tasks._default_discovery_query_plan(
        taxonomy_bricks=[{"name": "Legacy brick"}],
        geo_scope={"region": "US"},
        vertical_focus=["legacy_vertical"],
        scope_hints=scope_hints,
    )

    precision_texts = [entry["query_text"] for entry in plan["precision_queries"]]
    recall_texts = [entry["query_text"] for entry in plan["recall_queries"]]
    assert any("Portfolio analytics" in text for text in precision_texts)
    assert any("Voting rights workflow" in text for text in recall_texts)
    assert all("Legacy brick" not in text for text in precision_texts)

    queries, summary = workspace_tasks._build_external_search_queries_from_plan(plan)
    assert any(query["scope_bucket"] == "core" for query in queries)
    assert any(query["scope_bucket"] == "adjacent" for query in queries)
    assert "private equity" in queries[0]["must_include_terms"]
    assert summary["scope_buckets"] == ["core", "adjacent"]


def test_scope_hints_query_plan_prioritizes_canonical_adjacency_boxes_and_seed_urls():
    scope_hints = {
        "source_capabilities": ["Clinical staffing"],
        "source_customer_segments": ["Hospital operators"],
        "named_account_anchors": ["CHU Lille"],
        "adjacency_boxes": [
            {
                "label": "Workforce planning",
                "adjacency_kind": "adjacent_capability",
                "status": "corroborated_expansion",
                "priority_tier": "core_adjacent",
                "confidence": 0.81,
                "likely_customer_segments": ["Hospital operators"],
                "likely_workflows": ["Shift planning"],
                "retrieval_query_seeds": ["hospital workforce planning software"],
            },
            {
                "label": "Cafeteria menu tooling",
                "adjacency_kind": "adjacent_capability",
                "status": "corroborated_expansion",
                "priority_tier": "edge_case",
                "confidence": 0.62,
                "likely_customer_segments": ["Hospital operators"],
                "likely_workflows": ["Menu scheduling"],
            },
        ],
        "company_seeds": [
            {
                "name": "PlannerCo",
                "website": "https://plannerco.example.com",
                "status": "corroborated_expansion",
                "fit_to_adjacency_box_ids": ["adj_box_staffing"],
            }
        ],
        "company_seed_urls": ["https://plannerco.example.com"],
        "comparator_seed_urls": ["https://hublo.example.com"],
        "confirmed": True,
    }

    plan = workspace_tasks._default_discovery_query_plan(
        taxonomy_bricks=[{"name": "Legacy brick"}],
        geo_scope={"region": "EU"},
        vertical_focus=["legacy_vertical"],
        scope_hints=scope_hints,
    )

    recall_texts = [entry["query_text"] for entry in plan["recall_queries"]]
    assert any("Workforce planning" in text for text in recall_texts)
    assert all("Cafeteria menu tooling" not in text for text in recall_texts)
    assert "https://plannerco.example.com" in plan["seed_urls"]
    assert "https://hublo.example.com" in plan["seed_urls"]


def test_known_discovery_blocked_domains_excludes_buyer_and_first_party_seed_domains():
    profile = SimpleNamespace(
        buyer_company_url="https://4tpm.fr/?lang=en",
        comparator_seed_urls=[
            "https://known-competitor.example.com",
            "https://www.linkedin.com/company/known-competitor",
        ],
    )

    blocked = workspace_tasks._known_discovery_blocked_domains(
        profile,
        ["https://plannerco.example.com", "https://www.crunchbase.com/organization/plannerco"],
    )

    assert "4tpm.fr" in blocked
    assert "known-competitor.example.com" in blocked
    assert "plannerco.example.com" in blocked
    assert "linkedin.com" not in blocked
    assert "crunchbase.com" not in blocked


def test_known_entity_suppression_drops_exact_known_domain_and_name():
    profile = SimpleNamespace(
        buyer_company_url="https://4tpm.fr",
        comparator_seed_urls=[],
    )
    suppression = workspace_tasks._known_entity_suppression_profile(
        profile,
        existing_companies=[SimpleNamespace(name="Known Competitor", website="https://known-competitor.example.com")],
        previous_entities=[],
        previous_aliases=[],
    )

    kept, stats = workspace_tasks._suppress_known_entity_candidates(
        [
            {"name": "Known Competitor", "website": "https://known-competitor.example.com"},
            {"name": "Fresh Vendor", "website": "https://fresh-vendor.example.com"},
        ],
        suppression,
    )

    assert [candidate["name"] for candidate in kept] == ["Fresh Vendor"]
    assert stats["dropped_count"] == 1
    assert stats["known_entity_domain"] == 1


def test_known_entity_suppression_drops_third_party_page_about_known_entity():
    profile = SimpleNamespace(
        buyer_company_url="https://4tpm.fr",
        comparator_seed_urls=[],
    )
    suppression = workspace_tasks._known_entity_suppression_profile(
        profile,
        existing_companies=[],
        previous_entities=[],
        previous_aliases=[],
    )

    kept, stats = workspace_tasks._suppress_known_entity_candidates(
        [
            {
                "name": "4TPM - Overview, News & Competitors",
                "website": "https://www.zoominfo.com/c/4tpm/123",
                "discovery_url": "https://www.zoominfo.com/c/4tpm/123",
                "first_party_domains": [],
            },
            {
                "name": "New France TPM Vendor",
                "website": "https://new-vendor.example.com",
                "discovery_url": "https://new-vendor.example.com",
                "first_party_domains": ["new-vendor.example.com"],
            },
        ],
        suppression,
    )

    assert [candidate["name"] for candidate in kept] == ["New France TPM Vendor"]
    assert stats["dropped_count"] == 1
    assert stats["known_entity_third_party_page"] == 1
