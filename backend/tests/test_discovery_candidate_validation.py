import app.workers.workspace_tasks as workspace_tasks


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


def test_lane_driven_query_plan_overrides_legacy_brick_hints():
    search_lanes = [
        {
            "lane_type": "core",
            "title": "Core lane",
            "capabilities": ["Portfolio analytics", "Fund reporting"],
            "customer_tags": ["private equity"],
            "must_include_terms": ["SaaS"],
            "must_exclude_terms": ["ERP"],
            "seed_urls": ["https://comp-one.example.com"],
            "status": "confirmed",
        },
        {
            "lane_type": "adjacent",
            "title": "Adjacent lane",
            "capabilities": ["Voting rights workflow"],
            "customer_tags": ["fund ops"],
            "must_include_terms": [],
            "must_exclude_terms": [],
            "seed_urls": [],
            "status": "confirmed",
        },
    ]

    plan = workspace_tasks._default_discovery_query_plan(
        taxonomy_bricks=[{"name": "Legacy brick"}],
        geo_scope={"region": "US"},
        vertical_focus=["legacy_vertical"],
        search_lanes=search_lanes,
    )

    precision_texts = [entry["query_text"] for entry in plan["precision_queries"]]
    recall_texts = [entry["query_text"] for entry in plan["recall_queries"]]
    assert any("Portfolio analytics" in text for text in precision_texts)
    assert any("Voting rights workflow" in text for text in recall_texts)
    assert all("Legacy brick" not in text for text in precision_texts)

    queries, summary = workspace_tasks._build_external_search_queries_from_plan(plan)
    assert any(query["lane_type"] == "core" for query in queries)
    assert any(query["lane_type"] == "adjacent" for query in queries)
    assert "private equity" in queries[0]["must_include_terms"]
    assert summary["lane_types"] == ["core", "adjacent"]
