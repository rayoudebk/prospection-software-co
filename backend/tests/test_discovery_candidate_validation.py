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


def test_scope_hints_driven_query_plan_overrides_legacy_brick_hints():
    scope_hints = {
        "source_capabilities": ["Portfolio analytics", "Fund reporting"],
        "adjacent_capabilities": ["Voting rights workflow"],
        "source_customer_segments": ["private equity"],
        "adjacent_customer_segments": ["fund ops"],
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
