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
    assert stats["dropped_non_first_party_website"] == 0


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


def test_closed_world_candidate_validation_drops_non_first_party_profile_url():
    retrieval_results = [
        {
            "url": "https://www.thewealthmosaic.com/vendors/vendor-x/platform/",
            "normalized_url": "https://www.thewealthmosaic.com/vendors/vendor-x/platform/",
            "provider": "exa",
            "query_id": "precision_1",
            "query_type": "precision",
            "rank": 1,
            "snippet": "Directory profile",
        }
    ]
    candidates = [
        {
            "name": "Vendor X",
            "website": "https://www.thewealthmosaic.com/vendors/vendor-x/platform/",
            "why_relevant": [
                {
                    "text": "Directory profile",
                    "citation_url": "https://www.thewealthmosaic.com/vendors/vendor-x/platform/",
                }
            ],
        }
    ]

    validated, stats = workspace_tasks._validate_closed_world_candidates(candidates, retrieval_results)
    assert validated == []
    assert stats["dropped_non_first_party_website"] == 1


def test_closed_world_candidate_validation_drops_non_vendor_retrieval_result():
    retrieval_results = [
        {
            "url": "https://nucamp.co/blog/coding-bootcamp-belgium-bel-healthcare-how-ai-is-helping-healthcare-companies-in-belgium-cut-costs-and-improve-efficiency",
            "normalized_url": "https://nucamp.co/blog/coding-bootcamp-belgium-bel-healthcare-how-ai-is-helping-healthcare-companies-in-belgium-cut-costs-and-improve-efficiency",
            "provider": "exa",
            "query_id": "precision_3",
            "query_type": "precision",
            "rank": 7,
            "title": "How AI is helping healthcare companies in Belgium cut costs and improve efficiency",
            "snippet": "Coding bootcamp industry blog post.",
        }
    ]
    candidates = [
        {
            "name": "Nucamp",
            "website": "https://nucamp.co/blog/coding-bootcamp-belgium-bel-healthcare-how-ai-is-helping-healthcare-companies-in-belgium-cut-costs-and-improve-efficiency",
            "why_relevant": [
                {
                    "text": "Candidate synthesized from retrieval context.",
                    "citation_url": "https://nucamp.co/blog/coding-bootcamp-belgium-bel-healthcare-how-ai-is-helping-healthcare-companies-in-belgium-cut-costs-and-improve-efficiency",
                }
            ],
        }
    ]

    validated, stats = workspace_tasks._validate_closed_world_candidates(candidates, retrieval_results)
    assert validated == []
    assert stats["dropped_non_vendor_result"] == 1


def test_scope_hints_driven_query_plan_overrides_legacy_brick_hints():
    scope_hints = {
        "source_capabilities": ["Portfolio analytics", "Fund reporting"],
        "source_customer_segments": ["private equity"],
        "adjacency_boxes": [
            {
                "id": "adj_reporting",
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
    assert any("software vendor" in text for text in precision_texts + recall_texts)

    queries, summary = workspace_tasks._build_external_search_queries_from_plan(
        plan,
        normalized_scope=scope_hints,
    )
    assert any(query["scope_bucket"] == "core" for query in queries)
    assert any(query["scope_bucket"] == "adjacent" for query in queries)
    assert "private equity" in queries[0]["must_include_terms"]
    assert summary["scope_buckets"] == ["core", "adjacent"]
    adjacent_query = next(query for query in queries if query["scope_bucket"] == "adjacent")
    assert adjacent_query["fit_to_adjacency_box_ids"] == ["adj_reporting"]
    assert adjacent_query["fit_to_adjacency_box_labels"] == ["Voting rights workflow"]


def test_default_discovery_query_plan_adds_plain_source_company_competitor_queries():
    plan = workspace_tasks._default_discovery_query_plan(
        taxonomy_bricks=[{"name": "Clinical staffing"}],
        geo_scope={"region": "EU+UK", "include_countries": ["France"], "languages": ["fr"]},
        vertical_focus=["healthcare staffing"],
        scope_hints={
            "source_capabilities": ["Clinical staffing"],
            "source_customer_segments": ["Hospitals"],
        },
        source_company_url="https://hublo.com",
    )

    precision_texts = [entry["query_text"] for entry in plan["precision_queries"]]
    recall_texts = [entry["query_text"] for entry in plan["recall_queries"]]

    assert "Hublo competitors" in precision_texts
    assert "Hublo alternatives" in precision_texts
    assert "companies like Hublo" in precision_texts
    assert any(text.startswith("best alternatives to Hublo") for text in recall_texts)


def test_stabilize_discovery_query_plan_restores_source_company_queries():
    fallback = workspace_tasks._default_discovery_query_plan(
        taxonomy_bricks=[{"name": "Clinical staffing"}],
        geo_scope={"region": "EU+UK", "include_countries": ["France"], "languages": ["fr"]},
        vertical_focus=["healthcare staffing"],
        scope_hints={
            "source_capabilities": ["Clinical staffing"],
            "source_customer_segments": ["Hospitals"],
        },
        source_company_url="https://hublo.com",
    )
    plan = {
        "precision_queries": [
            {"query_text": "Clinical staffing competitors Hospitals France", "query_intent": "competitor_direct", "scope_bucket": "core"},
            {"query_text": "Clinical staffing software vendor Hospitals EU+UK", "query_intent": "category_vendor", "scope_bucket": "core"},
        ],
        "recall_queries": [
            {"query_text": "best healthcare staffing software France", "query_intent": "comparative_source", "scope_bucket": "core"},
        ],
    }

    stabilized, adjustments = workspace_tasks._stabilize_discovery_query_plan(
        plan,
        fallback,
        normalized_scope={"source_capabilities": ["Clinical staffing"], "source_customer_segments": ["Hospitals"]},
        vertical_focus=["healthcare staffing"],
        brick_names=["Clinical staffing"],
        source_company_url="https://hublo.com",
    )

    texts = [entry["query_text"] for entry in stabilized["precision_queries"] + stabilized["recall_queries"]]
    assert any("Hublo competitors" in text for text in texts)
    assert adjustments


def test_build_external_search_queries_keeps_competitor_families_lightweight():
    plan = {
        "precision_queries": [
            {
                "query_text": "workforce planning software vendor hospitals EU+UK",
                "query_family": "category_vendor",
                "query_intent": "category_vendor",
                "scope_bucket": "core",
            }
        ],
        "recall_queries": [
            {
                "query_text": "hublo healthcare competitors france",
                "query_family": "competitor_direct",
                "query_intent": "competitor_direct",
                "scope_bucket": "core",
            }
        ],
        "must_include_terms": ["hospital"],
        "must_exclude_terms": ["careers", "jobs", "consulting", "pdf"],
        "domain_allowlist": [],
        "domain_blocklist": ["hublo.com"],
    }

    queries, _summary = workspace_tasks._build_external_search_queries_from_plan(
        plan,
        preferred_countries=["France", "Belgium"],
        preferred_languages=["fr"],
    )

    category_query = next(item for item in queries if item["query_family"] == "category_vendor")
    competitor_query = next(item for item in queries if item["query_family"] == "competitor_direct")

    assert category_query["query_text"].endswith("France Belgium fr")
    assert category_query["must_include_terms"] == ["hospital"]
    assert category_query["must_exclude_terms"] == ["careers", "jobs", "consulting", "pdf"]

    assert competitor_query["query_text"] == "hublo healthcare competitors france"
    assert competitor_query["must_include_terms"] == []
    assert competitor_query["must_exclude_terms"] == ["careers", "jobs"]


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
    assert plan["seed_urls"] == []
    assert not any("plannerco.example.com" in url for url in plan["seed_urls"])
    assert not any("PlannerCo alternatives" in text for text in recall_texts)
    assert not any("CHU Lille" in text for text in recall_texts + [entry["query_text"] for entry in plan["precision_queries"]])
    assert not any("wealthtech" in text.lower() for text in recall_texts + [entry["query_text"] for entry in plan["precision_queries"]])
    assert "bloomberg" not in [item.lower() for item in plan["must_exclude_terms"]]


def test_planner_scope_hints_keeps_raw_company_seeds_but_filters_planning_subset():
    scope_hints = {
        "company_seeds": [
            {
                "name": "PlannerCo",
                "website": "https://plannerco.example.com",
                "status": "hypothesis",
                "confidence": 0.4,
                "evidence": [{"url": "https://example.com/plannerco"}],
            },
            {
                "name": "HealthRota",
                "website": "https://healthrota.co.uk",
                "status": "confirmed",
                "confidence": 0.92,
                "evidence": [{"url": "https://www.healthrota.co.uk/about"}],
            },
        ]
    }

    normalized = workspace_tasks._normalize_scope_hints(scope_hints)
    planner_scope = workspace_tasks._planner_scope_hints(scope_hints)

    assert [seed["name"] for seed in normalized["company_seeds"]] == ["PlannerCo", "HealthRota"]
    assert [seed["name"] for seed in planner_scope["company_seeds"]] == ["HealthRota"]


def test_backfill_comparative_retrieval_context_fetches_only_targeted_low_snippet_rows(monkeypatch):
    fetched: list[str] = []

    def _fake_fetch(url: str):
        fetched.append(url)
        return {
            "provider": "fast_fetch",
            "content": "Brigad Mediflash StaffMatch healthcare staffing marketplace platform for hospitals in France.",
        }

    monkeypatch.setattr(workspace_tasks, "fetch_page_fast", _fake_fetch)

    rows = [
        {
            "url": "https://example.com/hublo-competitors",
            "normalized_url": "https://example.com/hublo-competitors",
            "query_family": "comparative_source",
            "query_text": "hublo competitors healthcare staffing france",
            "snippet": "",
        },
        {
            "url": "https://vendor.example.com",
            "normalized_url": "https://vendor.example.com",
            "query_family": "category_vendor",
            "query_text": "healthcare staffing software vendor france",
            "snippet": "",
        },
    ]

    updated, stats = workspace_tasks._backfill_comparative_retrieval_context(rows, cap=4)

    assert fetched == ["https://example.com/hublo-competitors"]
    assert updated[0]["snippet_backfilled"] is True
    assert "Brigad" in updated[0]["snippet"]
    assert updated[1]["snippet"] == ""
    assert stats["attempted"] == 1
    assert stats["updated"] == 1


def test_parse_discovery_candidates_accepts_comparative_candidate_without_website():
    raw = """
    [
      {
        "name": "Brigad",
        "discovery_url": "https://example.com/hublo-competitors",
        "hq_country": "FR",
        "why_relevant": [
          {
            "text": "Brigad is listed among Hublo alternatives.",
            "citation_url": "https://example.com/hublo-competitors"
          }
        ]
      }
    ]
    """

    parsed = workspace_tasks._parse_discovery_candidates_from_text(raw)

    assert len(parsed) == 1
    assert parsed[0]["name"] == "Brigad"
    assert parsed[0]["website"] is None
    assert parsed[0]["discovery_url"] == "https://example.com/hublo-competitors"


def test_seed_candidates_from_mentions_dedupes_directory_names_and_attaches_scope_context():
    mentions = [
        {
            "company_name": "Vendor X",
            "listing_url": "https://directory.example.com/needs/portfolio-management/",
            "profile_url": "https://directory.example.com/vendors/vendor-x/",
            "source_name": "directory",
            "source_run_id": 1,
            "category_tags": ["Portfolio management"],
            "listing_text_snippets": ["Portfolio management software for private banks."],
        },
        {
            "company_name": "Vendor X",
            "listing_url": "https://directory.example.com/needs/portfolio-management/",
            "profile_url": "https://directory.example.com/vendors/vendor-x/",
            "source_name": "directory",
            "source_run_id": 1,
            "category_tags": ["Portfolio management"],
            "listing_text_snippets": ["Trusted by wealth managers."],
        },
    ]
    scope_hints = {
        "source_capabilities": ["Portfolio management"],
        "adjacency_boxes": [
            {
                "id": "adj_reporting",
                "label": "Client reporting",
                "retrieval_query_seeds": ["client reporting software"],
                "likely_workflows": ["Reporting workflow"],
            }
        ],
    }

    seeded = workspace_tasks._seed_candidates_from_mentions(
        mentions,
        normalized_scope=scope_hints,
    )

    assert len(seeded) == 1
    assert seeded[0]["name"] == "Vendor X"
    assert len(seeded[0]["why_relevant"]) == 2
    origin_metadata = seeded[0]["_origins"][0]["metadata"]
    assert origin_metadata["scope_bucket"] == "core"
    assert origin_metadata["source_capability_matches"] == ["Portfolio management"]
    assert origin_metadata["fit_to_adjacency_box_ids"] == []


def test_seed_candidates_from_mentions_matches_adjacency_box_terms():
    mentions = [
        {
            "company_name": "PlannerCo",
            "listing_url": "https://directory.example.com/needs/workforce-planning/",
            "source_name": "directory",
            "source_run_id": 2,
            "category_tags": ["Staffing"],
            "listing_text_snippets": ["Hospital workforce planning and shift planning software."],
        }
    ]
    scope_hints = {
        "source_capabilities": ["Clinical staffing"],
        "adjacency_boxes": [
            {
                "id": "adj_workforce",
                "label": "Workforce planning",
                "retrieval_query_seeds": ["hospital workforce planning software"],
                "likely_workflows": ["Shift planning"],
            }
        ],
    }

    seeded = workspace_tasks._seed_candidates_from_mentions(
        mentions,
        normalized_scope=scope_hints,
    )

    origin_metadata = seeded[0]["_origins"][0]["metadata"]
    assert origin_metadata["scope_bucket"] == "adjacent"
    assert origin_metadata["fit_to_adjacency_box_ids"] == ["adj_workforce"]
    assert origin_metadata["fit_to_adjacency_box_labels"] == ["Workforce planning"]


def test_candidates_from_retrieval_results_use_homepage_as_canonical_website():
    candidates = workspace_tasks._candidates_from_retrieval_results(
        [
            {
                "url": "https://vendor.example.com/platform/reporting",
                "provider": "exa",
                "query_id": "precision_1",
                "query_type": "precision",
                "scope_bucket": "core",
                "rank": 1,
                "snippet": "Reporting platform for private banks.",
            }
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["website"] == "https://vendor.example.com"
    assert candidates[0]["official_website_url"] == "https://vendor.example.com"
    assert candidates[0]["discovery_url"] == "https://vendor.example.com/platform/reporting"
    assert candidates[0]["why_relevant"][0]["citation_url"] == "https://vendor.example.com/platform/reporting"


def test_candidates_from_retrieval_results_preserve_lane_metadata():
    candidates = workspace_tasks._candidates_from_retrieval_results(
        [
            {
                "url": "https://vendor.example.com/platform/reporting",
                "provider": "exa",
                "query_id": "precision_1",
                "query_type": "precision",
                "scope_bucket": "adjacent",
                "brick_name": "Workforce planning",
                "fit_to_adjacency_box_ids": ["adj_workforce"],
                "fit_to_adjacency_box_labels": ["Workforce planning"],
                "source_capability_matches": [],
                "rank": 1,
                "snippet": "Workforce planning platform for hospitals.",
            }
        ]
    )

    metadata = candidates[0]["_origins"][0]["metadata"]
    assert metadata["fit_to_adjacency_box_ids"] == ["adj_workforce"]
    assert metadata["fit_to_adjacency_box_labels"] == ["Workforce planning"]


def test_scope_metadata_for_query_entry_matches_query_text_to_adjacency_box():
    scope_hints = {
        "source_capabilities": ["Clinical staffing"],
        "adjacency_boxes": [
            {
                "id": "adj_workforce",
                "label": "Workforce planning",
                "retrieval_query_seeds": ["hospital workforce planning software"],
                "likely_workflows": ["Shift planning"],
            }
        ],
    }

    metadata = workspace_tasks._scope_metadata_for_query_entry(
        {
            "query_text": "hospital workforce planning software vendor France Belgium",
            "query_intent": "find adjacent workflow vendors",
            "scope_bucket": "adjacent",
        },
        normalized_scope=scope_hints,
    )

    assert metadata["fit_to_adjacency_box_ids"] == ["adj_workforce"]
    assert metadata["fit_to_adjacency_box_labels"] == ["Workforce planning"]


def test_candidate_validation_lane_metadata_prefers_origin_lane_labels():
    lane_ids, lane_labels = workspace_tasks._candidate_validation_lane_metadata(
        {
            "origins": [
                {
                    "origin_type": "external_search_seed",
                    "metadata": {
                        "scope_bucket": "adjacent",
                        "brick_name": "Workforce planning",
                        "fit_to_adjacency_box_ids": ["adj_workforce"],
                        "fit_to_adjacency_box_labels": ["Workforce planning"],
                    },
                }
            ]
        },
        scope_buckets=["adjacent"],
    )

    assert lane_ids == ["adj_workforce"]
    assert lane_labels == ["Workforce planning"]


def test_candidates_from_retrieval_results_skip_aggregator_profile_domains():
    candidates = workspace_tasks._candidates_from_retrieval_results(
        [
            {
                "url": "https://thewealthmosaic.com/vendors/dorsum",
                "provider": "exa",
                "query_id": "precision_1",
                "query_type": "precision",
                "scope_bucket": "core",
                "rank": 1,
                "snippet": "Aggregator vendor profile.",
            }
        ]
    )

    assert candidates == []


def test_candidates_from_retrieval_results_skip_known_non_vendor_hosts():
    candidates = workspace_tasks._candidates_from_retrieval_results(
        [
            {
                "url": "https://www.citywire.com/funds-insider/news/private-bank-breaks-new-ground-with-aladdin-deal/a2452607",
                "provider": "exa",
                "query_id": "precision_1",
                "query_type": "precision",
                "scope_bucket": "core",
                "rank": 1,
                "title": "Private bank breaks new ground with Aladdin deal",
                "snippet": "Publisher article.",
            }
        ]
    )

    assert candidates == []


def test_is_persistable_vendor_entity_rejects_blocked_and_training_blog_entities():
    assert not workspace_tasks._is_persistable_vendor_entity(
        {
            "canonical_name": "Reddit",
            "canonical_website": "https://reddit.com",
            "discovery_primary_url": "https://reddit.com/r/ValueInvesting/comments/example",
            "why_relevant": [{"text": "Discussion thread"}],
        }
    )
    assert not workspace_tasks._is_persistable_vendor_entity(
        {
            "canonical_name": "Nucamp",
            "canonical_website": "https://www.nucamp.co/",
            "discovery_primary_url": "https://nucamp.co/blog/coding-bootcamp-belgium-bel-healthcare-how-ai-is-helping-healthcare-companies-in-belgium-cut-costs-and-improve-efficiency",
            "why_relevant": [{"text": "Coding bootcamp industry blog post."}],
        }
    )


def test_is_persistable_vendor_entity_rejects_consulting_and_social_entities_without_vendor_signal():
    assert not workspace_tasks._is_persistable_vendor_entity(
        {
            "canonical_name": "INCITE Consulting Solutions",
            "canonical_website": "https://inciteconsultingsolutions.com",
            "discovery_primary_url": "https://mhca.com/vendors/vendor-showcase",
            "why_relevant": [{"text": "Healthcare consulting services and advisory support."}],
        }
    )
    assert not workspace_tasks._is_persistable_vendor_entity(
        {
            "canonical_name": "Flickr",
            "canonical_website": "https://www.flickr.com/photos/11157300@N08/",
            "discovery_primary_url": "https://mhca.com/vendors/vendor-showcase",
            "why_relevant": [{"text": "Listed in directory showcase."}],
        }
    )


def test_is_persistable_vendor_entity_allows_fr_registry_candidate_without_official_website():
    assert workspace_tasks._is_persistable_vendor_entity(
        {
            "canonical_name": "Brigad",
            "canonical_website": None,
            "discovery_primary_url": "https://annuaire-entreprises.data.gouv.fr/entreprise/the-working-company-814956744",
            "origin_types": ["registry_fr_base"],
            "registry_id": "814956744",
            "registry_source": "fr_recherche_entreprises",
            "registry_country": "FR",
            "registry_fields": {
                "commercial_names": ["BRIGAD"],
                "activity_code": "62.01Z",
            },
            "why_relevant": [
                {
                    "text": "French registry candidate surfaced from 62.01Z neighborhood for healthcare staffing workflows.",
                }
            ],
        }
    )


def test_dedupe_strings_skips_none_values():
    assert workspace_tasks._dedupe_strings(["FR", None, " ", "fr", "Belgium"]) == ["FR", "Belgium"]



def test_external_candidate_name_prefers_domain_label_for_article_like_titles():
    assert (
        workspace_tasks._external_candidate_name(
            "Private Bank Portfolio Management Software",
            "https://masttro.com/institutions",
        )
        == "Masttro"
    )


def test_external_candidate_name_demotes_generic_geo_solution_titles():
    assert (
        workspace_tasks._external_candidate_name(
            "Perfect Patient Management System in Belgium",
            "https://sarutech.com",
        )
        == "Sarutech"
    )


def test_external_candidate_name_prefers_brand_segment_when_present():
    assert (
        workspace_tasks._external_candidate_name(
            "FNZ - End-to-end wealth management platform",
            "https://fnz.com",
        )
        == "FNZ"
    )
    assert (
        workspace_tasks._external_candidate_name(
            "Allvue Systems: AI-Powered Alternative Investment Software",
            "https://allvuesystems.com",
        )
        == "Allvue Systems"
    )
    assert (
        workspace_tasks._external_candidate_name(
            "Objectway growth Platform for Wealth Management",
            "https://objectway.com",
        )
        == "Objectway"
    )


def test_external_candidate_name_prefers_company_name_from_third_party_profile_titles():
    assert (
        workspace_tasks._external_candidate_name(
            "Mediflash - 2025 Company Profile - Tracxn",
            "https://tracxn.com/d/companies/mediflash/__example",
        )
        == "Mediflash"
    )


def test_candidates_from_retrieval_results_keep_discovery_url_for_third_party_profile_rows():
    candidates = workspace_tasks._candidates_from_retrieval_results(
        [
            {
                "url": "https://tracxn.com/d/companies/mediflash/__example",
                "normalized_url": "https://tracxn.com/d/companies/mediflash/__example",
                "provider": "exa",
                "query_id": "recall_1",
                "query_type": "recall",
                "query_family": "alternatives",
                "scope_bucket": "core",
                "rank": 1,
                "title": "Mediflash - 2025 Company Profile - Tracxn",
                "snippet": "Mediflash company profile and competitor tracking page.",
            }
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["name"] == "Mediflash"
    assert candidates[0]["website"] is None
    assert candidates[0]["official_website_url"] is None
    assert candidates[0]["discovery_url"] == "https://tracxn.com/d/companies/mediflash/__example"
    assert candidates[0]["first_party_domains"] == []
    assert (
        workspace_tasks._external_candidate_name(
            "Orchestrade Home",
            "https://orchestrade.com",
        )
        == "Orchestrade"
    )
    assert (
        workspace_tasks._external_candidate_name(
            "Portfolio Analytics for Asset Managers",
            "https://venn.twosigma.com",
        )
        == "Venn"
    )


def test_discovery_source_names_for_healthcare_workspace_selects_healthcare_directory_only():
    profile = SimpleNamespace(
        buyer_company_url="https://www.hublo.com/en",
        context_pack_markdown="Hospital staffing and workforce planning software for healthcare providers.",
    )
    source_names = workspace_tasks._discovery_source_names_for_workspace(
        profile,
        ["healthcare staffing", "workforce planning"],
        normalized_scope={
            "source_capabilities": ["shift replacement", "pool management"],
            "source_customer_segments": ["Hospitals", "Clinics"],
            "adjacency_box_labels": ["Internal mobility", "Vendor management system"],
        },
    )
    assert "healthcare_vendor_showcase_seed" in source_names
    assert "wealth_mosaic" not in source_names


def test_looks_like_vendor_candidate_result_rejects_editorial_report_hit():
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://www.ey.com/en_gl/insights/wealth-management/report-2025",
            "title": "EY GCC Wealth Management Industry Report 2025",
            "snippet": "Research report on the industry.",
        }
    )


def test_looks_like_vendor_candidate_result_accepts_vendor_homepage_like_hit():
    assert workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://orchestrade.com/",
            "title": "Orchestrade Home",
            "snippet": "Cross-asset trading and post-trade platform for capital markets.",
        }
    )


def test_looks_like_vendor_candidate_result_rejects_institution_homepage_hit():
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://ing.com/",
            "title": "ING global company website",
            "snippet": "Global banking and financial services institution.",
        }
    )


def test_looks_like_vendor_candidate_result_rejects_pdf_and_public_health_hosts():
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6697511/pdf/main.pdf",
            "title": "The French Health Data Hub and the German Medical Informatics Initiative",
            "snippet": "Research article.",
        }
    )
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://aphp.fr/",
            "title": "Greater Paris University Hospitals - AP-HP",
            "snippet": "Public hospital system in France.",
        }
    )


def test_looks_like_vendor_candidate_result_rejects_media_rankings_and_bootcamp_articles():
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://morismedia.in/belgiums-top-10-healthcare-scheduling-software-belgium",
            "title": "Belgium's Top 10 Healthcare Scheduling Software",
            "snippet": "Ranking of leading tools in the market.",
        }
    )
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://nucamp.co/blog/coding-bootcamp-belgium-bel-healthcare-how-ai-is-helping-healthcare-companies-in-belgium-cut-costs-and-improve-efficiency",
            "title": "How AI is helping healthcare companies in Belgium cut costs and improve efficiency",
            "snippet": "Coding bootcamp industry blog post.",
        }
    )


def test_looks_like_vendor_candidate_result_rejects_service_provider_pages_without_product_signal():
    assert not workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://tateeda.com/services/medical-hr-software-development-company",
            "title": "Medical HR Software Development Company",
            "snippet": "Custom software development and consulting services for healthcare HR teams.",
        }
    )


def test_capability_signals_from_reason_items_extracts_product_and_service_context():
    signals = workspace_tasks._capability_signals_from_reason_items(
        [
            {"text": "Hospital workforce planning platform for staff scheduling.", "dimension": "product"},
            {"text": "Implementation and onboarding services for rostering teams.", "dimension": "services"},
            {"text": "Named customer evidence: CHU Lille", "dimension": "customer"},
        ]
    )
    assert signals[:2] == [
        "Hospital workforce planning platform for staff scheduling.",
        "Implementation and onboarding services for rostering teams.",
    ]


def test_looks_like_vendor_candidate_result_accepts_vendor_blog_when_brand_is_explicit():
    assert workspace_tasks._looks_like_vendor_candidate_result(
        {
            "url": "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "title": "Zopa Bank partners with Upvest",
            "snippet": "Investment API platform for embedded finance and wealth products.",
        }
    )


def test_candidates_from_retrieval_results_skips_known_publisher_domains():
    candidates = workspace_tasks._candidates_from_retrieval_results(
        [
            {
                "url": "https://ibsintelligence.com/ibsi-news/wealthtech-trends/",
                "title": "IBS Intelligence wealthtech trends",
                "snippet": "Publisher coverage of the market.",
                "provider": "exa",
                "query_id": "q1",
                "query_type": "precision",
                "scope_bucket": "adjacent",
                "rank": 1,
            }
        ]
    )

    assert candidates == []


def test_looks_like_vendor_candidate_result_rejects_non_vendor_hosts():
    for url in (
        "https://wealthbriefing.com/some-article",
        "https://privatebankerinternational.com/news/story",
        "https://www.ey.com/en_gl/wealth-management",
    ):
        assert not workspace_tasks._looks_like_vendor_candidate_result(
            {
                "url": url,
                "title": "Wealth management article",
                "snippet": "Industry commentary for private banks and asset managers.",
            }
        )


def test_adaptive_hint_discovery_disabled_without_crawl_budget():
    assert not workspace_tasks._adaptive_hint_discovery_enabled(
        first_party_enrichment_enabled=True,
        first_party_hint_crawl_budget_default=0,
        first_party_crawl_budget_default=0,
    )
    assert workspace_tasks._adaptive_hint_discovery_enabled(
        first_party_enrichment_enabled=True,
        first_party_hint_crawl_budget_default=1,
        first_party_crawl_budget_default=0,
    )


def test_stabilize_discovery_query_plan_drops_restrictive_allowlist_and_merges_fallback():
    fallback_plan = {
        "precision_queries": [
            {
                "query_text": "Portfolio modeling software vendor private bank EU",
                "query_intent": "capability_discovery",
                "brick_name": "Portfolio modeling",
                "scope_bucket": "core",
            }
        ],
        "recall_queries": [
            {
                "query_text": "Objectway alternatives private bank software EU",
                "query_intent": "alternatives",
                "brick_name": "Objectway",
                "scope_bucket": "adjacent",
            }
        ],
        "seed_urls": ["https://objectway.com/"],
        "must_include_terms": ["private bank"],
        "must_exclude_terms": ["bloomberg"],
        "preferred_countries": ["FR"],
        "preferred_languages": ["en"],
        "domain_allowlist": [],
        "domain_blocklist": [],
    }
    model_plan = {
        "precision_queries": [
            {
                "query_text": "wealth management software europe",
                "query_intent": "capability",
                "brick_name": "Portfolio modeling",
                "scope_bucket": "core",
            }
        ],
        "recall_queries": [],
        "seed_urls": ["https://objectway.com/"],
        "must_include_terms": ["private bank"],
        "must_exclude_terms": [],
        "preferred_countries": [],
        "preferred_languages": [],
        "domain_allowlist": ["objectway.com", "thewealthmosaic.com"],
        "domain_blocklist": [],
    }

    stabilized, adjustments = workspace_tasks._stabilize_discovery_query_plan(model_plan, fallback_plan)

    assert stabilized["domain_allowlist"] == []
    assert any("software vendor" in entry["query_text"] for entry in stabilized["precision_queries"])
    assert any("alternatives" in entry["query_text"] for entry in stabilized["recall_queries"])
    assert "dropped_domain_allowlist" in adjustments
    assert "fallback_precision_queries" in adjustments
    assert "fallback_recall_queries" in adjustments


def test_stabilize_discovery_query_plan_strips_wealth_excludes_for_non_wealth_scope_and_placeholder_seeds():
    fallback_plan = {
        "precision_queries": [
            {
                "query_text": "Workforce planning software vendor hospitals EU",
                "query_intent": "capability_discovery",
                "brick_name": "Workforce planning",
                "scope_bucket": "core",
            }
        ],
        "recall_queries": [],
        "seed_urls": ["https://plannerco.example.com/", "https://healthrota.co.uk/"],
        "must_include_terms": ["hospital"],
        "must_exclude_terms": ["bloomberg", "careers", "jobs"],
        "preferred_countries": [],
        "preferred_languages": [],
        "domain_allowlist": [],
        "domain_blocklist": [],
    }

    stabilized, adjustments = workspace_tasks._stabilize_discovery_query_plan(
        fallback_plan,
        fallback_plan,
        normalized_scope={
            "source_capabilities": ["Clinical staffing"],
            "source_customer_segments": ["Hospitals"],
            "adjacency_boxes": [
                {
                    "label": "Workforce planning",
                    "status": "corroborated_expansion",
                    "priority_tier": "core_adjacent",
                }
            ],
        },
        vertical_focus=["healthcare"],
        brick_names=["Workforce planning"],
    )

    assert "bloomberg" not in [item.lower() for item in stabilized["must_exclude_terms"]]
    assert "careers" in [item.lower() for item in stabilized["must_exclude_terms"]]
    assert "https://plannerco.example.com/" not in stabilized["seed_urls"]
    assert "https://healthrota.co.uk/" in stabilized["seed_urls"]
    assert "dropped_wealth_excludes" in adjustments


def test_candidate_priority_score_prefers_external_first_party_over_directory_only():
    external_entity = {
        "origin_types": ["external_search_seed"],
        "canonical_website": "https://vendor.example.com",
        "identity_confidence": "medium",
        "why_relevant": [{"text": "Vendor result"}],
        "capability_signals": ["Portfolio modeling"],
    }
    directory_only_entity = {
        "origin_types": ["directory_seed"],
        "canonical_website": None,
        "identity_confidence": "medium",
        "why_relevant": [{"text": "Directory listing"}],
        "capability_signals": ["Portfolio modeling"],
    }

    assert (
        workspace_tasks._candidate_priority_score(external_entity)
        > workspace_tasks._candidate_priority_score(directory_only_entity)
    )


def test_select_scoring_entities_reserves_external_or_first_party_candidates():
    entities = []
    for idx in range(12):
        entities.append(
            {
                "canonical_name": f"External {idx}",
                "canonical_website": f"https://external-{idx}.example.com",
                "origin_types": ["external_search_seed"],
                "why_relevant": [{"text": "External vendor"}],
                "capability_signals": ["Portfolio modeling"],
            }
        )
    for idx in range(40):
        entities.append(
            {
                "canonical_name": f"Directory {idx}",
                "canonical_website": None,
                "origin_types": ["directory_seed"],
                "why_relevant": [{"text": "Directory listing"}],
                "capability_signals": ["Portfolio modeling"],
            }
        )

    selected, meta = workspace_tasks._select_scoring_entities(entities, cap=20)

    external_count = sum(
        1 for entity in selected if "external_search_seed" in set(entity.get("origin_types") or [])
    )
    assert len(selected) == 20
    assert external_count == 12
    assert meta["reserved_external_or_first_party"] == 12
    assert meta["reserved_directory_backfill"] == 8


def test_fr_registry_source_record_keeps_brand_and_legal_identity_separate():
    row = {
        "nom_complet": "THE WORKING COMPANY",
        "nom_raison_sociale": "THE WORKING COMPANY",
        "siren": "814956744",
        "activite_principale": "62.01Z",
        "activite_principale_naf25": "62.01Z",
        "etat_administratif": "A",
        "siege": {
            "nom_commercial": None,
            "liste_enseignes": None,
            "caractere_employeur": "O",
            "tranche_effectif_salarie": "12",
        },
        "matching_etablissements": [
            {
                "nom_commercial": "BRIGAD",
                "liste_enseignes": None,
            }
        ],
        "complements": {},
    }

    record = workspace_tasks._fr_registry_source_record(row, query_hint="THE WORKING COMPANY")

    assert record["display_name"] == "BRIGAD"
    assert record["legal_name"] == "THE WORKING COMPANY"
    assert record["brand_names"] == ["BRIGAD"]
    assert record["registry_fields"]["commercial_names"] == ["BRIGAD"]


def test_fr_registry_code_neighborhood_expands_beyond_source_code():
    source_record = {
        "activity_code": "63.11Z",
        "activity_code_naf25": "58.29Y",
    }
    semantic_terms = {"terms": ["software", "staffing", "healthcare"]}

    neighborhood = workspace_tasks._fr_registry_code_neighborhood(
        source_record,
        semantic_terms=semantic_terms,
    )

    codes = {entry["code"] for entry in neighborhood}
    assert "63.11Z" in codes
    assert "62.01Z" in codes
    assert "78.10Z" in codes


def test_build_france_registry_universe_candidates_keeps_missing_website_and_brand_first(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={
            "sites": [
                {
                    "company_name": "Hublo",
                    "summary": "Healthcare staffing and replacement platform for hospitals.",
                    "url": "https://hublo.com",
                }
            ]
        },
    )
    normalized_scope = {
        "source_capabilities": ["Clinical staffing"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [
            {
                "id": "shift_replacement",
                "label": "Shift replacement",
                "likely_customer_segments": ["Hospitals"],
                "likely_workflows": ["staff replacement"],
            }
        ],
        "adjacency_box_labels": ["Shift replacement"],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "63.11Z",
        "activity_code_naf25": "58.29Y",
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale == "63.11Z":
            return [
                {
                    "nom_complet": "MEDIFLASH",
                    "nom_raison_sociale": "MEDIFLASH",
                    "siren": "111111111",
                    "activite_principale": "63.11Z",
                    "activite_principale_naf25": "63.11Z",
                    "etat_administratif": "A",
                    "siege": {
                        "nom_commercial": "MEDIFLASH",
                        "caractere_employeur": "O",
                        "tranche_effectif_salarie": "11",
                        "libelle_commune": "PARIS",
                    },
                    "complements": {},
                }
            ], None
        if activite_principale == "62.01Z":
            return [
                {
                    "nom_complet": "THE WORKING COMPANY",
                    "nom_raison_sociale": "THE WORKING COMPANY",
                    "siren": "814956744",
                    "activite_principale": "62.01Z",
                    "activite_principale_naf25": "62.01Z",
                    "etat_administratif": "A",
                    "siege": {
                        "nom_commercial": None,
                        "caractere_employeur": "O",
                        "tranche_effectif_salarie": "12",
                        "libelle_commune": "PARIS",
                    },
                    "matching_etablissements": [
                        {"nom_commercial": "BRIGAD", "liste_enseignes": None}
                    ],
                    "complements": {},
                }
            ], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
    )

    assert diagnostics["registry_candidate_count"] >= 2
    brigad = next(candidate for candidate in candidates if candidate["registry_id"] == "814956744")
    assert brigad["name"] == "BRIGAD"
    assert brigad["legal_name"] == "THE WORKING COMPANY"
    assert brigad["official_website_url"] is None
    assert brigad["registry_fields"]["commercial_names"] == ["BRIGAD"]
    assert brigad["discovery_url"] == "https://annuaire-entreprises.data.gouv.fr/entreprise/814956744"


def test_fr_registry_semantic_score_observation_text_like_merger_note_increases_relevance():
    scope_pack = {
        "core": {"terms": ["healthcare", "staffing", "hospital"], "phrases": []},
        "adjacent": {"terms": ["replacement", "pool", "management"], "phrases": []},
        "entity_seed": {"terms": [], "phrases": []},
    }
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [
            {
                "id": "pool_mgmt",
                "label": "Pool management",
                "likely_customer_segments": ["Hospitals"],
                "likely_workflows": ["replacement", "pool management"],
            }
        ],
    }
    base_record = {
        "display_name": "Alpha Digital",
        "name": "Alpha Digital",
        "context_text": "solutions digitales pour entreprises",
        "industry_codes": ["62.01Z"],
        "registry_fields": {},
        "website": None,
        "is_employer": False,
        "employee_band": None,
    }
    observation_record = {
        **base_record,
        "context_text": (
            "solutions digitales pour entreprises. "
            "observations: fusion avec une plateforme de replacement et de pool management "
            "pour le personnel healthcare en hospital."
        ),
        "registry_fields": {"observation_count": 1},
    }

    base_score, base_meta = workspace_tasks._fr_registry_semantic_score(
        base_record,
        scope_pack=scope_pack,
        normalized_scope=normalized_scope,
        code_distance=1,
    )
    observation_score, observation_meta = workspace_tasks._fr_registry_semantic_score(
        observation_record,
        scope_pack=scope_pack,
        normalized_scope=normalized_scope,
        code_distance=1,
    )

    assert observation_score > base_score
    assert "replacement" in observation_meta["matched_terms"]
    assert "pool" in observation_meta["matched_terms"]
    assert observation_meta["lane_labels"] == ["Pool management"]
    assert base_meta["scope_bucket"] in {"broad_market", "out_of_scope"}
    assert observation_meta["scope_bucket"] in {"adjacent", "core"}


def test_fr_registry_source_observation_counterparty_surfaces_as_candidate(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={
            "sites": [
                {
                    "company_name": "Hublo",
                    "summary": "Healthcare staffing, shift replacement, and pool management platform.",
                    "url": "https://hublo.com",
                }
            ]
        },
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_workflows": ["Shift replacement"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [
            {
                "id": "pool_mgmt",
                "label": "Pool management",
                "likely_customer_segments": ["Hospitals"],
                "likely_workflows": ["replacement", "pool management"],
            }
        ],
        "adjacency_box_labels": ["Pool management"],
        "named_account_anchors": [],
        "geography_expansions": [],
        "company_seeds": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "registry_source": "fr_recherche_entreprises",
        "registry_url": "https://annuaire-entreprises.data.gouv.fr/entreprise/822276986",
        "activity_code": "63.11Z",
        "activity_code_naf25": "58.29Y",
        "registry_fields": {
            "object_text": "plateforme pour professionnels de santé et établissements de santé",
            "observations": [
                "Opération de fusion à compter du 14/12/2023. Société(s) ayant participé à l'opération : MEDIKSTAFF, Société par actions simplifiée."
            ],
            "observation_count": 1,
        },
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)
    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", lambda **kwargs: ([], None))

    candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
    )

    assert diagnostics["registry_candidate_count"] == 0
    assert diagnostics["registry_source_path_counts"].get("observation_counterparty_lookup", 0) == 0


def test_fr_registry_seed_queries_skip_source_identity_and_generic_customer_terms():
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        context_pack_json={
            "sites": [
                {
                    "company_name": "Hublo",
                    "summary": "Healthcare staffing, shift replacement, and pool management platform.",
                    "url": "https://hublo.com",
                }
            ]
        },
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_workflows": ["Shift replacement"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [],
        "adjacency_box_labels": [],
        "company_seeds": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "HUBLO",
        "legal_name": "HUBLO",
        "brand_names": ["HUBLO"],
        "registry_fields": {
            "observations": [
                "Opération de fusion. Société(s) ayant participé à l'opération : MEDIKSTAFF."
            ]
        },
    }

    scope_pack = workspace_tasks._fr_registry_scope_pack(profile, normalized_scope, source_record=source_record)
    specs = workspace_tasks._fr_registry_seed_query_specs(
        profile,
        normalized_scope,
        scope_pack,
        source_record=source_record,
    )

    queries = {spec["query"] for spec in specs}
    query_types = {spec["query_type"] for spec in specs}

    assert "Hublo" not in queries
    assert "Hospitals" not in queries
    assert "MEDIKSTAFF" in queries
    assert "source_brand" not in query_types


def test_fr_registry_semantic_score_ignores_low_signal_management_terms_for_core_fit():
    normalized_scope = {
        "source_capabilities": ["Pool management"],
        "source_workflows": ["Shift replacement"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [],
        "adjacency_box_labels": [],
        "company_seeds": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing platform."}]},
    )
    scope_pack = workspace_tasks._fr_registry_scope_pack(profile, normalized_scope)
    record = {
        "display_name": "R2T PLACEMENT ET MANAGEMENT",
        "name": "R2T PLACEMENT ET MANAGEMENT",
        "legal_name": "R2T PLACEMENT ET MANAGEMENT",
        "context_text": "placement et management recrutement",
        "industry_codes": ["78.10Z"],
        "registry_fields": {},
        "website": None,
        "is_employer": False,
        "employee_band": None,
    }

    _score, meta = workspace_tasks._fr_registry_semantic_score(
        record,
        scope_pack=scope_pack,
        normalized_scope=normalized_scope,
        code_distance=1,
    )

    assert "manage" not in (meta["node_fit_summary"].get("matched_core_terms") or [])


def test_normalize_scope_hints_extracts_labels_from_structured_anchor_entries():
    normalized = workspace_tasks._normalize_scope_hints(
        {
            "named_account_anchors": [
                {"id": "anchor-ap-hp", "label": "AP-HP"},
                {"id": "anchor-chu", "name": "CHU Lille"},
            ],
            "geography_expansions": [
                {"id": "geo-belgium", "label": "Belgium"},
            ],
        }
    )

    assert normalized["named_account_anchors"] == ["AP-HP", "CHU Lille"]
    assert normalized["geography_expansions"] == ["Belgium"]


def test_fetch_fr_registry_detail_record_preserves_existing_normalized_fields_when_no_detail_found(monkeypatch):
    base_record = {
        "name": "BRIGAD",
        "display_name": "BRIGAD",
        "legal_name": "THE WORKING COMPANY",
        "website": None,
        "registry_id": "814956744",
        "registry_source": "fr_recherche_entreprises",
        "registry_url": "https://annuaire-entreprises.data.gouv.fr/entreprise/814956744",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "activity_label": "Programmation informatique",
        "context_text": "Programmation informatique BRIGAD THE WORKING COMPANY",
        "brand_names": ["BRIGAD"],
        "industry_codes": ["62.01Z"],
        "industry_keywords": ["technology"],
        "registry_fields": {
            "commercial_names": ["BRIGAD"],
            "activity_code": "62.01Z",
            "object_text_present": False,
            "observation_count": 0,
        },
    }

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", lambda **kwargs: ([], None))
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_inpi_detail_fields", lambda siren, timeout_seconds=6: {})

    enriched = workspace_tasks._fetch_fr_registry_detail_record(base_record)

    assert enriched["activity_code"] == "62.01Z"
    assert enriched["activity_label"] == "Programmation informatique"
    assert enriched["brand_names"] == ["BRIGAD"]
    assert enriched["registry_fields"]["commercial_names"] == ["BRIGAD"]


def test_fetch_fr_inpi_detail_fields_uses_authenticated_companies_api(monkeypatch):
    payload = [
        {
            "id": "company-1",
            "siren": "822276986",
            "formality": {
                "historique": [{"libelleEvenement": "Ancienne dénomination BRIGAD"}],
                "content": {
                    "personneMorale": {
                        "identite": {
                            "description": {
                                "objet": "Plateforme de mise en relation et de recrutement pour les établissements de santé.",
                            }
                        },
                        "etablissementPrincipal": {
                            "descriptionEtablissement": {
                                "nomCommercial": "HUBLO",
                                "enseigne": "Hublo",
                            },
                            "activites": [
                                {"descriptionDetaillee": "Plateforme digitale pour remplacement et gestion administrative."}
                            ],
                            "nomsDeDomaine": [{"nomDomaine": "hublo.com"}],
                        },
                        "observations": {
                            "rcs": [
                                {"texte": "Opération de fusion à compter du 14/12/2023. Société(s) ayant participé à l'opération : MEDIKSTAFF"}
                            ]
                        },
                    }
                },
            },
        }
    ]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            assert url == "https://registre-national-entreprises.inpi.fr/api/companies"
            assert params == {"page": 1, "pageSize": 1, "siren[]": "822276986"}
            return FakeResponse()

    monkeypatch.setattr(workspace_tasks.settings, "inpi_token", "test-token")
    monkeypatch.setattr(workspace_tasks.httpx, "Client", FakeClient)

    details = workspace_tasks._fetch_fr_inpi_detail_fields("822276986")

    assert details["object_text"].startswith("Plateforme de mise en relation")
    assert details["activity_description"].startswith("Plateforme digitale")
    assert details["commercial_names"] == ["HUBLO"]
    assert details["domains"] == ["hublo.com"]
    assert details["observations"] == [
        "Opération de fusion à compter du 14/12/2023. Société(s) ayant participé à l'opération : MEDIKSTAFF"
    ]
    assert details["history_labels"] == ["Ancienne dénomination BRIGAD"]
    assert details["inpi_company_id"] == "company-1"
    assert details["inpi_url"] == "https://data.inpi.fr/entreprises/822276986"


def test_fetch_fr_inpi_detail_fields_reports_expired_token_without_credentials(monkeypatch):
    expired_token = (
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9."
        "eyJleHAiOjEwLCJ1c2VyVHlwZSI6IkZPIiwiY29ubmVjdGlvblR5cGUiOiJBUEkifQ."
        "signature"
    )
    monkeypatch.setattr(workspace_tasks.settings, "inpi_token", expired_token)
    monkeypatch.setattr(workspace_tasks.settings, "inpi_username", "")
    monkeypatch.setattr(workspace_tasks.settings, "inpi_password", "")

    details = workspace_tasks._fetch_fr_inpi_detail_fields("822276986")

    assert details["auth_error"] == "inpi_credentials_missing"


def test_fr_registry_secondary_lookup_specs_collects_commercial_and_observation_aliases():
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        context_pack_json={"sites": [{"company_name": "Hublo", "url": "https://hublo.com"}]},
    )
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_fields": {
            "observations": [
                "Opération de fusion à compter du 14/12/2023. Société(s) ayant participé à l'opération : MEDIKSTAFF"
            ]
        },
    }
    detailed_records = [
        {
            "display_name": "THE WORKING COMPANY",
            "legal_name": "THE WORKING COMPANY",
            "registry_fields": {
                "commercial_names": ["BRIGAD"],
                "history_labels": ["Ancienne dénomination BRIGAD"],
            },
        }
    ]

    specs = workspace_tasks._fr_registry_secondary_lookup_specs(profile, source_record, detailed_records)
    pairs = {(item["query_type"], item["query"]) for item in specs}

    assert ("commercial_name_lookup", "BRIGAD") in pairs
    assert ("observation_counterparty_lookup", "MEDIKSTAFF") in pairs


def test_fr_registry_semantic_score_demotes_generic_infra_rows_without_mandate_signals():
    scope_pack = {
        "core": {"terms": ["healthcare", "staffing", "hospital"], "phrases": []},
        "adjacent": {"terms": ["replacement", "pool", "management"], "phrases": []},
        "entity_seed": {"terms": ["brigad"], "phrases": ["Brigad"]},
    }
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [],
    }
    infra_record = {
        "display_name": "HEXA INFRA SERVICES",
        "name": "HEXA INFRA SERVICES",
        "legal_name": "HEXA INFRA SERVICES",
        "context_text": "infogerance cloud hebergement sauvegarde infrastructure digitale pour entreprises",
        "industry_codes": ["63.11Z"],
        "registry_fields": {},
        "website": None,
        "is_employer": False,
        "employee_band": None,
    }
    relevant_record = {
        **infra_record,
        "display_name": "BRIGAD",
        "name": "BRIGAD",
        "legal_name": "THE WORKING COMPANY",
        "context_text": (
            "plateforme de mise en relation et recrutement pour professionnels de santé "
            "et établissements de santé avec remplacement et pool management"
        ),
        "registry_fields": {
            "commercial_names": ["BRIGAD"],
            "object_text": "plateforme de mise en relation et recrutement pour professionnels de santé",
            "object_text_present": True,
            "observation_count": 1,
            "observations": ["fusion avec medikstaff"],
        },
    }

    infra_score, infra_meta = workspace_tasks._fr_registry_semantic_score(
        infra_record,
        scope_pack=scope_pack,
        normalized_scope=normalized_scope,
        code_distance=1,
    )
    relevant_score, relevant_meta = workspace_tasks._fr_registry_semantic_score(
        relevant_record,
        scope_pack=scope_pack,
        normalized_scope=normalized_scope,
        code_distance=1,
    )

    assert infra_score < relevant_score
    assert infra_meta["generic_infra_hits"]
    assert relevant_meta["object_observation_relevance"] > infra_meta["object_observation_relevance"]


def test_build_france_registry_universe_candidates_prefers_relevant_brand_and_semantic_context_over_generic_same_code_infra(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={
            "sites": [
                {
                    "company_name": "Hublo",
                    "summary": "Healthcare staffing, shift replacement, and pool management software for hospitals.",
                    "url": "https://hublo.com",
                }
            ]
        },
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [
            {
                "id": "shift_replacement",
                "label": "Shift replacement",
                "likely_customer_segments": ["Hospitals"],
                "likely_workflows": ["replacement", "pool", "pool management"],
            }
        ],
        "adjacency_box_labels": ["Shift replacement"],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale != "62.01Z":
            return [], None
        return [
            {
                "nom_complet": "THE WORKING COMPANY",
                "nom_raison_sociale": "THE WORKING COMPANY",
                "siren": "814956744",
                "activite_principale": "62.01Z",
                "activite_principale_naf25": "62.01Z",
                "libelle_activite_principale": "Programmation informatique",
                "etat_administratif": "A",
                "siege": {
                    "nom_commercial": None,
                    "caractere_employeur": "O",
                    "tranche_effectif_salarie": "12",
                    "libelle_commune": "PARIS",
                },
                "matching_etablissements": [{"nom_commercial": "BRIGAD", "liste_enseignes": None}],
                "complements": {},
            },
            {
                "nom_complet": "HEXA INFRA SERVICES",
                "nom_raison_sociale": "HEXA INFRA SERVICES",
                "siren": "222222222",
                "activite_principale": "62.01Z",
                "activite_principale_naf25": "62.01Z",
                "libelle_activite_principale": "Programmation informatique",
                "etat_administratif": "A",
                "siege": {
                    "nom_commercial": None,
                    "caractere_employeur": "O",
                    "tranche_effectif_salarie": "20",
                    "libelle_commune": "PARIS",
                },
                "complements": {},
            },
        ], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    def fake_detail(record):
        if record.get("registry_id") == "814956744":
            return {
                **record,
                    "context_text": (
                        f"{record.get('context_text', '')} "
                        "objet social: plateforme saas de replacement et de pool management "
                        "pour staffing hospitals. observations: fusion medgo whoog."
                    ).strip(),
                    "registry_fields": {
                        **(record.get("registry_fields") or {}),
                    "commercial_names": ["BRIGAD"],
                    "observation_count": 1,
                },
            }
        return {
            **record,
            "context_text": (
                f"{record.get('context_text', '')} "
                "objet social: infogerance, hebergement, cloud, infrastructure digitale."
            ).strip(),
            "registry_fields": {
                **(record.get("registry_fields") or {}),
                "observation_count": 0,
            },
        }

    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", fake_detail)

    candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
    )

    assert diagnostics["registry_candidate_count"] >= 2
    assert candidates[0]["registry_id"] == "814956744"
    assert candidates[0]["display_name"] == "BRIGAD"
    assert candidates[0]["legal_name"] == "THE WORKING COMPANY"
    assert "replacement" in candidates[0]["registry_fields"]["matched_terms"]
    assert "pool" in candidates[0]["registry_fields"]["matched_terms"]
    assert candidates[0]["precomputed_discovery_score"] > candidates[1]["precomputed_discovery_score"]


def test_build_france_registry_universe_candidates_recovers_seed_lookup_entity_absent_from_code_crawl(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing software.", "url": "https://hublo.com"}]},
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_customer_segments": ["Hospitals"],
        "source_workflows": ["replacement"],
        "company_seeds": [{"name": "Brigad"}],
        "adjacency_boxes": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "registry_fields": {"observations": []},
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale == "62.01Z":
            return [], None
        if query == "Brigad":
            return [
                {
                    "nom_complet": "THE WORKING COMPANY",
                    "nom_raison_sociale": "THE WORKING COMPANY",
                    "siren": "814956744",
                    "activite_principale": "62.01Z",
                    "activite_principale_naf25": "62.01Z",
                    "libelle_activite_principale": "Programmation informatique",
                    "etat_administratif": "A",
                    "siege": {"caractere_employeur": "O", "tranche_effectif_salarie": "12"},
                    "matching_etablissements": [{"nom_commercial": "BRIGAD"}],
                    "complements": {},
                }
            ], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(profile, normalized_scope)
    by_id = {candidate["registry_id"]: candidate for candidate in candidates}

    assert "814956744" in by_id
    assert by_id["814956744"]["display_name"] == "BRIGAD"
    assert diagnostics["registry_source_path_counts"]["seed_name_registry_lookup"] >= 1


def test_build_france_registry_universe_candidates_recovers_observation_counterparty_via_registry_resolution(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing software.", "url": "https://hublo.com"}]},
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_customer_segments": ["Hospitals"],
        "source_workflows": ["replacement"],
        "company_seeds": [],
        "adjacency_boxes": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "registry_fields": {
            "observations": [
                "Opération de fusion à compter du 14/12/2023. Société(s) ayant participé à l'opération : MEDIKSTAFF"
            ],
            "object_text": "Plateforme de mise en relation et recrutement pour les établissements de santé.",
        },
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale == "62.01Z":
            return [], None
        if query == "MEDIKSTAFF":
            return [
                {
                    "nom_complet": "MEDIKSTAFF (MSTAFF)",
                    "nom_raison_sociale": "MEDIKSTAFF",
                    "siren": "820625564",
                    "activite_principale": "62.01Z",
                    "activite_principale_naf25": "62.01Z",
                    "libelle_activite_principale": "Programmation informatique",
                    "etat_administratif": "C",
                    "siege": {"nom_commercial": "MSTAFF", "caractere_employeur": "N"},
                    "matching_etablissements": [],
                    "complements": {},
                }
            ], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(profile, normalized_scope)
    by_id = {candidate["registry_id"]: candidate for candidate in candidates}

    assert "820625564" in by_id
    assert diagnostics["registry_source_path_counts"]["observation_counterparty_lookup"] >= 1
    medikstaff_hit = next(item for item in diagnostics["benchmark_hits"] if item["registry_id"] == "820625564")
    assert medikstaff_hit["final_found"] is True


def test_build_france_registry_universe_candidates_extends_code_pages_when_signal_continues(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing software.", "url": "https://hublo.com"}]},
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_workflows": ["replacement"],
        "source_customer_segments": ["Hospitals"],
        "company_seeds": [{"name": "Brigad"}],
        "adjacency_boxes": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "registry_fields": {"observations": []},
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale == "62.01Z":
            if page == 1:
                return [
                    {
                        "nom_complet": "THE WORKING COMPANY",
                        "nom_raison_sociale": "THE WORKING COMPANY",
                        "siren": "814956744",
                        "activite_principale": "62.01Z",
                        "activite_principale_naf25": "62.01Z",
                        "libelle_activite_principale": "Programmation informatique",
                        "etat_administratif": "A",
                        "matching_etablissements": [{"nom_commercial": "BRIGAD"}],
                        "siege": {"caractere_employeur": "O"},
                        "complements": {},
                    }
                ], None
            if page == 2:
                return [
                    {
                        "nom_complet": "MEDIFLASH",
                        "nom_raison_sociale": "MEDIFLASH",
                        "siren": "887656270",
                        "activite_principale": "63.12Z",
                        "activite_principale_naf25": "63.12Z",
                        "libelle_activite_principale": "Portails Internet",
                        "etat_administratif": "A",
                        "matching_etablissements": [{"nom_commercial": "MEDIFLASH"}],
                        "siege": {"caractere_employeur": "O"},
                        "complements": {},
                    }
                ], None
            return [], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    _candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
        budget_overrides={
            "pages_per_code": 1,
            "max_pages_per_code": 3,
            "candidate_cap": 50,
            "detail_cap": 0,
            "page_extension_min_hits": 1,
            "page_stop_after_no_signal": 2,
        },
    )

    stats = next(item for item in diagnostics["code_fetch_stats"] if item["code"] == "62.01Z")
    assert stats["pages_fetched"] == 2
    assert stats["extended_pages"] >= 1


def test_build_france_registry_universe_candidates_stops_early_after_no_signal_pages(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing software.", "url": "https://hublo.com"}]},
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_workflows": ["replacement"],
        "source_customer_segments": ["Hospitals"],
        "company_seeds": [],
        "adjacency_boxes": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "registry_fields": {"observations": []},
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale == "62.01Z" and page in {1, 2, 3, 4}:
            return [
                {
                    "nom_complet": "DIGITAL HOSTING FACTORY",
                    "nom_raison_sociale": "DIGITAL HOSTING FACTORY",
                    "siren": f"44444444{page}",
                    "activite_principale": "62.01Z",
                    "activite_principale_naf25": "62.01Z",
                    "libelle_activite_principale": "Programmation informatique",
                    "etat_administratif": "A",
                    "siege": {"caractere_employeur": "O"},
                    "complements": {},
                }
            ], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    _candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
        budget_overrides={
            "pages_per_code": 2,
            "max_pages_per_code": 5,
            "candidate_cap": 50,
            "detail_cap": 0,
            "page_extension_min_hits": 2,
            "page_stop_after_no_signal": 2,
        },
    )

    stats = next(item for item in diagnostics["code_fetch_stats"] if item["code"] == "62.01Z")
    assert stats["pages_fetched"] == 2
    assert stats["stopped_early"] is True


def test_run_france_registry_recall_benchmark_returns_tiered_diagnostics(monkeypatch):
    profile = SimpleNamespace()
    normalized_scope = {"source_capabilities": [], "adjacency_boxes": []}

    def fake_builder(profile, normalized_scope, *, budget_overrides=None):
        pages = int((budget_overrides or {}).get("pages_per_code", 0))
        return (
            [{"display_name": f"Candidate {pages}", "legal_name": None, "registry_id": str(pages), "directness": "broad_market"}],
            {
                "registry_raw_candidate_count": pages * 10,
                "registry_scored_candidate_count": pages * 8,
                "registry_candidate_count": pages,
                "registry_queries_count": pages * 2,
                "registry_seed_query_count": 1,
                "registry_secondary_query_count": 1,
                "registry_source_path_counts": {"code_neighborhood_crawl": pages},
                "detail_stats": {"detail_api_hits": pages},
                "benchmark_hits": [],
                "code_fetch_stats": [],
            },
        )

    monkeypatch.setattr(workspace_tasks, "_build_france_registry_universe_candidates", fake_builder)

    benchmark = workspace_tasks._run_france_registry_recall_benchmark(profile, normalized_scope)

    assert [item["budget"]["pages_per_code"] for item in benchmark] == [5, 8, 12]
    assert benchmark[0]["registry_raw_candidate_count"] == 50
    assert benchmark[2]["top_candidates"][0]["registry_id"] == "12"


def test_build_france_registry_universe_candidates_respects_global_query_budget(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing software.", "url": "https://hublo.com"}]},
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_workflows": ["replacement"],
        "source_customer_segments": ["Hospitals"],
        "company_seeds": [{"name": "Mediflash"}],
        "adjacency_boxes": [],
        "named_account_anchors": ["Medikstaff"],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "registry_fields": {"observations": []},
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    query_calls = []

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        query_calls.append({"query": query, "code": activite_principale, "page": page})
        if activite_principale:
            return [
                {
                    "nom_complet": f"Candidate {activite_principale} {page}",
                    "nom_raison_sociale": f"Candidate {activite_principale} {page}",
                    "siren": f"{page:09d}",
                    "activite_principale": activite_principale,
                    "activite_principale_naf25": activite_principale,
                    "libelle_activite_principale": "Programmation informatique",
                    "etat_administratif": "A",
                    "siege": {"caractere_employeur": "O"},
                    "complements": {},
                }
            ], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    _candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
        budget_overrides={
            "pages_per_code": 5,
            "max_pages_per_code": 12,
            "candidate_cap": 200,
            "detail_cap": 0,
            "max_total_queries": 2,
        },
    )

    assert diagnostics["registry_queries_count"] == 2
    assert diagnostics["query_budget_exhausted"] is True
    assert len(query_calls) == 2


def test_build_france_registry_universe_candidates_reserves_seed_budget_before_code_crawl(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={"sites": [{"company_name": "Hublo", "summary": "Healthcare staffing software.", "url": "https://hublo.com"}]},
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_workflows": ["replacement"],
        "source_customer_segments": ["Hospitals"],
        "company_seeds": [{"name": "Mediflash"}],
        "adjacency_boxes": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
        "registry_fields": {"observations": []},
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )
    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", lambda record: record)

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if query == "Mediflash":
            return [
                {
                    "nom_complet": "MEDIFLASH",
                    "nom_raison_sociale": "MEDIFLASH",
                    "siren": "887656270",
                    "activite_principale": "63.12Z",
                    "activite_principale_naf25": "63.12Z",
                    "libelle_activite_principale": "Portails internet",
                    "etat_administratif": "A",
                    "siege": {"nom_commercial": "MEDIFLASH", "caractere_employeur": "O"},
                    "complements": {},
                }
            ], None
        if activite_principale:
            return [
                {
                    "nom_complet": f"Candidate {activite_principale} {page}",
                    "nom_raison_sociale": f"Candidate {activite_principale} {page}",
                    "siren": f"{page:09d}",
                    "activite_principale": activite_principale,
                    "activite_principale_naf25": activite_principale,
                    "libelle_activite_principale": "Programmation informatique",
                    "etat_administratif": "A",
                    "siege": {"caractere_employeur": "O"},
                    "complements": {},
                }
            ], None
        return [], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    candidates, diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
        budget_overrides={
            "pages_per_code": 5,
            "max_pages_per_code": 5,
            "candidate_cap": 200,
            "detail_cap": 0,
            "max_total_queries": 3,
            "seed_query_cap": 1,
            "seed_query_reserve": 1,
            "secondary_query_cap": 0,
            "secondary_query_reserve": 0,
        },
    )

    assert diagnostics["registry_seed_query_count"] == 1
    assert diagnostics["registry_code_query_count"] == 2
    assert diagnostics["registry_queries_count"] == 3
    assert diagnostics["registry_source_path_counts"]["seed_name_registry_lookup"] >= 1
    assert diagnostics["seed_query_specs"][0]["query"] == "Mediflash"
    assert diagnostics["executed_lookup_attempts"][0]["phase"] == "seed"
    assert diagnostics["executed_lookup_attempts"][0]["matched_count"] == 1
    assert any(candidate["registry_id"] == "887656270" for candidate in candidates)


def test_build_france_registry_universe_candidates_demotes_generic_digital_hosting_rows_for_healthcare_staffing_scope(monkeypatch):
    profile = SimpleNamespace(
        buyer_company_url="https://hublo.com",
        geo_scope={"include_countries": ["France"]},
        context_pack_json={
            "sites": [
                {
                    "company_name": "Hublo",
                    "summary": "Healthcare staffing, shift replacement, and pool management software for hospitals.",
                    "url": "https://hublo.com",
                }
            ]
        },
    )
    normalized_scope = {
        "source_capabilities": ["Healthcare staffing"],
        "source_customer_segments": ["Hospitals"],
        "adjacency_boxes": [
            {
                "id": "pool_mgmt",
                "label": "Pool management",
                "likely_customer_segments": ["Hospitals"],
                "likely_workflows": ["replacement", "pool", "pool management"],
            }
        ],
        "adjacency_box_labels": ["Pool management"],
        "named_account_anchors": [],
        "geography_expansions": [],
    }
    source_record = {
        "display_name": "Hublo",
        "legal_name": "HUBLO",
        "registry_id": "822276986",
        "activity_code": "62.01Z",
        "activity_code_naf25": "62.01Z",
    }

    monkeypatch.setattr(
        workspace_tasks,
        "_resolve_fr_source_registry_record",
        lambda profile, normalized_scope: (source_record, {"source_company_name": "Hublo"}),
    )

    def fake_fr_search(*, query=None, activite_principale=None, page=1, per_page=25, only_active=True, timeout_seconds=6):
        if activite_principale != "62.01Z":
            return [], None
        return [
            {
                "nom_complet": "HEALTHROTA",
                "nom_raison_sociale": "HEALTHROTA SAS",
                "siren": "333333333",
                "activite_principale": "62.01Z",
                "activite_principale_naf25": "62.01Z",
                "libelle_activite_principale": "Programmation informatique",
                "etat_administratif": "A",
                "siege": {
                    "nom_commercial": "HealthRota",
                    "caractere_employeur": "O",
                    "tranche_effectif_salarie": "15",
                    "libelle_commune": "PARIS",
                },
                "complements": {},
            },
            {
                "nom_complet": "DIGITAL HOSTING FACTORY",
                "nom_raison_sociale": "DIGITAL HOSTING FACTORY",
                "siren": "444444444",
                "activite_principale": "62.01Z",
                "activite_principale_naf25": "62.01Z",
                "libelle_activite_principale": "Programmation informatique",
                "etat_administratif": "A",
                "siege": {
                    "nom_commercial": None,
                    "caractere_employeur": "O",
                    "tranche_effectif_salarie": "20",
                    "libelle_commune": "PARIS",
                },
                "complements": {},
            },
        ], None

    monkeypatch.setattr(workspace_tasks, "_fr_registry_search", fake_fr_search)

    def fake_detail(record):
        if record.get("registry_id") == "333333333":
                return {
                    **record,
                    "context_text": (
                        f"{record.get('context_text', '')} "
                        "platform for replacement staffing and pool management "
                        "for hospitals and clinics."
                    ).strip(),
                }
        return {
            **record,
            "context_text": (
                f"{record.get('context_text', '')} "
                "hebergement cloud, infrastructure digitale, infogerance et services web."
            ).strip(),
        }

    monkeypatch.setattr(workspace_tasks, "_fetch_fr_registry_detail_record", fake_detail)

    candidates, _diagnostics = workspace_tasks._build_france_registry_universe_candidates(
        profile,
        normalized_scope,
    )

    by_id = {candidate["registry_id"]: candidate for candidate in candidates}
    assert by_id["333333333"]["directness"] in {"direct", "adjacent"}
    assert by_id["333333333"]["precomputed_discovery_score"] > by_id["444444444"]["precomputed_discovery_score"]
    assert by_id["444444444"]["registry_fields"]["generic_infra_hits"]


def test_limit_directory_seed_candidates_caps_listing_and_non_website_volume():
    candidates = []
    for idx in range(8):
        candidates.append(
            {
                "name": f"Vendor {idx}",
                "website": None,
                "official_website_url": None,
                "entity_type": "solution",
                "_origins": [
                    {
                        "origin_type": "directory_seed",
                        "origin_url": "https://example.com/listing/a",
                        "source_name": "wealth_mosaic",
                        "metadata": {},
                    }
                ],
                "why_relevant": [{"text": "Directory listing"}],
            }
        )
    for idx in range(6):
        candidates.append(
            {
                "name": f"Official {idx}",
                "website": f"https://official-{idx}.example.com",
                "official_website_url": f"https://official-{idx}.example.com",
                "entity_type": "company",
                "_origins": [
                    {
                        "origin_type": "directory_seed",
                        "origin_url": "https://example.com/listing/b",
                        "source_name": "wealth_mosaic",
                        "metadata": {},
                    }
                ],
                "why_relevant": [{"text": "Directory listing"}],
            }
        )

    kept, stats = workspace_tasks._limit_directory_seed_candidates(
        candidates,
        max_total=20,
        max_per_listing=4,
        max_per_source=20,
        max_without_website=2,
    )

    assert len(kept) == 6
    assert sum(1 for item in kept if item.get("official_website_url")) == 4
    assert sum(1 for item in kept if not item.get("official_website_url")) == 2
    assert stats["dropped_per_listing_cap"] >= 2
    assert stats["dropped_without_website_cap"] >= 1


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
