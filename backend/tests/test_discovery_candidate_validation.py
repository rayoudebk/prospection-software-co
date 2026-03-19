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
    assert plan["seed_urls"] == []
    assert not any("plannerco.example.com" in url for url in plan["seed_urls"])
    assert any("PlannerCo alternatives" in text for text in recall_texts)
    assert not any("CHU Lille" in text for text in recall_texts + [entry["query_text"] for entry in plan["precision_queries"]])
    assert not any("wealthtech" in text.lower() for text in recall_texts + [entry["query_text"] for entry in plan["precision_queries"]])
    assert "bloomberg" not in [item.lower() for item in plan["must_exclude_terms"]]


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


def test_external_candidate_name_prefers_domain_label_for_article_like_titles():
    assert (
        workspace_tasks._external_candidate_name(
            "Private Bank Portfolio Management Software",
            "https://masttro.com/institutions",
        )
        == "Masttro"
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
