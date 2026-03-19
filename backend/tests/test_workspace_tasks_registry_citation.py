from types import SimpleNamespace

import app.workers.workspace_tasks as workspace_tasks
from app.services.crawler.models import ContentBlock, ContextPack, CrawledPage, CustomerEvidence, Evidence, Signal


def _seed_entity(country: str = "UK") -> dict:
    return {
        "canonical_name": "Seed Vendor",
        "origin_types": ["reference_seed"],
        "registry_identity": {"country": country},
        "registry_country": country,
        "country": country,
        "industry_signature": {"industry_codes": [], "industry_keywords": ["wealth"]},
        "why_relevant": [{"text": "Institutional wealth workflow context."}],
    }


def test_expand_registry_neighbors_rejects_missing_official_website(monkeypatch):
    monkeypatch.setattr(workspace_tasks, "_registry_queries_for_entity", lambda name, industry_signature=None: ["seed query"])

    def fake_run_registry_search(country: str, query: str):
        return (
            [
                {
                    "name": "No Website Ltd",
                    "registry_id": "12345678",
                    "website": "",
                    "status": "active",
                    "registry_source": "uk_companies_house",
                    "registry_url": "https://find-and-update.company-information.service.gov.uk/company/12345678",
                    "industry_codes": ["62012"],
                    "industry_keywords": ["wealthtech"],
                }
            ],
            None,
            "uk_companies_house",
        )

    monkeypatch.setattr(workspace_tasks, "_run_registry_search", fake_run_registry_search)

    accepted, diagnostics, query_logs = workspace_tasks._expand_registry_neighbors(
        [_seed_entity(country="UK")],
        run_id="test-run",
        max_queries=1,
        max_neighbors=10,
    )

    assert accepted == []
    assert diagnostics["registry_neighbors_with_first_party_website_count"] == 0
    assert diagnostics["registry_neighbors_dropped_missing_official_website_count"] == 1
    assert diagnostics["registry_reject_reason_breakdown"].get("missing_official_website") == 1
    assert diagnostics["registry_origin_screening_counts"]["records_rejected"] == 1
    assert query_logs[0]["reject_reasons_json"].get("missing_official_website") == 1


def test_registry_expansion_countries_include_western_europe_set():
    expected = {"FR", "UK", "DE", "BE", "NL", "LU", "CH", "MC"}
    assert expected.issubset(workspace_tasks.REGISTRY_EXPANSION_COUNTRIES)


def test_run_registry_search_routes_to_gleif_for_new_countries(monkeypatch):
    called: list[tuple[str, str]] = []

    def fake_gleif(country: str, query: str, max_hits: int = 10, timeout_seconds: int = 6):
        called.append((country, query))
        return (
            [
                {
                    "name": "Aixigo AG",
                    "country": country,
                    "registry_id": "LEI123",
                    "registry_source": f"{country.lower()}_gleif_lei",
                    "registry_url": "https://search.gleif.org/#/record/LEI123",
                    "status": "active",
                    "is_active": True,
                    "industry_codes": [],
                    "industry_keywords": ["wealth"],
                }
            ],
            None,
        )

    monkeypatch.setattr(workspace_tasks, "_search_gleif_registry_neighbors", fake_gleif)

    records, error, source = workspace_tasks._run_registry_search("LU", "aixigo")
    assert error is None
    assert source == "lu_gleif_lei"
    assert records and records[0]["registry_source"] == "lu_gleif_lei"
    assert called == [("LU", "aixigo")]


def test_registry_lookup_reasons_include_gleif_for_new_countries():
    reasons = workspace_tasks._registry_lookup_reasons("Aixigo", "CH")
    assert reasons
    assert any("GLEIF LEI search" in str(item.get("text") or "") for item in reasons)
    assert any("api.gleif.org" in str(item.get("citation_url") or "") for item in reasons)


def test_expand_registry_neighbors_accepts_valid_official_website(monkeypatch):
    monkeypatch.setattr(workspace_tasks, "_registry_queries_for_entity", lambda name, industry_signature=None: ["seed query"])

    def fake_run_registry_search(country: str, query: str):
        return (
            [
                {
                    "name": "Rich Evidence AG",
                    "registry_id": "HRB123",
                    "website": "https://rich-evidence.example",
                    "status": "active",
                    "registry_source": "de_registry",
                    "registry_url": "https://handelsregister.de/rich-evidence",
                    "industry_codes": ["62012"],
                    "industry_keywords": ["wealth", "software"],
                }
            ],
            None,
            "de_registry",
        )

    monkeypatch.setattr(workspace_tasks, "_run_registry_search", fake_run_registry_search)
    monkeypatch.setattr(
        workspace_tasks,
        "_score_registry_neighbor",
        lambda seed_entity, record: (
            workspace_tasks.REGISTRY_NEIGHBOR_MIN_SCORE + 5.0,
            {"industry_code_match": 1.0},
            ["wealth"],
        ),
    )

    accepted, diagnostics, _ = workspace_tasks._expand_registry_neighbors(
        [_seed_entity(country="DE")],
        run_id="test-run",
        max_queries=1,
        max_neighbors=10,
    )

    assert len(accepted) == 1
    row = accepted[0]
    assert row["website"] == "https://rich-evidence.example"
    assert row["official_website_url"] == "https://rich-evidence.example"
    assert row["discovery_url"] == "https://handelsregister.de/rich-evidence"
    assert diagnostics["registry_neighbors_with_first_party_website_count"] == 1
    assert diagnostics["registry_neighbors_dropped_missing_official_website_count"] == 0
    assert diagnostics["registry_origin_screening_counts"]["records_accepted"] == 1


def test_registry_profile_only_identity_never_marked_high(monkeypatch):
    profile_url = "https://find-and-update.company-information.service.gov.uk/company/12345678"

    def fake_resolver(url: str, timeout_seconds: int = 4):
        return {
            "profile_url": url,
            "official_website": url,
            "identity_confidence": "high",
            "captured_at": "2026-01-01T00:00:00",
            "error": None,
        }

    monkeypatch.setattr(workspace_tasks, "resolve_external_website_from_profile", fake_resolver)

    candidate = {"name": "Registry-only Co", "website": profile_url, "first_party_domains": []}
    workspace_tasks._resolve_identities_for_candidates([candidate], max_fetches=1)

    assert candidate["official_website_url"] is None
    assert candidate["website"] is None
    assert candidate["identity"]["identity_confidence"] == "low"
    assert candidate["identity"]["input_domain"] == "find-and-update.company-information.service.gov.uk"


def test_resolve_identities_canonicalizes_redirected_first_party_domain(monkeypatch):
    def fake_direct_identity(url: str, timeout_seconds: int = 3):
        return {
            "profile_url": url,
            "official_website": "https://cwan.com/",
            "identity_confidence": "high",
            "captured_at": "2026-01-01T00:00:00",
            "error": None,
            "resolved_via_redirect": True,
        }

    monkeypatch.setattr(workspace_tasks, "_resolve_direct_website_identity", fake_direct_identity)

    candidate = {"name": "Jump Technology", "website": "https://www.jump-technology.com", "first_party_domains": []}
    stats = workspace_tasks._resolve_identities_for_candidates([candidate], max_fetches=5)

    assert stats["identity_resolved_count"] == 1
    assert stats["identity_fetch_count"] == 1
    assert candidate["official_website_url"] == "https://cwan.com/"
    assert candidate["website"] == "https://cwan.com/"
    assert candidate["identity"]["resolved_via_redirect"] is True
    assert "cwan.com" in candidate["first_party_domains"]
    assert "jump-technology.com" in candidate["first_party_domains"]


def test_resolve_directory_profile_seed_candidates_only_targets_directory_profiles(monkeypatch):
    calls: list[str] = []

    def fake_resolver(url: str, timeout_seconds: int = 4):
        calls.append(url)
        return {
            "profile_url": url,
            "official_website": "https://vendor.example.com",
            "identity_confidence": "high",
            "captured_at": "2026-01-01T00:00:00",
            "error": None,
        }

    monkeypatch.setattr(workspace_tasks, "resolve_external_website_from_profile", fake_resolver)

    directory_candidate = {
        "name": "Vendor Profile",
        "profile_url": "https://www.thewealthmosaic.com/vendors/vendor-profile/platform/",
        "discovery_url": "https://www.thewealthmosaic.com/vendors/vendor-profile/platform/",
        "website": None,
        "official_website_url": None,
        "_origins": [{"origin_type": "directory_seed", "source_name": "wealth_mosaic"}],
    }
    external_candidate = {
        "name": "External Vendor",
        "website": "https://vendor.example.com",
        "official_website_url": "https://vendor.example.com",
        "_origins": [{"origin_type": "external_search_seed", "source_name": "exa"}],
    }

    stats = workspace_tasks._resolve_directory_profile_seed_candidates(
        [directory_candidate, external_candidate],
        max_fetches=5,
    )

    assert stats["candidates_considered"] == 1
    assert stats["candidates_selected"] == 1
    assert stats["identity_resolved_count"] == 1
    assert calls == ["https://www.thewealthmosaic.com/vendors/vendor-profile/platform/"]
    assert directory_candidate["official_website_url"] == "https://vendor.example.com"
    assert directory_candidate["website"] == "https://vendor.example.com"


def test_build_first_party_hint_url_map_uses_evidence_urls_and_path_company_urls():
    profile = SimpleNamespace(
        supporting_evidence_urls=[
            "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "https://upvest.co/blog/boerse-stuttgart-and-upvest/",
        ],
        comparator_seed_urls=[
            "https://upvest.co/blog/liqid-enters-partnership-with-upvest-for-its-eltif-offering",
            "https://upvest.co",  # root URL should be ignored for path-required vendor hints
        ],
    )

    hint_map = workspace_tasks._build_first_party_hint_url_map(profile, include_benchmark_hints=True)
    assert "upvest.co" in hint_map
    assert "https://upvest.co/blog/zopa-bank-partners-with-upvest" in hint_map["upvest.co"]
    assert "https://upvest.co/blog/boerse-stuttgart-and-upvest/" in hint_map["upvest.co"]
    assert "https://upvest.co/blog/liqid-enters-partnership-with-upvest-for-its-eltif-offering" in hint_map["upvest.co"]
    assert "https://upvest.co/" not in hint_map["upvest.co"]


def test_auto_first_party_hint_urls_include_client_stories():
    hints = workspace_tasks._auto_first_party_hint_urls_for_domains(["cwan.com"])
    assert "https://cwan.com/client-stories/" in hints
    assert "https://cwan.com/customers/" in hints
    assert "https://cwan.com/case-studies/" in hints


def test_discover_adaptive_hint_urls_for_domain_combines_homepage_and_sitemap(monkeypatch):
    homepage = """
    <html>
      <body>
        <nav>
          <a href="/news/">News</a>
          <a href="/client-stories/">Client Stories</a>
        </nav>
        <footer>
          <a href="/customers/acme-bank/">Acme story</a>
        </footer>
      </body>
    </html>
    """
    sitemap = """
    <urlset>
      <url><loc>https://example.com/blog/acme-bank-partners</loc></url>
      <url><loc>https://example.com/case-studies/wealth-suite</loc></url>
      <url><loc>https://example.com/assets/logo.png</loc></url>
    </urlset>
    """

    def fake_fetch(url: str, timeout_seconds: int = 6):
        if url.endswith("/robots.txt"):
            return "Sitemap: https://example.com/sitemap.xml", "text/plain", url
        if url.endswith("/sitemap.xml"):
            return sitemap, "application/xml", url
        return homepage, "text/html", "https://example.com/"

    monkeypatch.setattr(workspace_tasks, "_fetch_hint_document", fake_fetch)

    hints = workspace_tasks._discover_adaptive_hint_urls_for_domain(
        "example.com",
        timeout_seconds=1,
        max_urls=20,
    )

    assert "https://example.com/client-stories/" in hints
    assert "https://example.com/customers/acme-bank/" in hints
    assert "https://example.com/blog/acme-bank-partners" in hints
    assert "https://example.com/case-studies/wealth-suite" in hints
    assert all(not hint.endswith(".png") for hint in hints)


def test_hint_url_score_prefers_target_career_paths_over_finance_roles():
    engineering_score = workspace_tasks._hint_url_score(
        "https://example.com/jobs/senior-solutions-engineer"
    )
    finance_score = workspace_tasks._hint_url_score(
        "https://example.com/jobs/financial-controller"
    )

    assert engineering_score > finance_score


def test_extract_first_party_signals_from_crawl_uses_hint_urls(monkeypatch):
    captured: dict[str, list[str]] = {}

    class FakeCrawler:
        def __init__(self, max_pages: int, timeout: int):
            self.max_pages = max_pages
            self.timeout = timeout

        async def crawl_for_context(self, url: str, start_urls=None):
            captured["start_urls"] = list(start_urls or [])
            hinted = "https://upvest.co/blog/zopa-bank-partners-with-upvest"
            return ContextPack(
                company_name="Upvest",
                website=url,
                pages=[
                    CrawledPage(
                        url=hinted,
                        title="Zopa Bank partners with Upvest",
                        page_type="customers",
                        blocks=[
                            ContentBlock(
                                type="paragraph",
                                content="Zopa Bank partners with Upvest to launch new institutional investment workflows.",
                            )
                        ],
                    ),
                    CrawledPage(
                        url="https://upvest.co/",
                        title="Upvest",
                        page_type="home",
                        blocks=[],
                    ),
                ],
                signals=[
                    Signal(
                        type="customer",
                        value="Zopa Bank",
                        evidence=Evidence(
                            source_url=hinted,
                            snippet="Zopa Bank partners with Upvest to power investment offerings for bank users.",
                        ),
                    )
                ],
                customer_evidence=[
                    CustomerEvidence(
                        name="Zopa Bank",
                        source_url=hinted,
                        evidence_type="case_study",
                        context="Partnership announcement",
                    )
                ],
            )

    import app.services.crawler as crawler_module

    monkeypatch.setattr(crawler_module, "UnifiedCrawler", FakeCrawler)

    hint_url = "https://upvest.co/blog/zopa-bank-partners-with-upvest"
    reasons, capabilities, meta, error = workspace_tasks._extract_first_party_signals_from_crawl(
        website="https://upvest.co",
        candidate_name="Upvest",
        max_pages=4,
        hint_urls=[hint_url],
    )

    assert error is None
    assert captured["start_urls"] == [hint_url]
    assert meta["hint_urls_used_count"] == 1
    assert meta["hint_pages_crawled"] >= 1
    assert meta["pages_crawled"] == 2
    assert reasons
    assert any(reason["citation_url"] == hint_url for reason in reasons)
    assert any("Zopa Bank" in reason["text"] for reason in reasons)
    assert capabilities == []


def test_extract_first_party_signals_from_crawl_uses_rendered_browser_fallback(monkeypatch):
    class FakeCrawler:
        def __init__(self, max_pages: int, timeout: int):
            self.max_pages = max_pages
            self.timeout = timeout

        async def crawl_for_context(self, url: str, start_urls=None):
            return ContextPack(
                company_name="CWAN",
                website=url,
                pages=[
                    CrawledPage(
                        url="https://cwan.com/",
                        title="CWAN",
                        page_type="home",
                        blocks=[],
                    )
                ],
                signals=[],
                customer_evidence=[],
            )

    def fake_browser_fallback(website: str, candidate_name: str, hint_urls=None, max_pages=2):
        return (
            [
                {
                    "text": "Client story: Acme Bank adopted CWAN workflows for institutional portfolio servicing.",
                    "citation_url": "https://cwan.com/client-stories/acme-bank",
                    "dimension": "customer",
                    "source_kind": "rendered_browser",
                }
            ],
            [],
            {
                "method": "chrome_devtools_mcp",
                "pages_crawled": 1,
                "hint_pages_crawled": 1,
                "hint_hit_urls": ["https://cwan.com/client-stories/acme-bank"],
                "provider": "chrome_devtools_mcp_playwright",
            },
            None,
        )

    import app.services.crawler as crawler_module

    monkeypatch.setattr(crawler_module, "UnifiedCrawler", FakeCrawler)
    monkeypatch.setattr(workspace_tasks, "_extract_first_party_signals_via_rendered_browser", fake_browser_fallback)
    monkeypatch.setattr(workspace_tasks.settings, "chrome_mcp_enabled", True, raising=False)

    reasons, _, meta, error = workspace_tasks._extract_first_party_signals_from_crawl(
        website="https://cwan.com",
        candidate_name="CWAN",
        max_pages=4,
        hint_urls=["https://cwan.com/client-stories/acme-bank"],
    )

    assert error is None
    assert reasons
    assert any(reason.get("source_kind") == "rendered_browser" for reason in reasons)
    assert meta["browser_fallback_attempted"] is True
    assert meta["method"] == "crawler_plus_rendered_browser"
    assert meta["hint_pages_crawled"] >= 1


def test_extract_first_party_signals_prioritizes_customer_reasons(monkeypatch):
    class FakeCrawler:
        def __init__(self, max_pages: int, timeout: int):
            self.max_pages = max_pages
            self.timeout = timeout

        async def crawl_for_context(self, url: str, start_urls=None):
            product_signals = [
                Signal(
                    type="capability",
                    value=f"Capability {index}",
                    evidence=Evidence(
                        source_url="https://cwan.com/products/beacon",
                        snippet=f"Product capability detail {index} for institutional workflows and analytics.",
                    ),
                )
                for index in range(25)
            ]
            return ContextPack(
                company_name="CWAN",
                website=url,
                pages=[
                    CrawledPage(
                        url="https://cwan.com/products/beacon",
                        title="Beacon",
                        page_type="product",
                        blocks=[],
                    )
                ],
                signals=product_signals,
                customer_evidence=[
                    CustomerEvidence(
                        name="Major Asset Manager",
                        source_url="https://cwan.com/client-stories/",
                        evidence_type="case_study",
                        context="Client story",
                    )
                ],
            )

    import app.services.crawler as crawler_module

    monkeypatch.setattr(crawler_module, "UnifiedCrawler", FakeCrawler)

    reasons, _, meta, error = workspace_tasks._extract_first_party_signals_from_crawl(
        website="https://cwan.com",
        candidate_name="CWAN",
        max_pages=6,
        hint_urls=["https://cwan.com/client-stories/"],
    )

    assert error is None
    assert meta["customer_evidence_count"] == 1
    assert any(reason["dimension"] == "customer" for reason in reasons)
    assert any("Major Asset Manager" in reason["text"] for reason in reasons)


def test_citation_summary_v1_sentence_to_pill_contract():
    claims = [
        {
            "claim_text": "Upvest partners with Zopa Bank for investment workflows.",
            "source_url": "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "source_type": "first_party_website",
            "source_tier": "tier1_vendor",
            "claim_group": "traction",
            "captured_at": "2026-01-01T00:00:00",
        },
        {
            "claim_text": "Upvest partners with Zopa Bank for investment workflows.",
            "source_url": "https://upvest.co/blog/zopa-bank-partners-with-upvest/",
            "source_type": "first_party_website",
            "source_tier": "tier1_vendor",
            "claim_group": "traction",
            "captured_at": "2026-01-01T00:00:01",
        },
        {
            "claim_text": "Upvest partners with Zopa Bank for investment workflows.",
            "source_url": "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "source_type": "first_party_website",
            "source_tier": "tier1_vendor",
            "claim_group": "vertical_workflow",
            "captured_at": "2026-01-01T00:00:02",
        },
        {
            "claim_text": "Registry filing confirms active legal entity.",
            "source_url": "https://find-and-update.company-information.service.gov.uk/company/12345678",
            "source_type": "official_registry_filing",
            "source_tier": "tier0_registry",
            "claim_group": "identity_scope",
            "captured_at": "2026-01-01T00:00:03",
        },
    ]

    summary = workspace_tasks._build_citation_summary_v1(claims)
    assert summary is not None
    assert summary["version"] == "v1"
    assert summary["sentences"]
    assert summary["source_pills"]

    pill_ids = {pill["pill_id"] for pill in summary["source_pills"]}
    assert len(summary["source_pills"]) == 3  # deduped by normalized URL + claim_group
    assert len(pill_ids) == len(summary["source_pills"])

    for sentence in summary["sentences"]:
        assert sentence["citation_pill_ids"]
        assert all(pill_id in pill_ids for pill_id in sentence["citation_pill_ids"])


def test_extract_price_floor_ignores_fundraising_currency_mentions():
    candidate = {"qualification": {}}
    reasons = [
        {
            "text": "Upvest closes a €30m fundraising round with participation from BlackRock and existing investors.",
            "citation_url": "https://upvest.co/blog/upvest-partners-with-blackrock",
            "dimension": "traction",
        }
    ]

    assert workspace_tasks._extract_price_floor_usd(candidate, reasons) is None
    passed_gate, reject_reasons, gate_meta = workspace_tasks._evaluate_enterprise_b2b_fit(candidate, reasons)
    assert passed_gate is True
    assert "low_ticket_public_pricing" not in reject_reasons
    assert gate_meta["public_price_floor_usd_month"] is None


def test_extract_price_floor_ignores_dollar_fundraising_suffix_mentions():
    candidate = {"qualification": {}}
    reasons = [
        {
            "text": "The company raised $50M in Series C to expand product operations.",
            "citation_url": "https://example.com/news/series-c",
            "dimension": "traction",
        }
    ]

    assert workspace_tasks._extract_price_floor_usd(candidate, reasons) is None


def test_extract_price_floor_parses_suffix_when_pricing_context_present():
    candidate = {"qualification": {}}
    reasons = [
        {
            "text": "Public pricing starts at €1.5k per month for platform access.",
            "citation_url": "https://example.com/pricing",
            "dimension": "pricing_gtm",
        }
    ]

    price_floor = workspace_tasks._extract_price_floor_usd(candidate, reasons)
    assert price_floor is not None
    assert int(price_floor) == 1500


def test_extract_price_floor_keeps_low_ticket_detection_for_actual_pricing():
    candidate = {"qualification": {}}
    reasons = [
        {
            "text": "Plans start at €49 per month and are self-serve online.",
            "citation_url": "https://example.com/pricing",
            "dimension": "pricing_gtm",
        }
    ]

    passed_gate, reject_reasons, gate_meta = workspace_tasks._evaluate_enterprise_b2b_fit(candidate, reasons)
    assert passed_gate is False
    assert "low_ticket_public_pricing" in reject_reasons
    assert gate_meta["public_price_floor_usd_month"] == 49.0


def test_low_direct_price_requires_pricing_evidence_to_trigger_gate():
    candidate = {"qualification": {"public_price_floor_usd_month": 49}}
    reasons = [
        {
            "text": "Zopa Bank partners with Upvest to launch institutional investment workflows.",
            "citation_url": "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "dimension": "customer",
        }
    ]

    passed_gate, reject_reasons, gate_meta = workspace_tasks._evaluate_enterprise_b2b_fit(candidate, reasons)
    assert passed_gate is True
    assert "low_ticket_public_pricing" not in reject_reasons
    assert gate_meta["public_price_floor_usd_month"] is None


def test_build_claim_records_maps_product_icp_signal_to_vertical_workflow():
    candidate = {
        "name": "Upvest",
        "website": "https://upvest.co",
        "official_website_url": "https://upvest.co",
    }
    trusted_reasons = [
        {
            "text": "Infrastructure platform for banks and wealth managers to launch investment workflows.",
            "citation_url": "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "dimension": "product",
        }
    ]

    records = workspace_tasks._build_claim_records(
        workspace_id=1,
        company_id=None,
        company_screening_id=1,
        candidate=candidate,
        trusted_reasons=trusted_reasons,
        matched_mentions=[],
        first_party_domains=["upvest.co"],
    )

    assert records
    first_reason_claim = records[0]
    assert first_reason_claim["dimension"] == "product"
    assert first_reason_claim["claim_group"] == "vertical_workflow"


def test_build_claim_records_keeps_generic_product_as_product_depth():
    candidate = {
        "name": "Generic Platform",
        "website": "https://example.com",
        "official_website_url": "https://example.com",
    }
    trusted_reasons = [
        {
            "text": "Cloud reporting module with configurable dashboards and export options.",
            "citation_url": "https://example.com/product/reporting",
            "dimension": "product",
        }
    ]

    records = workspace_tasks._build_claim_records(
        workspace_id=1,
        company_id=None,
        company_screening_id=1,
        candidate=candidate,
        trusted_reasons=trusted_reasons,
        matched_mentions=[],
        first_party_domains=["example.com"],
    )

    assert records
    first_reason_claim = records[0]
    assert first_reason_claim["claim_group"] == "product_depth"


def _size_score_fixture():
    candidate = {"name": "FixtureCo", "website": "https://fixture.example"}
    reasons = [
        {
            "text": "Institutional workflow platform for private banks and asset managers.",
            "citation_url": "https://fixture.example/platform",
            "dimension": "product",
        },
        {
            "text": "Case study: Bank Alpha migrated portfolio workflows to FixtureCo.",
            "citation_url": "https://fixture.example/case-study/bank-alpha",
            "dimension": "customer",
        },
        {
            "text": "Implementation and integration services for compliance and onboarding.",
            "citation_url": "https://fixture.example/services",
            "dimension": "services",
        },
    ]
    capabilities = [
        "Institutional platform suite",
        "Workflow orchestration",
        "Compliance automation",
        "OMS/PMS integration",
    ]
    gate_meta = {
        "has_institutional_signal": True,
        "target_customer": "asset_managers",
        "go_to_market": "b2b_enterprise",
        "pricing_model": "enterprise_quote",
        "public_price_floor_usd_month": None,
        "hard_fail": False,
    }
    return candidate, reasons, capabilities, gate_meta


def test_score_buy_side_candidate_applies_size_similarity_boost(monkeypatch):
    monkeypatch.setattr(workspace_tasks.settings, "size_fit_window_ratio", 0.30, raising=False)
    monkeypatch.setattr(workspace_tasks.settings, "size_fit_boost_points", 8.0, raising=False)
    monkeypatch.setattr(workspace_tasks.settings, "size_large_company_threshold", 200, raising=False)
    monkeypatch.setattr(workspace_tasks.settings, "size_large_company_penalty_points", 10.0, raising=False)

    candidate, reasons, capabilities, gate_meta = _size_score_fixture()

    score_without_size, _, penalties_without_size, _ = workspace_tasks._score_buy_side_candidate(
        candidate=candidate,
        reasons=reasons,
        capability_signals=capabilities,
        gate_meta=gate_meta,
        reject_reasons=[],
        candidate_employee_estimate=None,
        buyer_employee_estimate=50,
    )
    score_with_boost, _, penalties_with_boost, _ = workspace_tasks._score_buy_side_candidate(
        candidate=candidate,
        reasons=reasons,
        capability_signals=capabilities,
        gate_meta=gate_meta,
        reject_reasons=[],
        candidate_employee_estimate=60,  # within +/- 30% of buyer size 50
        buyer_employee_estimate=50,
    )

    assert score_with_boost > score_without_size
    assert any(
        p.get("reason") == "buyer_size_similarity_boost" and float(p.get("points", 0.0)) < 0
        for p in penalties_with_boost
    )
    assert all(p.get("reason") != "buyer_size_similarity_boost" for p in penalties_without_size)


def test_score_buy_side_candidate_applies_large_company_penalty(monkeypatch):
    monkeypatch.setattr(workspace_tasks.settings, "size_fit_window_ratio", 0.30, raising=False)
    monkeypatch.setattr(workspace_tasks.settings, "size_fit_boost_points", 8.0, raising=False)
    monkeypatch.setattr(workspace_tasks.settings, "size_large_company_threshold", 200, raising=False)
    monkeypatch.setattr(workspace_tasks.settings, "size_large_company_penalty_points", 10.0, raising=False)

    candidate, reasons, capabilities, gate_meta = _size_score_fixture()

    score_reference, _, _, _ = workspace_tasks._score_buy_side_candidate(
        candidate=candidate,
        reasons=reasons,
        capability_signals=capabilities,
        gate_meta=gate_meta,
        reject_reasons=[],
        candidate_employee_estimate=120,
        buyer_employee_estimate=90,
    )
    score_large, _, penalties_large, _ = workspace_tasks._score_buy_side_candidate(
        candidate=candidate,
        reasons=reasons,
        capability_signals=capabilities,
        gate_meta=gate_meta,
        reject_reasons=[],
        candidate_employee_estimate=260,
        buyer_employee_estimate=90,
    )

    assert score_large < score_reference
    assert any(p.get("reason") == "oversized_employee_count" for p in penalties_large)
    assert all(p.get("reason") != "buyer_size_similarity_boost" for p in penalties_large)


def test_resolve_buyer_employee_estimate_prefers_policy_override():
    workspace = SimpleNamespace(
        decision_policy_json={
            "buyer_employee_estimate": 42,
            "buyer_size": {"employee_estimate": 80},
        }
    )
    profile = SimpleNamespace(
        context_pack_markdown="We are a team of 19 employees.",
        context_pack_json={},
        comparator_seed_summaries={},
    )

    assert workspace_tasks._resolve_buyer_employee_estimate(workspace, profile) == 42


def test_resolve_buyer_employee_estimate_reads_context_text():
    workspace = SimpleNamespace(decision_policy_json={})
    profile = SimpleNamespace(
        context_pack_markdown="Independent fintech team with 55 employees across product and ops.",
        context_pack_json={},
        comparator_seed_summaries={},
    )

    assert workspace_tasks._resolve_buyer_employee_estimate(workspace, profile) == 55
