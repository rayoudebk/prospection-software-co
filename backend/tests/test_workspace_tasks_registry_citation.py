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


def test_build_first_party_hint_url_map_uses_evidence_urls_and_path_vendor_urls():
    profile = SimpleNamespace(
        reference_evidence_urls=[
            "https://upvest.co/blog/zopa-bank-partners-with-upvest",
            "https://upvest.co/blog/boerse-stuttgart-and-upvest/",
        ],
        reference_vendor_urls=[
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
