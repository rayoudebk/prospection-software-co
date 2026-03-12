from app.models.workspace import CompanyProfile
from app.workers import workspace_tasks
from app.workers.workspace_tasks import _context_pack_start_urls_for_site


def test_context_pack_start_urls_include_buyer_supporting_evidence_same_domain():
    profile = CompanyProfile(
        workspace_id=1,
        buyer_company_url="https://4tpm.fr/offers/patio-oms/?lang=en",
        reference_company_urls=[
            "https://wealth-dynamix.com/",
            "https://4tpm.fr/company/about",
        ],
        reference_evidence_urls=[
            "https://4tpm.fr/technology-services/services/?lang=en",
            "https://4tpm.fr/4tpm/references/?lang=en",
            "https://cwan.com/customers",
        ],
    )

    start_urls = _context_pack_start_urls_for_site(profile, "https://4tpm.fr/offers/patio-oms/?lang=en")

    assert "https://4tpm.fr/technology-services/services?lang=en" in start_urls
    assert "https://4tpm.fr/4tpm/references?lang=en" in start_urls
    assert "https://4tpm.fr/company/about" in start_urls
    assert all("cwan.com" not in url for url in start_urls)
    assert all("wealth-dynamix.com" not in url for url in start_urls)


def test_context_pack_start_urls_include_adaptive_buyer_hints(monkeypatch):
    profile = CompanyProfile(
        workspace_id=1,
        buyer_company_url="https://4tpm.fr/offers/patio-oms/?lang=en",
        reference_company_urls=[
            "https://wealth-dynamix.com/",
            "https://cwan.com/",
        ],
        reference_evidence_urls=[],
    )

    monkeypatch.setattr(
        workspace_tasks,
        "_discover_adaptive_hint_urls_for_domain",
        lambda domain, timeout_seconds=5, max_urls=18: [
            "https://4tpm.fr/platform/front-office",
            "https://4tpm.fr/platform/back-office",
            "https://4tpm.fr/solutions/online-brokerage",
        ] if domain == "4tpm.fr" else [],
    )

    start_urls = _context_pack_start_urls_for_site(profile, "https://4tpm.fr/offers/patio-oms/?lang=en")

    assert "https://4tpm.fr/platform/front-office" in start_urls
    assert "https://4tpm.fr/platform/back-office" in start_urls
    assert "https://4tpm.fr/solutions/online-brokerage" in start_urls


def test_context_pack_start_urls_include_adaptive_comparator_hints(monkeypatch):
    profile = CompanyProfile(
        workspace_id=1,
        buyer_company_url="https://4tpm.fr/offers/patio-oms/?lang=en",
        reference_company_urls=[
            "https://wealth-dynamix.com/",
            "https://cwan.com/",
        ],
        reference_evidence_urls=[],
    )

    monkeypatch.setattr(
        workspace_tasks,
        "_discover_adaptive_hint_urls_for_domain",
        lambda domain, timeout_seconds=5, max_urls=12: [
            "https://wealth-dynamix.com/solutions/client-lifecycle-management",
            "https://wealth-dynamix.com/capabilities/client-engagement-portal",
        ] if domain == "wealth-dynamix.com" else [],
    )

    start_urls = _context_pack_start_urls_for_site(profile, "https://wealth-dynamix.com/")

    assert "https://wealth-dynamix.com/solutions/client-lifecycle-management" in start_urls
    assert "https://wealth-dynamix.com/capabilities/client-engagement-portal" in start_urls
    assert all("4tpm.fr" not in url for url in start_urls)
