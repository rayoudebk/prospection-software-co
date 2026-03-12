from app.models.workspace import CompanyProfile
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
