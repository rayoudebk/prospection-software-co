from app.services.crawler.preview import PageScorer
from app.services.crawler.models import PagePreview
from app.services.crawler.unified import UnifiedCrawler


def test_preview_scoring_prefers_target_career_roles_over_finance_roles():
    scorer = PageScorer()

    engineering_preview = PagePreview(
        url="https://example.com/careers/senior-software-engineer",
        title="Senior Software Engineer",
        meta_description="Build workflow automation and platform integrations for enterprise customers.",
        h1="Senior Software Engineer",
        headings=["Engineering", "Platform", "Workflow Automation"],
        path_depth=2,
    )
    finance_preview = PagePreview(
        url="https://example.com/careers/financial-controller",
        title="Financial Controller",
        meta_description="Own accounting close, forecasting, and treasury operations.",
        h1="Financial Controller",
        headings=["Finance", "Accounting", "Treasury"],
        path_depth=2,
    )

    engineering_score = scorer._score_preview(engineering_preview)
    finance_score = scorer._score_preview(finance_preview)

    assert engineering_score > finance_score


def test_url_priority_prefers_target_career_roles_over_finance_roles():
    engineering_url = "https://example.com/jobs/senior-product-manager"
    finance_url = "https://example.com/jobs/assistant-controller"

    assert UnifiedCrawler._url_priority_score(engineering_url) > UnifiedCrawler._url_priority_score(finance_url)
