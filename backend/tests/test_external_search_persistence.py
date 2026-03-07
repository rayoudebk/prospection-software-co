from app.models.external_search import ExternalSearchRun, ExternalSearchResult


def test_external_search_run_columns_present():
    cols = ExternalSearchRun.__table__.columns
    for name in [
        "workspace_id",
        "job_id",
        "run_id",
        "created_at",
        "provider_order",
        "caps_json",
        "query_plan_json",
        "query_plan_hash",
    ]:
        assert name in cols


def test_external_search_result_columns_present():
    cols = ExternalSearchResult.__table__.columns
    for name in [
        "run_id",
        "provider",
        "query_id",
        "query_type",
        "query_text",
        "rank",
        "url",
        "url_fingerprint",
        "domain_fingerprint",
        "title",
        "snippet",
        "retrieved_at",
    ]:
        assert name in cols
