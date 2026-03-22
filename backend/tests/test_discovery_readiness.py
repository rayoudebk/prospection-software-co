from app.services import discovery_readiness


def test_assert_critical_runtime_ready_raises_for_schema_or_redis(monkeypatch):
    monkeypatch.setattr(
        discovery_readiness,
        "compute_runtime_discovery_readiness",
        lambda **kwargs: {
            "execution_mode": "live",
            "db_schema_ok": False,
            "redis_available": False,
            "worker_available": False,
            "retrieval_provider_available": False,
            "model_available": False,
            "reasons_blocked": ["db_schema_invalid", "redis_unavailable"],
        },
    )

    try:
        discovery_readiness.assert_critical_runtime_ready(database_url_sync="sqlite:///tmp/test.sqlite3")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "db_schema_invalid" in str(exc)
        assert "redis_unavailable" in str(exc)


def test_build_workspace_discovery_readiness_adds_expansion_block(monkeypatch):
    monkeypatch.setattr(
        discovery_readiness,
        "compute_runtime_discovery_readiness",
        lambda **kwargs: {
            "runnable": True,
            "execution_mode": "live",
            "db_schema_ok": True,
            "redis_available": True,
            "worker_available": True,
            "retrieval_provider_available": True,
            "model_available": True,
            "reasons_blocked": [],
            "available_retrieval_providers": ["exa"],
            "available_model_providers": ["gemini"],
            "schema_missing_columns": [],
        },
    )

    payload = discovery_readiness.build_workspace_discovery_readiness(expansion_confirmed=False)

    assert payload["runnable"] is False
    assert payload["execution_mode"] == "live"
    assert payload["expansion_confirmed"] is False
    assert payload["reasons_blocked"] == ["expansion_not_confirmed"]


def test_compute_runtime_discovery_readiness_fixture_mode_ignores_missing_live_keys(monkeypatch):
    monkeypatch.setattr(
        discovery_readiness,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "database_url_sync": "sqlite:///tmp/test.sqlite3",
                "redis_url": "redis://localhost:6379/0",
                "discovery_execution_mode": "fixture",
                "discovery_retrieval_provider_order": "exa,brave",
                "gemini_api_key": "",
                "openai_api_key": "",
                "anthropic_api_key": "",
                "exa_api_key": "",
                "brave_api_key": "",
                "tavily_api_key": "",
                "serper_api_key": "",
                "serpapi_api_key": "",
                "firecrawl_api_key": "",
                "jina_api_key": "",
                "stage_model_routes": staticmethod(lambda stage: [("gemini", "gemini-2.0-flash")]),
            },
        )(),
    )
    monkeypatch.setattr(discovery_readiness, "_schema_missing_columns", lambda **kwargs: [])
    monkeypatch.setattr(discovery_readiness, "_redis_available", lambda: True)
    monkeypatch.setattr(discovery_readiness, "_worker_available", lambda: True)

    payload = discovery_readiness.compute_runtime_discovery_readiness()

    assert payload["execution_mode"] == "fixture"
    assert payload["runnable"] is True
    assert payload["reasons_blocked"] == []


def test_compute_runtime_discovery_readiness_allows_france_registry_only(monkeypatch):
    monkeypatch.setattr(
        discovery_readiness,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "database_url_sync": "sqlite:///tmp/test.sqlite3",
                "redis_url": "redis://localhost:6379/0",
                "discovery_execution_mode": "live",
                "discovery_retrieval_provider_order": "exa,brave",
                "gemini_api_key": "gm-key",
                "openai_api_key": "",
                "anthropic_api_key": "",
                "exa_api_key": "",
                "brave_api_key": "",
                "tavily_api_key": "",
                "serper_api_key": "",
                "serpapi_api_key": "",
                "firecrawl_api_key": "",
                "jina_api_key": "",
                "inpi_token": "test-token",
                "inpi_username": "",
                "inpi_password": "",
                "stage_model_routes": staticmethod(lambda stage: [("gemini", "gemini-2.0-flash")]),
            },
        )(),
    )
    monkeypatch.setattr(discovery_readiness, "_schema_missing_columns", lambda **kwargs: [])
    monkeypatch.setattr(discovery_readiness, "_redis_available", lambda: True)
    monkeypatch.setattr(discovery_readiness, "_worker_available", lambda: True)

    payload = discovery_readiness.compute_runtime_discovery_readiness(
        allow_registry_only_retrieval=True,
    )

    assert payload["runnable"] is True
    assert payload["retrieval_provider_available"] is True
    assert payload["registry_retrieval_available"] is True
    assert "inpi_registry" in payload["available_retrieval_providers"]


def test_build_workspace_discovery_readiness_detects_france_registry_request(monkeypatch):
    monkeypatch.setattr(
        discovery_readiness,
        "compute_runtime_discovery_readiness",
        lambda **kwargs: {
            "runnable": True,
            "execution_mode": "live",
            "db_schema_ok": True,
            "redis_available": True,
            "worker_available": True,
            "retrieval_provider_available": True,
            "registry_retrieval_available": kwargs.get("allow_registry_only_retrieval"),
            "model_available": True,
            "reasons_blocked": [],
            "available_retrieval_providers": ["inpi_registry"] if kwargs.get("allow_registry_only_retrieval") else [],
            "available_model_providers": ["gemini"],
            "schema_missing_columns": [],
        },
    )

    payload = discovery_readiness.build_workspace_discovery_readiness(
        expansion_confirmed=True,
        geo_scope={"include_countries": ["FR"]},
    )

    assert payload["runnable"] is True
    assert payload["registry_retrieval_available"] is True


def test_worker_available_requires_namespaced_discovery_queues(monkeypatch):
    class _Inspector:
        @staticmethod
        def active_queues():
            return {
                "worker-a": [
                    {"name": discovery_readiness.DISCOVERY_QUEUE_NAMES["search"]},
                    {"name": discovery_readiness.DISCOVERY_QUEUE_NAMES["score"]},
                ]
            }

    monkeypatch.setattr(
        discovery_readiness.celery_app.control,
        "inspect",
        lambda timeout=1.0: _Inspector(),
    )

    assert discovery_readiness._worker_available() is True


def test_worker_available_rejects_unscoped_foreign_queues(monkeypatch):
    class _Inspector:
        @staticmethod
        def active_queues():
            return {
                "worker-a": [
                    {"name": "discovery.search"},
                    {"name": "discovery.score"},
                ]
            }

    monkeypatch.setattr(
        discovery_readiness.celery_app.control,
        "inspect",
        lambda timeout=1.0: _Inspector(),
    )

    expected_search = discovery_readiness.DISCOVERY_QUEUE_NAMES["search"]
    if expected_search == "discovery.search":
        assert discovery_readiness._worker_available() is True
    else:
        assert discovery_readiness._worker_available() is False
