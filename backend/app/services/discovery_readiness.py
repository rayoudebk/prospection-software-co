from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Connection

import redis

from app.config import get_settings
from app.workers.celery_app import DISCOVERY_QUEUE_NAMES, celery_app

REQUIRED_DISCOVERY_SCHEMA_COLUMNS: Dict[str, tuple[str, ...]] = {
    "jobs": ("company_id",),
    "company_profiles": ("buyer_company_url",),
    "company_context_packs": ("expansion_brief_json",),
    "company_screenings": ("ranking_eligible",),
}


def normalize_sync_database_url(database_url: str) -> str:
    normalized = str(database_url or "").strip()
    return normalized.replace("+asyncpg", "").replace("+aiosqlite", "")


def _provider_key_available(provider: str) -> bool:
    settings = get_settings()
    provider_name = str(provider or "").strip().lower()
    if provider_name == "gemini":
        return bool(settings.gemini_api_key)
    if provider_name == "openai":
        return bool(settings.openai_api_key)
    if provider_name == "anthropic":
        return bool(settings.anthropic_api_key)
    if provider_name == "exa":
        return bool(settings.exa_api_key)
    if provider_name == "brave":
        return bool(settings.brave_api_key)
    if provider_name == "tavily":
        return bool(settings.tavily_api_key)
    if provider_name == "serper":
        return bool(settings.serper_api_key)
    if provider_name == "serpapi":
        return bool(settings.serpapi_api_key)
    if provider_name == "firecrawl":
        return bool(settings.firecrawl_api_key)
    if provider_name == "jina":
        return bool(settings.jina_api_key)
    return False


def _available_retrieval_providers() -> List[str]:
    settings = get_settings()
    configured = [
        str(provider or "").strip().lower()
        for provider in str(settings.discovery_retrieval_provider_order or "").split(",")
        if str(provider or "").strip()
    ]
    return [provider for provider in configured if _provider_key_available(provider)]


def _available_model_providers() -> List[str]:
    settings = get_settings()
    providers: List[str] = []
    for provider, _model in settings.stage_model_routes("discovery_retrieval"):
        if provider not in providers and _provider_key_available(provider):
            providers.append(provider)
    return providers


def _schema_missing_columns(
    *,
    database_url_sync: Optional[str] = None,
    connection: Optional[Connection] = None,
) -> List[str]:
    missing: List[str] = []

    def _inspect_connection(conn: Connection) -> List[str]:
        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())
        unresolved: List[str] = []
        for table_name, required_columns in REQUIRED_DISCOVERY_SCHEMA_COLUMNS.items():
            if table_name not in table_names:
                unresolved.extend([f"{table_name}.{column}" for column in required_columns])
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name in required_columns:
                if column_name not in existing_columns:
                    unresolved.append(f"{table_name}.{column_name}")
        return unresolved

    if connection is not None:
        return _inspect_connection(connection)

    settings = get_settings()
    resolved_url = normalize_sync_database_url(database_url_sync or settings.database_url_sync)
    if not resolved_url:
        return [f"{table}.{column}" for table, columns in REQUIRED_DISCOVERY_SCHEMA_COLUMNS.items() for column in columns]
    engine = create_engine(resolved_url)
    try:
        with engine.connect() as conn:
            missing = _inspect_connection(conn)
    finally:
        engine.dispose()
    return missing


def _redis_available() -> bool:
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        return bool(client.ping())
    except Exception:
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


def _worker_available() -> bool:
    try:
        inspector = celery_app.control.inspect(timeout=1.0)
        if not inspector:
            return False
        active_queues = inspector.active_queues() or {}
        expected = {
            DISCOVERY_QUEUE_NAMES["search"],
            DISCOVERY_QUEUE_NAMES["score"],
        }
        for queues in active_queues.values():
            names = {
                str(item.get("name") or "").strip()
                for item in (queues or [])
                if isinstance(item, dict)
            }
            if expected.issubset(names):
                return True
        return False
    except Exception:
        return False


def compute_runtime_discovery_readiness(
    *,
    database_url_sync: Optional[str] = None,
    connection: Optional[Connection] = None,
    check_worker: bool = True,
) -> Dict[str, Any]:
    settings = get_settings()
    execution_mode = str(getattr(settings, "discovery_execution_mode", "live") or "live").strip().lower() or "live"
    if execution_mode not in {"live", "fixture"}:
        execution_mode = "live"
    schema_missing_columns: List[str] = []
    db_schema_ok = False
    try:
        schema_missing_columns = _schema_missing_columns(
            database_url_sync=database_url_sync,
            connection=connection,
        )
        db_schema_ok = not schema_missing_columns
    except Exception:
        schema_missing_columns = ["schema_introspection_failed"]
        db_schema_ok = False

    redis_available = _redis_available()
    available_retrieval_providers = _available_retrieval_providers()
    available_model_providers = _available_model_providers()
    worker_available = _worker_available() if check_worker else True

    reasons_blocked: List[str] = []
    if not db_schema_ok:
        reasons_blocked.append("db_schema_invalid")
    if not redis_available:
        reasons_blocked.append("redis_unavailable")
    if check_worker and not worker_available:
        reasons_blocked.append("worker_unavailable")
    if execution_mode == "live" and not available_retrieval_providers:
        reasons_blocked.append("retrieval_provider_unavailable")
    if execution_mode == "live" and not available_model_providers:
        reasons_blocked.append("model_provider_unavailable")

    return {
        "runnable": len(reasons_blocked) == 0,
        "execution_mode": execution_mode,
        "db_schema_ok": db_schema_ok,
        "redis_available": redis_available,
        "worker_available": worker_available,
        "retrieval_provider_available": bool(available_retrieval_providers),
        "model_available": bool(available_model_providers),
        "reasons_blocked": reasons_blocked,
        "available_retrieval_providers": available_retrieval_providers,
        "available_model_providers": available_model_providers,
        "schema_missing_columns": schema_missing_columns,
    }


def build_workspace_discovery_readiness(
    *,
    expansion_confirmed: bool,
    database_url_sync: Optional[str] = None,
    connection: Optional[Connection] = None,
    check_worker: bool = True,
) -> Dict[str, Any]:
    payload = compute_runtime_discovery_readiness(
        database_url_sync=database_url_sync,
        connection=connection,
        check_worker=check_worker,
    )
    reasons_blocked = list(payload.get("reasons_blocked") or [])
    if not expansion_confirmed:
        reasons_blocked.append("expansion_not_confirmed")
    payload["expansion_confirmed"] = bool(expansion_confirmed)
    payload["reasons_blocked"] = reasons_blocked
    payload["runnable"] = len(reasons_blocked) == 0
    return payload


def assert_critical_runtime_ready(*, database_url_sync: Optional[str] = None) -> Dict[str, Any]:
    payload = compute_runtime_discovery_readiness(
        database_url_sync=database_url_sync,
        check_worker=False,
    )
    critical_failures: List[str] = []
    if not payload.get("db_schema_ok"):
        critical_failures.append("db_schema_invalid")
    if not payload.get("redis_available"):
        critical_failures.append("redis_unavailable")
    if critical_failures:
        raise RuntimeError(
            "Discovery runtime startup check failed: "
            + ", ".join(critical_failures)
        )
    return payload
