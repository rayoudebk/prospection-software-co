from pathlib import Path

from sqlalchemy import create_engine

from app import startup_migrations
from app.models.base import Base
import app.models  # noqa: F401


def test_run_startup_migrations_uses_sync_database_url(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _Settings:
        database_url_sync = "postgresql://example/test"

    monkeypatch.setattr(startup_migrations, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        startup_migrations,
        "migrate_workspace_policy_v1",
        lambda *, database_url: calls.append(("workspace_policy", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_company_profile_reference_evidence_v1",
        lambda *, database_url: calls.append(("reference_evidence", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_company_profile_context_pack_v1",
        lambda *, database_url: calls.append(("context_pack", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_company_profile_context_split_v1",
        lambda *, database_url: calls.append(("context_split", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_remove_legacy_buyer_context_summary_v1",
        lambda *, database_url: calls.append(("remove_legacy_summary", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_market_map_brief_v1",
        lambda *, database_url: calls.append(("market_map_brief", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_expansion_brief_v1",
        lambda *, database_url: calls.append(("expansion_brief", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_company_context_graph_v1",
        lambda *, database_url: calls.append(("company_context_graph", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_company_context_storage_v1",
        lambda *, database_url: calls.append(("company_context_storage", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_remove_company_context_bridge_fields_v1",
        lambda *, database_url: calls.append(("remove_company_context_bridges", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_remove_company_profile_brief_fields_v1",
        lambda *, database_url: calls.append(("remove_company_profile_brief_fields", database_url)),
    )
    monkeypatch.setattr(
        startup_migrations,
        "migrate_remove_search_lanes_v1",
        lambda *, database_url: calls.append(("remove_search_lanes", database_url)),
    )

    startup_migrations.run_startup_migrations()

    assert calls == [
        ("workspace_policy", "postgresql://example/test"),
        ("reference_evidence", "postgresql://example/test"),
        ("context_pack", "postgresql://example/test"),
        ("context_split", "postgresql://example/test"),
        ("remove_legacy_summary", "postgresql://example/test"),
        ("market_map_brief", "postgresql://example/test"),
        ("expansion_brief", "postgresql://example/test"),
        ("company_context_graph", "postgresql://example/test"),
        ("company_context_storage", "postgresql://example/test"),
        ("remove_company_context_bridges", "postgresql://example/test"),
        ("remove_company_profile_brief_fields", "postgresql://example/test"),
        ("remove_search_lanes", "postgresql://example/test"),
    ]


def test_run_startup_migrations_accepts_current_schema_sqlite(tmp_path: Path):
    db_path = tmp_path / "startup-migrations.sqlite3"
    database_url = f"sqlite:///{db_path}"
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)

    startup_migrations.run_startup_migrations(database_url=database_url)
