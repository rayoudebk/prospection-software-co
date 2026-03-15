"""
Remove legacy taxonomy schema after context-pack/scope-review rollout.

This migration:
1. Ensures company-context tables exist.
2. Backfills company-context packs for older workspaces.
3. Normalizes workspace policy JSON to use `scope_review` instead of `brick_model`.
4. Drops `brick_mappings`, `brick_taxonomies`, and `vendors.tags_vertical`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.workspace import Workspace
from app.services.evidence_policy import DEFAULT_EVIDENCE_POLICY, normalize_policy
from migrations.migrate_company_context_backfill_v1 import (
    _ensure_company_context_tables,
    backfill_company_context,
)

settings = get_settings()


def _normalize_workspace_policies(session) -> int:
    updated = 0
    for workspace in session.query(Workspace).order_by(Workspace.id.asc()).all():
        normalized = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)
        gate_requirements = normalized.get("gate_requirements") or {}
        if not isinstance(gate_requirements, dict):
            gate_requirements = {}
            normalized["gate_requirements"] = gate_requirements
        gate_requirements.pop("brick_model", None)
        gate_requirements.pop("search_lanes", None)
        gate_requirements.setdefault(
            "scope_review",
            DEFAULT_EVIDENCE_POLICY["gate_requirements"]["scope_review"],
        )
        if workspace.decision_policy_json != normalized:
            workspace.decision_policy_json = normalized
            updated += 1
    if updated:
        session.commit()
    return updated


def _drop_legacy_schema(conn) -> dict[str, bool]:
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    result = {
        "dropped_brick_mappings": False,
        "dropped_brick_taxonomies": False,
        "dropped_tags_vertical": False,
    }

    if "brick_mappings" in tables:
        conn.execute(text("DROP TABLE IF EXISTS brick_mappings"))
        result["dropped_brick_mappings"] = True

    if "brick_taxonomies" in tables:
        conn.execute(text("DROP TABLE IF EXISTS brick_taxonomies"))
        result["dropped_brick_taxonomies"] = True

    refreshed = inspect(conn)
    vendor_columns = {col["name"] for col in refreshed.get_columns("vendors")} if "vendors" in refreshed.get_table_names() else set()
    if "tags_vertical" in vendor_columns:
        dialect = conn.engine.dialect.name
        if dialect == "postgresql":
            conn.execute(text("ALTER TABLE vendors DROP COLUMN IF EXISTS tags_vertical"))
        elif dialect == "sqlite":
            conn.execute(text("ALTER TABLE vendors DROP COLUMN tags_vertical"))
        else:
            raise RuntimeError(f"Unsupported dialect for dropping vendors.tags_vertical: {dialect}")
        result["dropped_tags_vertical"] = True

    return result


def migrate_remove_legacy_taxonomy_v1() -> dict[str, int | bool]:
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _ensure_company_context_tables(conn)

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with SessionLocal() as session:
        company_context_summary = backfill_company_context(session)
        policies_updated = _normalize_workspace_policies(session)

    with engine.begin() as conn:
        drop_summary = _drop_legacy_schema(conn)

    return {
        **company_context_summary,
        "workspace_policies_updated": policies_updated,
        **drop_summary,
    }


if __name__ == "__main__":
    print("Starting legacy taxonomy cleanup migration (v1)...")
    summary = migrate_remove_legacy_taxonomy_v1()
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print("✅ Migration complete")
