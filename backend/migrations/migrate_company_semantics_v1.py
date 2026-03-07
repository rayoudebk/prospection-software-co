"""
Rename vendor-era schema to company-era semantics.

This migration is idempotent and intentionally uses raw SQL / reflected tables
instead of ORM models because the current application models already expect the
renamed schema.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.evidence_policy import normalize_policy

settings = get_settings()


TABLE_RENAMES = [
    ("vendors", "companies"),
    ("vendor_dossiers", "company_dossiers"),
    ("vendor_mentions", "company_mentions"),
    ("vendor_screenings", "company_screenings"),
    ("vendor_claims", "company_claims"),
    ("vendor_facts", "company_facts"),
    ("workspace_evidence", "source_evidence"),
]

COLUMN_RENAMES = {
    "company_profiles": [("reference_vendor_urls", "reference_company_urls")],
    "company_dossiers": [("vendor_id", "company_id")],
    "company_screenings": [("vendor_id", "company_id")],
    "company_claims": [("vendor_id", "company_id"), ("screening_id", "company_screening_id")],
    "company_facts": [("vendor_id", "company_id")],
    "source_evidence": [("vendor_id", "company_id")],
    "jobs": [("vendor_id", "company_id")],
    "workspace_feedback_events": [("vendor_id", "company_id"), ("screening_id", "company_screening_id")],
    "report_snapshot_items": [("vendor_id", "company_id")],
    "evaluation_sample_results": [("vendor_id", "company_id")],
}


def _table_exists(conn, table_name: str) -> bool:
    return table_name in inspect(conn).get_table_names()


def _column_names(conn, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    return {column["name"] for column in inspect(conn).get_columns(table_name)}


def _rename_table_if_needed(conn, old_name: str, new_name: str) -> bool:
    if not _table_exists(conn, old_name) or _table_exists(conn, new_name):
        return False
    conn.execute(text(f'ALTER TABLE "{old_name}" RENAME TO "{new_name}"'))
    return True


def _rename_column_if_needed(conn, table_name: str, old_name: str, new_name: str) -> bool:
    columns = _column_names(conn, table_name)
    if old_name not in columns or new_name in columns:
        return False
    conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}"'))
    return True


def _drop_column_if_needed(conn, table_name: str, column_name: str) -> bool:
    columns = _column_names(conn, table_name)
    if column_name not in columns:
        return False
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        conn.execute(text(f'ALTER TABLE "{table_name}" DROP COLUMN IF EXISTS "{column_name}"'))
    elif dialect == "sqlite":
        conn.execute(text(f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"'))
    else:
        raise RuntimeError(f"Unsupported dialect for dropping {table_name}.{column_name}: {dialect}")
    return True


def _rename_status_enum_if_needed(conn) -> bool:
    if conn.engine.dialect.name != "postgresql":
        return False
    existing_types = {
        row[0]
        for row in conn.execute(
            text("SELECT typname FROM pg_type WHERE typname IN ('vendorstatus', 'companystatus')")
        ).fetchall()
    }
    if "vendorstatus" not in existing_types or "companystatus" in existing_types:
        return False
    conn.execute(text("ALTER TYPE vendorstatus RENAME TO companystatus"))
    return True


def _normalize_workspace_policies(conn) -> int:
    if not _table_exists(conn, "workspaces"):
        return 0
    metadata = MetaData()
    workspaces = Table("workspaces", metadata, autoload_with=conn)
    updated = 0
    rows = conn.execute(select(workspaces.c.id, workspaces.c.decision_policy_json)).all()
    for row in rows:
        normalized = normalize_policy(row.decision_policy_json or {})
        if row.decision_policy_json != normalized:
            conn.execute(
                workspaces.update()
                .where(workspaces.c.id == row.id)
                .values(decision_policy_json=normalized)
            )
            updated += 1
    return updated


def _normalize_job_payloads(conn) -> int:
    if not _table_exists(conn, "jobs") or "result_json" not in _column_names(conn, "jobs"):
        return 0
    metadata = MetaData()
    jobs = Table("jobs", metadata, autoload_with=conn)
    updated = 0
    rows = conn.execute(select(jobs.c.id, jobs.c.result_json)).all()
    for row in rows:
        payload = row.result_json if isinstance(row.result_json, dict) else None
        if not payload:
            continue
        changed = False
        normalized: dict[str, Any] = dict(payload)
        if "max_vendors" in normalized and "max_companies" not in normalized:
            normalized["max_companies"] = normalized.pop("max_vendors")
            changed = True
        if "vendors_created" in normalized and "companies_created" not in normalized:
            normalized["companies_created"] = normalized.pop("vendors_created")
            changed = True
        if changed:
            conn.execute(
                jobs.update().where(jobs.c.id == row.id).values(result_json=normalized)
            )
            updated += 1
    return updated


def migrate_company_semantics_v1() -> dict[str, Any]:
    engine = create_engine(settings.database_url_sync, echo=True)
    summary: dict[str, Any] = {
        "tables_renamed": [],
        "columns_renamed": [],
        "columns_dropped": [],
        "status_enum_renamed": False,
        "workspace_policies_updated": 0,
        "job_payloads_updated": 0,
    }

    with engine.begin() as conn:
        for old_name, new_name in TABLE_RENAMES:
            if _rename_table_if_needed(conn, old_name, new_name):
                summary["tables_renamed"].append(f"{old_name}->{new_name}")

        for table_name, renames in COLUMN_RENAMES.items():
            for old_name, new_name in renames:
                if _rename_column_if_needed(conn, table_name, old_name, new_name):
                    summary["columns_renamed"].append(f"{table_name}.{old_name}->{new_name}")

        if _drop_column_if_needed(conn, "source_evidence", "brick_ids"):
            summary["columns_dropped"].append("source_evidence.brick_ids")

        summary["status_enum_renamed"] = _rename_status_enum_if_needed(conn)
        summary["workspace_policies_updated"] = _normalize_workspace_policies(conn)
        summary["job_payloads_updated"] = _normalize_job_payloads(conn)

    return summary


if __name__ == "__main__":
    print("Starting company semantics rename migration (v1)...")
    result = migrate_company_semantics_v1()
    for key, value in result.items():
        print(f"- {key}: {value}")
    print("✅ Migration complete")
