"""
Rename remaining thesis/reference-era storage names to company-context semantics.

This migration is idempotent and safe to run on SQLite and Postgres.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

settings = get_settings()


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


def _drop_index_if_exists(conn, index_name: str) -> bool:
    existing = {index["name"] for index in inspect(conn).get_indexes("company_context_packs")} if _table_exists(conn, "company_context_packs") else set()
    if index_name not in existing:
        return False
    conn.execute(text(f'DROP INDEX "{index_name}"'))
    return True


def _create_index_if_missing(conn, index_name: str, table_name: str, column_name: str) -> bool:
    existing = {index["name"] for index in inspect(conn).get_indexes(table_name)} if _table_exists(conn, table_name) else set()
    if index_name in existing:
        return False
    conn.execute(text(f'CREATE INDEX "{index_name}" ON "{table_name}" ("{column_name}")'))
    return True


def migrate_company_context_storage_v1(database_url: str | None = None) -> dict[str, Any]:
    engine = create_engine(database_url or settings.database_url_sync, echo=False)
    summary: dict[str, Any] = {
        "tables_renamed": [],
        "columns_renamed": [],
        "indexes_dropped": [],
        "indexes_created": [],
    }

    with engine.begin() as conn:
        if _rename_table_if_needed(conn, "buyer_thesis_packs", "company_context_packs"):
            summary["tables_renamed"].append("buyer_thesis_packs->company_context_packs")

        for old_name, new_name in [
            ("reference_company_urls", "comparator_seed_urls"),
            ("reference_evidence_urls", "supporting_evidence_urls"),
            ("reference_summaries", "comparator_seed_summaries"),
            ("market_map_brief_json", "sourcing_brief_json"),
        ]:
            target_table = "company_context_packs" if old_name == "market_map_brief_json" else "company_profiles"
            if _rename_column_if_needed(conn, target_table, old_name, new_name):
                summary["columns_renamed"].append(f"{target_table}.{old_name}->{new_name}")

        if _drop_index_if_exists(conn, "ix_buyer_thesis_packs_company_context_graph_ref"):
            summary["indexes_dropped"].append("ix_buyer_thesis_packs_company_context_graph_ref")
        if _create_index_if_missing(
            conn,
            "ix_company_context_packs_company_context_graph_ref",
            "company_context_packs",
            "company_context_graph_ref",
        ):
            summary["indexes_created"].append("ix_company_context_packs_company_context_graph_ref")

    return summary


if __name__ == "__main__":
    result = migrate_company_context_storage_v1()
    for key, value in result.items():
        print(f"- {key}: {value}")
