"""
Drop legacy company-context bridge columns that are no longer canonical.

This removes old thesis-era storage from company_context_packs now that
sourcing_brief_json and graph-backed payloads are canonical.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

settings = get_settings()


def _column_names(conn, table_name: str) -> set[str]:
    inspector = inspect(conn)
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _drop_column_if_exists(conn, table_name: str, column_name: str) -> bool:
    if column_name not in _column_names(conn, table_name):
        return False
    conn.execute(text(f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"'))
    return True


def migrate_remove_company_context_bridge_fields_v1(database_url: str | None = None) -> dict[str, Any]:
    engine = create_engine(database_url or settings.database_url_sync, echo=False)
    dropped: list[str] = []
    with engine.begin() as conn:
        for column_name in ("summary", "claims_json", "source_pills_json", "open_questions_json"):
            if _drop_column_if_exists(conn, "company_context_packs", column_name):
                dropped.append(column_name)
    return {"dropped_columns": dropped}


if __name__ == "__main__":
    result = migrate_remove_company_context_bridge_fields_v1()
    print(result)
