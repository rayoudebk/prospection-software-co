"""Drop legacy company-profile brief cache fields now that company-context is canonical."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings

settings = get_settings()


def _column_names(conn, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(conn).get_columns(table_name)}


def _drop_column_if_present(conn, table_name: str, column_name: str) -> bool:
    if column_name not in _column_names(conn, table_name):
        return False
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        conn.execute(text(f'ALTER TABLE "{table_name}" DROP COLUMN IF EXISTS "{column_name}"'))
    elif dialect == "sqlite":
        conn.execute(text(f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"'))
    else:
        raise RuntimeError(f"Unsupported dialect for dropping {table_name}.{column_name}: {dialect}")
    return True


def migrate_remove_company_profile_brief_fields_v1(database_url: str | None = None) -> dict[str, list[str]]:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    removed: list[str] = []
    with engine.begin() as conn:
        for column_name in ("manual_brief_text", "generated_context_summary"):
            if _drop_column_if_present(conn, "company_profiles", column_name):
                removed.append(column_name)
    return {"removed_columns": removed}


if __name__ == "__main__":
    print("Starting company profile brief field removal (v1)...")
    summary = migrate_remove_company_profile_brief_fields_v1()
    print(summary)
    print("✅ Migration complete")
