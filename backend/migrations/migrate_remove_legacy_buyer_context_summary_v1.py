"""
Migration script to remove legacy buyer_context_summary after context split rollout.

This migration is intentionally safe to run on its own:
1. Ensures the explicit replacement columns exist
2. Backfills them from buyer_context_summary when needed
3. Drops company_profiles.buyer_context_summary
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings
from migrations.migrate_company_profile_context_split_v1 import (
    _add_column_if_missing,
    _backfill_context_fields,
)

settings = get_settings()


def _column_names(conn, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(conn).get_columns(table_name)}


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


def migrate_remove_legacy_buyer_context_summary_v1(database_url: str | None = None) -> None:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _add_column_if_missing(
            conn,
            "company_profiles",
            "manual_brief_text TEXT",
            "manual_brief_text",
        )
        _add_column_if_missing(
            conn,
            "company_profiles",
            "generated_context_summary TEXT",
            "generated_context_summary",
        )
        _backfill_context_fields(conn)
        removed = _drop_column_if_needed(conn, "company_profiles", "buyer_context_summary")
        if removed:
            print("Dropped column: company_profiles.buyer_context_summary")
        else:
            print("Column already removed: company_profiles.buyer_context_summary")


if __name__ == "__main__":
    print("Starting legacy buyer_context_summary removal migration (v1)...")
    migrate_remove_legacy_buyer_context_summary_v1()
    print("✅ Migration complete")
