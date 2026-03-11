"""
Migration script for explicit company-profile brief/summary fields.

Adds and backfills:
- company_profiles.manual_brief_text (TEXT)
- company_profiles.generated_context_summary (TEXT)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings

settings = get_settings()


def _add_column_if_missing(conn, table_name: str, column_sql: str, column_name: str) -> None:
    cols = {c["name"] for c in inspect(conn).get_columns(table_name)}
    if column_name in cols:
        print(f"Column already exists: {table_name}.{column_name}")
        return
    print(f"Adding column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))


def _backfill_context_fields(conn) -> None:
    conn.execute(
        text(
            """
            UPDATE company_profiles
            SET
              manual_brief_text = CASE
                WHEN manual_brief_text IS NULL
                  AND buyer_context_summary IS NOT NULL
                  AND TRIM(buyer_context_summary) <> ''
                  AND (buyer_company_url IS NULL OR TRIM(buyer_company_url) = '')
                THEN buyer_context_summary
                ELSE manual_brief_text
              END,
              generated_context_summary = CASE
                WHEN generated_context_summary IS NULL
                  AND buyer_context_summary IS NOT NULL
                  AND TRIM(buyer_context_summary) <> ''
                  AND buyer_company_url IS NOT NULL
                  AND TRIM(buyer_company_url) <> ''
                THEN buyer_context_summary
                ELSE generated_context_summary
              END
            """
        )
    )


def migrate_company_profile_context_split_v1(database_url: str | None = None) -> None:
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


if __name__ == "__main__":
    print("Starting company profile context split migration (v1)...")
    migrate_company_profile_context_split_v1()
    print("✅ Migration complete")
