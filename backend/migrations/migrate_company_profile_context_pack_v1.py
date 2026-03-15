"""
Migration script for company profile context-pack fields.

Adds and backfills:
- company_profiles.reference_company_urls (JSON)
- company_profiles.reference_summaries (JSON)
- company_profiles.context_pack_markdown (TEXT)
- company_profiles.context_pack_json (JSON)
- company_profiles.context_pack_generated_at (TIMESTAMP)
- company_profiles.product_pages_found (INTEGER)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings

settings = get_settings()


def _column_names(conn, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(conn).get_columns(table_name)}


def _rename_column_if_needed(conn, table_name: str, old_name: str, new_name: str) -> None:
    cols = _column_names(conn, table_name)
    if new_name in cols or old_name not in cols:
        return
    conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}"'))


def _add_column_if_missing(conn, table_name: str, column_sql: str, column_name: str) -> None:
    cols = _column_names(conn, table_name)
    if column_name in cols:
        print(f"Column already exists: {table_name}.{column_name}")
        return
    print(f"Adding column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))


def migrate_company_profile_context_pack_v1(database_url: str | None = None) -> None:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _rename_column_if_needed(conn, "company_profiles", "reference_vendor_urls", "reference_company_urls")

        _add_column_if_missing(
            conn,
            "company_profiles",
            "reference_company_urls JSON",
            "reference_company_urls",
        )
        _add_column_if_missing(
            conn,
            "company_profiles",
            "reference_summaries JSON",
            "reference_summaries",
        )
        _add_column_if_missing(
            conn,
            "company_profiles",
            "context_pack_markdown TEXT",
            "context_pack_markdown",
        )
        _add_column_if_missing(
            conn,
            "company_profiles",
            "context_pack_json JSON",
            "context_pack_json",
        )
        _add_column_if_missing(
            conn,
            "company_profiles",
            "context_pack_generated_at TIMESTAMP",
            "context_pack_generated_at",
        )
        _add_column_if_missing(
            conn,
            "company_profiles",
            "product_pages_found INTEGER",
            "product_pages_found",
        )

        conn.execute(
            text(
                """
                UPDATE company_profiles
                SET
                  reference_company_urls = COALESCE(reference_company_urls, :empty_list),
                  reference_summaries = COALESCE(reference_summaries, :empty_object),
                  context_pack_json = COALESCE(context_pack_json, :empty_object),
                  product_pages_found = COALESCE(product_pages_found, 0)
                """
            ),
            {
                "empty_list": json.dumps([]),
                "empty_object": json.dumps({}),
            },
        )


if __name__ == "__main__":
    print("Starting company profile context-pack migration (v1)...")
    migrate_company_profile_context_pack_v1()
    print("✅ Migration complete")
