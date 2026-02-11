"""
Migration script for company profile reference evidence links.

Adds and backfills:
- company_profiles.reference_evidence_urls (JSON)
"""
import json
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


def migrate_company_profile_reference_evidence_v1() -> None:
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _add_column_if_missing(
            conn,
            "company_profiles",
            "reference_evidence_urls JSON",
            "reference_evidence_urls",
        )
        conn.execute(
            text(
                """
                UPDATE company_profiles
                SET reference_evidence_urls = COALESCE(reference_evidence_urls, :empty_list)
                """
            ),
            {"empty_list": json.dumps([])},
        )


if __name__ == "__main__":
    print("Starting company profile reference evidence migration (v1)...")
    migrate_company_profile_reference_evidence_v1()
    print("✅ Migration complete")
