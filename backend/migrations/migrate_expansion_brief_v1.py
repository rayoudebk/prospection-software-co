"""
Migration script for expansion-brief artifacts.

Adds:
- company_context_packs.expansion_brief_json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings

settings = get_settings()


def _resolve_company_context_table(conn) -> str | None:
    tables = set(inspect(conn).get_table_names())
    if "company_context_packs" in tables:
        return "company_context_packs"
    if "buyer_thesis_packs" in tables:
        return "buyer_thesis_packs"
    return None


def _add_column_if_missing(conn, table_name: str, column_sql: str, column_name: str) -> None:
    cols = {c["name"] for c in inspect(conn).get_columns(table_name)}
    if column_name in cols:
        print(f"Column already exists: {table_name}.{column_name}")
        return
    print(f"Adding column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))


def migrate_expansion_brief_v1(database_url: str | None = None) -> None:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        table_name = _resolve_company_context_table(conn)
        if not table_name:
            return
        _add_column_if_missing(
            conn,
            table_name,
            "expansion_brief_json JSON",
            "expansion_brief_json",
        )
        conn.execute(
            text(
                f"""
                UPDATE {table_name}
                SET expansion_brief_json = COALESCE(expansion_brief_json, :empty_object)
                """
            ),
            {
                "empty_object": json.dumps({}),
            },
        )


if __name__ == "__main__":
    print("Starting expansion brief migration (v1)...")
    migrate_expansion_brief_v1()
    print("✅ Migration complete")
