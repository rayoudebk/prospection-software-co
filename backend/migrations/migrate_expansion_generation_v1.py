"""Add explicit expansion generation status fields to company_context_packs."""
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


def migrate_expansion_generation_v1(database_url: str | None = None) -> None:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        tables = set(inspect(conn).get_table_names())
        if "company_context_packs" not in tables:
            return
        _add_column_if_missing(
            conn,
            "company_context_packs",
            "expansion_status VARCHAR(32) DEFAULT 'not_generated' NOT NULL",
            "expansion_status",
        )
        _add_column_if_missing(
            conn,
            "company_context_packs",
            "expansion_error TEXT",
            "expansion_error",
        )
        _add_column_if_missing(
            conn,
            "company_context_packs",
            "expansion_generated_at TIMESTAMP",
            "expansion_generated_at",
        )
        conn.execute(
            text(
                """
                UPDATE company_context_packs
                SET expansion_status = COALESCE(NULLIF(expansion_status, ''), 'not_generated')
                """
            )
        )


if __name__ == "__main__":
    print("Starting expansion generation migration (v1)...")
    migrate_expansion_generation_v1()
    print("✅ Migration complete")
