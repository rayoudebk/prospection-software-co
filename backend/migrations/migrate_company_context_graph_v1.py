"""
Migration script for phase-1 company-context graph metadata.

Adds bridge metadata on buyer_thesis_packs while Neo4j becomes the canonical
store for phase-1 company context.
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


def _add_index_if_missing(conn, table_name: str, index_name: str, index_sql: str) -> None:
    indexes = {item["name"] for item in inspect(conn).get_indexes(table_name)}
    if index_name in indexes:
        print(f"Index already exists: {index_name}")
        return
    print(f"Adding index: {index_name}")
    conn.execute(text(index_sql))


def migrate_company_context_graph_v1(database_url: str | None = None) -> None:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        table_name = _resolve_company_context_table(conn)
        if not table_name:
            return
        _add_column_if_missing(
            conn,
            table_name,
            "company_context_graph_ref VARCHAR(255)",
            "company_context_graph_ref",
        )
        _add_column_if_missing(
            conn,
            table_name,
            "company_context_graph_cache_json JSON",
            "company_context_graph_cache_json",
        )
        _add_column_if_missing(
            conn,
            table_name,
            "graph_sync_status VARCHAR(32) DEFAULT 'not_synced' NOT NULL",
            "graph_sync_status",
        )
        _add_column_if_missing(
            conn,
            table_name,
            "graph_sync_error TEXT",
            "graph_sync_error",
        )
        _add_column_if_missing(
            conn,
            table_name,
            "graph_stats_json JSON",
            "graph_stats_json",
        )
        _add_column_if_missing(
            conn,
            table_name,
            "graph_synced_at TIMESTAMP",
            "graph_synced_at",
        )
        index_name = (
            "ix_company_context_packs_company_context_graph_ref"
            if table_name == "company_context_packs"
            else "ix_buyer_thesis_packs_company_context_graph_ref"
        )
        _add_index_if_missing(
            conn,
            table_name,
            index_name,
            f"CREATE INDEX {index_name} ON {table_name} (company_context_graph_ref)",
        )
        conn.execute(
            text(
                f"""
                UPDATE {table_name}
                SET company_context_graph_cache_json = COALESCE(company_context_graph_cache_json, :empty_object),
                    graph_stats_json = COALESCE(graph_stats_json, :empty_object),
                    graph_sync_status = COALESCE(NULLIF(graph_sync_status, ''), 'not_synced')
                """
            ),
            {"empty_object": json.dumps({})},
        )


if __name__ == "__main__":
    print("Starting company context graph migration (v1)...")
    migrate_company_context_graph_v1()
    print("✅ Migration complete")
