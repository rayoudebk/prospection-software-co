"""
Migration script for market-map brief artifacts.

Adds:
- buyer_thesis_packs.market_map_brief_json
- buyer_thesis_packs.taxonomy_nodes_json
- buyer_thesis_packs.taxonomy_edges_json
- buyer_thesis_packs.lens_seeds_json
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


def migrate_market_map_brief_v1(database_url: str | None = None) -> None:
    engine = create_engine(database_url or settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _add_column_if_missing(
            conn,
            "buyer_thesis_packs",
            "market_map_brief_json JSON",
            "market_map_brief_json",
        )
        _add_column_if_missing(
            conn,
            "buyer_thesis_packs",
            "taxonomy_nodes_json JSON",
            "taxonomy_nodes_json",
        )
        _add_column_if_missing(
            conn,
            "buyer_thesis_packs",
            "taxonomy_edges_json JSON",
            "taxonomy_edges_json",
        )
        _add_column_if_missing(
            conn,
            "buyer_thesis_packs",
            "lens_seeds_json JSON",
            "lens_seeds_json",
        )
        conn.execute(
            text(
                """
                UPDATE buyer_thesis_packs
                SET market_map_brief_json = COALESCE(market_map_brief_json, :empty_object),
                    taxonomy_nodes_json = COALESCE(taxonomy_nodes_json, :empty_list),
                    taxonomy_edges_json = COALESCE(taxonomy_edges_json, :empty_list),
                    lens_seeds_json = COALESCE(lens_seeds_json, :empty_list)
                """
            ),
            {
                "empty_object": json.dumps({}),
                "empty_list": json.dumps([]),
            },
        )


if __name__ == "__main__":
    print("Starting market map brief migration (v1)...")
    migrate_market_map_brief_v1()
    print("✅ Migration complete")
