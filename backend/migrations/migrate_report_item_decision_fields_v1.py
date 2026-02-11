"""
Migration script to add decision metadata fields to report_snapshot_items.
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


def migrate_report_item_decision_fields_v1():
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _add_column_if_missing(conn, "report_snapshot_items", "decision_classification VARCHAR(40)", "decision_classification")
        _add_column_if_missing(conn, "report_snapshot_items", "reason_codes_json JSON", "reason_codes_json")
        _add_column_if_missing(conn, "report_snapshot_items", "evidence_summary_json JSON", "evidence_summary_json")
        conn.execute(
            text(
                """
                UPDATE report_snapshot_items
                SET reason_codes_json = COALESCE(reason_codes_json, '{}'),
                    evidence_summary_json = COALESCE(evidence_summary_json, '{}')
                """
            )
        )


if __name__ == "__main__":
    print("Starting report snapshot decision fields migration...")
    migrate_report_item_decision_fields_v1()
    print("✅ Migration complete")
