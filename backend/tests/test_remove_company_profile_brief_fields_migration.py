from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from migrations.migrate_remove_company_profile_brief_fields_v1 import (
    migrate_remove_company_profile_brief_fields_v1,
)


def test_remove_company_profile_brief_fields_migration_drops_legacy_columns(tmp_path: Path):
    db_path = tmp_path / "remove-company-profile-brief-fields.sqlite3"
    database_url = f"sqlite:///{db_path}"
    engine = create_engine(database_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE company_profiles (
                    id INTEGER PRIMARY KEY,
                    workspace_id INTEGER NOT NULL,
                    buyer_company_url TEXT,
                    manual_brief_text TEXT,
                    generated_context_summary TEXT
                )
                """
            )
        )

    summary = migrate_remove_company_profile_brief_fields_v1(database_url)

    with engine.begin() as conn:
        columns = {column["name"] for column in inspect(conn).get_columns("company_profiles")}

    assert "manual_brief_text" not in columns
    assert "generated_context_summary" not in columns
    assert summary["removed_columns"] == ["manual_brief_text", "generated_context_summary"]
