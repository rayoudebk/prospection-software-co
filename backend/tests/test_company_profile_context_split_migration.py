from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from migrations.migrate_company_profile_context_split_v1 import (
    migrate_company_profile_context_split_v1,
)


def test_company_profile_context_split_migration_adds_columns_and_backfills(tmp_path: Path):
    db_path = tmp_path / "context-split.sqlite3"
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
                    buyer_context_summary TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO company_profiles (id, workspace_id, buyer_company_url, buyer_context_summary)
                VALUES
                    (1, 101, NULL, 'Manual thesis about healthcare software'),
                    (2, 102, 'https://acme.example.com', 'Generated company summary for Acme')
                """
            )
        )

    migrate_company_profile_context_split_v1(database_url)

    with engine.begin() as conn:
        columns = {column["name"] for column in inspect(conn).get_columns("company_profiles")}
        assert "manual_brief_text" in columns
        assert "generated_context_summary" in columns

        rows = conn.execute(
            text(
                """
                SELECT id, manual_brief_text, generated_context_summary
                FROM company_profiles
                ORDER BY id ASC
                """
            )
        ).fetchall()

    assert rows[0] == (1, "Manual thesis about healthcare software", None)
    assert rows[1] == (2, None, "Generated company summary for Acme")
