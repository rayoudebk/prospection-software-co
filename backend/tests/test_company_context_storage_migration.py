from sqlalchemy import create_engine, inspect, text

from migrations.migrate_company_context_storage_v1 import migrate_company_context_storage_v1


def test_company_context_storage_migration_renames_table_and_columns(tmp_path):
    db_path = tmp_path / "company-context-storage.sqlite3"
    database_url = f"sqlite:///{db_path}"
    engine = create_engine(database_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE buyer_thesis_packs (
                    id INTEGER PRIMARY KEY,
                    workspace_id INTEGER NOT NULL,
                    company_context_graph_ref TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE company_profiles (
                    id INTEGER PRIMARY KEY,
                    workspace_id INTEGER NOT NULL,
                    reference_company_urls JSON,
                    reference_evidence_urls JSON,
                    reference_summaries JSON
                )
                """
            )
        )
        conn.execute(
            text(
                'CREATE INDEX "ix_buyer_thesis_packs_company_context_graph_ref" '
                'ON "buyer_thesis_packs" ("company_context_graph_ref")'
            )
        )

    summary = migrate_company_context_storage_v1(database_url=database_url)

    with engine.begin() as conn:
        tables = set(inspect(conn).get_table_names())
        assert "company_context_packs" in tables
        assert "buyer_thesis_packs" not in tables

        columns = {column["name"] for column in inspect(conn).get_columns("company_profiles")}
        assert "comparator_seed_urls" in columns
        assert "supporting_evidence_urls" in columns
        assert "comparator_seed_summaries" in columns
        assert "reference_company_urls" not in columns
        assert "reference_evidence_urls" not in columns
        assert "reference_summaries" not in columns

        index_names = {index["name"] for index in inspect(conn).get_indexes("company_context_packs")}
        assert "ix_company_context_packs_company_context_graph_ref" in index_names
        assert "ix_buyer_thesis_packs_company_context_graph_ref" not in index_names

    assert "buyer_thesis_packs->company_context_packs" in summary["tables_renamed"]
