from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.models.workspace import CompanyProfile, Workspace
from app.models.company_context import CompanyContextPack
from migrations.migrate_company_context_backfill_v1 import (
    _ensure_company_context_tables,
    backfill_company_context,
)


def test_company_context_backfill_creates_tables_and_backfills_workspace():
    engine = create_engine("sqlite:///:memory:")
    Workspace.__table__.create(bind=engine)
    CompanyProfile.__table__.create(bind=engine)

    with engine.begin() as conn:
        _ensure_company_context_tables(conn)
        tables = set(inspect(conn).get_table_names())
        assert "company_context_packs" in tables

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with SessionLocal() as session:
        workspace = Workspace(name="Migration test", region_scope="US", decision_policy_json={})
        session.add(workspace)
        session.flush()
        session.add(
            CompanyProfile(
                workspace_id=workspace.id,
                buyer_company_url="https://acme.example.com",
                comparator_seed_urls=["https://comp-one.example.com"],
                supporting_evidence_urls=["https://acme.example.com/customers"],
                context_pack_markdown="# Acme\n\nReporting workflow platform.",
                context_pack_json={
                    "sites": [
                        {
                            "url": "https://acme.example.com",
                            "company_name": "Acme",
                            "summary": "Reporting workflow platform.",
                            "signals": [
                                {
                                    "type": "capability",
                                    "value": "Reporting workflow",
                                    "source_url": "https://acme.example.com/platform",
                                },
                                {
                                    "type": "service",
                                    "value": "Implementation services",
                                    "source_url": "https://acme.example.com/services",
                                },
                            ],
                            "customer_evidence": [],
                            "pages": [],
                        }
                    ]
                },
                product_pages_found=4,
                geo_scope={"region": "US", "include_countries": ["US"], "exclude_countries": []},
            )
        )
        session.commit()

    with SessionLocal() as session:
        summary = backfill_company_context(session)
        assert summary["company_context_packs_created"] == 1

    with SessionLocal() as session:
        company_context_pack = session.query(CompanyContextPack).one()
        assert company_context_pack.sourcing_brief_json.get("source_summary")
        assert company_context_pack.taxonomy_nodes_json
        assert company_context_pack.graph_sync_status == "not_synced"
