from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.models.workspace import BrickTaxonomy, CompanyProfile, Workspace
from app.models.thesis import BuyerThesisPack, SearchLane
from migrations.migrate_thesis_sourcing_v1 import _ensure_thesis_tables, backfill_thesis_sourcing


def test_thesis_sourcing_migration_creates_tables_and_backfills_workspace():
    engine = create_engine("sqlite:///:memory:")
    Workspace.__table__.create(bind=engine)
    CompanyProfile.__table__.create(bind=engine)
    BrickTaxonomy.__table__.create(bind=engine)

    with engine.begin() as conn:
        _ensure_thesis_tables(conn)
        tables = set(inspect(conn).get_table_names())
        assert "buyer_thesis_packs" in tables
        assert "search_lanes" in tables

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with SessionLocal() as session:
        workspace = Workspace(name="Migration test", region_scope="US", decision_policy_json={})
        session.add(workspace)
        session.flush()
        session.add(
            CompanyProfile(
                workspace_id=workspace.id,
                buyer_company_url="https://acme.example.com",
                buyer_context_summary="Acme is a SaaS reporting platform with implementation services.",
                reference_vendor_urls=["https://comp-one.example.com"],
                reference_evidence_urls=["https://acme.example.com/customers"],
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
        session.add(
            BrickTaxonomy(
                workspace_id=workspace.id,
                bricks=[{"id": "brick-1", "name": "Reporting workflow"}],
                priority_brick_ids=["brick-1"],
                vertical_focus=["private_equity"],
                confirmed=False,
            )
        )
        session.commit()

    with SessionLocal() as session:
        summary = backfill_thesis_sourcing(session)
        assert summary["thesis_packs_created"] == 1
        assert summary["search_lanes_created"] == 2

    with SessionLocal() as session:
        thesis_pack = session.query(BuyerThesisPack).one()
        search_lanes = session.query(SearchLane).order_by(SearchLane.lane_type.asc()).all()
        assert thesis_pack.summary
        assert any(claim["section"] == "core_capability" for claim in (thesis_pack.claims_json or []))
        assert {lane.lane_type for lane in search_lanes} == {"core", "adjacent"}
