import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401
from app.api import workspaces
from app.models.base import Base
from app.models.workspace import CompanyProfile


def _build_test_client(tmp_path: Path):
    db_path = tmp_path / "thesis-pack-test.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", future=True)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def init_models() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    asyncio.run(init_models())

    app = FastAPI()
    app.include_router(workspaces.router, prefix="/workspaces", tags=["workspaces"])
    app.dependency_overrides[workspaces.get_db] = override_get_db

    return TestClient(app), session_maker


def _seed_company_profile(session_maker: async_sessionmaker[AsyncSession], workspace_id: int) -> None:
    async def seed() -> None:
        async with session_maker() as session:
            result = await session.execute(
                select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
            )
            profile = result.scalar_one()
            profile.buyer_company_url = "https://acme.example.com"
            profile.buyer_context_summary = (
                "Acme is a SaaS fund operations workflow platform with implementation services "
                "and API integrations for private equity and fund teams."
            )
            profile.reference_vendor_urls = ["https://comp-one.example.com"]
            profile.reference_evidence_urls = ["https://acme.example.com/customers"]
            profile.context_pack_markdown = "# Acme\n\nPortfolio analytics and reporting for fund operations teams."
            profile.product_pages_found = 4
            profile.context_pack_json = {
                "sites": [
                    {
                        "url": "https://acme.example.com",
                        "company_name": "Acme",
                        "summary": "Portfolio analytics and reporting for fund operations teams.",
                        "signals": [
                            {
                                "type": "capability",
                                "value": "Portfolio analytics",
                                "source_url": "https://acme.example.com/platform",
                            },
                            {
                                "type": "service",
                                "value": "Implementation services",
                                "source_url": "https://acme.example.com/services",
                            },
                            {
                                "type": "integration",
                                "value": "API integrations",
                                "source_url": "https://acme.example.com/integrations",
                            },
                        ],
                        "customer_evidence": [
                            {
                                "name": "Northwind Capital",
                                "source_url": "https://acme.example.com/customers",
                            }
                        ],
                        "pages": [
                            {
                                "url": "https://acme.example.com/platform",
                                "title": "Platform",
                                "signals": [],
                                "customer_evidence": [],
                            }
                        ],
                    }
                ]
            }
            await session.commit()

    asyncio.run(seed())


def test_thesis_pack_bootstrap_update_and_adjustment_contract(tmp_path: Path):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Acme sourcing", "region_scope": "US"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)

    thesis_response = client.get(f"/workspaces/{workspace_id}/thesis-pack")
    assert thesis_response.status_code == 200
    thesis_payload = thesis_response.json()
    assert thesis_payload["summary"]
    assert thesis_payload["source_pills"]
    assert any(claim["section"] == "core_capability" for claim in thesis_payload["claims"])

    first_claim = thesis_payload["claims"][0]
    updated_claims = [
        {
            **claim,
            "user_status": "confirmed" if claim["id"] == first_claim["id"] else claim["user_status"],
        }
        for claim in thesis_payload["claims"]
    ]
    patch_response = client.patch(
        f"/workspaces/{workspace_id}/thesis-pack",
        json={"claims": updated_claims, "confirmed": True},
    )
    assert patch_response.status_code == 200
    patched_payload = patch_response.json()
    patched_first_claim = next(claim for claim in patched_payload["claims"] if claim["id"] == first_claim["id"])
    assert patched_first_claim["user_status"] == "confirmed"
    assert patched_payload["confirmed_at"] is not None

    adjustment_response = client.post(
        f"/workspaces/{workspace_id}/thesis-pack:apply-adjustment",
        json={
            "operations": [
                {"op": "add_claim", "section": "adjacent_capability", "value": "Voting rights workflow"},
                {"op": "add_open_question", "value": "What company-size window matters most?"},
            ]
        },
    )
    assert adjustment_response.status_code == 200
    adjustment_payload = adjustment_response.json()
    assert adjustment_payload["applied_operations"]
    assert any(
        claim["value"] == "Voting rights workflow"
        for claim in adjustment_payload["thesis_pack"]["claims"]
    )
    assert "What company-size window matters most?" in adjustment_payload["thesis_pack"]["open_questions"]


def test_search_lanes_contract_supports_patch_and_confirm(tmp_path: Path):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Lane workspace", "region_scope": "US"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)

    lanes_response = client.get(f"/workspaces/{workspace_id}/search-lanes")
    assert lanes_response.status_code == 200
    lanes_payload = lanes_response.json()
    assert {lane["lane_type"] for lane in lanes_payload["lanes"]} == {"core", "adjacent"}

    patched_lanes = []
    for lane in lanes_payload["lanes"]:
        if lane["lane_type"] == "core":
            patched_lanes.append(
                {
                    **lane,
                    "capabilities": ["Portfolio analytics", "Fund reporting"],
                    "must_include_terms": ["private equity", "SaaS"],
                }
            )
        else:
            patched_lanes.append(
                {
                    **lane,
                    "capabilities": ["Voting rights workflow"],
                    "must_exclude_terms": ["ERP"],
                }
            )

    patch_response = client.patch(
        f"/workspaces/{workspace_id}/search-lanes",
        json={"lanes": patched_lanes},
    )
    assert patch_response.status_code == 200
    patched_payload = patch_response.json()
    core_lane = next(lane for lane in patched_payload["lanes"] if lane["lane_type"] == "core")
    assert "Fund reporting" in core_lane["capabilities"]

    confirm_response = client.post(f"/workspaces/{workspace_id}/search-lanes:confirm")
    assert confirm_response.status_code == 200
    confirmed_payload = confirm_response.json()
    assert all(lane["status"] == "confirmed" for lane in confirmed_payload["lanes"])

    gates_response = client.get(f"/workspaces/{workspace_id}/gates")
    assert gates_response.status_code == 200
    gates_payload = gates_response.json()
    assert gates_payload["context_pack"] is True
    assert gates_payload["search_lanes"] is True
