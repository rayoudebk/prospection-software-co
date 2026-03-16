import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401
from app.api import workspaces
from app.models.base import Base
from app.models.job import Job, JobProvider, JobState, JobType
from app.models.company_context import CompanyContextPack
from app.models.workspace import CompanyProfile


@pytest.fixture(autouse=True)
def _disable_live_llm(monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: SimpleNamespace(
            gemini_api_key="",
            openai_api_key="",
            anthropic_api_key="",
        ),
    )


def _build_test_client(tmp_path: Path):
    db_path = tmp_path / "company-context-test.sqlite3"
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
            profile.comparator_seed_urls = ["https://comp-one.example.com"]
            profile.supporting_evidence_urls = ["https://acme.example.com/customers"]
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


def _seed_expansion_brief(session_maker: async_sessionmaker[AsyncSession], workspace_id: int) -> None:
    from app.services.company_context import build_company_context_artifacts, build_expansion_artifacts

    async def seed() -> None:
        async with session_maker() as session:
            profile_result = await session.execute(
                select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
            )
            profile = profile_result.scalar_one()
            pack_result = await session.execute(
                select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
            )
            pack = pack_result.scalar_one_or_none()
            if pack is None:
                sourcing_payload = build_company_context_artifacts(profile)
                pack = CompanyContextPack(
                    workspace_id=workspace_id,
                    sourcing_brief_json=sourcing_payload.get("sourcing_brief") or {},
                    taxonomy_nodes_json=sourcing_payload.get("taxonomy_nodes") or [],
                    taxonomy_edges_json=sourcing_payload.get("taxonomy_edges") or [],
                    lens_seeds_json=sourcing_payload.get("lens_seeds") or [],
                )
                session.add(pack)
                await session.flush()
            expansion_payload = build_expansion_artifacts(
                profile,
                sourcing_brief=pack.sourcing_brief_json or {},
                taxonomy_nodes=pack.taxonomy_nodes_json or [],
            )
            expansion_brief = expansion_payload.get("expansion_brief") or {}
            if not expansion_brief.get("adjacent_capabilities"):
                expansion_brief["adjacent_capabilities"] = [
                    {
                        "id": "expansion_adjacent_reporting",
                        "label": "Client reporting",
                        "expansion_type": "adjacent_capability",
                        "status": "corroborated_expansion",
                        "confidence": 0.62,
                        "why_it_matters": "Common adjacent workflow for adjacent discovery.",
                        "evidence_urls": ["https://comp-one.example.com/reporting"],
                        "supporting_node_ids": [],
                        "source_entity_names": ["Comp One"],
                        "market_importance": "medium",
                        "operational_centrality": "meaningful",
                        "workflow_criticality": "medium",
                        "daily_operator_usage": "medium",
                        "switching_cost_intensity": "medium",
                        "priority_tier": "meaningful_adjacent",
                    }
                ]
            if not expansion_brief.get("named_account_anchors"):
                expansion_brief["named_account_anchors"] = [
                    {
                        "id": "expansion_account_northwind",
                        "label": "Northwind Capital",
                        "expansion_type": "named_account_anchor",
                        "status": "source_grounded",
                        "confidence": 0.74,
                        "why_it_matters": "Named account anchor from source-company proof.",
                        "evidence_urls": ["https://acme.example.com/customers"],
                        "supporting_node_ids": [],
                        "source_entity_names": ["Acme"],
                        "market_importance": "medium",
                        "operational_centrality": "meaningful",
                        "workflow_criticality": "medium",
                        "daily_operator_usage": "medium",
                        "switching_cost_intensity": "medium",
                        "priority_tier": "meaningful_adjacent",
                    }
                ]
            pack.expansion_brief_json = expansion_brief
            pack.expansion_status = "ready"
            pack.expansion_generated_at = expansion_payload.get("generated_at")
            await session.commit()

    asyncio.run(seed())


def test_company_context_bootstrap_update_and_adjustment_contract(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Acme sourcing", "region_scope": "US"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)

    company_context_response = client.get(f"/workspaces/{workspace_id}/company-context")
    assert company_context_response.status_code == 200
    company_context_payload = company_context_response.json()
    assert company_context_payload["sourcing_brief"]["source_summary"]
    assert company_context_payload["source_documents"]
    assert company_context_payload["buyer_evidence"]["status"] == "sufficient"
    assert company_context_payload["graph_status"] in {"success", "not_configured", "failed"}
    assert "sourcing_brief" in company_context_payload
    assert company_context_payload["sourcing_report"]["artifact_type"] == "report_artifact"
    assert company_context_payload["sourcing_report"]["report_kind"] == "sourcing_brief"
    assert "taxonomy_nodes" in company_context_payload
    assert "lens_seeds" in company_context_payload
    assert "expansion_brief" not in company_context_payload
    assert "expansion_report" not in company_context_payload

    patch_response = client.patch(
        f"/workspaces/{workspace_id}/company-context",
        json={"confirmed": True},
    )
    assert patch_response.status_code == 200
    patched_payload = patch_response.json()
    assert patched_payload["confirmed_at"] is not None

    taxonomy_patch_response = client.patch(
        f"/workspaces/{workspace_id}/company-context",
        json={
            "taxonomy_nodes": [
                *(
                    [
                        {
                            **company_context_payload["taxonomy_nodes"][0],
                            "phrase": "Private equity operations team",
                            "aliases": ["fund operations team", "PE ops"],
                        }
                    ]
                    if company_context_payload["taxonomy_nodes"]
                    else [
                        {
                            "id": "taxonomy_manual_customer",
                            "layer": "customer_archetype",
                            "phrase": "Private equity operations team",
                            "aliases": ["fund operations team", "PE ops"],
                            "confidence": 0.81,
                            "evidence_ids": [],
                            "scope_status": "in_scope",
                        }
                    ]
                )
            ]
        },
    )
    assert taxonomy_patch_response.status_code == 200
    taxonomy_payload = taxonomy_patch_response.json()
    assert any(
        node["phrase"] == "Private equity operations team"
        for node in taxonomy_payload["taxonomy_nodes"]
    )

def test_scope_review_contract_updates_and_confirms_scope(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "app.services.company_context._reason_sourcing_brief",
        lambda **kwargs: {
            **kwargs["fallback_brief"],
            "reasoning_status": "success",
            "reasoning_warning": None,
            "reasoning_provider": "test",
            "reasoning_model": "stub",
        },
    )
    monkeypatch.setattr(
        "app.services.company_context.get_settings",
        lambda: type("S", (), {
            "gemini_api_key": "",
            "openai_api_key": "",
            "anthropic_api_key": "",
        })(),
    )

    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Scope workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)
    client.get(f"/workspaces/{workspace_id}/company-context")
    _seed_expansion_brief(session_maker, workspace_id)

    scope_response = client.get(f"/workspaces/{workspace_id}/scope-review")
    assert scope_response.status_code == 200
    scope_payload = scope_response.json()
    assert scope_payload["source_capabilities"]
    assert scope_payload["named_account_anchors"]

    source_capability_id = scope_payload["source_capabilities"][0]["id"]
    named_account_id = scope_payload["named_account_anchors"][0]["id"]

    patch_response = client.patch(
        f"/workspaces/{workspace_id}/scope-review",
        json={
            "decisions": [
                {"id": source_capability_id, "status": "user_removed"},
                {"id": named_account_id, "status": "user_deprioritized"},
            ]
        },
    )
    assert patch_response.status_code == 200
    patched_payload = patch_response.json()
    patched_source = next(item for item in patched_payload["source_capabilities"] if item["id"] == source_capability_id)
    patched_account = next(item for item in patched_payload["named_account_anchors"] if item["id"] == named_account_id)
    assert patched_source["status"] == "user_removed"
    assert patched_account["status"] == "user_deprioritized"

    confirm_response = client.post(f"/workspaces/{workspace_id}/scope-review:confirm")
    assert confirm_response.status_code == 200
    confirmed_payload = confirm_response.json()
    assert confirmed_payload["confirmed_at"] is not None

    gates_response = client.get(f"/workspaces/{workspace_id}/gates")
    assert gates_response.status_code == 200
    assert gates_response.json()["scope_review"] is True


def test_company_context_refresh_enqueues_worker_task_and_returns_refreshing(tmp_path: Path, monkeypatch):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Refresh workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)

    enqueued: list[int] = []

    class _FakeTaskResult:
        id = "task-123"

    def _fake_delay(workspace_id: int):
        enqueued.append(workspace_id)
        return _FakeTaskResult()

    monkeypatch.setattr("app.workers.workspace_tasks.run_company_context_refresh.delay", _fake_delay)

    response = client.post(f"/workspaces/{workspace_id}/company-context:refresh")
    assert response.status_code == 200
    payload = response.json()

    assert payload["graph_status"] == "refreshing"
    assert enqueued == [workspace_id]


def test_expansion_generate_schedules_inline_task_and_returns_generating(tmp_path: Path, monkeypatch):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Expansion workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)
    client.get(f"/workspaces/{workspace_id}/company-context")

    scheduled: list[str] = []

    def _fake_create_task(coro):
        scheduled.append(getattr(getattr(coro, "cr_code", None), "co_name", "unknown"))
        coro.close()
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    monkeypatch.setattr("app.api.workspaces.asyncio.create_task", _fake_create_task)

    response = client.post(f"/workspaces/{workspace_id}/expansion-brief:generate")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "generating"
    assert "_run_expansion_refresh_inline" in scheduled


def test_company_context_refresh_resets_stale_refreshing_state(tmp_path: Path, monkeypatch):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Refresh workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)

    async def seed_stale_pack() -> None:
        async with session_maker() as session:
            result = await session.execute(
                select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
            )
            pack = result.scalar_one_or_none()
            if pack is None:
                pack = CompanyContextPack(workspace_id=workspace_id)
                session.add(pack)
                await session.flush()
            pack.graph_sync_status = "refreshing"
            pack.graph_sync_error = None
            pack.updated_at = datetime.utcnow() - timedelta(minutes=10)
            await session.commit()

    asyncio.run(seed_stale_pack())

    enqueued: list[int] = []

    class _FakeTaskResult:
        id = "task-456"

    def _fake_delay(value: int):
        enqueued.append(value)
        return _FakeTaskResult()

    monkeypatch.setattr("app.workers.workspace_tasks.run_company_context_refresh.delay", _fake_delay)

    response = client.post(f"/workspaces/{workspace_id}/company-context:refresh")
    assert response.status_code == 200
    payload = response.json()

    assert payload["graph_status"] == "refreshing"
    assert enqueued == [workspace_id]


def test_company_context_refresh_falls_back_inline_when_enqueue_fails(tmp_path: Path, monkeypatch):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Refresh workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]
    _seed_company_profile(session_maker, workspace_id)

    scheduled: list[str] = []

    def _fake_create_task(coro):
        scheduled.append(getattr(getattr(coro, "cr_code", None), "co_name", "unknown"))
        coro.close()
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    monkeypatch.setattr(
        "app.workers.workspace_tasks.run_company_context_refresh.delay",
        lambda _workspace_id: (_ for _ in ()).throw(RuntimeError("broker down")),
    )
    monkeypatch.setattr("app.api.workspaces.asyncio.create_task", _fake_create_task)

    response = client.post(f"/workspaces/{workspace_id}/company-context:refresh")
    assert response.status_code == 200
    payload = response.json()

    assert payload["graph_status"] == "refreshing"
    assert "_run_company_context_refresh_inline" in scheduled


def test_company_context_uses_graph_ref_from_cache_when_column_empty(tmp_path: Path):
    pack = CompanyContextPack(
        id=1,
        workspace_id=42,
        company_context_graph_ref=None,
        company_context_graph_cache_json={
            "graph_ref": "workspace-test-acme",
            "source_documents": [],
            "nodes": [],
            "edges": [],
        },
        graph_sync_status="success",
        graph_stats_json={},
    )
    profile = CompanyProfile(
        workspace_id=42,
        buyer_company_url="https://acme.example.com",
        comparator_seed_urls=[],
        supporting_evidence_urls=[],
        comparator_seed_summaries={},
        context_pack_json={},
    )

    payload = workspaces._company_context_payload_from_pack(pack, profile=profile)
    assert payload["company_context_graph_ref"] == "workspace-test-acme"


def test_workspace_region_update_keeps_profile_geo_scope_in_sync(tmp_path: Path):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Geo sync workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]

    patch_context_response = client.patch(
        f"/workspaces/{workspace_id}/context-pack",
        json={
            "geo_scope": {
                "region": "EU+UK",
                "include_countries": ["DE", "FR"],
                "exclude_countries": ["RU"],
            }
        },
    )
    assert patch_context_response.status_code == 200

    update_workspace_response = client.patch(
        f"/workspaces/{workspace_id}",
        json={"region_scope": "US"},
    )
    assert update_workspace_response.status_code == 200
    assert update_workspace_response.json()["region_scope"] == "US"

    async def fetch_profile() -> CompanyProfile:
        async with session_maker() as session:
            result = await session.execute(
                select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
            )
            return result.scalar_one()

    profile = asyncio.run(fetch_profile())
    assert profile.geo_scope["region"] == "US"
    assert profile.geo_scope["include_countries"] == ["DE", "FR"]
    assert profile.geo_scope["exclude_countries"] == ["RU"]


def test_company_context_response_preserves_expansion_inputs():
    from app.api.workspaces import _company_context_response_from_payload

    payload = {
        "id": 1,
        "workspace_id": 7,
        "graph_status": "success",
        "graph_stats": {},
        "deep_research_handoff": {"graph_ref": "workspace-7-4tpm"},
        "source_documents": [],
        "context_pack_v2": {"sites": []},
        "expansion_inputs": [
            {"name": "CWAN", "website": "https://cwan.com"},
            {"name": "Wealth Dynamix", "website": "https://wealth-dynamix.com"},
        ],
        "taxonomy_nodes": [],
        "taxonomy_edges": [],
        "lens_seeds": [],
        "sourcing_brief": {
            "source_company": {"name": "4TPM", "website": "https://4tpm.fr"},
            "source_summary": "Summary",
            "customer_nodes": [],
            "workflow_nodes": [],
            "capability_nodes": [],
            "delivery_or_integration_nodes": [],
            "named_customer_proof": [],
            "partner_integration_proof": [],
            "secondary_evidence_proof": [],
            "customer_partner_corroboration": [],
            "directory_category_context": [],
            "other_secondary_context": [],
            "active_lenses": [],
            "adjacency_hypotheses": [],
            "strongest_evidence_buckets": [],
            "confidence_gaps": [],
            "open_questions": [],
            "unknowns_not_publicly_resolvable": [],
            "crawl_coverage": {},
            "confirmed_at": None,
        },
    }

    response = _company_context_response_from_payload(payload)

    assert response.deep_research_handoff["graph_ref"] == "workspace-7-4tpm"
    assert len(response.expansion_inputs) == 2
    assert response.expansion_inputs[0]["name"] == "CWAN"
    assert response.expansion_inputs[1]["name"] == "Wealth Dynamix"


def test_context_pack_refresh_runs_inline_when_enqueue_fails(tmp_path: Path):
    client, _session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Queue failure workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]

    with patch("app.workers.workspace_tasks.generate_context_pack_v2.delay", side_effect=RuntimeError("broker down")):
        with patch("app.api.workspaces._run_context_pack_inline", return_value=None):
            response = client.post(f"/workspaces/{workspace_id}/context-pack:refresh")

    assert response.status_code == 200


def test_context_pack_refresh_returns_clear_503_when_enqueue_and_inline_fail(tmp_path: Path):
    client, _session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Queue failure workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]

    with patch("app.workers.workspace_tasks.generate_context_pack_v2.delay", side_effect=RuntimeError("broker down")):
        with patch("app.api.workspaces._run_context_pack_inline", side_effect=RuntimeError("inline down")):
            response = client.post(f"/workspaces/{workspace_id}/context-pack:refresh")

    assert response.status_code == 503
    assert "Background crawl worker unavailable" in response.json()["detail"]


def test_context_pack_refresh_supersedes_existing_active_job(tmp_path: Path):
    client, session_maker = _build_test_client(tmp_path)

    create_response = client.post("/workspaces", json={"name": "Supersede workspace", "region_scope": "EU+UK"})
    assert create_response.status_code == 200
    workspace_id = create_response.json()["id"]

    async def seed_running_job() -> None:
        async with session_maker() as session:
            session.add(
                Job(
                    workspace_id=workspace_id,
                    job_type=JobType.context_pack,
                    state=JobState.running,
                    provider=JobProvider.crawler,
                )
            )
            await session.commit()

    asyncio.run(seed_running_job())

    with patch("app.workers.workspace_tasks.generate_context_pack_v2.delay", side_effect=RuntimeError("broker down")):
        with patch("app.api.workspaces._run_context_pack_inline", return_value=None):
            response = client.post(f"/workspaces/{workspace_id}/context-pack:refresh")

    assert response.status_code == 200

    async def fetch_jobs() -> list[Job]:
        async with session_maker() as session:
            result = await session.execute(
                select(Job).where(Job.workspace_id == workspace_id).order_by(Job.id.asc())
            )
            return list(result.scalars().all())

    jobs = asyncio.run(fetch_jobs())
    assert len(jobs) == 2
    assert jobs[0].state == JobState.failed
    assert jobs[0].error_message == "Superseded by newer sourcing brief run"
    assert jobs[1].state == JobState.queued
