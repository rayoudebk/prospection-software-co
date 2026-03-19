import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401
from app.api import workspaces
from app.models.base import Base
from app.models.company_context import CompanyContextPack
from app.models.intelligence import CandidateEntity, CompanyScreening
from app.models.workspace import Workspace, CompanyProfile


def _build_test_client(tmp_path: Path):
    db_path = tmp_path / "validation-api.sqlite3"
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


def _seed_validation_workspace(session_maker: async_sessionmaker[AsyncSession], workspace_id: int = 1) -> None:
    async def seed() -> None:
        async with session_maker() as session:
            workspace = Workspace(id=workspace_id, name="Validation Test", region_scope="EU+UK")
            profile = CompanyProfile(
                workspace_id=workspace_id,
                buyer_company_url="https://buyer.example.com",
                comparator_seed_urls=["https://dir.example.com"],
                supporting_evidence_urls=["https://buyer.example.com/platform"],
                geo_scope={"region": "EU+UK", "include_countries": [], "exclude_countries": []},
            )
            pack = CompanyContextPack(
                workspace_id=workspace_id,
                sourcing_brief_json={"source_summary": "We sell portfolio systems."},
                taxonomy_nodes_json=[{"id": "cap_1", "layer": "capability", "phrase": "Portfolio management"}],
                expansion_brief_json={
                    "confirmed_at": "2026-03-19T00:00:00Z",
                    "adjacency_boxes": [{"id": "adj_1", "label": "Portfolio management"}],
                },
                expansion_status="ready",
            )
            entity = CandidateEntity(
                id=11,
                workspace_id=workspace_id,
                canonical_name="QPLIX",
                canonical_website="https://www.qplix.com",
                canonical_domain="qplix.com",
                discovery_primary_url="https://www.qplix.com",
                entity_type="company",
                identity_confidence="high",
                metadata_json={
                    "validation": {
                        "status": "queued_for_validation",
                        "recommendation": "validated_keep",
                        "promoted_to_cards": False,
                        "lane_ids": ["adj_1"],
                        "lane_labels": ["Portfolio management"],
                        "query_families": ["exa::vendor::portfolio management"],
                        "source_families": ["search"],
                        "origin_types": ["external_search_seed"],
                        "priority_score": 77.0,
                    }
                },
            )
            screening = CompanyScreening(
                workspace_id=workspace_id,
                candidate_entity_id=11,
                candidate_name="QPLIX",
                candidate_website="https://www.qplix.com",
                candidate_discovery_url="https://www.qplix.com",
                candidate_official_website="https://www.qplix.com",
                screening_status="review",
                total_score=77.0,
                decision_classification="good_target",
                evidence_sufficiency="sufficient",
                ranking_eligible=True,
                screening_meta_json={
                    "screening_run_id": "run-1",
                    "candidate_hq_country": "Germany",
                    "entity_type": "company",
                    "capability_signals": ["Portfolio management"],
                    "scope_buckets": ["adjacent"],
                    "origin_types": ["external_search_seed"],
                    "registry_identity": {},
                    "citation_summary_v1": {
                        "version": "v1",
                        "sentences": [
                            {
                                "id": "s1",
                                "text": "QPLIX provides portfolio management software.",
                                "citation_pill_ids": ["p1"],
                                "claim_group": "product_depth",
                                "source_tier": "tier1_vendor",
                                "source_kind": "vendor_site",
                                "captured_at": None,
                            }
                        ],
                        "source_pills": [
                            {
                                "pill_id": "p1",
                                "label": "qplix.com",
                                "url": "https://www.qplix.com",
                                "source_tier": "tier1_vendor",
                                "source_kind": "vendor_site",
                                "captured_at": None,
                                "claim_group": "product_depth",
                            }
                        ],
                    },
                },
                source_summary_json={
                    "source_urls": ["https://www.qplix.com"],
                    "expansion_provenance": [
                        {
                            "provider": "exa",
                            "query_type": "vendor",
                            "query_text": "portfolio management software vendor",
                            "brick_name": "Portfolio management",
                            "scope_bucket": "adjacent",
                        }
                    ],
                },
            )
            session.add_all([workspace, profile, pack, entity, screening])
            await session.commit()

    asyncio.run(seed())


def test_validation_queue_and_promotion_flow(tmp_path: Path):
    client, session_maker = _build_test_client(tmp_path)
    _seed_validation_workspace(session_maker)

    queue_response = client.get("/workspaces/1/validation/queue")
    assert queue_response.status_code == 200
    payload = queue_response.json()
    assert len(payload) == 1
    assert payload[0]["candidate_entity_id"] == 11
    assert payload[0]["validation_status"] == "queued_for_validation"
    assert payload[0]["validation_lane_labels"] == ["Portfolio management"]

    update_response = client.patch(
        "/workspaces/1/validation/11",
        json={"status": "promoted_to_cards"},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["validation_status"] == "promoted_to_cards"
    assert updated["promoted_to_cards"] is True


def test_report_generation_requires_promoted_shortlist(tmp_path: Path):
    client, session_maker = _build_test_client(tmp_path)
    _seed_validation_workspace(session_maker)

    response = client.post("/workspaces/1/reports:generate", json={})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "cards_not_ready"


def test_validation_refresh_updates_identity_diagnostics(tmp_path: Path, monkeypatch):
    client, session_maker = _build_test_client(tmp_path)
    _seed_validation_workspace(session_maker)

    monkeypatch.setattr(
        workspaces,
        "_resolve_directory_profile_seed_candidates",
        lambda candidates, max_fetches: {
            "candidates_considered": len(candidates),
            "candidates_selected": len(candidates),
        },
    )

    def fake_resolve_identities(candidates, max_fetches, timeout_seconds=0, concurrency=0):
        for candidate in candidates:
            candidate["identity"] = {
                "official_website": candidate.get("official_website_url") or candidate.get("website"),
                "identity_confidence": "high",
                "resolved_via_redirect": True,
                "error": None,
            }
        return {"identity_resolved_count": len(candidates)}

    monkeypatch.setattr(workspaces, "_resolve_identities_for_candidates", fake_resolve_identities)
    monkeypatch.setattr(
        workspaces,
        "_extract_first_party_signals",
        lambda website, candidate_name: (
            [
                {
                    "text": f"{candidate_name} provides portfolio management platform software.",
                    "citation_url": website,
                    "dimension": "product",
                }
            ],
            None,
        ),
    )

    response = client.post("/workspaces/1/validation:refresh", json={"top_n": 1})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["identity_confidence"] == "high"
    assert payload[0]["vendor_classification"] == "vendor_candidate"
    assert payload[0]["identity_diagnostics"]["resolved_via_redirect"] is True
    assert payload[0]["identity_diagnostics"]["has_first_party_evidence"] is True
