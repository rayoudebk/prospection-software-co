#!/usr/bin/env python3
"""Bootstrap and sync a workspace company-context graph into Neo4j."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime

from sqlalchemy import select

from app.models.base import async_session_maker
from app.models.company_context import CompanyContextPack
from app.models.workspace import CompanyProfile
from app.services.company_context_graph import (
    Neo4jCompanyContextGraphStore,
    build_company_context_payload,
)
from app.services.company_context import build_company_context_artifacts
from app.startup_migrations import run_startup_migrations


async def _sync_workspace(workspace_id: int) -> int:
    run_startup_migrations()

    async with async_session_maker() as session:
        profile_result = await session.execute(
            select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
        )
        profile = profile_result.scalar_one_or_none()
        if not profile:
            print(f"Workspace {workspace_id} has no company profile", file=sys.stderr)
            return 1

        pack_result = await session.execute(
            select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
        )
        pack = pack_result.scalar_one_or_none()
        if not pack:
            pack = CompanyContextPack(workspace_id=workspace_id)
            session.add(pack)
            await session.flush()

        refreshed = build_company_context_artifacts(profile)
        pack.market_map_brief_json = refreshed.get("market_map_brief") or {}
        pack.expansion_brief_json = refreshed.get("expansion_brief") or {}
        pack.taxonomy_nodes_json = refreshed.get("taxonomy_nodes") or []
        pack.taxonomy_edges_json = refreshed.get("taxonomy_edges") or []
        pack.lens_seeds_json = refreshed.get("lens_seeds") or []
        pack.generated_at = refreshed.get("generated_at")
        pack.confirmed_at = None
        pack.updated_at = datetime.utcnow()

        payload = build_company_context_payload(pack, profile)
        graph_payload = payload.get("company_context_graph") or {}
        sync_result = Neo4jCompanyContextGraphStore().sync_graph(graph_payload)
        sync_status = str(sync_result.get("status") or "failed")

        pack.company_context_graph_ref = payload.get("company_context_graph_ref")
        pack.company_context_graph_cache_json = graph_payload
        pack.graph_stats_json = payload.get("graph_stats") or {}
        pack.graph_sync_status = sync_status
        pack.graph_sync_error = sync_result.get("error")
        pack.graph_synced_at = datetime.utcnow()
        pack.market_map_brief_json = payload.get("market_map_brief") or pack.market_map_brief_json or {}
        pack.expansion_brief_json = payload.get("expansion_brief") or pack.expansion_brief_json or {}

        await session.commit()

        result = {
            "workspace_id": workspace_id,
            "company_context_graph_ref": pack.company_context_graph_ref,
            "graph_status": sync_status,
            "graph_warning": pack.graph_sync_error,
            "graph_stats": pack.graph_stats_json or {},
            "market_map_brief": pack.market_map_brief_json or {},
            "expansion_brief": pack.expansion_brief_json or {},
        }
        print(json.dumps(result, indent=2, default=str))
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync a workspace company-context graph to Neo4j")
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to sync")
    args = parser.parse_args()
    return asyncio.run(_sync_workspace(args.workspace_id))


if __name__ == "__main__":
    raise SystemExit(main())
