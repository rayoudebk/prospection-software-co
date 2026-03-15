#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.services.company_context_graph import (
    Neo4jCompanyContextGraphStore,
    build_company_context_graph,
)
from app.services.company_context import build_company_context_artifacts


@dataclass
class APICompanyProfile:
    workspace_id: int
    buyer_company_url: Optional[str]
    comparator_seed_urls: list[str]
    supporting_evidence_urls: list[str]
    comparator_seed_summaries: dict[str, Any]
    geo_scope: dict[str, Any]
    context_pack_markdown: Optional[str]
    context_pack_json: dict[str, Any]
    context_pack_generated_at: Optional[str]
    product_pages_found: int


def _profile_from_api_payload(payload: dict[str, Any]) -> APICompanyProfile:
    return APICompanyProfile(
        workspace_id=int(payload["workspace_id"]),
        buyer_company_url=payload.get("buyer_company_url"),
        comparator_seed_urls=list(payload.get("comparator_seed_urls") or []),
        supporting_evidence_urls=list(payload.get("supporting_evidence_urls") or []),
        comparator_seed_summaries=dict(payload.get("comparator_seed_summaries") or {}),
        geo_scope=dict(payload.get("geo_scope") or {}),
        context_pack_markdown=payload.get("context_pack_markdown"),
        context_pack_json=dict(payload.get("context_pack_json") or {}),
        context_pack_generated_at=payload.get("context_pack_generated_at"),
        product_pages_found=int(payload.get("product_pages_found") or 0),
    )


def _fetch_context_pack(base_url: str, workspace_id: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/workspaces/{workspace_id}/context-pack"
    response = httpx.get(url, timeout=120.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a live company context payload from the API and sync it to Neo4j.")
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to fetch from the live API.")
    parser.add_argument(
        "--base-url",
        default="https://sourcing-os.up.railway.app",
        help="Base API URL. Defaults to the Railway deployment.",
    )
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print the built graph payload before syncing.",
    )
    args = parser.parse_args()

    payload = _fetch_context_pack(args.base_url, args.workspace_id)
    profile = _profile_from_api_payload(payload)
    company_context_payload = build_company_context_artifacts(profile)
    graph_payload = build_company_context_graph(profile, payload=company_context_payload)

    if args.print_graph:
        print(json.dumps(graph_payload, indent=2, ensure_ascii=False))

    sync_result = Neo4jCompanyContextGraphStore().sync_graph(graph_payload)
    output = {
        "workspace_id": args.workspace_id,
        "company_context_graph_ref": graph_payload.get("graph_ref"),
        "graph_status": sync_result.get("status"),
        "graph_warning": sync_result.get("error"),
        "graph_stats": graph_payload.get("graph_stats") or {},
        "market_map_brief": graph_payload.get("market_map_brief") or {},
        "expansion_brief": graph_payload.get("expansion_brief") or {},
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
