#!/usr/bin/env python3
"""Replay a workspace context pack through the local company-context builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.workspace import CompanyProfile  # noqa: E402
from app.services.company_context import build_company_context_artifacts  # noqa: E402


def _load_workspace_context(*, workspace_id: int, api_base: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/workspaces/{workspace_id}/context-pack"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _profile_from_context_pack(payload: dict[str, Any], *, workspace_id: int) -> CompanyProfile:
    return CompanyProfile(
        workspace_id=workspace_id,
        buyer_company_url=payload.get("buyer_company_url"),
        comparator_seed_urls=payload.get("comparator_seed_urls") or [],
        supporting_evidence_urls=payload.get("supporting_evidence_urls") or [],
        comparator_seed_summaries=payload.get("comparator_seed_summaries") or {},
        geo_scope=payload.get("geo_scope") or {},
        context_pack_markdown=payload.get("context_pack_markdown"),
        context_pack_json=payload.get("context_pack_json") or {},
        product_pages_found=payload.get("product_pages_found") or 0,
    )


def _compact_output(payload: dict[str, Any]) -> dict[str, Any]:
    brief = payload.get("market_map_brief") or {}
    return {
        "reasoning_status": brief.get("reasoning_status"),
        "reasoning_provider": brief.get("reasoning_provider"),
        "reasoning_model": brief.get("reasoning_model"),
        "source_summary": brief.get("source_summary"),
        "customer_nodes": [node.get("phrase") for node in brief.get("customer_nodes") or []],
        "workflow_nodes": [node.get("phrase") for node in brief.get("workflow_nodes") or []],
        "capability_nodes": [node.get("phrase") for node in brief.get("capability_nodes") or []],
        "delivery_or_integration_nodes": [
            node.get("phrase") for node in brief.get("delivery_or_integration_nodes") or []
        ],
        "active_lenses": [lens.get("lens_type") for lens in brief.get("active_lenses") or []],
        "open_questions": brief.get("open_questions") or [],
        "confidence_gaps": brief.get("confidence_gaps") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to replay locally")
    parser.add_argument(
        "--api-base",
        default="https://sourcing-os.up.railway.app",
        help="Base API URL for fetching the stored context pack",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full company-context payload instead of the compact summary view",
    )
    args = parser.parse_args()

    context_pack_payload = _load_workspace_context(workspace_id=args.workspace_id, api_base=args.api_base)
    profile = _profile_from_context_pack(context_pack_payload, workspace_id=args.workspace_id)
    bootstrap_payload = build_company_context_artifacts(profile)

    output = bootstrap_payload if args.full else _compact_output(bootstrap_payload)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
