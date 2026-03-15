#!/usr/bin/env python3
"""Print the bounded secondary-evidence query plan for a source-company workspace."""

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

from app.config import get_settings  # noqa: E402
from app.models.workspace import CompanyProfile  # noqa: E402
import app.services.company_context as company_context_mod  # noqa: E402
from app.services.company_context import build_company_context_artifacts  # noqa: E402
from app.services.company_context_graph import (  # noqa: E402
    _build_secondary_queries,
    build_primary_company_graph_from_context,
)


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


def _stub_market_map_reasoning() -> None:
    company_context_mod._reason_sourcing_brief = lambda **kwargs: {  # type: ignore[attr-defined]
        **kwargs["fallback_brief"],
        "reasoning_status": "success",
        "reasoning_warning": None,
        "reasoning_provider": "local_stub",
        "reasoning_model": "local_stub",
    }
    company_context_mod.build_expansion_brief = lambda **kwargs: {  # type: ignore[attr-defined]
        "reasoning_status": "not_run",
        "reasoning_warning": "Skipped in local query-plan inspection.",
        "adjacent_capabilities": [],
        "adjacent_customer_segments": [],
        "named_account_anchors": [],
        "geography_expansions": [],
    }


def _compact_query(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query.get("query_id"),
        "query_type": query.get("query_type"),
        "query_text": query.get("query_text"),
        "must_include_terms": query.get("must_include_terms") or [],
        "domain_allowlist": query.get("domain_allowlist") or [],
        "domain_blocklist": query.get("domain_blocklist") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to inspect")
    parser.add_argument(
        "--api-base",
        default="https://sourcing-os.up.railway.app",
        help="Base API URL for fetching the stored context pack",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every query object instead of the compact summary",
    )
    args = parser.parse_args()

    _stub_market_map_reasoning()
    context_pack_payload = _load_workspace_context(workspace_id=args.workspace_id, api_base=args.api_base)
    profile = _profile_from_context_pack(context_pack_payload, workspace_id=args.workspace_id)
    company_context_payload = build_company_context_artifacts(profile)
    primary_graph = build_primary_company_graph_from_context(profile, company_context_payload)
    queries = _build_secondary_queries(primary_graph, profile.comparator_seed_urls or [])

    settings = get_settings()
    output = {
        "workspace_id": args.workspace_id,
        "source_company": (company_context_payload.get("sourcing_brief") or {}).get("source_company") or {},
        "named_customers": [
            item.get("name")
            for item in ((company_context_payload.get("sourcing_brief") or {}).get("named_customer_proof") or [])
            if isinstance(item, dict) and item.get("name")
        ],
        "partners": [
            item.get("name")
            for item in ((company_context_payload.get("sourcing_brief") or {}).get("partner_integration_proof") or [])
            if isinstance(item, dict) and item.get("name")
        ],
        "secondary_budget": {
            "provider_order": [
                token.strip().lower()
                for token in str(settings.company_context_secondary_provider_order or "").split(",")
                if token.strip()
            ],
            "per_query_cap": int(settings.company_context_secondary_per_query_cap),
            "query_cap": int(settings.company_context_secondary_query_cap),
            "result_cap": int(settings.company_context_secondary_result_cap),
            "per_domain_cap": int(settings.company_context_secondary_per_domain_cap),
        },
        "query_count": len(queries),
        "query_type_counts": {
            query_type: sum(1 for query in queries if query.get("query_type") == query_type)
            for query_type in sorted({str(query.get("query_type") or "") for query in queries})
        },
        "queries": queries if args.full else [_compact_query(query) for query in queries],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
