#!/usr/bin/env python3
"""Run Gemini vs OpenAI expansion-brief generation locally for the same workspace context."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any

import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings  # noqa: E402
from app.models.workspace import CompanyProfile  # noqa: E402
from app.services.company_context import build_expansion_artifacts  # noqa: E402


DEFAULT_ROUTES = {
    "gemini": (
        "gemini:deep-research-pro-preview-12-2025"
    ),
    "openai": (
        "openai:o4-mini-deep-research"
    ),
}


def _load_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def _load_workspace_payloads(*, workspace_id: int, api_base: str) -> tuple[dict[str, Any], dict[str, Any]]:
    context_pack_url = f"{api_base.rstrip('/')}/workspaces/{workspace_id}/context-pack"
    company_context_url = f"{api_base.rstrip('/')}/workspaces/{workspace_id}/company-context"
    return _load_json(context_pack_url), _load_json(company_context_url)


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


@contextmanager
def _temporary_expansion_route(route: str):
    previous = os.environ.get("LLM_STAGE_EXPANSION_MODELS")
    os.environ["LLM_STAGE_EXPANSION_MODELS"] = route
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LLM_STAGE_EXPANSION_MODELS", None)
        else:
            os.environ["LLM_STAGE_EXPANSION_MODELS"] = previous
        get_settings.cache_clear()


@contextmanager
def _temporary_expansion_timeout(timeout_seconds: int):
    previous = os.environ.get("STAGE_EXPANSION_RESEARCH_TIMEOUT_SECONDS")
    os.environ["STAGE_EXPANSION_RESEARCH_TIMEOUT_SECONDS"] = str(timeout_seconds)
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("STAGE_EXPANSION_RESEARCH_TIMEOUT_SECONDS", None)
        else:
            os.environ["STAGE_EXPANSION_RESEARCH_TIMEOUT_SECONDS"] = previous
        get_settings.cache_clear()


def _compact_summary(expansion_brief: dict[str, Any]) -> dict[str, Any]:
    return {
        "reasoning_status": expansion_brief.get("reasoning_status"),
        "reasoning_provider": expansion_brief.get("reasoning_provider"),
        "reasoning_model": expansion_brief.get("reasoning_model"),
        "adjacency_boxes": [
            {
                "label": item.get("label"),
                "adjacency_kind": item.get("adjacency_kind"),
                "priority_tier": item.get("priority_tier"),
                "workflow_criticality": ((item.get("criticality") or {}).get("workflow_criticality")),
                "switching_cost_intensity": ((item.get("criticality") or {}).get("switching_cost_intensity")),
                "company_seed_ids": item.get("company_seed_ids") or [],
            }
            for item in (expansion_brief.get("adjacency_boxes") or [])
        ],
        "company_seeds": [
            {
                "name": item.get("name"),
                "seed_type": item.get("seed_type"),
                "fit_to_adjacency_box_ids": item.get("fit_to_adjacency_box_ids") or [],
            }
            for item in (expansion_brief.get("company_seeds") or [])
        ],
        "named_account_anchors": [item.get("label") for item in (expansion_brief.get("named_account_anchors") or [])],
        "geography_expansions": [item.get("label") for item in (expansion_brief.get("geography_expansions") or [])],
    }


def _run_variant(
    *,
    provider_name: str,
    route: str,
    timeout_seconds: int,
    profile: CompanyProfile,
    sourcing_brief: dict[str, Any],
    taxonomy_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    with _temporary_expansion_route(route), _temporary_expansion_timeout(timeout_seconds):
        payload = build_expansion_artifacts(
            profile,
            sourcing_brief=sourcing_brief,
            taxonomy_nodes=taxonomy_nodes,
        )
    brief = payload.get("expansion_brief") or {}
    return {
        "provider_variant": provider_name,
        "route": route,
        "generated_at": payload.get("generated_at"),
        "expansion_brief": brief,
        "summary": _compact_summary(brief),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to evaluate")
    parser.add_argument(
        "--api-base",
        default="https://sourcing-os.up.railway.app",
        help="Base API URL for fetching context-pack and company-context payloads",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/expansion-ab",
        help="Directory where comparison outputs should be written",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Research timeout to use for each provider variant",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "both"],
        default="both",
        help="Which provider variant to run",
    )
    args = parser.parse_args()

    context_pack_payload, company_context_payload = _load_workspace_payloads(
        workspace_id=args.workspace_id,
        api_base=args.api_base,
    )
    profile = _profile_from_context_pack(context_pack_payload, workspace_id=args.workspace_id)
    sourcing_brief = company_context_payload.get("sourcing_brief") or {}
    taxonomy_nodes = company_context_payload.get("taxonomy_nodes") or []
    if not sourcing_brief or not taxonomy_nodes:
        raise SystemExit("Workspace is missing sourcing_brief or taxonomy_nodes; refresh company context first.")

    output_dir = Path(args.output_dir).expanduser().resolve() / f"workspace-{args.workspace_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    gemini_result: dict[str, Any] | None = None
    openai_result: dict[str, Any] | None = None

    if args.provider in {"gemini", "both"}:
        gemini_result = _run_variant(
            provider_name="gemini",
            route=DEFAULT_ROUTES["gemini"],
            timeout_seconds=args.timeout_seconds,
            profile=profile,
            sourcing_brief=sourcing_brief,
            taxonomy_nodes=taxonomy_nodes,
        )
        _write_json(output_dir / "gemini-expansion.json", gemini_result)

    if args.provider in {"openai", "both"}:
        openai_result = _run_variant(
            provider_name="openai",
            route=DEFAULT_ROUTES["openai"],
            timeout_seconds=args.timeout_seconds,
            profile=profile,
            sourcing_brief=sourcing_brief,
            taxonomy_nodes=taxonomy_nodes,
        )
        _write_json(output_dir / "openai-expansion.json", openai_result)

    comparison = {
        "workspace_id": args.workspace_id,
        "api_base": args.api_base,
        "provider": args.provider,
        "outputs": {
            "gemini": {
                "file": str((output_dir / "gemini-expansion.json")),
                "reasoning_provider": (((gemini_result or {}).get("summary") or {}).get("reasoning_provider")),
                "reasoning_model": (((gemini_result or {}).get("summary") or {}).get("reasoning_model")),
            },
            "openai": {
                "file": str((output_dir / "openai-expansion.json")),
                "reasoning_provider": (((openai_result or {}).get("summary") or {}).get("reasoning_provider")),
                "reasoning_model": (((openai_result or {}).get("summary") or {}).get("reasoning_model")),
            },
        },
        "gemini_summary": (gemini_result or {}).get("summary") or {},
        "openai_summary": (openai_result or {}).get("summary") or {},
    }

    _write_json(output_dir / "comparison.json", comparison)

    print(json.dumps(comparison, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
