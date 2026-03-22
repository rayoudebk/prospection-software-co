#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.models.company_context import CompanyContextPack
from app.models.workspace import CompanyProfile
from app.workers.workspace_tasks import (
    SessionLocal,
    _build_france_registry_universe_candidates,
    _normalize_scope_hints,
)
from app.services.company_context import derive_discovery_scope_hints


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded France registry-universe smoke pass.")
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to inspect.")
    parser.add_argument("--pages-per-code", type=int, default=2)
    parser.add_argument("--max-pages-per-code", type=int, default=2)
    parser.add_argument("--candidate-cap", type=int, default=200)
    parser.add_argument("--detail-cap", type=int, default=2)
    parser.add_argument("--search-timeout-seconds", type=int, default=2)
    parser.add_argument("--detail-timeout-seconds", type=int, default=2)
    parser.add_argument("--seed-query-cap", type=int, default=4)
    parser.add_argument("--secondary-query-cap", type=int, default=4)
    parser.add_argument("--max-total-queries", type=int, default=12)
    parser.add_argument("--max-elapsed-seconds", type=int, default=12)
    parser.add_argument("--page-extension-min-hits", type=int, default=2)
    parser.add_argument("--page-stop-after-no-signal", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    settings = get_settings()

    budget = {
        "pages_per_code": args.pages_per_code,
        "max_pages_per_code": args.max_pages_per_code,
        "candidate_cap": args.candidate_cap,
        "detail_cap": args.detail_cap,
        "search_timeout_seconds": args.search_timeout_seconds,
        "detail_timeout_seconds": args.detail_timeout_seconds,
        "seed_query_cap": args.seed_query_cap,
        "secondary_query_cap": args.secondary_query_cap,
        "max_total_queries": args.max_total_queries,
        "max_elapsed_seconds": args.max_elapsed_seconds,
        "page_extension_min_hits": args.page_extension_min_hits,
        "page_stop_after_no_signal": args.page_stop_after_no_signal,
    }

    db = SessionLocal()
    try:
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == args.workspace_id).first()
        pack = db.query(CompanyContextPack).filter(CompanyContextPack.workspace_id == args.workspace_id).first()
        if profile is None:
            print(json.dumps({"error": "workspace_profile_not_found", "workspace_id": args.workspace_id}))
            return 3

        scope = _normalize_scope_hints(derive_discovery_scope_hints(pack, profile) if pack and profile else {})
        started_at = time.time()
        candidates, diagnostics = _build_france_registry_universe_candidates(profile, scope, budget_overrides=budget)
        payload = {
            "workspace_id": args.workspace_id,
            "elapsed_seconds": round(time.time() - started_at, 2),
            "source_record": diagnostics.get("source_record"),
            "budget": diagnostics.get("budget"),
            "query_budget_exhausted": diagnostics.get("query_budget_exhausted"),
            "elapsed_budget_exhausted": diagnostics.get("elapsed_budget_exhausted"),
            "estimated_max_query_ceiling": diagnostics.get("estimated_max_query_ceiling"),
            "registry_raw_candidate_count": diagnostics.get("registry_raw_candidate_count"),
            "registry_scored_candidate_count": diagnostics.get("registry_scored_candidate_count"),
            "registry_candidate_count": diagnostics.get("registry_candidate_count"),
            "registry_queries_count": diagnostics.get("registry_queries_count"),
            "registry_seed_query_count": diagnostics.get("registry_seed_query_count"),
            "registry_secondary_query_count": diagnostics.get("registry_secondary_query_count"),
            "registry_source_path_counts": diagnostics.get("registry_source_path_counts"),
            "seed_query_specs": diagnostics.get("seed_query_specs"),
            "initial_secondary_query_specs": diagnostics.get("initial_secondary_query_specs"),
            "post_detail_secondary_query_specs": diagnostics.get("post_detail_secondary_query_specs"),
            "executed_lookup_attempts": diagnostics.get("executed_lookup_attempts"),
            "detail_stats": diagnostics.get("detail_stats"),
            "benchmark_hits": diagnostics.get("benchmark_hits"),
            "top_candidates": [
                {
                    "display_name": candidate.get("display_name"),
                    "legal_name": candidate.get("legal_name"),
                    "registry_id": candidate.get("registry_id"),
                    "directness": candidate.get("directness"),
                    "lookup_source_paths": (candidate.get("registry_fields") or {}).get("lookup_source_paths"),
                }
                for candidate in candidates[:10]
            ],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered)
        print(rendered)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
