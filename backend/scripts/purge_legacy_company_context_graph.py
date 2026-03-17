#!/usr/bin/env python3
"""Purge a legacy unscoped company-context graph projection from Neo4j.

Use this only to clean historical projections written before graph namespaces
were introduced. It deletes nodes where `graph_ref` matches the legacy
workspace/company ref and `graph_namespace` is empty.
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.services.company_context_graph import (
    Neo4jCompanyContextGraphStore,
    legacy_company_context_graph_ref,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge a legacy unscoped Neo4j company-context graph projection.")
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id used by the legacy graph ref.")
    parser.add_argument(
        "--company-name",
        type=str,
        required=True,
        help="Company name slug source used by the legacy graph ref, for example '4TPM' or 'Hublo'.",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = Neo4jCompanyContextGraphStore()
    legacy_ref = legacy_company_context_graph_ref(args.workspace_id, args.company_name)
    result = store.purge_workspace_projection(
        graph_namespace="__unused__",
        workspace_id=args.workspace_id,
        legacy_graph_ref=legacy_ref,
        include_legacy_graph_ref=True,
    )
    result["graph_namespace"] = settings.graph_namespace or ""
    result["legacy_graph_ref"] = legacy_ref
    print(result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
