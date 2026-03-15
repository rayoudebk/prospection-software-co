"""
Migration script for company-context backfill artifacts.

Creates:
- company_context_packs

Backfills missing company-context packs for existing workspaces from:
- company_profiles
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.company_context import CompanyContextPack
from app.models.workspace import CompanyProfile, Workspace
from app.services.company_context import build_company_context_artifacts

settings = get_settings()


def _ensure_company_context_tables(conn) -> None:
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if "company_context_packs" not in tables:
        CompanyContextPack.__table__.create(bind=conn)


def _apply_company_context_payload(
    company_context_pack: CompanyContextPack,
    payload: dict,
    *,
    preserve_existing: bool,
) -> bool:
    changed = False
    if not preserve_existing or not (company_context_pack.market_map_brief_json or {}):
        company_context_pack.market_map_brief_json = payload.get("market_map_brief") or {}
        changed = True
    if not preserve_existing or not (company_context_pack.expansion_brief_json or {}):
        company_context_pack.expansion_brief_json = payload.get("expansion_brief") or {}
        changed = True
    if not preserve_existing or not (company_context_pack.taxonomy_nodes_json or []):
        company_context_pack.taxonomy_nodes_json = payload.get("taxonomy_nodes") or []
        changed = True
    if not preserve_existing or not (company_context_pack.taxonomy_edges_json or []):
        company_context_pack.taxonomy_edges_json = payload.get("taxonomy_edges") or []
        changed = True
    if not preserve_existing or not (company_context_pack.lens_seeds_json or []):
        company_context_pack.lens_seeds_json = payload.get("lens_seeds") or []
        changed = True
    if not preserve_existing or not company_context_pack.generated_at:
        company_context_pack.generated_at = payload.get("generated_at")
        changed = True
    return changed


def backfill_company_context(session) -> dict[str, int]:
    counts = {
        "workspaces_seen": 0,
        "workspaces_skipped_missing_profile": 0,
        "company_context_packs_created": 0,
        "company_context_packs_updated": 0,
    }

    workspaces = session.query(Workspace).order_by(Workspace.id.asc()).all()
    for workspace in workspaces:
        counts["workspaces_seen"] += 1
        profile = workspace.company_profile
        if not profile or not isinstance(profile, CompanyProfile):
            counts["workspaces_skipped_missing_profile"] += 1
            continue

        bootstrap_payload = build_company_context_artifacts(profile)
        company_context_pack = workspace.company_context_pack
        if company_context_pack is None:
            company_context_pack = CompanyContextPack(
                workspace_id=workspace.id,
                market_map_brief_json=bootstrap_payload.get("market_map_brief") or {},
                expansion_brief_json=bootstrap_payload.get("expansion_brief") or {},
                taxonomy_nodes_json=bootstrap_payload.get("taxonomy_nodes") or [],
                taxonomy_edges_json=bootstrap_payload.get("taxonomy_edges") or [],
                lens_seeds_json=bootstrap_payload.get("lens_seeds") or [],
                generated_at=bootstrap_payload.get("generated_at"),
                confirmed_at=bootstrap_payload.get("confirmed_at"),
            )
            session.add(company_context_pack)
            session.flush()
            counts["company_context_packs_created"] += 1
        elif _apply_company_context_payload(company_context_pack, bootstrap_payload, preserve_existing=True):
            counts["company_context_packs_updated"] += 1

    session.commit()
    return counts


def migrate_company_context_backfill_v1() -> dict[str, int]:
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _ensure_company_context_tables(conn)

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with SessionLocal() as session:
        return backfill_company_context(session)


if __name__ == "__main__":
    print("Starting company-context backfill migration (v1)...")
    summary = migrate_company_context_backfill_v1()
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print("✅ Migration complete")
