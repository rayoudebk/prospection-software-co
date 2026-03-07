"""
Migration script for thesis-first sourcing artifacts.

Creates:
- buyer_thesis_packs
- search_lanes

Backfills missing thesis packs and search lanes for existing workspaces from:
- company_profiles
- brick_taxonomies
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.thesis import BuyerThesisPack, SearchLane
from app.models.workspace import BrickTaxonomy, CompanyProfile, Workspace
from app.services.thesis import bootstrap_thesis_payload, derive_search_lane_payloads

settings = get_settings()


def _ensure_thesis_tables(conn) -> None:
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if "buyer_thesis_packs" not in tables:
        BuyerThesisPack.__table__.create(bind=conn)
    if "search_lanes" not in tables:
        SearchLane.__table__.create(bind=conn)


def _apply_thesis_payload(
    thesis_pack: BuyerThesisPack,
    payload: dict,
    *,
    preserve_existing: bool,
) -> bool:
    changed = False
    if not preserve_existing or not thesis_pack.summary:
        thesis_pack.summary = payload.get("summary")
        changed = True
    if not preserve_existing or not (thesis_pack.claims_json or []):
        thesis_pack.claims_json = payload.get("claims") or []
        changed = True
    if not preserve_existing or not (thesis_pack.source_pills_json or []):
        thesis_pack.source_pills_json = payload.get("source_pills") or []
        changed = True
    if not preserve_existing or not (thesis_pack.open_questions_json or []):
        thesis_pack.open_questions_json = payload.get("open_questions") or []
        changed = True
    if not preserve_existing or not thesis_pack.generated_at:
        thesis_pack.generated_at = payload.get("generated_at")
        changed = True
    return changed


def _apply_lane_payload(
    lane: SearchLane,
    payload: dict,
    *,
    preserve_existing: bool,
) -> bool:
    changed = False
    for attr, key in [
        ("title", "title"),
        ("intent", "intent"),
        ("capabilities_json", "capabilities"),
        ("customer_tags_json", "customer_tags"),
        ("must_include_terms_json", "must_include_terms"),
        ("must_exclude_terms_json", "must_exclude_terms"),
        ("seed_urls_json", "seed_urls"),
    ]:
        current_value = getattr(lane, attr)
        next_value = payload.get(key)
        if preserve_existing and current_value not in (None, [], "", {}):
            continue
        setattr(lane, attr, next_value)
        changed = True
    if not preserve_existing or not lane.status:
        lane.status = payload.get("status") or "draft"
        changed = True
    if (not preserve_existing or not lane.confirmed_at) and (payload.get("status") == "confirmed"):
        lane.confirmed_at = payload.get("confirmed_at")
        changed = True
    return changed


def backfill_thesis_sourcing(session) -> dict[str, int]:
    counts = {
        "workspaces_seen": 0,
        "workspaces_skipped_missing_profile": 0,
        "thesis_packs_created": 0,
        "thesis_packs_updated": 0,
        "search_lanes_created": 0,
        "search_lanes_updated": 0,
    }

    workspaces = session.query(Workspace).order_by(Workspace.id.asc()).all()
    for workspace in workspaces:
        counts["workspaces_seen"] += 1
        profile = workspace.company_profile
        taxonomy = workspace.brick_taxonomy
        if not profile or not isinstance(profile, CompanyProfile):
            counts["workspaces_skipped_missing_profile"] += 1
            continue
        if taxonomy is not None and not isinstance(taxonomy, BrickTaxonomy):
            taxonomy = None

        bootstrap_payload = bootstrap_thesis_payload(profile, taxonomy)
        thesis_pack = workspace.thesis_pack
        if thesis_pack is None:
            thesis_pack = BuyerThesisPack(
                workspace_id=workspace.id,
                summary=bootstrap_payload.get("summary"),
                claims_json=bootstrap_payload.get("claims") or [],
                source_pills_json=bootstrap_payload.get("source_pills") or [],
                open_questions_json=bootstrap_payload.get("open_questions") or [],
                generated_at=bootstrap_payload.get("generated_at"),
                confirmed_at=bootstrap_payload.get("confirmed_at"),
            )
            session.add(thesis_pack)
            session.flush()
            counts["thesis_packs_created"] += 1
        elif _apply_thesis_payload(thesis_pack, bootstrap_payload, preserve_existing=True):
            counts["thesis_packs_updated"] += 1

        existing_by_type = {lane.lane_type: lane for lane in (workspace.search_lanes or []) if lane.lane_type}
        lane_payloads = derive_search_lane_payloads(thesis_pack, profile, taxonomy)
        for lane_payload in lane_payloads:
            lane_type = str(lane_payload.get("lane_type") or "").strip().lower()
            if lane_type not in {"core", "adjacent"}:
                continue
            lane = existing_by_type.get(lane_type)
            if lane is None:
                lane = SearchLane(
                    workspace_id=workspace.id,
                    lane_type=lane_type,
                    title=lane_payload.get("title") or f"{lane_type.title()} sourcing lane",
                    intent=lane_payload.get("intent"),
                    capabilities_json=lane_payload.get("capabilities") or [],
                    customer_tags_json=lane_payload.get("customer_tags") or [],
                    must_include_terms_json=lane_payload.get("must_include_terms") or [],
                    must_exclude_terms_json=lane_payload.get("must_exclude_terms") or [],
                    seed_urls_json=lane_payload.get("seed_urls") or [],
                    status=lane_payload.get("status") or "draft",
                    confirmed_at=lane_payload.get("confirmed_at"),
                )
                session.add(lane)
                existing_by_type[lane_type] = lane
                counts["search_lanes_created"] += 1
                continue
            if _apply_lane_payload(lane, lane_payload, preserve_existing=True):
                counts["search_lanes_updated"] += 1

    session.commit()
    return counts


def migrate_thesis_sourcing_v1() -> dict[str, int]:
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        _ensure_thesis_tables(conn)

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with SessionLocal() as session:
        return backfill_thesis_sourcing(session)


if __name__ == "__main__":
    print("Starting thesis sourcing migration (v1)...")
    summary = migrate_thesis_sourcing_v1()
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print("✅ Migration complete")
