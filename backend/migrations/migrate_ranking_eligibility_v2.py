"""
Migration script for ranking eligibility v2 backfills.

Backfills:
- vendor_mentions.entity_type from vendor URL pattern.
- vendor_screenings.ranking_eligible based on official website + classification.
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

from app.config import get_settings

settings = get_settings()


DIRECTORY_HOST_TOKENS = (
    "thewealthmosaic.com",
    "thewealthmosaic.co.uk",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
)


def _normalize_host(url: str | None) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    try:
        host = (urlparse(raw).netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_directory_host(url: str | None) -> bool:
    host = _normalize_host(url)
    if not host:
        return False
    return any(host == token or host.endswith(f".{token}") for token in DIRECTORY_HOST_TOKENS)


def _extract_solution_slug(url: str | None) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    try:
        parts = [p for p in (urlparse(raw).path or "").split("/") if p]
        if len(parts) >= 3 and parts[0] in {"vendor", "vendors"}:
            return parts[2].strip() or None
    except Exception:
        return None
    return None


def migrate_ranking_eligibility_v2():
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        mention_rows = conn.execute(
            text("SELECT id, company_url, profile_url, solution_slug, entity_type FROM vendor_mentions")
        ).mappings().all()
        for row in mention_rows:
            solution_slug = str(row.get("solution_slug") or "").strip() or _extract_solution_slug(
                row.get("profile_url") or row.get("company_url")
            )
            entity_type = "solution" if solution_slug else "company"
            conn.execute(
                text(
                    """
                    UPDATE vendor_mentions
                    SET solution_slug = COALESCE(solution_slug, :solution_slug),
                        entity_type = :entity_type
                    WHERE id = :id
                    """
                ),
                {
                    "id": row["id"],
                    "solution_slug": solution_slug,
                    "entity_type": entity_type,
                },
            )

        screening_rows = conn.execute(
            text(
                """
                SELECT id, decision_classification, candidate_official_website, candidate_entity_id, source_summary_json
                FROM vendor_screenings
                """
            )
        ).mappings().all()

        # Build entity type map once for fast checks.
        entity_rows = conn.execute(
            text("SELECT id, entity_type FROM candidate_entities")
        ).mappings().all()
        entity_type_by_id = {int(row["id"]): str(row.get("entity_type") or "company") for row in entity_rows}

        for row in screening_rows:
            classification = str(row.get("decision_classification") or "insufficient_evidence")
            official = str(row.get("candidate_official_website") or "").strip() or None
            entity_type = entity_type_by_id.get(int(row.get("candidate_entity_id") or 0), "company")
            source_summary = row.get("source_summary_json")
            if not isinstance(source_summary, dict):
                try:
                    source_summary = json.loads(source_summary or "{}")
                except Exception:
                    source_summary = {}
            source_counts = source_summary.get("source_type_counts") if isinstance(source_summary.get("source_type_counts"), dict) else {}
            has_tier1_proxy = bool(source_counts.get("first_party_website"))

            ranking_eligible = bool(
                entity_type == "company"
                and official
                and not _is_directory_host(official)
                and classification != "not_good_target"
                and has_tier1_proxy
            )

            conn.execute(
                text(
                    """
                    UPDATE vendor_screenings
                    SET ranking_eligible = :ranking_eligible
                    WHERE id = :id
                    """
                ),
                {"id": row["id"], "ranking_eligible": ranking_eligible},
            )


if __name__ == "__main__":
    print("Starting ranking eligibility migration (v2)...")
    migrate_ranking_eligibility_v2()
    print("✅ Migration complete")

