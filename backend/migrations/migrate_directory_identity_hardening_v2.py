"""
Migration script for directory identity hardening v2.

Adds and backfills:
- vendor_mentions: profile_url, official_website_url, company_slug, solution_slug, entity_type
- candidate_entities: entity_type, first_party_domains_json, solutions_json, discovery_primary_url
- vendor_screenings: candidate_discovery_url, candidate_official_website, top_claim_json, ranking_eligible
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings

settings = get_settings()


DIRECTORY_HOST_TOKENS = (
    "thewealthmosaic.com",
    "thewealthmosaic.co.uk",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
)


def _add_column_if_missing(conn, table_name: str, column_sql: str, column_name: str) -> None:
    cols = {c["name"] for c in inspect(conn).get_columns(table_name)}
    if column_name in cols:
        print(f"Column already exists: {table_name}.{column_name}")
        return
    print(f"Adding column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))


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


def _extract_vendor_slugs(url: str | None) -> tuple[str | None, str | None]:
    raw = str(url or "").strip()
    if not raw:
        return None, None
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    try:
        parts = [p for p in (urlparse(raw).path or "").split("/") if p]
        if len(parts) >= 2 and parts[0] in {"vendor", "vendors"}:
            company_slug = parts[1].strip() or None
            solution_slug = parts[2].strip() if len(parts) >= 3 else None
            return company_slug, solution_slug or None
    except Exception:
        return None, None
    return None, None


def migrate_directory_identity_hardening_v2():
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        # vendor_mentions
        _add_column_if_missing(conn, "vendor_mentions", "profile_url VARCHAR(1000)", "profile_url")
        _add_column_if_missing(conn, "vendor_mentions", "official_website_url VARCHAR(1000)", "official_website_url")
        _add_column_if_missing(conn, "vendor_mentions", "company_slug VARCHAR(180)", "company_slug")
        _add_column_if_missing(conn, "vendor_mentions", "solution_slug VARCHAR(220)", "solution_slug")
        _add_column_if_missing(conn, "vendor_mentions", "entity_type VARCHAR(32)", "entity_type")

        # candidate_entities
        _add_column_if_missing(conn, "candidate_entities", "discovery_primary_url VARCHAR(1000)", "discovery_primary_url")
        _add_column_if_missing(conn, "candidate_entities", "entity_type VARCHAR(32)", "entity_type")
        _add_column_if_missing(conn, "candidate_entities", "first_party_domains_json JSON", "first_party_domains_json")
        _add_column_if_missing(conn, "candidate_entities", "solutions_json JSON", "solutions_json")

        # vendor_screenings
        _add_column_if_missing(conn, "vendor_screenings", "candidate_discovery_url VARCHAR(1000)", "candidate_discovery_url")
        _add_column_if_missing(conn, "vendor_screenings", "candidate_official_website VARCHAR(1000)", "candidate_official_website")
        _add_column_if_missing(conn, "vendor_screenings", "top_claim_json JSON", "top_claim_json")
        _add_column_if_missing(conn, "vendor_screenings", "ranking_eligible BOOLEAN", "ranking_eligible")

        # vendor_mentions backfill
        rows = conn.execute(
            text(
                """
                SELECT id, listing_url, company_url, profile_url, official_website_url, company_slug, solution_slug, entity_type
                FROM vendor_mentions
                """
            )
        ).mappings().all()
        for row in rows:
            listing_url = str(row.get("listing_url") or "").strip() or None
            company_url = str(row.get("company_url") or "").strip() or None
            profile_url = str(row.get("profile_url") or "").strip() or None
            official = str(row.get("official_website_url") or "").strip() or None
            company_slug = str(row.get("company_slug") or "").strip() or None
            solution_slug = str(row.get("solution_slug") or "").strip() or None
            entity_type = str(row.get("entity_type") or "").strip() or None

            profile_url = profile_url or company_url or listing_url
            if not company_slug and not solution_slug:
                inferred_company_slug, inferred_solution_slug = _extract_vendor_slugs(profile_url or company_url)
                company_slug = company_slug or inferred_company_slug
                solution_slug = solution_slug or inferred_solution_slug

            if not official and company_url and not _is_directory_host(company_url):
                official = company_url
            if official and _is_directory_host(official):
                official = None

            if not entity_type:
                entity_type = "solution" if solution_slug else "company"

            conn.execute(
                text(
                    """
                    UPDATE vendor_mentions
                    SET profile_url = :profile_url,
                        official_website_url = :official_website_url,
                        company_slug = :company_slug,
                        solution_slug = :solution_slug,
                        entity_type = :entity_type
                    WHERE id = :id
                    """
                ),
                {
                    "id": row["id"],
                    "profile_url": profile_url,
                    "official_website_url": official,
                    "company_slug": company_slug,
                    "solution_slug": solution_slug,
                    "entity_type": entity_type,
                },
            )

        # candidate_entities backfill
        entity_rows = conn.execute(
            text(
                """
                SELECT id, canonical_website, canonical_domain, discovery_primary_url, entity_type, first_party_domains_json, solutions_json
                FROM candidate_entities
                """
            )
        ).mappings().all()
        for row in entity_rows:
            website = str(row.get("canonical_website") or "").strip() or None
            domain = str(row.get("canonical_domain") or "").strip() or _normalize_host(website) or None
            discovery_primary_url = str(row.get("discovery_primary_url") or "").strip() or None
            entity_type = str(row.get("entity_type") or "").strip() or "company"

            first_party_domains = row.get("first_party_domains_json")
            if not isinstance(first_party_domains, list):
                first_party_domains = []
            if domain and not _is_directory_host(website or domain):
                if domain not in first_party_domains:
                    first_party_domains.append(domain)

            solutions = row.get("solutions_json")
            if not isinstance(solutions, list):
                solutions = []

            conn.execute(
                text(
                    """
                    UPDATE candidate_entities
                    SET canonical_domain = COALESCE(canonical_domain, :canonical_domain),
                        discovery_primary_url = COALESCE(discovery_primary_url, :discovery_primary_url),
                        entity_type = COALESCE(NULLIF(entity_type, ''), :entity_type),
                        first_party_domains_json = :first_party_domains_json,
                        solutions_json = :solutions_json
                    WHERE id = :id
                    """
                ),
                {
                    "id": row["id"],
                    "canonical_domain": domain,
                    "discovery_primary_url": discovery_primary_url,
                    "entity_type": entity_type,
                    "first_party_domains_json": json.dumps(first_party_domains),
                    "solutions_json": json.dumps(solutions),
                },
            )

        # vendor_screenings backfill
        screening_rows = conn.execute(
            text(
                """
                SELECT id, candidate_website, candidate_discovery_url, candidate_official_website, top_claim_json, ranking_eligible, screening_meta_json
                FROM vendor_screenings
                """
            )
        ).mappings().all()
        for row in screening_rows:
            meta = row.get("screening_meta_json")
            if not isinstance(meta, dict):
                meta = {}
            identity = meta.get("identity") if isinstance(meta.get("identity"), dict) else {}
            candidate_website = str(row.get("candidate_website") or "").strip() or None
            candidate_discovery_url = str(row.get("candidate_discovery_url") or "").strip() or None
            candidate_official_website = str(row.get("candidate_official_website") or "").strip() or None

            if not candidate_discovery_url:
                candidate_discovery_url = (
                    str(identity.get("input_website") or "").strip()
                    or str(meta.get("input_website") or "").strip()
                    or None
                )
            if not candidate_official_website:
                candidate_official_website = str(identity.get("official_website") or "").strip() or candidate_website
            if candidate_official_website and _is_directory_host(candidate_official_website):
                candidate_official_website = None

            top_claim = row.get("top_claim_json")
            if not isinstance(top_claim, dict):
                top_claim = {}
            ranking_eligible = bool(row.get("ranking_eligible") or False)

            conn.execute(
                text(
                    """
                    UPDATE vendor_screenings
                    SET candidate_discovery_url = COALESCE(candidate_discovery_url, :candidate_discovery_url),
                        candidate_official_website = :candidate_official_website,
                        top_claim_json = :top_claim_json,
                        ranking_eligible = COALESCE(ranking_eligible, :ranking_eligible)
                    WHERE id = :id
                    """
                ),
                {
                    "id": row["id"],
                    "candidate_discovery_url": candidate_discovery_url,
                    "candidate_official_website": candidate_official_website,
                    "top_claim_json": json.dumps(top_claim),
                    "ranking_eligible": ranking_eligible,
                },
            )


if __name__ == "__main__":
    print("Starting directory identity hardening migration (v2)...")
    migrate_directory_identity_hardening_v2()
    print("✅ Migration complete")

