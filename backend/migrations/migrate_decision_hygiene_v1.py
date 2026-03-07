"""
Migration script for decision hygiene v1 fields.

Adds:
- workspace_evidence source tier/kind/freshness columns
- vendor_claims claim-group/tier/contradiction/freshness columns
- vendor_screenings decision classification/rationale columns

Also backfills:
- workspace_evidence.source_tier/source_kind from source_url and content_type
- vendor_screenings.decision_classification from screening_status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings

settings = get_settings()


def _add_column_if_missing(conn, table_name: str, column_sql: str, column_name: str) -> None:
    # Re-inspect against the live connection so repeated calls in one transaction see newly added columns.
    cols = {c["name"] for c in inspect(conn).get_columns(table_name)}
    if column_name in cols:
        print(f"Column already exists: {table_name}.{column_name}")
        return
    print(f"Adding column: {table_name}.{column_name}")
    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))


def migrate_decision_hygiene_v1():
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        # workspace_evidence
        _add_column_if_missing(conn, "workspace_evidence", "retrieved_at TIMESTAMP", "retrieved_at")
        _add_column_if_missing(conn, "workspace_evidence", "freshness_ttl_days INTEGER", "freshness_ttl_days")
        _add_column_if_missing(conn, "workspace_evidence", "valid_through TIMESTAMP", "valid_through")
        _add_column_if_missing(conn, "workspace_evidence", "source_tier VARCHAR(32)", "source_tier")
        _add_column_if_missing(conn, "workspace_evidence", "source_kind VARCHAR(32)", "source_kind")
        _add_column_if_missing(conn, "workspace_evidence", "asserted_by VARCHAR(120)", "asserted_by")

        # vendor_claims
        _add_column_if_missing(conn, "vendor_claims", "source_tier VARCHAR(32)", "source_tier")
        _add_column_if_missing(conn, "vendor_claims", "source_evidence_id INTEGER", "source_evidence_id")
        _add_column_if_missing(conn, "vendor_claims", "claim_group VARCHAR(64)", "claim_group")
        _add_column_if_missing(conn, "vendor_claims", "claim_status VARCHAR(24)", "claim_status")
        _add_column_if_missing(conn, "vendor_claims", "contradiction_group_id VARCHAR(120)", "contradiction_group_id")
        _add_column_if_missing(conn, "vendor_claims", "freshness_ttl_days INTEGER", "freshness_ttl_days")
        _add_column_if_missing(conn, "vendor_claims", "valid_through TIMESTAMP", "valid_through")

        # vendor_screenings
        _add_column_if_missing(conn, "vendor_screenings", "positive_reason_codes_json JSON", "positive_reason_codes_json")
        _add_column_if_missing(conn, "vendor_screenings", "caution_reason_codes_json JSON", "caution_reason_codes_json")
        _add_column_if_missing(conn, "vendor_screenings", "reject_reason_codes_json JSON", "reject_reason_codes_json")
        _add_column_if_missing(conn, "vendor_screenings", "missing_claim_groups_json JSON", "missing_claim_groups_json")
        _add_column_if_missing(conn, "vendor_screenings", "unresolved_contradictions_count INTEGER", "unresolved_contradictions_count")
        _add_column_if_missing(conn, "vendor_screenings", "decision_classification VARCHAR(40)", "decision_classification")
        _add_column_if_missing(conn, "vendor_screenings", "evidence_sufficiency VARCHAR(24)", "evidence_sufficiency")
        _add_column_if_missing(conn, "vendor_screenings", "rationale_summary TEXT", "rationale_summary")
        _add_column_if_missing(conn, "vendor_screenings", "rationale_markdown TEXT", "rationale_markdown")
        _add_column_if_missing(conn, "vendor_screenings", "decision_engine_version VARCHAR(64)", "decision_engine_version")
        _add_column_if_missing(conn, "vendor_screenings", "gating_passed BOOLEAN", "gating_passed")

        # Backfill evidence defaults
        conn.execute(
            text(
                """
                UPDATE workspace_evidence
                SET source_tier = CASE
                    WHEN source_url LIKE '%companieshouse.gov.uk%' OR source_url LIKE '%company-information.service.gov.uk%'
                      OR source_url LIKE '%pappers.fr%' OR source_url LIKE '%infogreffe.fr%'
                      OR source_url LIKE '%inpi.fr%' OR source_url LIKE '%annuaire-entreprises.data.gouv.fr%'
                      OR source_url LIKE '%handelsregister.de%' OR source_url LIKE '%unternehmensregister.de%'
                      OR source_url LIKE '%gleif.org%' THEN 'tier0_registry'
                    WHEN source_url LIKE '%thewealthmosaic.com%' THEN 'tier4_discovery'
                    ELSE 'tier3_third_party'
                END
                WHERE source_tier IS NULL OR source_tier = ''
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE workspace_evidence
                SET source_kind = CASE
                    WHEN source_tier = 'tier0_registry' THEN 'registry'
                    WHEN source_tier = 'tier4_discovery' THEN 'directory'
                    WHEN content_type IN ('case_study') THEN 'customer_partner'
                    ELSE 'third_party'
                END
                WHERE source_kind IS NULL OR source_kind = ''
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE workspace_evidence
                SET retrieved_at = COALESCE(retrieved_at, captured_at)
                """
            )
        )

        # Backfill screening classification
        conn.execute(
            text(
                """
                UPDATE vendor_screenings
                SET decision_classification = CASE
                    WHEN screening_status = 'kept' THEN 'good_target'
                    WHEN screening_status = 'review' THEN 'borderline_watchlist'
                    WHEN screening_status = 'rejected' THEN 'not_good_target'
                    ELSE 'insufficient_evidence'
                END
                WHERE decision_classification IS NULL OR decision_classification = ''
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE vendor_screenings
                SET evidence_sufficiency = COALESCE(NULLIF(evidence_sufficiency, ''), 'insufficient'),
                    unresolved_contradictions_count = COALESCE(unresolved_contradictions_count, 0),
                    gating_passed = COALESCE(gating_passed, FALSE)
                """
            )
        )


if __name__ == "__main__":
    print("Starting decision hygiene v1 migration...")
    migrate_decision_hygiene_v1()
    print("✅ Migration complete")
