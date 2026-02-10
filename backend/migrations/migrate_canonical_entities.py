"""
Migration script to add canonical discovery entity tables and screening linkage.

This script:
1. Creates candidate_entities, candidate_entity_aliases, candidate_origin_edges, registry_query_logs tables
2. Adds candidate_entity_id column to vendor_screenings (if missing)
3. Adds supporting index on vendor_screenings.candidate_entity_id
4. Adds foreign key constraint to candidate_entities on PostgreSQL (best effort)

Run with:
    python -m migrations.migrate_canonical_entities
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.models.intelligence import (
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    RegistryQueryLog,
)

settings = get_settings()


def migrate_canonical_entities():
    """Create canonical entity tables and link vendor screenings to them."""
    engine = create_engine(settings.database_url_sync, echo=True)
    inspector = inspect(engine)

    with engine.begin() as conn:
        existing_tables = set(inspector.get_table_names())

        if "candidate_entities" not in existing_tables:
            print("Creating table: candidate_entities")
            CandidateEntity.__table__.create(bind=conn)
        else:
            print("Table already exists: candidate_entities")

        if "candidate_entity_aliases" not in existing_tables:
            print("Creating table: candidate_entity_aliases")
            CandidateEntityAlias.__table__.create(bind=conn)
        else:
            print("Table already exists: candidate_entity_aliases")

        if "candidate_origin_edges" not in existing_tables:
            print("Creating table: candidate_origin_edges")
            CandidateOriginEdge.__table__.create(bind=conn)
        else:
            print("Table already exists: candidate_origin_edges")

        if "registry_query_logs" not in existing_tables:
            print("Creating table: registry_query_logs")
            RegistryQueryLog.__table__.create(bind=conn)
        else:
            print("Table already exists: registry_query_logs")

        screening_columns = {col["name"] for col in inspector.get_columns("vendor_screenings")}
        if "candidate_entity_id" not in screening_columns:
            print("Adding column: vendor_screenings.candidate_entity_id")
            conn.execute(text("ALTER TABLE vendor_screenings ADD COLUMN candidate_entity_id INTEGER"))
        else:
            print("Column already exists: vendor_screenings.candidate_entity_id")

        print("Ensuring index: idx_vendor_screenings_candidate_entity_id")
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_vendor_screenings_candidate_entity_id "
                "ON vendor_screenings (candidate_entity_id)"
            )
        )

        if engine.dialect.name == "postgresql":
            print("Ensuring FK: vendor_screenings(candidate_entity_id) -> candidate_entities(id)")
            try:
                conn.execute(
                    text(
                        "ALTER TABLE vendor_screenings "
                        "ADD CONSTRAINT fk_vendor_screenings_candidate_entity_id "
                        "FOREIGN KEY (candidate_entity_id) REFERENCES candidate_entities(id)"
                    )
                )
            except SQLAlchemyError:
                # Constraint may already exist; keep migration idempotent.
                print("Foreign key already exists or could not be added; skipping.")


if __name__ == "__main__":
    print("Starting canonical entity migration...")
    print("=" * 60)
    migrate_canonical_entities()
    print("✅ Migration complete")
    print("=" * 60)
