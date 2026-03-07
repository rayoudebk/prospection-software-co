"""
Migration script for external search persistence tables.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect

from app.config import get_settings
from app.models.external_search import ExternalSearchRun, ExternalSearchResult

settings = get_settings()


def migrate_external_search_persistence_v1():
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        if "external_search_runs" not in tables:
            ExternalSearchRun.__table__.create(bind=conn)
        if "external_search_results" not in tables:
            ExternalSearchResult.__table__.create(bind=conn)


if __name__ == "__main__":
    print("Starting external search persistence migration...")
    migrate_external_search_persistence_v1()
    print("✅ Migration complete")
