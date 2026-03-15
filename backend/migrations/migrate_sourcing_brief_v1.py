"""
Migration script for sourcing-brief artifacts.

Maintains compatibility with older schemas that may still have
`market_map_brief_json` before company-context storage renames run.
"""
from __future__ import annotations

from migrations.migrate_market_map_brief_v1 import migrate_market_map_brief_v1


def migrate_sourcing_brief_v1(database_url: str | None = None) -> None:
    migrate_market_map_brief_v1(database_url=database_url)


if __name__ == "__main__":
    print("Starting sourcing brief migration (v1)...")
    migrate_sourcing_brief_v1()
    print("✅ Migration complete")
