"""Drop the legacy search_lanes table now that scope review is canonical."""
from __future__ import annotations

from sqlalchemy import create_engine, inspect, text


def migrate_remove_search_lanes_v1(*, database_url: str) -> dict[str, bool]:
    engine = create_engine(database_url, future=True)
    dropped = False
    with engine.begin() as conn:
        inspector = inspect(conn)
        if "search_lanes" in set(inspector.get_table_names()):
            conn.execute(text("DROP TABLE IF EXISTS search_lanes"))
            dropped = True
    return {"search_lanes_dropped": dropped}
