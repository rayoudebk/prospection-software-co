"""Safe startup migrations for the currently deployed workspace schema."""

from app.config import get_settings
from migrations.migrate_company_context_graph_v1 import migrate_company_context_graph_v1
from migrations.migrate_company_context_storage_v1 import migrate_company_context_storage_v1
from migrations.migrate_company_profile_reference_evidence_v1 import (
    migrate_company_profile_reference_evidence_v1,
)
from migrations.migrate_company_profile_context_pack_v1 import (
    migrate_company_profile_context_pack_v1,
)
from migrations.migrate_company_profile_context_split_v1 import (
    migrate_company_profile_context_split_v1,
)
from migrations.migrate_expansion_brief_v1 import migrate_expansion_brief_v1
from migrations.migrate_expansion_generation_v1 import migrate_expansion_generation_v1
from migrations.migrate_remove_legacy_buyer_context_summary_v1 import (
    migrate_remove_legacy_buyer_context_summary_v1,
)
from migrations.migrate_remove_company_context_bridge_fields_v1 import (
    migrate_remove_company_context_bridge_fields_v1,
)
from migrations.migrate_remove_company_profile_brief_fields_v1 import (
    migrate_remove_company_profile_brief_fields_v1,
)
from migrations.migrate_sourcing_brief_v1 import migrate_sourcing_brief_v1
from migrations.migrate_remove_search_lanes_v1 import migrate_remove_search_lanes_v1
from migrations.migrate_workspace_policy_v1 import migrate_workspace_policy_v1


def run_startup_migrations(database_url: str | None = None) -> None:
    """Apply idempotent migrations needed by the current workspace API."""
    database_url = database_url or get_settings().database_url_sync
    migrate_workspace_policy_v1(database_url=database_url)
    migrate_company_profile_reference_evidence_v1(database_url=database_url)
    migrate_company_profile_context_pack_v1(database_url=database_url)
    migrate_company_profile_context_split_v1(database_url=database_url)
    migrate_remove_legacy_buyer_context_summary_v1(database_url=database_url)
    migrate_sourcing_brief_v1(database_url=database_url)
    migrate_expansion_brief_v1(database_url=database_url)
    migrate_expansion_generation_v1(database_url=database_url)
    migrate_company_context_graph_v1(database_url=database_url)
    migrate_company_context_storage_v1(database_url=database_url)
    migrate_remove_company_context_bridge_fields_v1(database_url=database_url)
    migrate_remove_company_profile_brief_fields_v1(database_url=database_url)
    migrate_remove_search_lanes_v1(database_url=database_url)
