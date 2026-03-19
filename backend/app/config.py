from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Tuple
from urllib.parse import urlparse
import re


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/prospection"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/prospection"
    graph_namespace: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_queue_namespace: str = ""

    # Gemini
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Optional external retrieval connectors
    tavily_api_key: str = ""
    exa_api_key: str = ""
    serper_api_key: str = ""
    serpapi_api_key: str = ""
    firecrawl_api_key: str = ""
    jina_api_key: str = ""
    brave_api_key: str = ""

    # Optional registry API keys
    companies_house_api_key: str = ""

    # Neo4j company-context graph
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # LLM model routing (provider:model, comma-separated)
    llm_stage_discovery_models: str = "gemini:gemini-2.0-flash,openai:gpt-4.1-mini,anthropic:claude-3-5-haiku-latest"
    llm_stage_adjudication_models: str = "anthropic:claude-3-7-sonnet-latest,openai:gpt-4.1,gemini:gemini-2.0-flash"
    llm_stage_structured_models: str = "openai:gpt-4.1-mini,anthropic:claude-3-5-haiku-latest,gemini:gemini-2.0-flash"
    llm_stage_summary_models: str = "anthropic:claude-3-7-sonnet-latest,gemini:gemini-2.0-flash,openai:gpt-4.1-mini"
    llm_stage_crawler_triage_models: str = "gemini:gemini-2.0-flash,openai:gpt-4.1-mini,anthropic:claude-3-5-haiku-latest"
    llm_stage_market_map_models: str = "anthropic:claude-3-7-sonnet-latest,openai:gpt-4.1-mini,gemini:gemini-2.0-flash"
    llm_stage_expansion_models: str = "gemini:deep-research-pro-preview-12-2025,openai:o4-mini-deep-research,gemini:gemini-2.0-flash,anthropic:claude-3-7-sonnet-latest,openai:gpt-4.1-mini"

    # Stage budgets and retry policy
    stage_seed_ingest_timeout_seconds: int = 240
    stage_llm_discovery_timeout_seconds: int = 360
    stage_expansion_research_timeout_seconds: int = 600
    stage_registry_timeout_seconds: int = 360
    stage_enrichment_timeout_seconds: int = 540
    stage_scoring_timeout_seconds: int = 420
    stage_retry_max_attempts: int = 2
    stage_retry_backoff_seconds: float = 1.0

    # Discovery runtime and quality gates
    discovery_execution_mode: str = "live"
    discovery_global_timeout_seconds: int = 1800
    quality_min_claims_created: int = 3
    quality_min_ranking_eligible_count: int = 1
    size_fit_window_ratio: float = 0.30
    size_fit_boost_points: float = 8.0
    size_large_company_threshold: int = 200
    size_large_company_penalty_points: float = 10.0
    audit_max_fp_low_ticket_without_pricing_evidence: int = 0
    audit_max_fn_missing_vertical_with_institutional_text: int = 8
    audit_max_fp_registry_or_directory_overweight: int = 5
    audit_max_fn_customer_proof_but_thin_grouping: int = 8
    discovery_candidate_entity_cap: int = 2000
    discovery_scoring_entities_cap: int = 120
    discovery_validation_queue_limit: int = 36
    discovery_validation_lane_cap: int = 6
    discovery_validation_query_family_cap: int = 4
    discovery_validation_source_family_cap: int = 18
    discovery_directory_seed_total_cap: int = 2000
    discovery_directory_seed_per_listing_cap: int = 2000
    discovery_directory_seed_per_source_cap: int = 2000
    discovery_directory_seed_without_website_cap: int = 2000
    discovery_directory_profile_resolution_enabled: bool = True
    discovery_directory_profile_resolution_cap: int = 30

    # Registry expansion runtime controls
    registry_identity_top_seeds: int = 50
    registry_max_queries: int = 140
    registry_max_accepted_neighbors: int = 120
    registry_max_de_queries: int = 6
    registry_identity_max_seconds: int = 120
    registry_neighbor_max_seconds: int = 150

    # First-party enrichment runtime controls
    first_party_fetch_budget: int = 15
    first_party_crawl_budget: int = 0
    first_party_crawl_deep_budget: int = 0
    first_party_hint_crawl_budget: int = 0
    first_party_crawl_light_max_pages: int = 3
    first_party_crawl_deep_max_pages: int = 6
    first_party_min_priority_for_crawl: float = 55.0
    first_party_adaptive_hint_timeout_seconds: int = 6
    first_party_adaptive_hint_max_urls_per_domain: int = 40
    first_party_adaptive_hint_domain_budget: int = 25

    # Optional rendered-browser fallback (Chrome DevTools MCP style connector)
    chrome_mcp_enabled: bool = False
    chrome_mcp_endpoint: str = ""
    chrome_mcp_timeout_seconds: int = 25
    chrome_mcp_max_pages_per_domain: int = 2
    chrome_mcp_min_text_chars: int = 700

    # External retrieval limits / cache TTLs
    retrieval_search_cache_ttl_seconds: int = 43200
    retrieval_url_cache_ttl_seconds: int = 43200
    external_search_candidates_cap: int = 25
    discovery_retrieval_provider_order: str = "exa,brave,tavily,serpapi"
    discovery_retrieval_per_query_cap: int = 10
    discovery_retrieval_total_cap: int = 90
    discovery_retrieval_per_domain_cap: int = 3
    discovery_retrieval_similar_seed_cap: int = 6
    discovery_candidate_synthesis_enabled: bool = False
    discovery_identity_resolution_enabled: bool = True
    discovery_registry_expansion_enabled: bool = False
    discovery_first_party_enrichment_enabled: bool = True
    company_context_secondary_provider_order: str = "serper,brave"
    company_context_secondary_per_query_cap: int = 4
    company_context_secondary_query_cap: int = 18
    company_context_secondary_result_cap: int = 36
    company_context_secondary_per_domain_cap: int = 2
    company_context_secondary_max_seconds: int = 30

    # App
    debug: bool = True

    @staticmethod
    def _parse_provider_models(value: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for raw in str(value or "").split(","):
            token = raw.strip()
            if not token or ":" not in token:
                continue
            provider, model = token.split(":", 1)
            provider = provider.strip().lower()
            model = model.strip()
            if provider and model:
                pairs.append((provider, model))
        return pairs

    def resolved_celery_queue_namespace(self) -> str:
        explicit = str(self.celery_queue_namespace or "").strip().lower()
        if explicit:
            return re.sub(r"[^a-z0-9_-]+", "-", explicit).strip("-")
        parsed = urlparse(str(self.database_url_sync or ""))
        host = re.sub(r"[^a-z0-9_-]+", "-", str(parsed.hostname or "local").lower()).strip("-") or "local"
        db_name = re.sub(r"[^a-z0-9_-]+", "-", str(parsed.path or "").strip("/").lower()).strip("-") or "prospection"
        return f"{host}-{db_name}"

    def stage_model_routes(self, stage_name: str) -> List[Tuple[str, str]]:
        mapping = {
            "discovery_retrieval": self.llm_stage_discovery_models,
            "discovery_query_planning": self.llm_stage_discovery_models,
            "discovery_candidate_synthesis": self.llm_stage_discovery_models,
            "evidence_adjudication": self.llm_stage_adjudication_models,
            "structured_normalization": self.llm_stage_structured_models,
            "context_summary": self.llm_stage_summary_models,
            "crawler_triage": self.llm_stage_crawler_triage_models,
            "market_map_reasoning": self.llm_stage_market_map_models,
            "expansion_brief_reasoning": self.llm_stage_expansion_models,
        }
        return self._parse_provider_models(mapping.get(stage_name, self.llm_stage_discovery_models))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
