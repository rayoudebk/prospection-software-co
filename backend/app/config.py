from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Tuple


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/prospection"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/prospection"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Gemini
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Optional external retrieval connectors
    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    firecrawl_api_key: str = ""
    jina_api_key: str = ""

    # Optional registry API keys
    companies_house_api_key: str = ""

    # LLM model routing (provider:model, comma-separated)
    llm_stage_discovery_models: str = "gemini:gemini-2.0-flash,openai:gpt-4.1-mini,anthropic:claude-3-5-haiku-latest"
    llm_stage_adjudication_models: str = "anthropic:claude-3-7-sonnet-latest,openai:gpt-4.1,gemini:gemini-2.0-flash"
    llm_stage_structured_models: str = "openai:gpt-4.1-mini,anthropic:claude-3-5-haiku-latest,gemini:gemini-2.0-flash"
    llm_stage_summary_models: str = "anthropic:claude-3-7-sonnet-latest,gemini:gemini-2.0-flash,openai:gpt-4.1-mini"
    llm_stage_crawler_triage_models: str = "gemini:gemini-2.0-flash,openai:gpt-4.1-mini,anthropic:claude-3-5-haiku-latest"

    # Stage budgets and retry policy
    stage_seed_ingest_timeout_seconds: int = 240
    stage_llm_discovery_timeout_seconds: int = 360
    stage_registry_timeout_seconds: int = 360
    stage_enrichment_timeout_seconds: int = 540
    stage_scoring_timeout_seconds: int = 420
    stage_retry_max_attempts: int = 2
    stage_retry_backoff_seconds: float = 1.0

    # Discovery runtime and quality gates
    discovery_global_timeout_seconds: int = 1800
    quality_min_claims_created: int = 3
    quality_min_ranking_eligible_count: int = 1
    discovery_pre_score_universe_cap: int = 300
    discovery_scoring_entities_cap: int = 90

    # Registry expansion runtime controls
    registry_identity_top_seeds: int = 50
    registry_max_queries: int = 140
    registry_max_accepted_neighbors: int = 120
    registry_max_de_queries: int = 6
    registry_identity_max_seconds: int = 120
    registry_neighbor_max_seconds: int = 150

    # First-party enrichment runtime controls
    first_party_fetch_budget: int = 36
    first_party_crawl_budget: int = 20
    first_party_crawl_deep_budget: int = 8
    first_party_hint_crawl_budget: int = 20
    first_party_crawl_light_max_pages: int = 3
    first_party_crawl_deep_max_pages: int = 6
    first_party_min_priority_for_crawl: float = 55.0

    # External retrieval limits / cache TTLs
    retrieval_search_cache_ttl_seconds: int = 43200
    retrieval_url_cache_ttl_seconds: int = 43200
    external_search_candidates_cap: int = 25

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

    def stage_model_routes(self, stage_name: str) -> List[Tuple[str, str]]:
        mapping = {
            "discovery_retrieval": self.llm_stage_discovery_models,
            "evidence_adjudication": self.llm_stage_adjudication_models,
            "structured_normalization": self.llm_stage_structured_models,
            "context_summary": self.llm_stage_summary_models,
            "crawler_triage": self.llm_stage_crawler_triage_models,
        }
        return self._parse_provider_models(mapping.get(stage_name, self.llm_stage_discovery_models))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
