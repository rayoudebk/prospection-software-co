"""Celery tasks for workspace-based workflow."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from html import unescape
import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse
import unicodedata
import httpx
import redis
from selectolax.parser import HTMLParser
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from celery import chain

from app.workers.celery_app import celery_app
from app.config import get_settings
from app.models.workspace import Workspace, CompanyProfile
from app.models.company_context import CompanyContextPack
from app.models.company import Company, CompanyDossier, CompanyStatus
from app.models.job import DB_ACTIVE_JOB_STATES, Job, JobType, JobState
from app.models.source_evidence import SourceEvidence
from app.models.report import ReportSnapshot, ReportSnapshotItem, CompanyFact
from app.models.intelligence import (
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    ComparatorSourceRun,
    RegistryQueryLog,
    CompanyMention,
    CompanyScreening,
    CompanyClaim,
)
from app.services.comparator_sources import (
    SOURCE_REGISTRY,
    ingest_source,
    resolve_external_website_from_profile,
)
from app.services.reporting import (
    DISCOVERY_COUNTRIES,
    RELIABLE_FILINGS_COUNTRIES,
    classify_size_bucket,
    estimate_size_from_signals,
    extract_customers_and_integrations,
    extract_filing_facts_from_evidence,
    is_reliable_filing_source_url,
    is_trusted_source_url,
    modules_with_evidence,
    normalize_country,
    source_label_for_url,
)
from app.services.evidence_policy import (
    DEFAULT_EVIDENCE_POLICY,
    claim_group_for_dimension,
    infer_source_kind,
    infer_source_tier,
    normalize_policy,
    valid_through_from_claim_group,
)
from app.services.decision_engine import evaluate_decision
from app.services.claims_graph import rebuild_workspace_claims_graph
from app.services.discovery_candidate_graph import (
    Neo4jDiscoveryCandidateGraphStore,
    build_discovery_candidate_graph_payload,
)
from app.services.discovery_readiness import build_workspace_discovery_readiness
from app.services.discovery_validation import (
    VALIDATION_STATUS_PROMOTED,
    build_candidate_discovery_context,
    build_diversified_validation_queue,
    compute_discovery_score,
    discovery_source_role,
    normalize_discovery_query_family,
    normalize_discovery_source_family,
    VALIDATION_STATUS_QUEUED,
    VALIDATION_STATUS_KEEP,
    VALIDATION_STATUS_REJECT,
    VALIDATION_STATUS_WATCHLIST,
    set_validation_metadata,
    validation_metadata,
)
from app.services.llm.orchestrator import LLMOrchestrator
from app.services.llm.types import LLMRequest, LLMStage, LLMOrchestrationError
from app.services.retrieval.crawl_connectors import fetch_page_fast
from app.services.company_context_graph import sync_company_context_pack_graph
from app.models.external_search import ExternalSearchRun, ExternalSearchResult
from app.services.retrieval.search_orchestrator import run_external_search_queries
from app.services.retrieval.url_normalization import normalize_url
from app.services.retrieval.cache import RetrievalCache
from app.services.crawler.connectors.chrome_devtools_mcp import render_page_via_chrome_devtools_mcp
from app.services.crawler.career_priority import (
    career_excluded_keyword_hits,
    career_target_keyword_hits,
    is_career_page_url,
)
from app.services.quality_audit import (
    build_quality_audit_v1,
    normalize_quality_audit_v1,
    quality_audit_thresholds_from_settings,
)

logger = logging.getLogger(__name__)

_FR_INPI_LOGIN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}

MIN_PUBLIC_PRICE_USD = 250.0
MIN_SOFTWARE_HEAVINESS = 3
KEEP_SCORE_THRESHOLD = 45.0
REVIEW_SCORE_THRESHOLD = 30.0
MIN_TRUSTED_EVIDENCE_FOR_KEEP = 3
MIN_TRUSTED_EVIDENCE_FOR_REVIEW = 2
MAX_PENALTY_POINTS = 35.0
SIZE_FIT_WINDOW_RATIO = 0.30
SIZE_FIT_BOOST_POINTS = 8.0
SIZE_LARGE_COMPANY_THRESHOLD = 200
SIZE_LARGE_COMPANY_PENALTY_POINTS = 10.0
MAX_IDENTITY_FETCHES_PER_RUN = 600
IDENTITY_RESOLUTION_TIMEOUT_SECONDS = 4
IDENTITY_RESOLUTION_CONCURRENCY = 20
FIRST_PARTY_FETCH_BUDGET = 48
FIRST_PARTY_FETCH_TIMEOUT_SECONDS = 4
FIRST_PARTY_MAX_SIGNALS = 6
FIRST_PARTY_CRAWL_BUDGET = 42
FIRST_PARTY_CRAWL_DEEP_BUDGET = 18
FIRST_PARTY_HINT_CRAWL_BUDGET = 18
FIRST_PARTY_CRAWL_LIGHT_MAX_PAGES = 5
FIRST_PARTY_CRAWL_DEEP_MAX_PAGES = 10
FIRST_PARTY_CRAWL_MAX_REASONS = 16
FIRST_PARTY_CRAWL_DEEP_PRIORITY_THRESHOLD = 95.0
CANDIDATE_ENTITY_CAP_DEFAULT = 500
DIRECT_IDENTITY_RESOLUTION_TIMEOUT_SECONDS = 3

FIRST_PARTY_AUTO_HINT_PATHS = (
    "/platform/",
    "/platform/front-office",
    "/platform/back-office",
    "/platform/front-digital",
    "/platform/payments",
    "/solutions/",
    "/solution/",
    "/solutions/front-office",
    "/solutions/online-brokerage",
    "/solutions/private-banks",
    "/solutions/asset-managers",
    "/offers/",
    "/product/",
    "/products/",
    "/technology-services/",
    "/technology-services/services/",
    "/technology-services/technology/",
    "/technology-services/documentation-api/",
    "/integrations/",
    "/integration/",
    "/api/",
    "/documentation/",
    "/docs/",
    "/resources/",
    "/client-stories/",
    "/customers/",
    "/case-studies/",
    "/resources/case-studies/",
    "/success-stories/",
    "/careers/",
    "/jobs/",
    "/jobs/search/",
    "/careers/open-positions/",
)

FIRST_PARTY_HINT_URL_TOKENS = (
    "platform",
    "front-office",
    "back-office",
    "front-digital",
    "payments",
    "solution",
    "solutions",
    "product",
    "products",
    "offer",
    "offers",
    "technology-services",
    "technology",
    "services",
    "integrations",
    "integration",
    "api",
    "documentation",
    "docs",
    "features",
    "fonctionalites",
    "functionalities",
    "capabilities",
    "client-stories",
    "customer-story",
    "customers",
    "customer",
    "case-studies",
    "case-study",
    "success-stories",
    "stories",
    "testimonials",
    "partners",
    "partnership",
    "newsroom",
    "news",
    "press",
    "blog",
    "insights",
    "resources",
    "clients",
    "temoignage",
    "cas-client",
    "kunden",
)

REGISTRY_MAX_QUERIES = 360
REGISTRY_MAX_ACCEPTED_NEIGHBORS = 300
REGISTRY_MAX_RAW_HITS_PER_QUERY = 10
REGISTRY_MAX_NEIGHBORS_PER_ENTITY = 3
REGISTRY_IDENTITY_TOP_SEEDS = 120
REGISTRY_IDENTITY_MIN_SCORE = 0.68
REGISTRY_IDENTITY_BRAND_MATCH_MIN_SCORE = 0.62
REGISTRY_NEIGHBOR_MIN_SCORE = 28.0
REGISTRY_EXPANSION_COUNTRIES = {"FR", "UK", "DE", "BE", "NL", "LU", "CH", "MC"}
REGISTRY_MAX_DE_QUERIES = 10
REGISTRY_DE_ERROR_BREAKER = 3
REGISTRY_IDENTITY_MAX_SECONDS = 216
REGISTRY_NEIGHBOR_MAX_SECONDS = 288

HARD_FAIL_REASONS = {
    "go_to_market_b2c",
    "retail_only_icp",
    "consumer_language_without_institutional_icp",
}

AGGREGATOR_DOMAINS = {
    "thewealthmosaic.com",
    "thewealthmosaic.co.uk",
}

PLACEHOLDER_SEED_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "localhost",
}

WEALTH_DISCOVERY_EXCLUDE_TERMS = {
    "bloomberg",
    "fis",
    "fiserv",
    "broadridge",
    "ss&c",
    "refinitiv",
}

WEALTH_BENCHMARK_SEEDS = [
    {"name": "QPLIX", "website": "https://qplix.com", "hq_country": "DE"},
    {"name": "Avaloq", "website": "https://www.avaloq.com", "hq_country": "CH"},
    {"name": "SimCorp", "website": "https://www.simcorp.com", "hq_country": "DK"},
    {"name": "Upvest", "website": "https://upvest.co", "hq_country": "DE"},
]

WEALTH_BENCHMARK_EVIDENCE_URLS = {
    "upvest.co": [
        "https://upvest.co/blog/liqid-enters-partnership-with-upvest-for-its-eltif-offering",
        "https://upvest.co/blog/zopa-bank-partners-with-upvest",
        "https://upvest.co/blog/boerse-stuttgart-and-upvest",
    ]
}

WEALTH_CONTEXT_TOKENS = {
    "wealth",
    "portfolio",
    "asset management",
    "private bank",
    "wealth manager",
    "investment platform",
    "portfolio management",
    "pms",
}

HEALTHCARE_CONTEXT_TOKENS = {
    "healthcare",
    "hospital",
    "hospitals",
    "clinic",
    "clinics",
    "care provider",
    "care providers",
    "staffing",
    "shift",
    "shift replacement",
    "rostering",
    "workforce",
    "pool management",
    "internal mobility",
    "vendor management system",
    "vms",
}

SCREEN_WEIGHTS = {
    "institutional_icp_fit": 30.0,
    "platform_product_depth": 25.0,
    "services_implementation_complexity": 15.0,
    "named_customer_credibility": 10.0,
    "enterprise_gtm": 10.0,
    "defensibility_moat": 10.0,
}

INSTITUTIONAL_TOKENS = {
    "asset manager",
    "asset managers",
    "wealth manager",
    "wealth managers",
    "private bank",
    "private banks",
    "fund admin",
    "fund administration",
    "institutional",
    "advisory firm",
    "advisors",
    "broker",
    "custodian",
    "insurance",
}

B2C_TOKENS = {
    "retail investor",
    "retail investors",
    "consumer",
    "personal finance",
    "individual investor",
    "individual investors",
    "for individuals",
}

VERTICAL_WORKFLOW_HINT_TOKENS = {
    "workflow",
    "workflows",
    "infrastructure",
    "platform",
    "api",
    "apis",
    "custody",
    "portfolio",
    "wealth management",
    "asset management",
    "investment",
    "investing",
    "advisory",
    "securities",
    "trading",
    "fund",
    "funds",
    "bank",
    "banks",
    "asset manager",
    "asset managers",
    "wealth manager",
    "wealth managers",
    "private bank",
    "private banks",
}

# Sync engine for Celery workers
settings = get_settings()
sync_engine = create_engine(settings.database_url_sync, echo=settings.debug)
SessionLocal = sessionmaker(bind=sync_engine)


def _discovery_context_key(job_id: int) -> str:
    return f"discovery:pipeline:job:{int(job_id)}"


def _load_discovery_context(job_id: int) -> dict[str, Any]:
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    raw = client.get(_discovery_context_key(job_id))
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_discovery_context(job_id: int, payload: dict[str, Any], ttl_seconds: int = 86400) -> None:
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.setex(_discovery_context_key(job_id), max(60, int(ttl_seconds)), json.dumps(payload))


def _save_stage_checkpoint(job_id: int, stage_name: str, payload: dict[str, Any]) -> None:
    ctx = _load_discovery_context(job_id)
    checkpoints = ctx.get("stage_checkpoints") if isinstance(ctx.get("stage_checkpoints"), dict) else {}
    stage_key = str(stage_name)
    checkpoint_payload = {
        **(payload if isinstance(payload, dict) else {}),
        "captured_at": datetime.utcnow().isoformat(),
    }
    checkpoints[stage_key] = checkpoint_payload
    ctx["stage_checkpoints"] = checkpoints
    _save_discovery_context(job_id, ctx)
    # Mirror compact stage checkpoints into Job.result_json for DB-safe auditability.
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        result = dict(job.result_json) if isinstance(job.result_json, dict) else {}
        persisted = result.get("stage_checkpoints") if isinstance(result.get("stage_checkpoints"), dict) else {}
        persisted[stage_key] = checkpoint_payload
        result["stage_checkpoints"] = persisted
        screening_run_id = str(
            (result.get("screening_run_id") if isinstance(result, dict) else None)
            or ctx.get("screening_run_id")
            or ""
        ).strip()
        if screening_run_id:
            result["screening_run_id"] = screening_run_id
            result["stage_checkpoint_key"] = f"job:{int(job_id)}:run:{screening_run_id}"
        else:
            result["stage_checkpoint_key"] = f"job:{int(job_id)}"
        job.result_json = result
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _stage_queue_field(stage_name: str) -> str:
    return str(stage_name or "").strip().lower()


def _mark_stage_enqueued(job_id: int, stage_name: str, when: Optional[datetime] = None) -> None:
    ctx = _load_discovery_context(job_id)
    queue_times = ctx.get("stage_enqueued_at") if isinstance(ctx.get("stage_enqueued_at"), dict) else {}
    queue_times[_stage_queue_field(stage_name)] = (when or datetime.utcnow()).isoformat()
    ctx["stage_enqueued_at"] = queue_times
    _save_discovery_context(job_id, ctx)


def _record_stage_queue_wait(job_id: int, stage_name: str, started_at: Optional[datetime] = None) -> None:
    ctx = _load_discovery_context(job_id)
    queue_times = ctx.get("stage_enqueued_at") if isinstance(ctx.get("stage_enqueued_at"), dict) else {}
    enqueued_raw = queue_times.get(_stage_queue_field(stage_name))
    if not enqueued_raw:
        return
    try:
        enqueued_at = datetime.fromisoformat(str(enqueued_raw))
    except Exception:
        return
    now = started_at or datetime.utcnow()
    queue_wait_ms = max(0, int((now - enqueued_at).total_seconds() * 1000))
    wait_map = ctx.get("queue_wait_ms_by_stage") if isinstance(ctx.get("queue_wait_ms_by_stage"), dict) else {}
    wait_map[str(stage_name)] = queue_wait_ms
    ctx["queue_wait_ms_by_stage"] = wait_map
    _save_discovery_context(job_id, ctx)


def _increment_stage_retry(job_id: int, stage_name: str) -> None:
    ctx = _load_discovery_context(job_id)
    retry_counts = ctx.get("stage_retry_counts") if isinstance(ctx.get("stage_retry_counts"), dict) else {}
    retry_counts[str(stage_name)] = int(retry_counts.get(str(stage_name), 0)) + 1
    ctx["stage_retry_counts"] = retry_counts
    _save_discovery_context(job_id, ctx)


def _is_retryable_stage_exception(exc: Exception) -> bool:
    text = str(exc or "").lower()
    retry_tokens = ("timeout", "timed out", "429", "rate limit", "connection reset", "connection aborted", "502", "503", "504", "temporary")
    if any(token in text for token in retry_tokens):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    return False


def _read_retrieval_cache_stats_snapshot() -> dict[str, int]:
    try:
        return RetrievalCache().stats_snapshot()
    except Exception:
        return {}


def _compute_cache_hit_rates(
    start_stats: Optional[dict[str, int]],
    end_stats: Optional[dict[str, int]],
) -> dict[str, float]:
    start = start_stats if isinstance(start_stats, dict) else {}
    end = end_stats if isinstance(end_stats, dict) else {}

    def _rate(namespace: str) -> float:
        hit_key = f"{namespace}:hit"
        miss_key = f"{namespace}:miss"
        hits = max(0, int(end.get(hit_key, 0)) - int(start.get(hit_key, 0)))
        misses = max(0, int(end.get(miss_key, 0)) - int(start.get(miss_key, 0)))
        total = hits + misses
        if total <= 0:
            return 0.0
        return round(hits / total, 4)

    return {
        "search_cache_hit_rate": _rate("search"),
        "url_cache_hit_rate": _rate("url_content"),
    }


def _fail_discovery_job(job_id: int, message: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        job.state = JobState.failed
        job.error_message = str(message)[:1000]
        job.finished_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def _is_deadlock_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    deadlock_markers = (
        "deadlock detected",
        "psycopg2.errors.deadlockdetected",
        "lock timeout",
        "serialization failure",
    )
    return any(marker in text for marker in deadlock_markers)


def _expire_stale_running_discovery_jobs(db, exclude_job_id: Optional[int] = None) -> int:
    timeout_seconds = max(60, int(settings.discovery_global_timeout_seconds))
    cutoff = datetime.utcnow() - timedelta(seconds=timeout_seconds)
    stale_jobs = (
        db.query(Job)
        .filter(
            Job.job_type == JobType.discovery_universe,
            Job.state == JobState.running,
            Job.started_at.isnot(None),
            Job.started_at < cutoff,
        )
        .all()
    )
    stale_count = 0
    for stale_job in stale_jobs:
        if exclude_job_id is not None and stale_job.id == exclude_job_id:
            continue
        stale_job.state = JobState.failed
        stale_job.error_message = "stale_run_timeout_cleanup"
        stale_job.finished_at = datetime.utcnow()
        stale_job.progress_message = "Failed by stale run cleanup"
        stale_count += 1
    if stale_count:
        db.commit()
    return stale_count


def _screening_run_id_from_screening(screening: CompanyScreening) -> str:
    meta = screening.screening_meta_json if isinstance(screening.screening_meta_json, dict) else {}
    return str(meta.get("screening_run_id") or "").strip()


def _collect_run_screenings_and_claims(
    db,
    workspace_id: int,
    screening_run_id: str,
    *,
    max_screenings: int = 5000,
) -> tuple[list[CompanyScreening], dict[int, list[CompanyClaim]]]:
    target_run_id = str(screening_run_id or "").strip()
    if not target_run_id:
        return [], {}
    screening_rows = (
        db.query(CompanyScreening)
        .filter(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .limit(max_screenings)
        .all()
    )
    run_screenings = [
        row for row in screening_rows
        if _screening_run_id_from_screening(row) == target_run_id
    ]
    screening_ids = [row.id for row in run_screenings if row.id]
    if not screening_ids:
        return run_screenings, {}
    claim_rows = db.query(CompanyClaim).filter(CompanyClaim.company_screening_id.in_(screening_ids)).all()
    claims_by_screening: dict[int, list[CompanyClaim]] = {}
    for claim in claim_rows:
        if not claim.company_screening_id:
            continue
        claims_by_screening.setdefault(int(claim.company_screening_id), []).append(claim)
    return run_screenings, claims_by_screening


def _build_quality_audit_for_workspace_run(
    db,
    workspace_id: int,
    screening_run_id: str,
) -> dict[str, Any]:
    run_screenings, claims_by_screening = _collect_run_screenings_and_claims(
        db,
        workspace_id,
        screening_run_id,
    )
    return build_quality_audit_v1(
        screenings=run_screenings,
        claims_by_screening=claims_by_screening,
        run_id=screening_run_id,
        thresholds=quality_audit_thresholds_from_settings(settings),
    )


def _load_previous_completed_run_quality_audit(
    db,
    workspace_id: int,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    previous_job = (
        db.query(Job)
        .filter(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.discovery_universe,
            Job.state == JobState.completed,
        )
        .order_by(Job.finished_at.desc(), Job.created_at.desc())
        .first()
    )
    if not previous_job:
        return None, None

    previous_result = previous_job.result_json if isinstance(previous_job.result_json, dict) else {}
    previous_run_id = str(previous_result.get("screening_run_id") or "").strip()
    if not previous_run_id:
        return None, None

    normalized = normalize_quality_audit_v1(previous_result.get("quality_audit_v1"))
    if normalized:
        return normalized, previous_run_id

    rebuilt = _build_quality_audit_for_workspace_run(
        db=db,
        workspace_id=workspace_id,
        screening_run_id=previous_run_id,
    )
    normalized_rebuilt = normalize_quality_audit_v1(rebuilt)
    if normalized_rebuilt:
        return normalized_rebuilt, previous_run_id
    return rebuilt, previous_run_id


def normalize_domain(url: str | None) -> str | None:
    """Extract and normalize domain from URL."""
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url.lower() if url else None


def _is_aggregator_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    return any(domain == agg or domain.endswith(f".{agg}") for agg in AGGREGATOR_DOMAINS)


def _infer_country_from_domain(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    lowered = domain.lower()
    if lowered.endswith(".fr"):
        return "FR"
    if lowered.endswith(".co.uk") or lowered.endswith(".uk"):
        return "UK"
    if lowered.endswith(".ie"):
        return "IE"
    if lowered.endswith(".de"):
        return "DE"
    if lowered.endswith(".ch"):
        return "CH"
    if lowered.endswith(".mc"):
        return "MC"
    if lowered.endswith(".es"):
        return "ES"
    if lowered.endswith(".pt"):
        return "PT"
    if lowered.endswith(".be"):
        return "BE"
    if lowered.endswith(".nl"):
        return "NL"
    if lowered.endswith(".lu"):
        return "LU"
    return None


def _is_registry_profile_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    lowered = domain.lower()
    return any(lowered == item or lowered.endswith(f".{item}") for item in REGISTRY_PROFILE_DOMAINS)


def _is_non_first_party_profile_domain(domain: Optional[str]) -> bool:
    return _is_aggregator_domain(domain) or _is_registry_profile_domain(domain)


def _is_known_third_party_context_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    lowered = str(domain).lower()
    return any(lowered == item or lowered.endswith(f".{item}") for item in KNOWN_THIRD_PARTY_CONTEXT_DOMAINS)


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return None


def _country_hints_from_reasons(reasons: list[dict[str, Any]]) -> list[str]:
    combined = " ".join(
        str(reason.get("text") or "")
        for reason in reasons
        if isinstance(reason, dict)
    ).lower()
    hints: list[str] = []
    for country, tokens in COUNTRY_HINT_TOKENS.items():
        if any(token in combined for token in tokens):
            hints.append(country)
    return _dedupe_strings(hints)


def _infer_country_from_text(text: str) -> Optional[str]:
    lowered = str(text or "").lower()
    for country, tokens in COUNTRY_HINT_TOKENS.items():
        if any(token in lowered for token in tokens):
            return country
    return None


def _looks_active_status(status: Optional[str]) -> bool:
    normalized = str(status or "").strip().lower()
    if not normalized:
        return True
    return normalized in {"active", "a", "open", "registered", "aktuell", "bestehend"}


def _normalize_fr_registry_code(value: Any) -> Optional[str]:
    token = str(value or "").strip().upper()
    if not token:
        return None
    if re.match(r"^\d{2}\.\d{2}[A-Z]$", token):
        return token
    return None


def _fr_registry_search(
    *,
    query: Optional[str] = None,
    activite_principale: Optional[str] = None,
    page: int = 1,
    per_page: int = REGISTRY_MAX_RAW_HITS_PER_QUERY,
    only_active: bool = True,
    timeout_seconds: Optional[int] = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    url = "https://recherche-entreprises.api.gouv.fr/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "application/json",
    }
    params: dict[str, Any] = {
        "page": max(1, int(page)),
        "per_page": max(1, int(per_page)),
    }
    if query and str(query).strip():
        params["q"] = str(query).strip()
    if activite_principale and str(activite_principale).strip():
        params["activite_principale"] = str(activite_principale).strip().upper()
    if only_active:
        params["etat_administratif"] = "A"
    effective_timeout = max(1, int(timeout_seconds or getattr(settings, "discovery_fr_registry_search_timeout_seconds", 3)))
    try:
        with httpx.Client(timeout=effective_timeout, headers=headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return [], f"fr_registry_search_failed:{exc}"
    results = payload.get("results")
    if not isinstance(results, list):
        return [], None
    return [row for row in results if isinstance(row, dict)], None


def _clean_legal_name_from_nom_complet(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.match(r"^(.*?)\s+\(([^)]+)\)\s*$", raw)
    if match:
        prefix = str(match.group(1) or "").strip()
        if prefix:
            return prefix
    return raw


def _fr_registry_brand_names(row: dict[str, Any]) -> list[str]:
    brands: list[str] = []

    def _add(value: Any) -> None:
        normalized = str(value or "").strip()
        if not normalized:
            return
        brands.append(normalized)

    nom_complet = str(row.get("nom_complet") or "").strip()
    parenthetical = re.findall(r"\(([^)]+)\)", nom_complet)
    for item in parenthetical:
        _add(item)
    _add(row.get("nom_commercial"))
    siege = row.get("siege") if isinstance(row.get("siege"), dict) else {}
    _add(siege.get("nom_commercial"))
    for item in (siege.get("liste_enseignes") or []):
        _add(item)
    for etab in row.get("matching_etablissements") or []:
        if not isinstance(etab, dict):
            continue
        _add(etab.get("nom_commercial"))
        for item in (etab.get("liste_enseignes") or []):
            _add(item)
    legal_name = _fr_registry_legal_name(row)
    return [
        item
        for item in _dedupe_strings(brands)
        if item and item.lower() != str(legal_name or "").strip().lower()
    ]


def _fr_registry_legal_name(row: dict[str, Any]) -> Optional[str]:
    return _first_non_empty(
        row.get("nom_raison_sociale"),
        _clean_legal_name_from_nom_complet(row.get("nom_complet")),
        row.get("nom"),
    )


def _fr_registry_display_name(row: dict[str, Any]) -> Optional[str]:
    brands = _fr_registry_brand_names(row)
    if brands:
        return brands[0]
    return _fr_registry_legal_name(row)


def _fr_registry_website(row: dict[str, Any]) -> Optional[str]:
    website = row.get("site_internet") or row.get("site_web")
    if isinstance(website, list):
        website = website[0] if website else None
    if not website:
        siege = row.get("siege") if isinstance(row.get("siege"), dict) else {}
        website = siege.get("site_internet") or siege.get("site_web")
        if isinstance(website, list):
            website = website[0] if website else None
    normalized = str(website or "").strip()
    if normalized and not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    return normalized or None


def _fr_registry_context_text(row: dict[str, Any]) -> str:
    siege = row.get("siege") if isinstance(row.get("siege"), dict) else {}
    complements = row.get("complements") if isinstance(row.get("complements"), dict) else {}
    complement_tokens = [
        key.replace("est_", "").replace("_", " ")
        for key, value in complements.items()
        if value is True and str(key).startswith("est_")
    ]
    text_parts: list[str] = [
        str(_fr_registry_legal_name(row) or ""),
        str(_fr_registry_display_name(row) or ""),
        " ".join(_fr_registry_brand_names(row)),
        str(row.get("libelle_activite_principale") or ""),
        str(row.get("activite_principale") or ""),
        str(row.get("activite_principale_naf25") or ""),
        str(row.get("section_activite_principale") or ""),
        str(row.get("libelle_nature_juridique") or row.get("nature_juridique") or ""),
        str(row.get("categorie_entreprise") or ""),
        str(siege.get("libelle_commune") or ""),
        str(siege.get("departement") or ""),
        str(siege.get("region") or ""),
        " ".join(complement_tokens),
    ]
    return " ".join(part for part in text_parts if str(part).strip()).strip()


def _normalize_html_text(value: Any) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_natural_person_name(value: Any) -> bool:
    text = _normalize_html_text(value)
    if not text:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    lowered = text.lower()
    company_markers = (
        "sas",
        "sasu",
        "sarl",
        "eurl",
        "holding",
        "groupe",
        "solutions",
        "software",
        "technologies",
        "technology",
        "systems",
        "services",
        "plateforme",
        "platform",
    )
    if any(marker in lowered for marker in company_markers):
        return False
    tokens = [token for token in re.split(r"[\s'-]+", text) if token]
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    return all(re.fullmatch(r"[A-Za-zÀ-ÿ]+", token) for token in tokens)


def _fr_registry_source_record(row: dict[str, Any], *, query_hint: Optional[str] = None) -> dict[str, Any]:
    legal_name = _fr_registry_legal_name(row)
    display_name = _fr_registry_display_name(row)
    siren = str(row.get("siren") or "").strip()
    website = _fr_registry_website(row)
    complements = row.get("complements") if isinstance(row.get("complements"), dict) else {}
    activity_code = _first_non_empty(
        row.get("activite_principale"),
        row.get("activite_principale_naf25"),
    )
    section_code = _first_non_empty(row.get("section_activite_principale"))
    activity_label = _first_non_empty(row.get("libelle_activite_principale"), activity_code)
    status = _first_non_empty(row.get("etat_administratif"), "A")
    registry_url = f"https://annuaire-entreprises.data.gouv.fr/entreprise/{siren}" if siren else None
    industry_codes = _dedupe_strings(
        [
            str(code).upper()
            for code in [
                _normalize_fr_registry_code(activity_code),
                _normalize_fr_registry_code(row.get("activite_principale_naf25")),
                str(section_code or "").strip().upper(),
            ]
            if str(code or "").strip()
        ]
    )
    record = {
        "name": display_name or legal_name or "",
        "display_name": display_name or legal_name or "",
        "legal_name": legal_name,
        "brand_names": _fr_registry_brand_names(row),
        "website": website,
        "country": "FR",
        "registry_id": siren or None,
        "registry_source": "fr_recherche_entreprises",
        "registry_url": registry_url or f"https://recherche-entreprises.api.gouv.fr/search?q={quote_plus(str(query_hint or display_name or legal_name or '').strip())}",
        "status": status,
        "is_active": _looks_active_status(status),
        "context_text": _fr_registry_context_text(row),
        "industry_codes": industry_codes,
        "activity_code": _normalize_fr_registry_code(activity_code),
        "activity_code_naf25": _normalize_fr_registry_code(row.get("activite_principale_naf25")),
        "activity_label": activity_label,
        "section_code": str(section_code or "").strip().upper() or None,
        "employee_band": _first_non_empty(row.get("tranche_effectif_salarie"), (row.get("siege") or {}).get("tranche_effectif_salarie")),
        "is_employer": str((row.get("siege") or {}).get("caractere_employeur") or row.get("caractere_employeur") or "").strip().upper() == "O",
        "registry_fields": {
            "ape_code": _normalize_fr_registry_code(activity_code),
            "naf25_code": _normalize_fr_registry_code(row.get("activite_principale_naf25")),
            "section_code": str(section_code or "").strip().upper() or None,
            "active_status": status,
            "commercial_names": _fr_registry_brand_names(row),
            "object_text_present": bool(str(_fr_registry_context_text(row) or "").strip()),
            "observation_count": 0,
            "employee_band": _first_non_empty(row.get("tranche_effectif_salarie"), (row.get("siege") or {}).get("tranche_effectif_salarie")),
            "is_employer": str((row.get("siege") or {}).get("caractere_employeur") or row.get("caractere_employeur") or "").strip().upper() == "O",
            "is_service_public": bool(complements.get("est_service_public")),
        },
    }
    record["industry_keywords"] = _industry_keywords_from_record(record)
    return record


def _fr_registry_semantic_terms(profile: CompanyProfile, normalized_scope: dict[str, Any]) -> dict[str, Any]:
    context_pack_json = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    primary_site = ((context_pack_json.get("sites") or [{}])[0] if isinstance(context_pack_json.get("sites"), list) else {}) or {}
    phrases: list[str] = []
    phrases.extend(normalized_scope.get("source_capabilities") or [])
    phrases.extend(normalized_scope.get("source_customer_segments") or [])
    phrases.extend(normalized_scope.get("adjacency_box_labels") or [])
    for box in (normalized_scope.get("adjacency_boxes") or []):
        if not isinstance(box, dict):
            continue
        phrases.append(str(box.get("label") or "").strip())
        phrases.extend(box.get("likely_customer_segments") or [])
        phrases.extend(box.get("likely_workflows") or [])
        phrases.extend(box.get("retrieval_query_seeds") or [])
    phrases.extend(normalized_scope.get("named_account_anchors") or [])
    phrases.extend(normalized_scope.get("geography_expansions") or [])
    phrases.append(str(primary_site.get("summary") or "").strip())
    phrases.append(str(primary_site.get("company_name") or "").strip())
    phrases = [phrase for phrase in phrases if str(phrase or "").strip()]

    terms: list[str] = []
    for phrase in phrases:
        lowered = unicodedata.normalize("NFKD", str(phrase).lower())
        lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
        for token in re.findall(r"[a-z0-9]{3,}", lowered):
            normalized = TOKEN_SYNONYMS.get(token, token)
            if len(normalized) < 3:
                continue
            terms.append(normalized)
            for expanded in FR_REGISTRY_SEMANTIC_EXPANSIONS.get(normalized, []):
                terms.append(expanded)
    return {
        "phrases": _dedupe_strings(phrases)[:80],
        "terms": _dedupe_strings(terms)[:160],
    }


def _fr_registry_scope_phrases(phrases: list[str]) -> dict[str, Any]:
    clean_phrases = [str(phrase or "").strip() for phrase in phrases if str(phrase or "").strip()]
    terms: list[str] = []
    for phrase in clean_phrases:
        lowered = unicodedata.normalize("NFKD", str(phrase).lower())
        lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
        for token in re.findall(r"[a-z0-9]{3,}", lowered):
            normalized = TOKEN_SYNONYMS.get(token, token)
            if (
                len(normalized) < 3
                or normalized in NAME_STOPWORDS
                or normalized in FR_REGISTRY_SCOPE_LOW_SIGNAL_TERMS
            ):
                continue
            terms.append(normalized)
            for expanded in FR_REGISTRY_SEMANTIC_EXPANSIONS.get(normalized, []):
                if expanded in NAME_STOPWORDS or expanded in FR_REGISTRY_SCOPE_LOW_SIGNAL_TERMS:
                    continue
                terms.append(expanded)
    return {
        "phrases": _dedupe_strings(clean_phrases)[:80],
        "terms": _dedupe_strings(terms)[:160],
    }


def _fr_registry_extract_observation_entities(observations: list[str]) -> list[str]:
    names: list[str] = []
    blocked_tokens = {
        "RADIATION",
        "COMPTER",
        "OPERATION",
        "OPÉRATION",
        "FUSION",
        "PARTICIPE",
        "PARTICIPÉ",
        "PARTICIPEE",
        "PARTICIPEES",
        "SOCIETE",
        "SOCIÉTÉ",
        "RCS",
        "TRIBUNAL",
        "ACTIVITE",
        "ACTIVITÉ",
    }
    for observation in observations or []:
        text = _normalize_html_text(observation)
        if not text:
            continue
        for match in re.finditer(
            r"(?:particip(?:e|é|ee|ée|ees|ées)s?\s+à\s+l[’']op[ée]ration\s*:\s*|fusion avec\s+|absorb(?:e|é|ee)\s+|apport partiel à\s+)([^.;:()]+)",
            text,
            flags=re.IGNORECASE,
        ):
            candidate = re.split(r"\s+(?:soci[eé]t[eé]|sas|sarl|sa|scop|rcs)\b", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
            candidate_upper = candidate.upper()
            candidate_tokens = [token for token in re.split(r"[\s,/()-]+", candidate_upper) if token]
            if (
                len(candidate) >= 3
                and candidate_tokens
                and not any(token in blocked_tokens for token in candidate_tokens)
                and not re.search(r"\b\d{2}[-/]\d{2}[-/]\d{2,4}\b", candidate_upper)
            ):
                names.append(candidate.upper())
    return _dedupe_strings(names)[:16]


def _fr_registry_scope_pack(
    profile: CompanyProfile,
    normalized_scope: dict[str, Any],
    *,
    source_record: dict[str, Any] | None = None,
    include_source_identity: bool = True,
) -> dict[str, Any]:
    context_pack_json = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    primary_site = ((context_pack_json.get("sites") or [{}])[0] if isinstance(context_pack_json.get("sites"), list) else {}) or {}
    core_phrases: list[str] = []
    core_phrases.extend(normalized_scope.get("source_capabilities") or [])
    core_phrases.extend(normalized_scope.get("source_workflows") or [])
    core_phrases.extend(normalized_scope.get("source_customer_segments") or [])
    core_phrases.append(str(primary_site.get("summary") or "").strip())

    adjacency_phrases: list[str] = []
    adjacency_phrases.extend(normalized_scope.get("adjacency_box_labels") or [])
    for box in (normalized_scope.get("adjacency_boxes") or []):
        if not isinstance(box, dict):
            continue
        adjacency_phrases.append(str(box.get("label") or "").strip())
        adjacency_phrases.extend(box.get("likely_customer_segments") or [])
        adjacency_phrases.extend(box.get("likely_workflows") or [])
        adjacency_phrases.extend(box.get("retrieval_query_seeds") or [])

    entity_seed_phrases: list[str] = []
    for seed in (normalized_scope.get("company_seeds") or []):
        if isinstance(seed, dict):
            entity_seed_phrases.append(str(seed.get("name") or "").strip())
    entity_seed_phrases.extend(normalized_scope.get("named_account_anchors") or [])
    entity_seed_phrases.extend(normalized_scope.get("geography_expansions") or [])
    if include_source_identity:
        entity_seed_phrases.append(str(primary_site.get("company_name") or "").strip())
    if include_source_identity and isinstance(source_record, dict):
        entity_seed_phrases.extend(source_record.get("brand_names") or [])
        entity_seed_phrases.extend(
            _fr_registry_extract_observation_entities(
                list(((source_record.get("registry_fields") or {}).get("observations") or []))
            )
        )

    return {
        "core": _fr_registry_scope_phrases(core_phrases),
        "adjacent": _fr_registry_scope_phrases(adjacency_phrases),
        "entity_seed": _fr_registry_scope_phrases(entity_seed_phrases),
        "geography_terms": _dedupe_strings(normalized_scope.get("geography_expansions") or [])[:12],
    }


def _looks_like_registry_company_seed(value: Any) -> bool:
    text = _normalize_html_text(value)
    if not text:
        return False
    lowered = text.lower()
    if _looks_like_natural_person_name(text):
        return False
    blocked = (
        "ap-hp",
        "aphp",
        "assistance publique",
        "centre hospitalier",
        "hopital",
        "hôpital",
        "chu",
        "cpts",
        "mairie",
        "commune",
        "universit",
        "minister",
        "minist",
        "prefecture",
        "préfecture",
    )
    if any(token in lowered for token in blocked):
        return False
    tokens = [token for token in re.split(r"[\s,/()-]+", text) if token]
    return 1 <= len(tokens) <= 8


def _looks_like_vendor_category_phrase(value: Any) -> bool:
    text = _normalize_html_text(value).lower()
    if not text:
        return False
    vendorish = (
        "software",
        "logiciel",
        "plateforme",
        "platform",
        "solution",
        "solutions",
        "marketplace",
        "place de marche",
        "place de marché",
        "outil",
        "application",
        "applications",
        "vendor",
        "vms",
    )
    return any(token in text for token in vendorish)


def _fr_registry_seed_query_matches_record(record: dict[str, Any], query: str) -> bool:
    query_norm = _normalize_name_for_matching(query)
    if not query_norm:
        return False
    query_tokens = {token for token in query_norm.split() if len(token) >= 3 and token not in NAME_STOPWORDS}
    candidate_names = _dedupe_strings(
        [
            str(record.get("display_name") or "").strip(),
            str(record.get("legal_name") or "").strip(),
            *list(record.get("brand_names") or []),
            *list(((record.get("registry_fields") or {}).get("commercial_names") or [])),
        ]
    )
    for raw_name in candidate_names:
        name_norm = _normalize_name_for_matching(raw_name)
        if not name_norm:
            continue
        if query_norm == name_norm or query_norm in name_norm or name_norm in query_norm:
            return True
        if query_tokens:
            name_tokens = set(name_norm.split())
            if query_tokens.issubset(name_tokens):
                return True
        if _name_similarity(query_norm, name_norm) >= 0.86:
            return True
    return False


def _fr_registry_seed_query_specs(
    profile: CompanyProfile,
    normalized_scope: dict[str, Any],
    scope_pack: dict[str, Any],
    *,
    source_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    source_name = _source_company_name_from_profile(profile)
    source_identity_norms = {
        value
        for value in [
            _normalize_name_for_matching(source_name),
            _normalize_name_for_matching(_company_label_from_url(profile.buyer_company_url) or ""),
        ]
        if value
    }
    if isinstance(source_record, dict):
        source_identity_norms.update(
            value
            for value in (
                _normalize_name_for_matching(str(source_record.get("display_name") or "").strip()),
                _normalize_name_for_matching(str(source_record.get("legal_name") or "").strip()),
            )
            if value
        )

    def _add(query: Any, *, query_type: str, only_active: bool = True) -> None:
        text = _normalize_html_text(query)
        if not text:
            return
        normalized = _normalize_name_for_matching(text)
        if normalized in source_identity_norms:
            return
        key = (text.lower(), query_type, bool(only_active))
        if key in seen:
            return
        seen.add(key)
        specs.append({"query": text, "query_type": query_type, "only_active": bool(only_active)})

    if isinstance(source_record, dict):
        for alias in _fr_registry_extract_observation_entities(
            list(((source_record.get("registry_fields") or {}).get("observations") or []))
        ):
            if _looks_like_registry_company_seed(alias):
                _add(alias, query_type="source_observation_alias", only_active=False)

    for seed in normalized_scope.get("company_seeds") or []:
        if not isinstance(seed, dict):
            continue
        name = str(seed.get("name") or "").strip()
        if _looks_like_registry_company_seed(name):
            _add(name, query_type="company_seed")

    for raw in normalized_scope.get("named_account_anchors") or []:
        if _looks_like_registry_company_seed(raw):
            _add(raw, query_type="named_account")

    customer_segments = [str(item).strip() for item in (normalized_scope.get("source_customer_segments") or []) if str(item).strip()]
    for box in normalized_scope.get("adjacency_boxes") or []:
        if not isinstance(box, dict):
            continue
        label = str(box.get("label") or "").strip()
        if _looks_like_vendor_category_phrase(label):
            _add(label, query_type="adjacency_label")
        for seed in box.get("retrieval_query_seeds") or []:
            if _looks_like_vendor_category_phrase(seed):
                _add(seed, query_type="adjacency_vendor_phrase")

    return specs[:24]


def _fr_registry_query_type_to_source_path(query_type: str) -> str:
    normalized = str(query_type or "").strip().lower()
    if normalized in {"company_seed", "named_account", "adjacency_label", "adjacency_vendor_phrase"}:
        return "seed_name_registry_lookup"
    if normalized in {"commercial_name_lookup"}:
        return "commercial_name_lookup"
    if normalized in {"source_observation_alias", "observation_counterparty_lookup"}:
        return "observation_counterparty_lookup"
    if normalized in {"history_alias_lookup"}:
        return "history_alias_lookup"
    return "seed_name_registry_lookup"


def _fr_registry_secondary_lookup_specs(
    profile: CompanyProfile,
    source_record: dict[str, Any],
    detailed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    source_name = _source_company_name_from_profile(profile)
    source_identity_norms = {
        value
        for value in [
            _normalize_name_for_matching(source_name),
            _normalize_name_for_matching(_company_label_from_url(profile.buyer_company_url) or ""),
            _normalize_name_for_matching(str(source_record.get("display_name") or "").strip()),
            _normalize_name_for_matching(str(source_record.get("legal_name") or "").strip()),
        ]
        if value
    }

    def _add(query: Any, *, query_type: str, only_active: bool) -> None:
        text = _normalize_html_text(query)
        if not text:
            return
        normalized = _normalize_name_for_matching(text)
        if not normalized or normalized in source_identity_norms:
            return
        key = (normalized, query_type, bool(only_active))
        if key in seen:
            return
        seen.add(key)
        specs.append({"query": text, "query_type": query_type, "only_active": bool(only_active)})

    for record in [source_record, *list(detailed_records or [])]:
        if not isinstance(record, dict):
            continue
        registry_fields = record.get("registry_fields") if isinstance(record.get("registry_fields"), dict) else {}
        for name in registry_fields.get("commercial_names") or []:
            if _looks_like_registry_company_seed(name):
                _add(name, query_type="commercial_name_lookup", only_active=True)
        for alias in _fr_registry_extract_observation_entities(list(registry_fields.get("observations") or [])):
            if _looks_like_registry_company_seed(alias):
                _add(alias, query_type="observation_counterparty_lookup", only_active=False)
        for alias in _fr_registry_extract_history_aliases(list(registry_fields.get("history_labels") or [])):
            if _looks_like_registry_company_seed(alias):
                _add(alias, query_type="history_alias_lookup", only_active=False)

    return specs[:48]


def _fr_registry_lane_fit(
    record: dict[str, Any],
    normalized_scope: dict[str, Any],
    scope_pack: dict[str, Any],
) -> tuple[list[str], list[str], list[str], str, dict[str, Any]]:
    context_text = str(record.get("context_text") or "").lower()
    lane_ids: list[str] = []
    lane_labels: list[str] = []
    source_capability_matches: list[str] = []
    core_terms = set(scope_pack.get("core", {}).get("terms") or [])
    adjacent_terms = set(scope_pack.get("adjacent", {}).get("terms") or [])
    matched_core_terms = [term for term in core_terms if term and term in context_text]
    matched_adjacent_terms = [term for term in adjacent_terms if term and term in context_text and term not in matched_core_terms]
    for capability in normalized_scope.get("source_capabilities") or []:
        token = str(capability or "").strip()
        if token and token.lower() in context_text:
            source_capability_matches.append(token)
    for box in (normalized_scope.get("adjacency_boxes") or []):
        if not isinstance(box, dict):
            continue
        label = str(box.get("label") or "").strip()
        if not label:
            continue
        lane_terms = [label.lower()]
        lane_terms.extend(str(item).strip().lower() for item in (box.get("likely_customer_segments") or []) if str(item).strip())
        lane_terms.extend(str(item).strip().lower() for item in (box.get("likely_workflows") or []) if str(item).strip())
        if any(term and term in context_text for term in lane_terms):
            if str(box.get("id") or "").strip():
                lane_ids.append(str(box.get("id") or "").strip())
            lane_labels.append(label)
    core_match_count = len(_dedupe_strings(source_capability_matches)) + len(_dedupe_strings(matched_core_terms))
    adjacent_match_count = len(_dedupe_strings(lane_labels)) + len(_dedupe_strings(matched_adjacent_terms))
    if core_match_count > 0:
        scope_bucket = "core"
    elif adjacent_match_count > 0:
        scope_bucket = "adjacent"
    else:
        scope_bucket = "broad_market"
    matched_core_labels = _dedupe_strings(
        list(source_capability_matches)
        + ([str(item).strip() for item in (normalized_scope.get("source_capabilities") or []) if str(item).strip()] if matched_core_terms else [])
    )
    return (
        _dedupe_strings(lane_ids),
        _dedupe_strings(lane_labels),
        _dedupe_strings(source_capability_matches),
        scope_bucket,
        {
            "matched_node_ids": _dedupe_strings(lane_ids),
            "matched_node_labels": _dedupe_strings(matched_core_labels + lane_labels),
            "core_match_count": core_match_count,
            "adjacent_match_count": adjacent_match_count,
            "matched_core_terms": _dedupe_strings(matched_core_terms)[:16],
            "matched_adjacent_terms": _dedupe_strings(matched_adjacent_terms)[:16],
        },
    )


def _directness_from_node_fit_summary(
    summary: dict[str, Any] | None,
    *,
    fallback: Optional[str] = None,
) -> str:
    payload = summary if isinstance(summary, dict) else {}
    core = int(payload.get("core_match_count") or 0)
    adjacent = int(payload.get("adjacent_match_count") or 0)
    if core > 0:
        return "direct"
    if adjacent > 0:
        return "adjacent"
    normalized_fallback = str(fallback or "").strip().lower()
    if normalized_fallback in {"broad_market", "out_of_scope"}:
        return normalized_fallback
    return "broad_market"


def _merge_node_fit_summary(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    left = existing if isinstance(existing, dict) else {}
    right = incoming if isinstance(incoming, dict) else {}
    return {
        "matched_node_ids": _dedupe_strings(list(left.get("matched_node_ids") or []) + list(right.get("matched_node_ids") or [])),
        "matched_node_labels": _dedupe_strings(list(left.get("matched_node_labels") or []) + list(right.get("matched_node_labels") or [])),
        "core_match_count": max(int(left.get("core_match_count") or 0), int(right.get("core_match_count") or 0)),
        "adjacent_match_count": max(int(left.get("adjacent_match_count") or 0), int(right.get("adjacent_match_count") or 0)),
        "matched_core_terms": _dedupe_strings(list(left.get("matched_core_terms") or []) + list(right.get("matched_core_terms") or []))[:16],
        "matched_adjacent_terms": _dedupe_strings(list(left.get("matched_adjacent_terms") or []) + list(right.get("matched_adjacent_terms") or []))[:16],
        "node_fit_score": max(float(left.get("node_fit_score") or 0.0), float(right.get("node_fit_score") or 0.0)),
    }


def _summary_sentence(text: Any, limit: int = 240) -> Optional[str]:
    normalized = _normalize_html_text(text)
    if not normalized:
        return None
    sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0].strip()
    compact = sentence or normalized
    return compact[:limit] if compact else None


def _fr_registry_candidate_short_description(
    record: dict[str, Any],
    semantic_meta: dict[str, Any],
) -> Optional[str]:
    registry_fields = record.get("registry_fields") if isinstance(record.get("registry_fields"), dict) else {}
    object_text = _summary_sentence(registry_fields.get("object_text"))
    if object_text:
        return object_text
    activity_description = _summary_sentence(registry_fields.get("activity_description"))
    if activity_description:
        return activity_description
    observations = registry_fields.get("observations") if isinstance(registry_fields.get("observations"), list) else []
    for observation in observations:
        snippet = _summary_sentence(observation)
        if snippet:
            return snippet
    matched_nodes = [
        str(label).strip()
        for label in (((semantic_meta.get("node_fit_summary") or {}).get("matched_node_labels")) or [])
        if str(label).strip()
    ]
    activity_label = str(record.get("activity_label") or "").strip()
    if activity_label and matched_nodes:
        return f"{activity_label} aligned with {', '.join(matched_nodes[:2])}."[:240]
    if activity_label:
        return activity_label[:240]
    ape_code = str((registry_fields or {}).get("ape_code") or record.get("activity_code") or "").strip()
    return ape_code[:240] if ape_code else None


def _fr_registry_recall_signal(
    record: dict[str, Any],
    *,
    scope_pack: dict[str, Any],
    normalized_scope: dict[str, Any],
    code_distance: int,
) -> dict[str, Any]:
    score, semantic_meta = _fr_registry_semantic_score(
        record,
        scope_pack=scope_pack,
        normalized_scope=normalized_scope,
        code_distance=code_distance,
    )
    node_fit_summary = semantic_meta.get("node_fit_summary") if isinstance(semantic_meta.get("node_fit_summary"), dict) else {}
    node_fit_score = float(node_fit_summary.get("node_fit_score") or 0.0)
    matched_terms = list(semantic_meta.get("matched_terms") or [])
    has_signal = bool(
        node_fit_score > 0.0
        or semantic_meta.get("seed_lookup_match")
        or semantic_meta.get("seed_name_match")
        or semantic_meta.get("commercial_name_match")
        or matched_terms
        or list((record.get("registry_fields") or {}).get("commercial_names") or [])
    )
    return {
        "score": score,
        "semantic_meta": semantic_meta,
        "node_fit_score": node_fit_score,
        "matched_terms_count": len(matched_terms),
        "has_signal": has_signal,
    }


def _fr_registry_detail_priority(item: dict[str, Any]) -> tuple[float, int, int, int, int, float]:
    record = item.get("record") if isinstance(item.get("record"), dict) else {}
    semantic_meta = item.get("semantic_meta") if isinstance(item.get("semantic_meta"), dict) else {}
    registry_fields = record.get("registry_fields") if isinstance(record.get("registry_fields"), dict) else {}
    node_fit_summary = semantic_meta.get("node_fit_summary") if isinstance(semantic_meta.get("node_fit_summary"), dict) else {}
    node_fit_score = float(node_fit_summary.get("node_fit_score") or 0.0)
    source_paths = set(record.get("lookup_source_paths") or [])
    non_code_lookup = int(bool(source_paths - {"code_neighborhood_crawl"}))
    seed_lookup = int(bool(record.get("seed_lookup_match")))
    commercial_name_count = len(registry_fields.get("commercial_names") or [])
    matched_terms_count = len(semantic_meta.get("matched_terms") or [])
    score = float(item.get("score") or 0.0)
    return (
        node_fit_score,
        seed_lookup,
        non_code_lookup,
        commercial_name_count,
        matched_terms_count,
        score,
    )


def _fr_registry_semantic_score(
    record: dict[str, Any],
    *,
    scope_pack: dict[str, Any],
    normalized_scope: dict[str, Any],
    code_distance: int,
) -> tuple[float, dict[str, Any]]:
    context_text = unicodedata.normalize("NFKD", str(record.get("context_text") or "").lower())
    context_text = "".join(ch for ch in context_text if not unicodedata.combining(ch))
    registry_fields = record.get("registry_fields") if isinstance(record.get("registry_fields"), dict) else {}
    terms = _dedupe_strings(
        list(scope_pack.get("core", {}).get("terms") or [])
        + list(scope_pack.get("adjacent", {}).get("terms") or [])
        + list(scope_pack.get("entity_seed", {}).get("terms") or [])
    )
    matched_terms = [term for term in terms if term and term in context_text]
    matched_terms = _dedupe_strings(matched_terms)
    lane_ids, lane_labels, source_capability_matches, scope_bucket, node_fit_summary = _fr_registry_lane_fit(
        record,
        normalized_scope,
        scope_pack,
    )
    software_relevance = max(
        _software_signal_score_from_codes(record.get("industry_codes") or []),
        1.0 if any(token in context_text for token in ("logiciel", "plateforme", "platform", "saas", "application")) else 0.0,
    )
    observations = registry_fields.get("observations") if isinstance(registry_fields.get("observations"), list) else []
    object_text = str(registry_fields.get("object_text") or "").strip()
    activity_description = str(registry_fields.get("activity_description") or "").strip()
    has_registry_text = bool(object_text or activity_description or observations)
    commercial_names = registry_fields.get("commercial_names") if isinstance(registry_fields.get("commercial_names"), list) else []
    broad_infra_hits = [
        token for token in FR_REGISTRY_GENERIC_INFRA_TOKENS
        if token and token in context_text
    ]
    healthcare_staffing_hits = [
        token for token in (
            "sante",
            "santé",
            "professionnels de santé",
            "etablissements de santé",
            "établissements de santé",
            "recrutement",
            "mise en relation",
            "remplacement",
            "vacation",
            "vacataire",
            "planning",
            "pool",
            "interim",
            "intérim",
        )
        if token in context_text
    ]
    object_observation_relevance = min(
        1.0,
        (
            (0.6 if object_text else 0.0)
            + (0.35 if activity_description else 0.0)
            + min(0.4, 0.08 * len(observations))
            + min(0.6, 0.06 * len(healthcare_staffing_hits))
        ),
    )
    commercial_name_match = 1.0 if any(
        term and any(term in str(name or "").lower() for name in commercial_names)
        for term in terms
    ) else 0.0
    seed_name_match = 1.0 if any(
        phrase and (
            phrase.lower() in context_text
            or any(phrase.lower() in str(name or "").lower() for name in commercial_names)
            or phrase.lower() in str(record.get("display_name") or "").lower()
            or phrase.lower() in str(record.get("legal_name") or "").lower()
        )
        for phrase in (scope_pack.get("entity_seed", {}).get("phrases") or [])
    ) else 0.0
    seed_lookup_match = bool(record.get("seed_lookup_match"))
    natural_person_penalty = (
        1.0
        if _looks_like_natural_person_name(record.get("legal_name") or record.get("display_name") or record.get("name"))
        and not commercial_names
        and not record.get("website")
        else 0.0
    )
    node_fit_score = float(node_fit_summary.get("core_match_count") or 0) * 10.0 + float(node_fit_summary.get("adjacent_match_count") or 0) * 5.0
    score = 12.0
    score += node_fit_score
    score += commercial_name_match * 8.0
    score += seed_name_match * 8.0
    if seed_lookup_match:
        score += 12.0
    score += object_observation_relevance * 12.0
    score += min(20.0, float(len(matched_terms)) * 2.0)
    score += max(0.0, 8.0 - (2.0 * float(code_distance)))
    score += software_relevance * 8.0
    if record.get("website"):
        score += 3.0
    if record.get("is_employer"):
        score += 2.0
    if str(record.get("employee_band") or "").strip():
        score += 1.0
    lowered_name = str(record.get("display_name") or record.get("name") or "").strip().lower()
    if any(token in lowered_name for token in FR_REGISTRY_SOFT_BLOCKED_NAME_TOKENS):
        score -= 20.0
    if bool((record.get("registry_fields") or {}).get("is_service_public")):
        score -= 20.0
    if broad_infra_hits and not matched_terms and not healthcare_staffing_hits:
        score -= min(18.0, 4.5 * len(broad_infra_hits))
    if not has_registry_text and not matched_terms:
        score -= 10.0
    if scope_bucket == "broad_market" and len(matched_terms) <= 1 and not healthcare_staffing_hits:
        score -= 8.0
    if natural_person_penalty:
        score -= 20.0
    out_of_scope = bool(
        node_fit_score <= 0.0
        and (
            bool((record.get("registry_fields") or {}).get("is_service_public"))
            or natural_person_penalty
            or (broad_infra_hits and not healthcare_staffing_hits)
        )
    )
    if out_of_scope:
        scope_bucket = "out_of_scope"
        score -= 30.0
    node_fit_summary["node_fit_score"] = round(float(node_fit_score), 3)
    return (
        round(score, 3),
        {
            "matched_terms": matched_terms[:24],
            "lane_ids": lane_ids,
            "lane_labels": lane_labels,
            "source_capability_matches": source_capability_matches,
            "scope_bucket": scope_bucket,
            "node_fit_summary": node_fit_summary,
            "software_relevance": round(float(software_relevance), 3),
            "object_observation_relevance": round(float(object_observation_relevance), 3),
            "commercial_name_match": round(float(commercial_name_match), 3),
            "seed_name_match": round(float(seed_name_match), 3),
            "seed_lookup_match": seed_lookup_match,
            "generic_infra_hits": _dedupe_strings(broad_infra_hits)[:12],
            "healthcare_staffing_hits": _dedupe_strings(healthcare_staffing_hits)[:16],
            "natural_person_penalty": bool(natural_person_penalty),
            "code_distance": int(code_distance),
            "out_of_scope": out_of_scope,
        },
    )


def _fr_registry_code_neighborhood(
    source_record: dict[str, Any],
    *,
    semantic_terms: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(code: Any, distance: int, reason: str) -> None:
        normalized = _normalize_fr_registry_code(code)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append({"code": normalized, "distance": int(distance), "reason": reason})

    for code in [
        source_record.get("activity_code"),
        source_record.get("activity_code_naf25"),
    ]:
        _add(code, 0, "source_company_code")

    source_codes = {str(item.get("code") or "") for item in candidates}
    if any(term in set(semantic_terms.get("terms") or []) for term in ("staffing", "recruitment", "interim", "intérim", "replacement")):
        for code in FR_REGISTRY_RECRUITMENT_APE_CODES:
            _add(code, 1, "recruitment_family")
    if any(code.startswith(("58.", "62.", "63.")) for code in source_codes):
        for code in FR_REGISTRY_DIGITAL_APE_CODES:
            _add(code, 2, "digital_family")

    return candidates[:10]


def _source_company_name_from_profile(profile: CompanyProfile) -> str:
    context_pack_json = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    sites = context_pack_json.get("sites") if isinstance(context_pack_json.get("sites"), list) else []
    primary_site = sites[0] if sites else {}
    return (
        str((primary_site or {}).get("company_name") or "").strip()
        or str(_company_label_from_url(profile.buyer_company_url) or "").strip()
        or "Source Company"
    )


def _source_company_domain_from_profile(profile: CompanyProfile) -> Optional[str]:
    context_pack_json = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    sites = context_pack_json.get("sites") if isinstance(context_pack_json.get("sites"), list) else []
    primary_site = (sites[0] if sites else {}) or {}
    for raw in (
        (primary_site or {}).get("website"),
        (primary_site or {}).get("url"),
        profile.buyer_company_url,
    ):
        domain = normalize_domain(raw)
        if domain:
            return domain
    return None


def _strip_scope_identity_terms(scope_pack: dict[str, Any], identity_phrases: list[str]) -> dict[str, Any]:
    identity_tokens = set((_fr_registry_scope_phrases(identity_phrases) or {}).get("terms") or [])
    if not identity_tokens:
        return scope_pack
    sanitized = dict(scope_pack or {})
    for bucket in ("core", "adjacent", "entity_seed"):
        current = sanitized.get(bucket) if isinstance(sanitized.get(bucket), dict) else {}
        terms = [
            str(term).strip()
            for term in (current.get("terms") or [])
            if str(term).strip() and str(term).strip() not in identity_tokens
        ]
        phrases = [
            str(phrase).strip()
            for phrase in (current.get("phrases") or [])
            if str(phrase).strip()
            and not any(token in _fr_registry_scope_phrases([str(phrase).strip()]).get("terms", []) for token in identity_tokens)
        ]
        sanitized[bucket] = {
            "terms": _dedupe_strings(terms)[:160],
            "phrases": _dedupe_strings(phrases)[:80],
        }
    return sanitized


def _should_use_france_registry_universe(profile: CompanyProfile) -> bool:
    if not bool(getattr(settings, "discovery_fr_registry_first_enabled", True)):
        return False
    geo_scope = profile.geo_scope if isinstance(profile.geo_scope, dict) else {}
    include_countries = {
        normalize_country(country)
        for country in (geo_scope.get("include_countries") or [])
        if normalize_country(country)
    }
    if "FR" in include_countries:
        return True
    buyer_domain = normalize_domain(profile.buyer_company_url)
    if buyer_domain and buyer_domain.endswith(".fr"):
        return True
    context_pack_json = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    sites = context_pack_json.get("sites") if isinstance(context_pack_json.get("sites"), list) else []
    primary_url = str(((sites[0] if sites else {}) or {}).get("url") or "").strip()
    return bool(normalize_domain(primary_url) and normalize_domain(primary_url).endswith(".fr"))


def _resolve_fr_source_registry_record(
    profile: CompanyProfile,
    normalized_scope: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    source_name = _source_company_name_from_profile(profile)
    source_domain = _source_company_domain_from_profile(profile)
    rows, error = _fr_registry_search(query=source_name, page=1, per_page=12, only_active=True)
    diagnostics: dict[str, Any] = {
        "source_company_name": source_name,
        "source_company_domain": source_domain,
        "query_error": error,
        "query_hits": len(rows),
    }
    if error or not rows:
        return None, diagnostics
    scope_pack = _fr_registry_scope_pack(profile, normalized_scope, include_source_identity=False)
    identity_phrases = [source_name]
    if source_domain:
        identity_phrases.extend(
            [
                source_domain,
                source_domain.split(".")[0],
                _company_label_from_url(profile.buyer_company_url) or "",
            ]
        )
    scope_pack = _strip_scope_identity_terms(scope_pack, identity_phrases)
    company_name_norm = _normalize_name_for_matching(source_name)
    best: Optional[dict[str, Any]] = None
    best_score = -1.0
    for row in rows:
        record = _fr_registry_source_record(row, query_hint=source_name)
        display_norm = _normalize_name_for_matching(str(record.get("display_name") or record.get("name") or ""))
        legal_norm = _normalize_name_for_matching(str(record.get("legal_name") or ""))
        name_score = max(
            _name_similarity(company_name_norm, display_norm),
            _name_similarity(company_name_norm, legal_norm),
        )
        semantic_score, semantic_meta = _fr_registry_semantic_score(
            record,
            scope_pack=scope_pack,
            normalized_scope=normalized_scope,
            code_distance=0,
        )
        total_score = (name_score * 50.0) + semantic_score
        record_domain = normalize_domain(record.get("website"))
        if source_domain and record_domain and source_domain == record_domain:
            total_score += 35.0
        if record.get("website"):
            total_score += 2.0
        if total_score > best_score:
            best_score = total_score
            best = {
                **record,
                "semantic_meta": semantic_meta,
                "resolution_score": round(total_score, 3),
            }
    diagnostics["best_score"] = round(best_score, 3) if best_score >= 0 else None
    return best, diagnostics


def _fr_inpi_public_url(siren: str) -> str:
    return f"https://data.inpi.fr/entreprises/{str(siren or '').strip()}"


def _fr_registry_extract_history_aliases(history_labels: list[str]) -> list[str]:
    aliases: list[str] = []
    for label in history_labels or []:
        text = _normalize_html_text(label)
        if not text:
            continue
        if _looks_like_registry_company_seed(text):
            aliases.append(text)
        for match in re.findall(r"\b[A-Z][A-Z0-9&'’.-]{2,}(?:\s+[A-Z0-9&'’.-]{2,})*\b", text):
            normalized = re.sub(r"\s+", " ", match).strip(" ,.;:-")
            if _looks_like_registry_company_seed(normalized):
                aliases.append(normalized)
    return _dedupe_strings(aliases)[:16]


def _fr_registry_collect_commercial_names_from_inpi_company(company: dict[str, Any]) -> list[str]:
    formality = company.get("formality") if isinstance(company.get("formality"), dict) else {}
    content = formality.get("content") if isinstance(formality.get("content"), dict) else {}
    personne_morale = content.get("personneMorale") if isinstance(content.get("personneMorale"), dict) else {}
    names: list[str] = []

    def _add(value: Any) -> None:
        text = _normalize_html_text(value)
        if text:
            names.append(text)

    principal = personne_morale.get("etablissementPrincipal") if isinstance(personne_morale.get("etablissementPrincipal"), dict) else {}
    principal_desc = principal.get("descriptionEtablissement") if isinstance(principal.get("descriptionEtablissement"), dict) else {}
    _add(principal_desc.get("nomCommercial"))
    _add(principal_desc.get("enseigne"))

    for etab in personne_morale.get("autresEtablissements") or []:
        if not isinstance(etab, dict):
            continue
        desc = etab.get("descriptionEtablissement") if isinstance(etab.get("descriptionEtablissement"), dict) else {}
        _add(desc.get("nomCommercial"))
        _add(desc.get("enseigne"))

    return _dedupe_strings(names)[:16]


def _decode_jwt_without_verification(token: str) -> dict[str, Any]:
    raw = str(token or "").strip()
    if raw.count(".") < 2:
        return {}
    try:
        payload = raw.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode()).decode()
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _fr_inpi_token_expired(token: str, *, skew_seconds: int = 30) -> bool:
    payload = _decode_jwt_without_verification(token)
    exp = payload.get("exp")
    try:
        exp_ts = float(exp)
    except (TypeError, ValueError):
        return False
    return exp_ts <= (time.time() + float(skew_seconds))


def _fr_inpi_login(timeout_seconds: int, *, force_refresh: bool = False) -> tuple[Optional[str], Optional[str]]:
    cached_token = str(_FR_INPI_LOGIN_CACHE.get("token") or "").strip()
    cached_expires_at = float(_FR_INPI_LOGIN_CACHE.get("expires_at") or 0.0)
    if not force_refresh and cached_token and cached_expires_at > (time.time() + 30):
        return cached_token, None

    username = str(getattr(settings, "inpi_username", "") or "").strip()
    password = str(getattr(settings, "inpi_password", "") or "").strip()
    if not username or not password:
        return None, "inpi_credentials_missing"

    url = "https://registre-national-entreprises.inpi.fr/api/sso/login"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers, http2=False) as client:
            response = client.post(url, json={"username": username, "password": password})
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        return None, f"inpi_login_http_{exc.response.status_code}"
    except Exception:
        return None, "inpi_login_failed"

    token = ""
    if isinstance(payload, dict):
        token = str(payload.get("token") or "").strip()
    if not token:
        return None, "inpi_login_missing_token"

    exp_payload = _decode_jwt_without_verification(token)
    exp = exp_payload.get("exp")
    try:
        expires_at = float(exp)
    except (TypeError, ValueError):
        expires_at = time.time() + 3600
    _FR_INPI_LOGIN_CACHE["token"] = token
    _FR_INPI_LOGIN_CACHE["expires_at"] = expires_at
    return token, None


def _fr_inpi_bearer_token(timeout_seconds: int, *, force_refresh: bool = False) -> tuple[Optional[str], Optional[str]]:
    static_token = str(getattr(settings, "inpi_token", "") or "").strip()
    if static_token and not force_refresh and not _fr_inpi_token_expired(static_token):
        return static_token, None
    if static_token and _fr_inpi_token_expired(static_token):
        login_token, login_error = _fr_inpi_login(timeout_seconds, force_refresh=force_refresh)
        return login_token, login_error or "inpi_token_expired"
    if static_token and force_refresh:
        login_token, login_error = _fr_inpi_login(timeout_seconds, force_refresh=True)
        return login_token, login_error or "inpi_token_refresh_failed"
    login_token, login_error = _fr_inpi_login(timeout_seconds, force_refresh=force_refresh)
    if login_token:
        return login_token, None
    return None, login_error or "inpi_token_missing"


def _fetch_fr_inpi_detail_fields(siren: str, timeout_seconds: Optional[int] = None) -> dict[str, Any]:
    normalized_siren = str(siren or "").strip()
    if not normalized_siren:
        return {}
    url = "https://registre-national-entreprises.inpi.fr/api/companies"
    effective_timeout = max(1, int(timeout_seconds or getattr(settings, "discovery_fr_registry_detail_timeout_seconds", 4)))
    token, auth_error = _fr_inpi_bearer_token(effective_timeout)
    if not token:
        return {"auth_error": auth_error} if auth_error else {}

    payload: Any = None
    for attempt in range(2):
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        try:
            with httpx.Client(timeout=effective_timeout, follow_redirects=True, headers=headers, http2=False) as client:
                response = client.get(
                    url,
                    params={"page": 1, "pageSize": 1, "siren[]": normalized_siren},
                )
                if response.status_code == 401 and attempt == 0:
                    token, refresh_error = _fr_inpi_bearer_token(effective_timeout, force_refresh=True)
                    if token:
                        continue
                    return {"auth_error": refresh_error or "inpi_unauthorized"}
                response.raise_for_status()
                payload = response.json()
                break
        except httpx.HTTPStatusError as exc:
            return {"auth_error": f"inpi_http_{exc.response.status_code}"}
        except Exception:
            return {"auth_error": "inpi_request_failed"}
    if not isinstance(payload, list) or not payload:
        return {}
    company = next(
        (
            row
            for row in payload
            if isinstance(row, dict) and str(row.get("siren") or "").strip() == normalized_siren
        ),
        payload[0] if isinstance(payload[0], dict) else None,
    )
    if not isinstance(company, dict):
        return {}

    formality = company.get("formality") if isinstance(company.get("formality"), dict) else {}
    content = formality.get("content") if isinstance(formality.get("content"), dict) else {}
    personne_morale = content.get("personneMorale") if isinstance(content.get("personneMorale"), dict) else {}
    identite = personne_morale.get("identite") if isinstance(personne_morale.get("identite"), dict) else {}
    description = identite.get("description") if isinstance(identite.get("description"), dict) else {}
    principal = personne_morale.get("etablissementPrincipal") if isinstance(personne_morale.get("etablissementPrincipal"), dict) else {}
    principal_activities = principal.get("activites") if isinstance(principal.get("activites"), list) else []
    principal_domains = principal.get("nomsDeDomaine") if isinstance(principal.get("nomsDeDomaine"), list) else []
    observations_payload = personne_morale.get("observations") if isinstance(personne_morale.get("observations"), dict) else {}
    observations: list[str] = []
    for bucket in ("rcs", "rnm"):
        entries = observations_payload.get(bucket) if isinstance(observations_payload.get(bucket), list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = _normalize_html_text(entry.get("texte"))
            if text:
                observations.append(text)

    domains = _dedupe_strings(
        [
            str(item.get("nomDomaine") or "").strip().lower()
            for item in principal_domains
            if isinstance(item, dict) and str(item.get("nomDomaine") or "").strip()
        ]
    )
    activity_description = _first_non_empty(
        *[
            _normalize_html_text(activity.get("descriptionDetaillee"))
            for activity in principal_activities
            if isinstance(activity, dict)
        ]
    )
    history = formality.get("historique") if isinstance(formality.get("historique"), list) else []
    history_labels = _dedupe_strings(
        [
            _normalize_html_text(entry.get("libelleEvenement"))
            for entry in history
            if isinstance(entry, dict) and _normalize_html_text(entry.get("libelleEvenement"))
        ]
    )[:16]
    commercial_names = _fr_registry_collect_commercial_names_from_inpi_company(company)
    object_text = _normalize_html_text(description.get("objet"))

    return {
        "inpi_company_id": str(company.get("id") or "").strip() or None,
        "inpi_url": _fr_inpi_public_url(normalized_siren),
        "object_text": object_text or None,
        "activity_description": activity_description or None,
        "commercial_names": commercial_names,
        "domains": domains[:8],
        "observations": _dedupe_strings(observations)[:8],
        "history_labels": history_labels,
        "auth_error": None,
    }


def _merge_fr_registry_record_detail(base_record: dict[str, Any], detail_record: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_record or {})
    detail = dict(detail_record or {})
    for key in (
        "name",
        "display_name",
        "legal_name",
        "website",
        "country",
        "registry_id",
        "registry_source",
        "registry_url",
        "status",
        "activity_code",
        "activity_code_naf25",
        "activity_label",
        "context_text",
    ):
        detail_value = detail.get(key)
        if isinstance(detail_value, str):
            detail_value = detail_value.strip()
        if detail_value:
            merged[key] = detail_value

    merged["brand_names"] = _dedupe_strings(
        list(base_record.get("brand_names") or []) + list(detail.get("brand_names") or [])
    )
    merged["industry_codes"] = _dedupe_strings(
        list(base_record.get("industry_codes") or []) + list(detail.get("industry_codes") or [])
    )
    merged["industry_keywords"] = _dedupe_strings(
        list(base_record.get("industry_keywords") or []) + list(detail.get("industry_keywords") or [])
    )
    base_registry_fields = base_record.get("registry_fields") if isinstance(base_record.get("registry_fields"), dict) else {}
    detail_registry_fields = detail.get("registry_fields") if isinstance(detail.get("registry_fields"), dict) else {}
    merged_registry_fields = {**base_registry_fields, **detail_registry_fields}
    if isinstance(base_registry_fields.get("commercial_names"), list) or isinstance(detail_registry_fields.get("commercial_names"), list):
        merged_registry_fields["commercial_names"] = _dedupe_strings(
            list(base_registry_fields.get("commercial_names") or []) + list(detail_registry_fields.get("commercial_names") or [])
        )
    if isinstance(base_registry_fields.get("observations"), list) or isinstance(detail_registry_fields.get("observations"), list):
        merged_registry_fields["observations"] = _dedupe_strings(
            list(base_registry_fields.get("observations") or []) + list(detail_registry_fields.get("observations") or [])
        )[:8]
        merged_registry_fields["observation_count"] = len(merged_registry_fields["observations"])
    if isinstance(base_registry_fields.get("history_labels"), list) or isinstance(detail_registry_fields.get("history_labels"), list):
        merged_registry_fields["history_labels"] = _dedupe_strings(
            list(base_registry_fields.get("history_labels") or []) + list(detail_registry_fields.get("history_labels") or [])
        )[:16]
    if isinstance(base_registry_fields.get("domains"), list) or isinstance(detail_registry_fields.get("domains"), list):
        merged_registry_fields["domains"] = _dedupe_strings(
            list(base_registry_fields.get("domains") or []) + list(detail_registry_fields.get("domains") or [])
        )[:8]
    merged["registry_fields"] = merged_registry_fields
    return merged


def _fetch_fr_registry_detail_record(
    record: dict[str, Any],
    *,
    search_timeout_seconds: Optional[int] = None,
    detail_timeout_seconds: Optional[int] = None,
) -> dict[str, Any]:
    siren = str(record.get("registry_id") or "").strip()
    legal_name = str(record.get("legal_name") or record.get("display_name") or record.get("name") or "").strip()
    merged_row: dict[str, Any] = {}
    for query in [siren, legal_name]:
        if not query:
            continue
        rows, _error = _fr_registry_search(
            query=query,
            page=1,
            per_page=4,
            only_active=True,
            timeout_seconds=search_timeout_seconds,
        )
        for row in rows:
            row_siren = str(row.get("siren") or "").strip()
            if siren and row_siren and row_siren != siren:
                continue
            merged_row = {**merged_row, **row}
            existing_matching = merged_row.get("matching_etablissements") if isinstance(merged_row.get("matching_etablissements"), list) else []
            new_matching = row.get("matching_etablissements") if isinstance(row.get("matching_etablissements"), list) else []
            merged_row["matching_etablissements"] = existing_matching + [item for item in new_matching if isinstance(item, dict)]
            if row.get("siege") and not merged_row.get("siege"):
                merged_row["siege"] = row.get("siege")
            if row.get("complements") and not merged_row.get("complements"):
                merged_row["complements"] = row.get("complements")
    detailed_record = _fr_registry_source_record(merged_row, query_hint=siren or legal_name) if merged_row else dict(record)
    merged = _merge_fr_registry_record_detail(record, detailed_record)

    inpi_fields = _fetch_fr_inpi_detail_fields(siren, timeout_seconds=detail_timeout_seconds)
    auth_error = str(inpi_fields.get("auth_error") or "").strip() if isinstance(inpi_fields, dict) else ""
    if auth_error:
        registry_fields = merged.get("registry_fields") if isinstance(merged.get("registry_fields"), dict) else {}
        registry_fields["inpi_auth_error"] = auth_error
        merged["registry_fields"] = registry_fields
    if inpi_fields:
        object_text = str(inpi_fields.get("object_text") or "").strip()
        activity_description = str(inpi_fields.get("activity_description") or "").strip()
        observations = inpi_fields.get("observations") if isinstance(inpi_fields.get("observations"), list) else []
        commercial_names = inpi_fields.get("commercial_names") if isinstance(inpi_fields.get("commercial_names"), list) else []
        domains = inpi_fields.get("domains") if isinstance(inpi_fields.get("domains"), list) else []
        history_labels = inpi_fields.get("history_labels") if isinstance(inpi_fields.get("history_labels"), list) else []
        domain = str(domains[0] or "").strip() if domains else ""
        if domain and not str(merged.get("website") or "").strip():
            merged["website"] = _fr_registry_website({"site_internet": domain}) or f"https://{domain}"
        merged["brand_names"] = _dedupe_strings(list(merged.get("brand_names") or []) + commercial_names)
        extra_context = " ".join([object_text, activity_description] + observations + history_labels + commercial_names).strip()
        if extra_context:
            merged["context_text"] = " ".join(
                part for part in [str(merged.get("context_text") or "").strip(), extra_context] if part
            ).strip()
        registry_fields = merged.get("registry_fields") if isinstance(merged.get("registry_fields"), dict) else {}
        registry_fields["object_text_present"] = bool(object_text)
        if object_text:
            registry_fields["object_text"] = object_text[:1600]
        if activity_description:
            registry_fields["activity_description"] = activity_description[:1200]
            registry_fields["activity_description_present"] = True
        if observations:
            registry_fields["observations"] = observations[:8]
            registry_fields["observation_count"] = len(observations[:8])
        if commercial_names:
            registry_fields["commercial_names"] = _dedupe_strings(
                list(registry_fields.get("commercial_names") or []) + commercial_names
            )[:16]
        if history_labels:
            registry_fields["history_labels"] = history_labels[:16]
            registry_fields["history_count"] = len(history_labels[:16])
        if domain:
            registry_fields["domain"] = domain
        if domains:
            registry_fields["domains"] = domains[:8]
        if inpi_fields.get("inpi_company_id"):
            registry_fields["inpi_company_id"] = inpi_fields.get("inpi_company_id")
        registry_fields["inpi_url"] = inpi_fields.get("inpi_url")
        merged["registry_fields"] = registry_fields
    return merged


def _build_france_registry_universe_candidates(
    profile: CompanyProfile,
    normalized_scope: dict[str, Any],
    *,
    budget_overrides: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_record, source_diagnostics = _resolve_fr_source_registry_record(profile, normalized_scope)
    if source_record is None:
        return [], {"source_resolution": source_diagnostics, "errors": ["fr_registry_source_unresolved"]}

    budget = budget_overrides if isinstance(budget_overrides, dict) else {}
    per_page = max(5, int(getattr(settings, "discovery_fr_registry_per_page", 25)))
    pages_per_code = max(1, int(budget.get("pages_per_code", getattr(settings, "discovery_fr_registry_pages_per_code", 4))))
    max_pages_per_code = max(
        pages_per_code,
        int(budget.get("max_pages_per_code", getattr(settings, "discovery_fr_registry_max_pages_per_code", pages_per_code))),
    )
    candidate_cap = max(50, int(budget.get("candidate_cap", getattr(settings, "discovery_fr_registry_candidate_cap", 800))))
    detail_cap = max(0, int(budget.get("detail_cap", getattr(settings, "discovery_fr_registry_detail_cap", 80))))
    search_timeout_seconds = max(
        1,
        int(budget.get("search_timeout_seconds", getattr(settings, "discovery_fr_registry_search_timeout_seconds", 3))),
    )
    detail_timeout_seconds = max(
        1,
        int(budget.get("detail_timeout_seconds", getattr(settings, "discovery_fr_registry_detail_timeout_seconds", 4))),
    )
    max_total_queries = max(
        1,
        int(budget.get("max_total_queries", getattr(settings, "discovery_fr_registry_max_total_queries", 200))),
    )
    max_elapsed_seconds = max(
        1,
        int(budget.get("max_elapsed_seconds", getattr(settings, "discovery_fr_registry_max_elapsed_seconds", 45))),
    )
    page_extension_min_hits = max(
        1,
        int(budget.get("page_extension_min_hits", getattr(settings, "discovery_fr_registry_page_extension_min_hits", 2))),
    )
    page_stop_after_no_signal = max(
        1,
        int(budget.get("page_stop_after_no_signal", getattr(settings, "discovery_fr_registry_page_stop_after_no_signal", 2))),
    )

    raw_rows: dict[str, tuple[dict[str, Any], int, str]] = {}
    query_count = 0
    query_budget_exhausted = False
    elapsed_budget_exhausted = False
    started_at = time.perf_counter()
    code_fetch_stats: dict[str, dict[str, Any]] = {}
    seed_per_query = max(
        3,
        int(budget.get("seed_per_query", getattr(settings, "discovery_fr_registry_seed_per_query", 8))),
    )
    seed_query_cap = max(
        0,
        int(budget.get("seed_query_cap", getattr(settings, "discovery_fr_registry_seed_query_cap", 24))),
    )
    seed_query_reserve = max(
        0,
        int(budget.get("seed_query_reserve", getattr(settings, "discovery_fr_registry_seed_query_reserve", 4))),
    )
    secondary_seed_per_query = max(
        3,
        int(budget.get("secondary_seed_per_query", getattr(settings, "discovery_fr_registry_secondary_seed_per_query", 6))),
    )
    secondary_query_cap = max(
        0,
        int(budget.get("secondary_query_cap", getattr(settings, "discovery_fr_registry_secondary_query_cap", 48))),
    )
    secondary_query_reserve = max(
        0,
        int(budget.get("secondary_query_reserve", getattr(settings, "discovery_fr_registry_secondary_query_reserve", 4))),
    )
    seed_query_budget = min(seed_query_cap, seed_query_reserve)
    remaining_after_seed = max(0, max_total_queries - seed_query_budget)
    secondary_query_budget = min(secondary_query_cap, secondary_query_reserve, remaining_after_seed)
    code_query_budget = max(1, max_total_queries - seed_query_budget - secondary_query_budget)
    query_phase_counts = {"code": 0, "seed": 0, "secondary": 0}
    executed_lookup_keys: set[tuple[str, str, bool]] = set()
    detail_stats = {
        "detail_api_hits": 0,
        "detail_api_errors": 0,
        "object_text_count": 0,
        "activity_description_count": 0,
        "commercial_name_count": 0,
        "domain_count": 0,
        "observation_count": 0,
        "history_count": 0,
    }
    detail_cache: dict[str, dict[str, Any]] = {}
    executed_lookup_attempts: list[dict[str, Any]] = []

    def _store_raw_record(
        record: dict[str, Any],
        *,
        code_distance: int,
        code_reason: str,
        source_path: str,
        seed_query: Optional[str] = None,
        seed_query_type: Optional[str] = None,
    ) -> None:
        siren = str(record.get("registry_id") or "").strip()
        if not siren or siren == str(source_record.get("registry_id") or "").strip():
            return
        payload = dict(record)
        payload["lookup_source_paths"] = _dedupe_strings(list(payload.get("lookup_source_paths") or []) + [source_path])
        if seed_query:
            payload["seed_lookup_match"] = True
            payload["seed_lookup_queries"] = _dedupe_strings(list(payload.get("seed_lookup_queries") or []) + [seed_query])
            payload["seed_lookup_types"] = _dedupe_strings(list(payload.get("seed_lookup_types") or []) + ([seed_query_type] if seed_query_type else []))
        existing = raw_rows.get(siren)
        if existing:
            existing_record, existing_distance, existing_reason = existing
            merged = _merge_fr_registry_record_detail(existing_record, payload)
            merged["lookup_source_paths"] = _dedupe_strings(
                list(existing_record.get("lookup_source_paths") or []) + [source_path]
            )
            if seed_query:
                merged["seed_lookup_match"] = True
                merged["seed_lookup_queries"] = _dedupe_strings(list(existing_record.get("seed_lookup_queries") or []) + [seed_query])
                merged["seed_lookup_types"] = _dedupe_strings(list(existing_record.get("seed_lookup_types") or []) + ([seed_query_type] if seed_query_type else []))
            raw_rows[siren] = (
                merged,
                min(existing_distance, int(code_distance)),
                code_reason if source_path != "code_neighborhood_crawl" else existing_reason,
            )
            return
        raw_rows[siren] = (payload, int(code_distance), code_reason)

    def _elapsed_exhausted() -> bool:
        nonlocal elapsed_budget_exhausted
        if elapsed_budget_exhausted:
            return True
        if (time.perf_counter() - started_at) >= float(max_elapsed_seconds):
            elapsed_budget_exhausted = True
            return True
        return False

    def _budgeted_registry_search(*, phase: str, **kwargs: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
        nonlocal query_count, query_budget_exhausted
        if _elapsed_exhausted():
            return [], "fr_registry_elapsed_budget_exhausted"
        if query_count >= max_total_queries:
            query_budget_exhausted = True
            return [], "fr_registry_query_budget_exhausted"
        normalized_phase = str(phase or "code").strip().lower()
        phase_limit = code_query_budget
        if normalized_phase == "seed":
            phase_limit = seed_query_budget
        elif normalized_phase == "secondary":
            phase_limit = secondary_query_budget
        if int(query_phase_counts.get(normalized_phase, 0) or 0) >= phase_limit:
            return [], f"fr_registry_{normalized_phase}_budget_exhausted"
        kwargs.setdefault("timeout_seconds", search_timeout_seconds)
        rows, error = _fr_registry_search(**kwargs)
        query_count += 1
        query_phase_counts[normalized_phase] = int(query_phase_counts.get(normalized_phase, 0) or 0) + 1
        if query_count >= max_total_queries:
            query_budget_exhausted = True
        return rows, error

    def _detail_fetch_record(record: dict[str, Any]) -> dict[str, Any]:
        if _elapsed_exhausted():
            return record
        try:
            return _fetch_fr_registry_detail_record(
                record,
                search_timeout_seconds=search_timeout_seconds,
                detail_timeout_seconds=detail_timeout_seconds,
            )
        except TypeError:
            return _fetch_fr_registry_detail_record(record)

    source_record = _detail_fetch_record(source_record)
    scope_pack = _fr_registry_scope_pack(profile, normalized_scope, source_record=source_record)
    semantic_terms = _fr_registry_semantic_terms(profile, normalized_scope)
    code_neighborhood = _fr_registry_code_neighborhood(source_record, semantic_terms=semantic_terms)
    seed_queries = _fr_registry_seed_query_specs(profile, normalized_scope, scope_pack, source_record=source_record)[:seed_query_cap]
    initial_secondary_seed_queries = _fr_registry_secondary_lookup_specs(profile, source_record, [source_record])[:secondary_query_cap]

    def _detail_enrich_record(record: dict[str, Any]) -> dict[str, Any]:
        siren = str(record.get("registry_id") or "").strip()
        if not siren:
            return record
        cached = detail_cache.get(siren)
        if cached is not None:
            return cached
        enriched = _detail_fetch_record(record)
        detail_cache[siren] = enriched
        registry_fields = enriched.get("registry_fields") if isinstance(enriched.get("registry_fields"), dict) else {}
        detail_stats["detail_api_hits"] += 1
        if not any(
            [
                str(registry_fields.get("object_text") or "").strip(),
                str(registry_fields.get("activity_description") or "").strip(),
                list(registry_fields.get("commercial_names") or []),
                list(registry_fields.get("domains") or []),
                str(registry_fields.get("domain") or "").strip(),
                list(registry_fields.get("observations") or []),
                list(registry_fields.get("history_labels") or []),
            ]
        ):
            detail_stats["detail_api_errors"] += 1
        if registry_fields.get("object_text"):
            detail_stats["object_text_count"] += 1
        if registry_fields.get("activity_description"):
            detail_stats["activity_description_count"] += 1
        if registry_fields.get("commercial_names"):
            detail_stats["commercial_name_count"] += 1
        if registry_fields.get("domains") or registry_fields.get("domain"):
            detail_stats["domain_count"] += 1
        if registry_fields.get("observations"):
            detail_stats["observation_count"] += 1
        if registry_fields.get("history_labels"):
            detail_stats["history_count"] += 1
        return enriched

    def _score_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
        scored: list[dict[str, Any]] = []
        activity_counts: dict[str, int] = {}
        for siren, (record, code_distance, code_reason) in raw_rows.items():
            score, semantic_meta = _fr_registry_semantic_score(
                record,
                scope_pack=scope_pack,
                normalized_scope=normalized_scope,
                code_distance=code_distance,
            )
            scored.append(
                {
                    "record": record,
                    "score": score,
                    "semantic_meta": semantic_meta,
                    "code_distance": code_distance,
                    "code_reason": code_reason,
                }
            )
            activity_code = str(record.get("activity_code") or "").strip()
            if activity_code:
                activity_counts[activity_code] = activity_counts.get(activity_code, 0) + 1
        scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return scored, activity_counts

    def _primary_origin_type(record: dict[str, Any]) -> str:
        source_paths = set(record.get("lookup_source_paths") or [])
        if "commercial_name_lookup" in source_paths:
            return "registry_fr_commercial_lookup"
        if "observation_counterparty_lookup" in source_paths:
            return "registry_fr_observation_lookup"
        if "history_alias_lookup" in source_paths:
            return "registry_fr_history_lookup"
        if "seed_name_registry_lookup" in source_paths or bool(record.get("seed_lookup_match")):
            return "registry_fr_seed_lookup"
        return "registry_fr_base"

    seed_query_count = 0

    secondary_seed_query_count = 0

    def _execute_lookup_specs(
        specs: list[dict[str, Any]],
        *,
        phase: str,
        per_query: int,
    ) -> None:
        nonlocal seed_query_count, secondary_seed_query_count
        for seed_spec in specs:
            if query_budget_exhausted or elapsed_budget_exhausted:
                break
            query = str(seed_spec.get("query") or "").strip()
            query_type = str(seed_spec.get("query_type") or "").strip() or "secondary_lookup"
            only_active = bool(seed_spec.get("only_active", True))
            if not query:
                continue
            lookup_key = (query.lower(), query_type, only_active)
            if lookup_key in executed_lookup_keys:
                continue
            rows, _error = _budgeted_registry_search(
                phase=phase,
                query=query,
                page=1,
                per_page=per_query,
                only_active=only_active,
            )
            matched_rows = 0
            if _error in {
                "fr_registry_query_budget_exhausted",
                "fr_registry_elapsed_budget_exhausted",
                f"fr_registry_{phase}_budget_exhausted",
            }:
                executed_lookup_attempts.append(
                    {
                        "phase": phase,
                        "query": query,
                        "query_type": query_type,
                        "only_active": only_active,
                        "row_count": 0,
                        "matched_count": 0,
                        "error": _error,
                    }
                )
                break
            executed_lookup_keys.add(lookup_key)
            if phase == "seed":
                seed_query_count += 1
            elif phase == "secondary":
                secondary_seed_query_count += 1
            for row in rows:
                record = _fr_registry_source_record(row, query_hint=query)
                if not _fr_registry_seed_query_matches_record(record, query):
                    if phase == "seed" and query_type not in {"adjacency_label", "adjacency_vendor_phrase"}:
                        continue
                    if phase == "secondary":
                        continue
                matched_rows += 1
                _store_raw_record(
                    record,
                    code_distance=0,
                    code_reason="seed_lookup" if phase == "seed" else query_type,
                    source_path=_fr_registry_query_type_to_source_path(query_type),
                    seed_query=query,
                    seed_query_type=query_type,
                )
            executed_lookup_attempts.append(
                {
                    "phase": phase,
                    "query": query,
                    "query_type": query_type,
                    "only_active": only_active,
                    "row_count": len(rows),
                    "matched_count": matched_rows,
                    "error": _error,
                }
            )

    _execute_lookup_specs(seed_queries, phase="seed", per_query=seed_per_query)
    _execute_lookup_specs(initial_secondary_seed_queries, phase="secondary", per_query=secondary_seed_per_query)

    for code_entry in code_neighborhood:
        if query_budget_exhausted or elapsed_budget_exhausted:
            break
        code = str(code_entry.get("code") or "").strip()
        if not code:
            continue
        fetch_stats = code_fetch_stats.setdefault(
            code,
            {
                "code": code,
                "distance": int(code_entry.get("distance") or 0),
                "reason": str(code_entry.get("reason") or ""),
                "pages_fetched": 0,
                "rows_fetched": 0,
                "signal_rows": 0,
                "extended_pages": 0,
                "stopped_early": False,
            },
        )
        current_limit = pages_per_code
        consecutive_no_signal_pages = 0
        page = 1
        while page <= current_limit and page <= max_pages_per_code:
            if _elapsed_exhausted():
                break
            rows, _error = _budgeted_registry_search(
                phase="code",
                activite_principale=code,
                page=page,
                per_page=per_page,
                only_active=True,
            )
            if _error in {
                "fr_registry_query_budget_exhausted",
                "fr_registry_elapsed_budget_exhausted",
                "fr_registry_code_budget_exhausted",
            }:
                break
            if not rows:
                break
            fetch_stats["pages_fetched"] = int(fetch_stats.get("pages_fetched") or 0) + 1
            fetch_stats["rows_fetched"] = int(fetch_stats.get("rows_fetched") or 0) + len(rows)
            page_signal_hits = 0
            for row in rows:
                record = _fr_registry_source_record(row, query_hint=code)
                signal = _fr_registry_recall_signal(
                    record,
                    scope_pack=scope_pack,
                    normalized_scope=normalized_scope,
                    code_distance=int(code_entry.get("distance") or 0),
                )
                if signal.get("has_signal"):
                    page_signal_hits += 1
                _store_raw_record(
                    record,
                    code_distance=int(code_entry.get("distance") or 0),
                    code_reason=str(code_entry.get("reason") or ""),
                    source_path="code_neighborhood_crawl",
                )
            fetch_stats["signal_rows"] = int(fetch_stats.get("signal_rows") or 0) + int(page_signal_hits)
            if page_signal_hits > 0:
                consecutive_no_signal_pages = 0
            else:
                consecutive_no_signal_pages += 1
            if page_signal_hits >= page_extension_min_hits and current_limit < max_pages_per_code:
                current_limit += 1
                fetch_stats["extended_pages"] = int(fetch_stats.get("extended_pages") or 0) + 1
            if page >= pages_per_code and consecutive_no_signal_pages >= page_stop_after_no_signal:
                fetch_stats["stopped_early"] = True
                break
            if len(raw_rows) >= candidate_cap:
                break
            page += 1
        if len(raw_rows) >= candidate_cap or query_budget_exhausted or elapsed_budget_exhausted:
            break

    scored_rows, activity_code_counts = _score_rows()

    detail_priority_rows = sorted(scored_rows, key=_fr_registry_detail_priority, reverse=True)
    detail_registry_ids = {
        str((item.get("record") or {}).get("registry_id") or "").strip()
        for item in detail_priority_rows[:detail_cap]
        if str((item.get("record") or {}).get("registry_id") or "").strip()
    }
    enriched_rows: list[dict[str, Any]] = []
    for index, item in enumerate(scored_rows):
        record = item["record"]
        if str(record.get("registry_id") or "").strip() in detail_registry_ids:
            record = _detail_enrich_record(record)
            score, semantic_meta = _fr_registry_semantic_score(
                record,
                scope_pack=scope_pack,
                normalized_scope=normalized_scope,
                code_distance=int(item.get("code_distance") or 0),
            )
            item = {
                **item,
                "record": record,
                "score": score,
                "semantic_meta": semantic_meta,
            }
        enriched_rows.append(item)
    enriched_rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    detailed_seed_records = [item["record"] for item in enriched_rows[:detail_cap] if isinstance(item.get("record"), dict)]
    secondary_seed_queries = _fr_registry_secondary_lookup_specs(profile, source_record, detailed_seed_records)[:secondary_query_cap]
    _execute_lookup_specs(secondary_seed_queries, phase="secondary", per_query=secondary_seed_per_query)

    scored_rows, activity_code_counts = _score_rows()
    detail_priority_rows = sorted(scored_rows, key=_fr_registry_detail_priority, reverse=True)
    detail_registry_ids = {
        str((item.get("record") or {}).get("registry_id") or "").strip()
        for item in detail_priority_rows[:detail_cap]
        if str((item.get("record") or {}).get("registry_id") or "").strip()
    }
    enriched_rows = []
    for index, item in enumerate(scored_rows):
        record = item["record"]
        if str(record.get("registry_id") or "").strip() in detail_registry_ids:
            record = _detail_enrich_record(record)
            score, semantic_meta = _fr_registry_semantic_score(
                record,
                scope_pack=scope_pack,
                normalized_scope=normalized_scope,
                code_distance=int(item.get("code_distance") or 0),
            )
            item = {
                **item,
                "record": record,
                "score": score,
                "semantic_meta": semantic_meta,
            }
        enriched_rows.append(item)

    candidates: list[dict[str, Any]] = []
    enriched_rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

    for item in enriched_rows[:candidate_cap]:
        record = item["record"]
        semantic_meta = item["semantic_meta"] if isinstance(item.get("semantic_meta"), dict) else {}
        lane_ids = semantic_meta.get("lane_ids") or []
        lane_labels = semantic_meta.get("lane_labels") or []
        scope_bucket = str(semantic_meta.get("scope_bucket") or "broad_market")
        if scope_bucket == "out_of_scope":
            continue
        website = str(record.get("website") or "").strip() or None
        official_domain = normalize_domain(website)
        display_name = str(record.get("display_name") or record.get("name") or "").strip()
        legal_name = str(record.get("legal_name") or "").strip() or None
        registry_url = str(record.get("registry_url") or "").strip() or None
        node_fit_summary = semantic_meta.get("node_fit_summary") if isinstance(semantic_meta.get("node_fit_summary"), dict) else {}
        short_description = _fr_registry_candidate_short_description(record, semantic_meta)
        directness = _directness_from_node_fit_summary(node_fit_summary, fallback=scope_bucket)
        origin_type = _primary_origin_type(record)
        candidate = {
            "name": display_name,
            "display_name": display_name,
            "legal_name": legal_name,
            "brand_name": display_name if display_name and display_name != legal_name else None,
            "brand_names": record.get("brand_names") if isinstance(record.get("brand_names"), list) else [],
            "alias_names": _dedupe_strings(
                list(record.get("brand_names") or [])
                + list(((record.get("registry_fields") or {}).get("commercial_names") or []))
                + list(_fr_registry_extract_observation_entities(list(((record.get("registry_fields") or {}).get("observations") or [])))
                )
            )[:12],
            "website": website,
            "official_website_url": website,
            "discovery_url": registry_url,
            "profile_url": registry_url,
            "entity_type": "company",
            "registry_id": record.get("registry_id"),
            "registry_source": record.get("registry_source"),
            "registry_country": "FR",
            "registry_status": record.get("status"),
            "country": "FR",
            "identity": {
                "official_website": website,
                "identity_confidence": "medium" if website else "low",
                "error": None if website else "missing_website",
            },
            "first_party_domains": [official_domain] if official_domain else [],
            "industry_signature": _normalize_industry_signature(
                list(record.get("industry_codes") or []),
                list(record.get("industry_keywords") or []),
            ),
            "why_relevant": [
                {
                    "text": short_description or f"French registry candidate surfaced from {record.get('activity_code') or 'APE'} neighborhood.",
                    "citation_url": registry_url,
                    "dimension": "registry_lookup",
                }
            ],
            "precomputed_discovery_score": float(item.get("score") or 0.0),
            "discovery_score": float(item.get("score") or 0.0),
            "directness": directness,
            "node_fit_summary": node_fit_summary,
            "registry_fields": {
                **(record.get("registry_fields") if isinstance(record.get("registry_fields"), dict) else {}),
                "matched_terms": semantic_meta.get("matched_terms") or [],
                "generic_infra_hits": semantic_meta.get("generic_infra_hits") or [],
                "code_reason": item.get("code_reason"),
                "lookup_source_paths": list(record.get("lookup_source_paths") or []),
                "seed_lookup_match": bool(record.get("seed_lookup_match")),
                "seed_lookup_queries": list(record.get("seed_lookup_queries") or []),
                "seed_lookup_types": list(record.get("seed_lookup_types") or []),
            },
            "_origins": [
                {
                    "origin_type": origin_type,
                    "origin_url": registry_url,
                    "source_name": "fr_registry",
                    "source_run_id": None,
                    "metadata": {
                        "query_type": "registry_lookup",
                        "query_family": "registry_lookup",
                        "source_family": "registry",
                        "scope_bucket": scope_bucket,
                        "fit_to_adjacency_box_ids": lane_ids,
                        "fit_to_adjacency_box_labels": lane_labels,
                        "source_capability_matches": semantic_meta.get("source_capability_matches") or [],
                        "registry_id": record.get("registry_id"),
                        "registry_source": record.get("registry_source"),
                        "code_distance": int(item.get("code_distance") or 0),
                        "code_reason": item.get("code_reason"),
                        "lookup_source_paths": list(record.get("lookup_source_paths") or []),
                        "seed_lookup_match": bool(record.get("seed_lookup_match")),
                        "seed_lookup_queries": list(record.get("seed_lookup_queries") or []),
                        "seed_lookup_types": list(record.get("seed_lookup_types") or []),
                        "node_fit_summary": node_fit_summary,
                    },
                }
            ],
        }
        candidates.append(candidate)

    source_path_counts: dict[str, int] = {}
    for record, _code_distance, _code_reason in raw_rows.values():
        paths = list(record.get("lookup_source_paths") or ["code_neighborhood_crawl"])
        for path in paths:
            source_path_counts[path] = source_path_counts.get(path, 0) + 1

    def _candidate_names(payload: dict[str, Any]) -> set[str]:
        names = _dedupe_strings(
            [
                str(payload.get("display_name") or "").strip(),
                str(payload.get("legal_name") or "").strip(),
                *list(payload.get("brand_names") or []),
                *list(((payload.get("registry_fields") or {}).get("commercial_names") or [])),
                *list(payload.get("alias_names") or []),
            ]
        )
        return {
            _normalize_name_for_matching(name)
            for name in names
            if _normalize_name_for_matching(name)
        }

    benchmarks = [
        {"label": "THE WORKING COMPANY / 814956744", "registry_id": "814956744", "names": ["THE WORKING COMPANY", "BRIGAD"]},
        {"label": "MEDIFLASH / 887656270", "registry_id": "887656270", "names": ["MEDIFLASH"]},
        {"label": "MEDIKSTAFF / 820625564", "registry_id": "820625564", "names": ["MEDIKSTAFF", "MSTAFF"]},
    ]
    benchmark_hits: list[dict[str, Any]] = []
    final_candidates_by_id = {str(item.get("registry_id") or "").strip(): item for item in candidates if str(item.get("registry_id") or "").strip()}
    raw_candidates_by_id = {str(record.get("registry_id") or "").strip(): record for record, _dist, _reason in raw_rows.values() if str(record.get("registry_id") or "").strip()}
    for benchmark in benchmarks:
        normalized_names = {_normalize_name_for_matching(name) for name in benchmark["names"] if _normalize_name_for_matching(name)}
        raw_match = raw_candidates_by_id.get(benchmark["registry_id"])
        final_match = final_candidates_by_id.get(benchmark["registry_id"])
        if raw_match is None:
            for record, _dist, _reason in raw_rows.values():
                if _candidate_names(record) & normalized_names:
                    raw_match = record
                    break
        if final_match is None:
            for candidate in candidates:
                if _candidate_names(candidate) & normalized_names:
                    final_match = candidate
                    break
        benchmark_hits.append(
            {
                "label": benchmark["label"],
                "registry_id": benchmark["registry_id"],
                "raw_found": raw_match is not None,
                "final_found": final_match is not None,
                "raw_match_name": str((raw_match or {}).get("display_name") or (raw_match or {}).get("legal_name") or "").strip() or None,
                "final_match_name": str((final_match or {}).get("display_name") or (final_match or {}).get("legal_name") or "").strip() or None,
            }
        )

    alias_only_paths = {"commercial_name_lookup", "observation_counterparty_lookup", "history_alias_lookup"}
    alias_only_recovered_count = 0
    for record, _dist, _reason in raw_rows.values():
        source_paths = set(record.get("lookup_source_paths") or [])
        if source_paths and source_paths.issubset(alias_only_paths):
            alias_only_recovered_count += 1

    diagnostics = {
        "source_resolution": source_diagnostics,
        "source_record": {
            "display_name": source_record.get("display_name"),
            "legal_name": source_record.get("legal_name"),
            "registry_id": source_record.get("registry_id"),
            "activity_code": source_record.get("activity_code"),
            "activity_code_naf25": source_record.get("activity_code_naf25"),
            "detail_summary": {
                "object_text_present": bool(str(((source_record.get("registry_fields") or {}).get("object_text") or "")).strip()),
                "activity_description_present": bool(
                    str(((source_record.get("registry_fields") or {}).get("activity_description") or "")).strip()
                ),
                "inpi_auth_error": str(((source_record.get("registry_fields") or {}).get("inpi_auth_error") or "")).strip() or None,
                "commercial_names": list(((source_record.get("registry_fields") or {}).get("commercial_names") or []))[:8],
                "domains": list(((source_record.get("registry_fields") or {}).get("domains") or []))[:8],
                "observations": list(((source_record.get("registry_fields") or {}).get("observations") or []))[:4],
                "history_labels": list(((source_record.get("registry_fields") or {}).get("history_labels") or []))[:8],
            },
        },
        "code_neighborhood": code_neighborhood,
        "registry_queries_count": query_count,
        "registry_raw_candidate_count": len(raw_rows),
        "registry_scored_candidate_count": len(scored_rows),
        "registry_detail_fetch_count": min(detail_cap, len(scored_rows)),
        "registry_candidate_count": len(candidates),
        "registry_seed_query_count": seed_query_count,
        "registry_secondary_query_count": secondary_seed_query_count,
        "registry_code_query_count": int(query_phase_counts.get("code") or 0),
        "registry_source_path_counts": source_path_counts,
        "seed_query_specs": seed_queries[:12],
        "initial_secondary_query_specs": initial_secondary_seed_queries[:12],
        "post_detail_secondary_query_specs": secondary_seed_queries[:12],
        "executed_lookup_attempts": executed_lookup_attempts[:24],
        "detail_stats": detail_stats,
        "alias_only_recovered_count": alias_only_recovered_count,
        "benchmark_hits": benchmark_hits,
        "code_fetch_stats": sorted(
            code_fetch_stats.values(),
            key=lambda item: (
                -int(item.get("rows_fetched") or 0),
                str(item.get("code") or ""),
            ),
        ),
        "raw_activity_code_counts": [
            {"code": code, "count": count}
            for code, count in sorted(activity_code_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:20]
        ],
        "final_candidate_commercial_name_count": sum(
            1 for candidate in candidates if list((candidate.get("registry_fields") or {}).get("commercial_names") or [])
        ),
        "final_candidate_observation_count": sum(
            1 for candidate in candidates if list((candidate.get("registry_fields") or {}).get("observations") or [])
        ),
        "final_candidate_object_text_count": sum(
            1 for candidate in candidates if str((candidate.get("registry_fields") or {}).get("object_text") or "").strip()
        ),
        "scope_pack": {
            "core_node_terms_count": len(scope_pack.get("core", {}).get("terms") or []),
            "adjacency_node_terms_count": len(scope_pack.get("adjacent", {}).get("terms") or []),
            "entity_seed_terms_count": len(scope_pack.get("entity_seed", {}).get("terms") or []),
        },
        "budget": {
            "pages_per_code": pages_per_code,
            "max_pages_per_code": max_pages_per_code,
            "candidate_cap": candidate_cap,
            "detail_cap": detail_cap,
            "search_timeout_seconds": search_timeout_seconds,
            "detail_timeout_seconds": detail_timeout_seconds,
            "seed_query_cap": seed_query_cap,
            "seed_query_reserve": seed_query_budget,
            "secondary_query_cap": secondary_query_cap,
            "secondary_query_reserve": secondary_query_budget,
            "code_query_budget": code_query_budget,
            "max_total_queries": max_total_queries,
            "max_elapsed_seconds": max_elapsed_seconds,
            "page_extension_min_hits": page_extension_min_hits,
            "page_stop_after_no_signal": page_stop_after_no_signal,
        },
        "query_budget_exhausted": query_budget_exhausted,
        "elapsed_budget_exhausted": elapsed_budget_exhausted,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "estimated_max_query_ceiling": min(
            max_total_queries,
            int(max_pages_per_code * max(1, len(code_neighborhood))) + seed_query_cap + secondary_query_cap,
        ),
    }
    return candidates, diagnostics


def _run_france_registry_recall_benchmark(
    profile: CompanyProfile,
    normalized_scope: dict[str, Any],
    *,
    tiers: Optional[list[dict[str, int]]] = None,
) -> list[dict[str, Any]]:
    benchmark_tiers = tiers or [
        {"pages_per_code": 5, "max_pages_per_code": 5, "candidate_cap": 1200, "detail_cap": 60},
        {"pages_per_code": 8, "max_pages_per_code": 8, "candidate_cap": 1800, "detail_cap": 80},
        {"pages_per_code": 12, "max_pages_per_code": 12, "candidate_cap": 2500, "detail_cap": 120},
    ]
    results: list[dict[str, Any]] = []
    for tier in benchmark_tiers:
        started_at = time.perf_counter()
        candidates, diagnostics = _build_france_registry_universe_candidates(
            profile,
            normalized_scope,
            budget_overrides=tier,
        )
        results.append(
            {
                "budget": dict(tier),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "registry_raw_candidate_count": diagnostics.get("registry_raw_candidate_count"),
                "registry_scored_candidate_count": diagnostics.get("registry_scored_candidate_count"),
                "registry_candidate_count": diagnostics.get("registry_candidate_count"),
                "registry_queries_count": diagnostics.get("registry_queries_count"),
                "registry_seed_query_count": diagnostics.get("registry_seed_query_count"),
                "registry_secondary_query_count": diagnostics.get("registry_secondary_query_count"),
                "registry_source_path_counts": diagnostics.get("registry_source_path_counts"),
                "detail_stats": diagnostics.get("detail_stats"),
                "benchmark_hits": diagnostics.get("benchmark_hits"),
                "code_fetch_stats": diagnostics.get("code_fetch_stats"),
                "top_candidates": [
                    {
                        "display_name": item.get("display_name"),
                        "legal_name": item.get("legal_name"),
                        "registry_id": item.get("registry_id"),
                        "directness": item.get("directness"),
                    }
                    for item in candidates[:10]
                ],
            }
        )
    return results


def _software_signal_score_from_codes(codes: list[str]) -> float:
    normalized = [str(code or "").strip().lower() for code in codes if str(code or "").strip()]
    if not normalized:
        return 0.0
    score = 0.0
    for code in normalized:
        if code.startswith("62"):
            score = max(score, 1.0)
        elif code.startswith("63"):
            score = max(score, 0.7)
        elif code.startswith("66"):
            score = max(score, 0.5)
    return min(1.0, score)


def _industry_keywords_from_record(record: dict[str, Any]) -> list[str]:
    text_parts = [
        str(record.get("context_text") or ""),
        " ".join(str(code) for code in (record.get("industry_codes") or [])),
        str(record.get("name") or ""),
    ]
    combined = " ".join(text_parts).lower()
    keywords: list[str] = []
    for token in REGISTRY_RELEVANCE_TOKENS.union(SOFTWARE_SIGNAL_TOKENS):
        if token in combined:
            keywords.append(token)
    return _dedupe_strings(keywords)


def _normalize_industry_signature(codes: list[str], keywords: list[str]) -> dict[str, Any]:
    deduped_codes = _dedupe_strings([str(code).strip().upper() for code in codes if str(code).strip()])
    deduped_keywords = _dedupe_strings([str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()])
    software_relevance = max(
        _software_signal_score_from_codes(deduped_codes),
        1.0 if any(token in deduped_keywords for token in SOFTWARE_SIGNAL_TOKENS) else 0.0,
    )
    return {
        "industry_codes": deduped_codes[:20],
        "industry_keywords": deduped_keywords[:20],
        "software_relevance_score": round(float(software_relevance), 4),
    }


def _merge_industry_signatures(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    left_obj = left if isinstance(left, dict) else {}
    right_obj = right if isinstance(right, dict) else {}
    merged_codes = _dedupe_strings(
        [str(code) for code in (left_obj.get("industry_codes") or []) + (right_obj.get("industry_codes") or [])]
    )
    merged_keywords = _dedupe_strings(
        [str(keyword) for keyword in (left_obj.get("industry_keywords") or []) + (right_obj.get("industry_keywords") or [])]
    )
    left_score = float(left_obj.get("software_relevance_score") or 0.0)
    right_score = float(right_obj.get("software_relevance_score") or 0.0)
    merged = _normalize_industry_signature(merged_codes, merged_keywords)
    merged["software_relevance_score"] = round(max(left_score, right_score, float(merged.get("software_relevance_score") or 0.0)), 4)
    return merged


def _registry_lookup_reasons(candidate_name: str, country: Optional[str]) -> list[dict[str, str]]:
    normalized_country = normalize_country(country)
    if normalized_country not in REGISTRY_EXPANSION_COUNTRIES:
        return []
    if not candidate_name.strip():
        return []
    query = quote_plus(candidate_name.strip())
    if normalized_country == "UK":
        return [
            {
                "text": f"Deterministic registry lookup prepared for {candidate_name} on Companies House.",
                "citation_url": f"https://find-and-update.company-information.service.gov.uk/search/companies?q={query}",
                "dimension": "registry_lookup",
            }
        ]
    if normalized_country == "DE":
        return [
            {
                "text": f"Deterministic registry lookup prepared for {candidate_name} on Handelsregister.",
                "citation_url": f"https://www.handelsregister.de/rp_web/normalesuche/welcome.xhtml?query={query}",
                "dimension": "registry_lookup",
            }
        ]
    if normalized_country in {"BE", "NL", "LU", "CH", "MC"}:
        return [
            {
                "text": (
                    f"Deterministic registry lookup prepared for {candidate_name} "
                    f"on GLEIF LEI search ({normalized_country})."
                ),
                "citation_url": (
                    "https://api.gleif.org/api/v1/lei-records"
                    f"?filter[entity.legalAddress.country]={normalized_country}"
                    f"&filter[entity.legalName]={query}"
                ),
                "dimension": "registry_lookup",
            }
        ]
    return [
        {
            "text": f"Deterministic registry lookup prepared for {candidate_name} on Annuaire des Entreprises.",
            "citation_url": f"https://annuaire-entreprises.data.gouv.fr/rechercher?terme={query}",
            "dimension": "registry_lookup",
        },
        {
            "text": f"Deterministic registry lookup prepared for {candidate_name} on INPI data.",
            "citation_url": f"https://data.inpi.fr/recherche?q={query}",
            "dimension": "registry_lookup",
        },
    ]


def _resolve_candidate_identity(
    candidate: dict[str, Any],
    identity_cache: dict[str, dict[str, Any]],
    allow_network_resolution: bool = True,
) -> dict[str, Any]:
    website = str(candidate.get("website") or "").strip()
    if not website:
        candidate["identity"] = {
            "input_website": None,
            "official_website": None,
            "identity_confidence": "low",
            "identity_sources": [],
            "error": "missing_website",
        }
        return candidate

    if not website.startswith(("http://", "https://")):
        website = f"https://{website}"
    input_domain = normalize_domain(website)
    cache_key = website.lower()
    cached = identity_cache.get(cache_key)
    if cached is None:
        if input_domain and _is_non_first_party_profile_domain(input_domain) and allow_network_resolution:
            cached = resolve_external_website_from_profile(website)
        elif input_domain and _is_non_first_party_profile_domain(input_domain):
            cached = {
                "profile_url": website,
                "official_website": None,
                "identity_confidence": "low",
                "captured_at": datetime.utcnow().isoformat(),
                "error": "resolution_budget_exhausted",
            }
        else:
            cached = {
                "profile_url": website,
                "official_website": website,
                "identity_confidence": "high",
                "captured_at": datetime.utcnow().isoformat(),
                "error": None,
            }
        identity_cache[cache_key] = cached

    resolved = str(cached.get("official_website") or "").strip()
    if resolved and not resolved.startswith(("http://", "https://")):
        resolved = f"https://{resolved}"
    resolved_domain = normalize_domain(resolved) if resolved else None
    if resolved_domain and _is_non_first_party_profile_domain(resolved_domain):
        resolved = ""
        resolved_domain = None

    final_website = resolved or (
        website
        if input_domain and not _is_non_first_party_profile_domain(input_domain)
        else ""
    )
    final_domain = normalize_domain(final_website) if final_website else None
    identity_confidence = str(cached.get("identity_confidence", "low") or "low")
    if not final_website:
        identity_confidence = "low"

    candidate["original_website"] = website
    candidate["website"] = final_website or None
    candidate["official_website_url"] = final_website or None
    candidate["identity"] = {
        "input_website": website,
        "official_website": final_website or None,
        "input_domain": input_domain,
        "canonical_domain": final_domain or input_domain,
        "identity_confidence": identity_confidence,
        "identity_sources": [cached.get("profile_url")] if cached.get("profile_url") else [],
        "captured_at": cached.get("captured_at"),
        "error": cached.get("error"),
    }
    if not candidate.get("hq_country"):
        inferred_country = _infer_country_from_domain(final_domain or input_domain)
        if inferred_country:
            candidate["hq_country"] = inferred_country
    return candidate


LEGAL_SUFFIXES = {
    "sas",
    "sa",
    "sarl",
    "ltd",
    "limited",
    "llc",
    "inc",
    "plc",
    "gmbh",
    "bv",
    "b.v",
    "nv",
    "s.p.a",
    "spa",
    "ag",
    "group",
    "holding",
}

NAME_STOPWORDS = {
    "the",
    "and",
    "solutions",
    "solution",
    "technologies",
    "technology",
    "systems",
    "system",
    "software",
    "platform",
}

FR_REGISTRY_SCOPE_LOW_SIGNAL_TERMS = {
    "manage",
    "management",
    "internal",
    "mobility",
    "customer",
    "customers",
    "service",
    "services",
    "application",
    "applications",
}

TOKEN_SYNONYMS = {
    "financial": "finance",
    "financiere": "finance",
    "financier": "finance",
    "informatique": "technology",
    "technologie": "technology",
    "technologies": "technology",
    "tech": "technology",
}

REGISTRY_RELEVANCE_TOKENS = {
    "wealth",
    "portfolio",
    "asset",
    "investment",
    "investments",
    "securities",
    "fund",
    "funds",
    "trading",
    "order management",
    "oms",
    "pms",
    "compliance",
    "risk",
    "attribution",
}

SOFTWARE_SIGNAL_TOKENS = {
    "software",
    "platform",
    "saas",
    "fintech",
    "wealthtech",
    "investment",
    "portfolio",
    "asset",
    "order management",
    "pms",
    "oms",
    "compliance",
    "risk",
    "custody",
}

COUNTRY_HINT_TOKENS = {
    "FR": {"france", "paris", "lyon", "marseille", "french"},
    "UK": {"united kingdom", "uk", "london", "manchester", "edinburgh", "british"},
    "DE": {"germany", "deutschland", "german", "berlin", "munich", "muenchen", "frankfurt", "hamburg", "gmbh"},
    "BE": {"belgium", "belgian", "brussels", "bruxelles", "antwerp", "anvers"},
    "NL": {"netherlands", "dutch", "amsterdam", "rotterdam", "the hague"},
    "LU": {"luxembourg", "luxembourg city", "letzebuerg"},
    "CH": {"switzerland", "swiss", "zurich", "geneva", "lausanne", "zug"},
    "MC": {"monaco", "monegasque", "monte carlo"},
}

FR_REGISTRY_DIGITAL_APE_CODES = (
    "58.21Z",
    "58.29A",
    "58.29B",
    "58.29C",
    "58.29Y",
    "62.01Z",
    "62.02A",
    "62.02B",
    "62.03Z",
    "62.09Z",
    "63.11Z",
    "63.12Z",
)

FR_REGISTRY_RECRUITMENT_APE_CODES = (
    "78.10Z",
    "78.20Z",
    "78.30Z",
)

FR_REGISTRY_SOFT_BLOCKED_NAME_TOKENS = (
    "minist",
    "mairie",
    "ville de",
    "commune",
    "centre hospitalier",
    "hopitaux",
    "hôpitaux",
    "onisep",
    "universit",
    "chambre de commerce",
    "conseil departemental",
    "département",
    "prefecture",
    "préfecture",
)

FR_REGISTRY_GENERIC_INFRA_TOKENS = (
    "hebergement",
    "hébergement",
    "cloud",
    "datacenter",
    "data center",
    "hosting",
    "outsourcing",
    "bpo",
    "archivage",
    "sauvegarde",
    "backup",
    "ged",
    "dematerialisation",
    "dématérialisation",
    "copie",
    "photocopie",
    "impression",
    "messagerie",
)

FR_REGISTRY_SEMANTIC_EXPANSIONS = {
    "software": ["logiciel", "saas", "plateforme", "application", "applications", "edition", "édition"],
    "platform": ["plateforme", "marketplace", "place de marche", "place de marché", "reseau", "réseau"],
    "healthcare": ["sante", "santé", "medical", "médical", "hospitalier", "hospitaliere", "hospitalière"],
    "hospital": ["hopital", "hôpital", "clinique", "ehpad", "etablissement de sante", "établissement de santé"],
    "staffing": ["recrutement", "interim", "intérim", "vacation", "vacataire", "remplacement", "mise en relation"],
    "recruitment": ["recrutement", "mise en relation", "talent", "embauche"],
    "workforce": ["personnel", "effectif", "effectifs", "ressources humaines", "rh", "gestion administrative"],
    "scheduling": ["planning", "planification", "horaire", "horaires", "shift", "roster"],
    "marketplace": ["marketplace", "place de marche", "place de marché", "mise en relation"],
    "payments": ["paiement", "facturation"],
}

REGISTRY_PROFILE_DOMAINS = {
    "annuaire-entreprises.data.gouv.fr",
    "find-and-update.company-information.service.gov.uk",
    "recherche-entreprises.api.gouv.fr",
    "data.inpi.fr",
    "pappers.fr",
    "handelsregister.de",
    "unternehmensregister.de",
    "api.gleif.org",
    "search.gleif.org",
}

KNOWN_THIRD_PARTY_CONTEXT_DOMAINS = {
    "linkedin.com",
    "crunchbase.com",
    "zoominfo.com",
    "pitchbook.com",
    "societeinfo.com",
    "rubypayeur.com",
    "annuaire-entreprises.data.gouv.fr",
    "pappers.fr",
    "infogreffe.fr",
    "data.inpi.fr",
}

DISCOVERY_NON_VENDOR_HOSTS = {
    "afme.eu",
    "appsruntheworld.com",
    "citywire.com",
    "cooley.com",
    "ec.europa.eu",
    "financialit.net",
    "globalcustodian.com",
    "healthmanagement.org",
    "ibsintelligence.com",
    "introspectivemarketresearch.com",
    "kenresearch.com",
    "link.springer.com",
    "mdpi.com",
    "medrxiv.org",
    "mordorintelligence.com",
    "ncbi.nlm.nih.gov",
    "slashdot.org",
    "pmc.ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "toolradar.com",
    "datos-insights.com",
    "openbankingtracker.com",
    "onlinelibrary.wiley.com",
    "pharmiweb.com",
    "privatebankerinternational.com",
    "regcompliancewatch.com",
    "researchgate.net",
    "thefinrate.com",
    "thehedgefundjournal.com",
    "talentbusinesspartners.com",
    "verifiedmarketresearch.com",
    "wealthbriefing.com",
    "reddit.com",
    "inven.ai",
    "ey.com",
    "europa.eu",
    "flickr.com",
    "klasresearch.com",
}

DISCOVERY_VENDORISH_TOKENS = {
    "software",
    "platform",
    "solution",
    "solutions",
    "vendor",
    "vendors",
    "system",
    "systems",
    "suite",
    "api",
    "apis",
    "saas",
    "workflow",
    "reporting",
    "compliance",
    "proxy voting",
    "corporate actions",
    "order management",
    "post-trade",
    "portfolio management",
}

DISCOVERY_STRONG_VENDOR_TOKENS = {
    "software",
    "platform",
    "solution",
    "solutions",
    "vendor",
    "vendors",
    "suite",
    "api",
    "apis",
    "saas",
}

DISCOVERY_EDITORIAL_TOKENS = {
    " report",
    " review",
    " reviews",
    " matrix",
    " best ",
    " top ",
    " page ",
    " article",
    " articles",
    " news",
    " press",
    " press release",
    " guide",
    " guides",
    " insights",
    " introduces ",
    " breaks new ground ",
    " pioneers ",
    " disclosure",
    " conference",
    " rankings",
    " research",
}

DISCOVERY_SERVICE_PROVIDER_TOKENS = {
    " development company",
    " software development",
    " custom software",
    " app developers",
    " app development",
    " outsourcing",
    " consulting",
    " consultancy",
    " agency",
    " recruitment",
    " staffing agency",
    " locum",
    " contractor",
    " temporary staffing",
    " hire developers",
    " hire developer",
}

DISCOVERY_INVESTOR_TOKENS = {
    " investors",
    " investment firm",
    " venture capital",
    " private equity",
    " investor list",
    " top investors",
    " funding",
    " portfolio company",
}

DISCOVERY_INSTITUTION_TOKENS = {
    " official website",
    " global company website",
    " annual report",
    " investor relations",
    " corporate governance",
    " sustainability report",
    " shareholders",
    " for investors",
    " private client bank",
    " investment management",
    " asset management",
    " hospital",
    " hospitals",
    " clinic",
    " clinics",
    " university",
    " university hospital",
    " university hospitals",
    " foundation",
    " association",
    " ministry",
    " public health",
    " health system",
    " healthcare system",
    " nhs",
    " medical center",
    " medical centres",
    " advisory firm",
    " consulting firm",
}

DISCOVERY_NON_VENDOR_FILE_SUFFIXES = (
    ".pdf",
    ".xlsx",
    ".xls",
    ".csv",
    ".ppt",
    ".pptx",
    ".doc",
    ".docx",
)

DISCOVERY_NON_VENDOR_PATH_TOKENS = (
    "/wp-content/uploads/",
    "/sites/default/files/",
    "/bitstream/",
    "/legal-content/",
    "/content/pdf/",
    "/science/article/",
    "/doi/full/",
)

DISCOVERY_ARTICLE_PATH_TOKENS = (
    "/article/",
    "/articles/",
    "/blog/",
    "/blogs/",
    "/insights/",
    "/news/",
    "/press/",
    "/press-release/",
    "/press-releases/",
)

DISCOVERY_NON_VENDOR_DOMAIN_LABEL_TOKENS = {
    "academy",
    "bootcamp",
    "journal",
    "magazine",
    "media",
    "news",
    "press",
    "school",
}

DISCOVERY_GENERIC_VENDOR_TOKENS = {
    "software",
    "solution",
    "solutions",
}


def _normalize_name_for_matching(name: str) -> str:
    lowered = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [token.strip() for token in lowered.split() if token.strip()]
    normalized_tokens = [TOKEN_SYNONYMS.get(token, token) for token in tokens]
    filtered = [
        token
        for token in normalized_tokens
        if token not in LEGAL_SUFFIXES and token not in NAME_STOPWORDS
    ]
    return " ".join(filtered)


def _normalize_name_phrase(name: str) -> str:
    lowered = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _domains_conflict(domain_a: Optional[str], domain_b: Optional[str]) -> bool:
    if not domain_a or not domain_b:
        return False
    if _is_non_first_party_profile_domain(domain_a) or _is_non_first_party_profile_domain(domain_b):
        return False
    return domain_a != domain_b


def _can_bridge_registry_profile(
    candidate: dict[str, Any],
    entity: dict[str, Any],
    name_similarity: float,
) -> bool:
    candidate_domain = normalize_domain(candidate.get("website"))
    entity_domain = normalize_domain(entity.get("canonical_website"))
    if not candidate_domain or not entity_domain:
        return False
    if candidate_domain == entity_domain:
        return True
    if not (_is_registry_profile_domain(candidate_domain) or _is_registry_profile_domain(entity_domain)):
        return False
    if _is_registry_profile_domain(candidate_domain) and _is_registry_profile_domain(entity_domain):
        return False
    if name_similarity < 0.92:
        return False

    candidate_country = _canonical_country(candidate)
    entity_country = normalize_country(entity.get("country"))
    if candidate_country and entity_country and candidate_country != entity_country:
        return False

    candidate_registry_id = _registry_identifier(candidate)
    entity_registry_id = str(entity.get("registry_id") or "").strip() or None
    if candidate_registry_id and entity_registry_id and candidate_registry_id != entity_registry_id:
        return False
    return True


def _origin_entries(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    origins = candidate.get("_origins")
    if isinstance(origins, list):
        valid: list[dict[str, Any]] = []
        for origin in origins:
            if not isinstance(origin, dict):
                continue
            if not origin.get("origin_type"):
                continue
            valid.append(origin)
        return valid
    return []


def _resolve_direct_website_identity(
    website_url: str,
    timeout_seconds: int = DIRECT_IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    normalized = str(website_url or "").strip()
    now = datetime.utcnow().isoformat()
    if not normalized:
        return {
            "profile_url": None,
            "official_website": None,
            "identity_confidence": "low",
            "captured_at": now,
            "error": "missing_website",
        }
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0; +https://example.com/bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers, http2=False) as client:
            response = client.get(normalized)
            response.raise_for_status()
            resolved = str(response.url).strip() or normalized
            resolved_domain = normalize_domain(resolved)
            if not resolved_domain or _is_non_first_party_profile_domain(resolved_domain):
                resolved = normalized
            return {
                "profile_url": normalized,
                "official_website": resolved,
                "identity_confidence": "high",
                "captured_at": now,
                "error": None,
                "resolved_via_redirect": normalize_domain(resolved) != normalize_domain(normalized),
            }
    except Exception as exc:
        # Redirect resolution failures should not zero out identity for otherwise valid domains.
        return {
            "profile_url": normalized,
            "official_website": normalized,
            "identity_confidence": "high",
            "captured_at": now,
            "error": f"redirect_resolution_failed:{exc}",
            "resolved_via_redirect": False,
        }


def _apply_identity_payload(candidate: dict[str, Any], payload: dict[str, Any]) -> None:
    existing_official = str(
        candidate.get("official_website_url")
        or candidate.get("website")
        or ""
    ).strip()
    discovery_seed = str(
        candidate.get("discovery_url")
        or candidate.get("profile_url")
        or payload.get("profile_url")
        or existing_official
        or ""
    ).strip()
    normalized_input = (
        discovery_seed
        if discovery_seed.startswith(("http://", "https://"))
        else (f"https://{discovery_seed}" if discovery_seed else "")
    )
    input_domain = normalize_domain(normalized_input)

    resolved = str(payload.get("official_website") or "").strip()
    if resolved and not resolved.startswith(("http://", "https://")):
        resolved = f"https://{resolved}"

    existing_domain = normalize_domain(existing_official)
    final_official = resolved or (
        existing_official
        if existing_domain and not _is_non_first_party_profile_domain(existing_domain)
        else None
    )
    final_domain = normalize_domain(final_official) if final_official else None
    if final_domain and _is_non_first_party_profile_domain(final_domain):
        final_official = None
        final_domain = None

    # Persist explicit separation between discovery/profile URL and first-party official URL.
    if normalized_input:
        if _is_non_first_party_profile_domain(input_domain):
            candidate["profile_url"] = normalized_input
            candidate["discovery_url"] = candidate.get("discovery_url") or normalized_input
        elif not candidate.get("discovery_url"):
            candidate["discovery_url"] = normalized_input
    candidate["official_website_url"] = final_official
    candidate["original_website"] = normalized_input or existing_official
    candidate["website"] = final_official

    first_party_domains = _normalize_domain_list(candidate.get("first_party_domains") or [])
    if input_domain and not _is_non_first_party_profile_domain(input_domain) and input_domain not in first_party_domains:
        first_party_domains.append(input_domain)
    if final_domain and final_domain not in first_party_domains:
        first_party_domains.append(final_domain)
    candidate["first_party_domains"] = first_party_domains

    identity_confidence = str(payload.get("identity_confidence", "low") or "low")
    if not final_official:
        identity_confidence = "low"

    identity_sources = payload.get("identity_sources") if isinstance(payload.get("identity_sources"), list) else []
    if payload.get("profile_url"):
        identity_sources.append(payload.get("profile_url"))
    identity_sources = _dedupe_strings([str(item) for item in identity_sources if str(item).strip()])

    candidate["identity"] = {
        "input_website": normalized_input or None,
        "official_website": final_official,
        "input_domain": input_domain,
        "canonical_domain": final_domain or input_domain,
        "identity_confidence": identity_confidence,
        "identity_sources": identity_sources,
        "captured_at": payload.get("captured_at"),
        "error": payload.get("error"),
        "resolved_via_redirect": bool(payload.get("resolved_via_redirect")),
    }
    if not candidate.get("hq_country"):
        inferred_country = _infer_country_from_domain(final_domain or input_domain)
        if inferred_country:
            candidate["hq_country"] = inferred_country


def _resolve_identities_for_candidates(
    candidates: list[dict[str, Any]],
    max_fetches: int = MAX_IDENTITY_FETCHES_PER_RUN,
    timeout_seconds: int = IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
    concurrency: int = IDENTITY_RESOLUTION_CONCURRENCY,
) -> dict[str, Any]:
    """Resolve identity for all candidates with bounded network fetches."""
    aggregator_urls: list[str] = []
    direct_urls: list[str] = []
    aggregator_seen: set[str] = set()
    direct_seen: set[str] = set()
    for candidate in candidates:
        identity_url = str(
            candidate.get("website")
            or candidate.get("official_website_url")
            or candidate.get("profile_url")
            or candidate.get("discovery_url")
            or ""
        ).strip()
        if not identity_url:
            payload = {
                "profile_url": None,
                "official_website": None,
                "identity_confidence": "low",
                "captured_at": datetime.utcnow().isoformat(),
                "error": "missing_website",
            }
            _apply_identity_payload(candidate, payload)
            continue
        normalized = identity_url if identity_url.startswith(("http://", "https://")) else f"https://{identity_url}"
        domain = normalize_domain(normalized)
        if not domain:
            payload = {
                "profile_url": normalized,
                "official_website": None,
                "identity_confidence": "low",
                "captured_at": datetime.utcnow().isoformat(),
                "error": "invalid_website",
            }
            _apply_identity_payload(candidate, payload)
            continue
        if not _is_non_first_party_profile_domain(domain):
            if normalized.lower() not in direct_seen:
                direct_seen.add(normalized.lower())
                direct_urls.append(normalized)
            continue
        if normalized.lower() not in aggregator_seen:
            aggregator_seen.add(normalized.lower())
            aggregator_urls.append(normalized)

    fetch_urls = aggregator_urls[: max(0, max_fetches)]
    exhausted_urls = set(aggregator_urls[max(0, max_fetches) :])
    resolved_payloads: dict[str, dict[str, Any]] = {}
    fetch_errors: dict[str, str] = {}

    if fetch_urls:
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, 50))) as executor:
            futures = {
                executor.submit(resolve_external_website_from_profile, url, timeout_seconds): url
                for url in fetch_urls
            }
            for future in as_completed(futures):
                source_url = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "profile_url": source_url,
                        "official_website": None,
                        "identity_confidence": "low",
                        "captured_at": datetime.utcnow().isoformat(),
                        "error": f"profile_fetch_failed:{exc}",
                    }
                if result.get("error"):
                    fetch_errors[normalize_domain(source_url) or source_url] = str(result["error"])
                resolved_payloads[source_url.lower()] = result

    direct_fetch_urls = direct_urls[: max(0, max_fetches)]
    direct_exhausted = set(direct_urls[max(0, max_fetches) :])
    direct_payloads: dict[str, dict[str, Any]] = {}
    if direct_fetch_urls:
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, 50))) as executor:
            futures = {
                executor.submit(_resolve_direct_website_identity, url, DIRECT_IDENTITY_RESOLUTION_TIMEOUT_SECONDS): url
                for url in direct_fetch_urls
            }
            for future in as_completed(futures):
                source_url = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "profile_url": source_url,
                        "official_website": source_url,
                        "identity_confidence": "high",
                        "captured_at": datetime.utcnow().isoformat(),
                        "error": f"redirect_resolution_failed:{exc}",
                        "resolved_via_redirect": False,
                    }
                if result.get("error"):
                    fetch_errors[normalize_domain(source_url) or source_url] = str(result["error"])
                direct_payloads[source_url.lower()] = result

    for candidate in candidates:
        identity_url = str(
            candidate.get("website")
            or candidate.get("official_website_url")
            or candidate.get("profile_url")
            or candidate.get("discovery_url")
            or ""
        ).strip()
        if not identity_url:
            continue
        normalized = identity_url if identity_url.startswith(("http://", "https://")) else f"https://{identity_url}"
        domain = normalize_domain(normalized)
        if not domain:
            continue
        if not _is_non_first_party_profile_domain(domain):
            if normalized.lower() in direct_payloads:
                _apply_identity_payload(candidate, direct_payloads[normalized.lower()])
                continue
            error = "resolution_budget_exhausted" if normalized in direct_exhausted else "identity_not_resolved"
            _apply_identity_payload(
                candidate,
                {
                    "profile_url": normalized,
                    "official_website": normalized,
                    "identity_confidence": "high",
                    "captured_at": datetime.utcnow().isoformat(),
                    "error": error,
                    "resolved_via_redirect": False,
                },
            )
            continue
        if normalized.lower() in resolved_payloads:
            _apply_identity_payload(candidate, resolved_payloads[normalized.lower()])
            continue
        error = "resolution_budget_exhausted" if normalized in exhausted_urls else "external_website_not_found"
        payload = {
            "profile_url": normalized,
            "official_website": None,
            "identity_confidence": "low",
            "captured_at": datetime.utcnow().isoformat(),
            "error": error,
        }
        _apply_identity_payload(candidate, payload)

    resolved_high = 0
    for candidate in candidates:
        identity = candidate.get("identity") if isinstance(candidate.get("identity"), dict) else {}
        if identity.get("identity_confidence") == "high":
            resolved_high += 1
    return {
        "identity_resolved_count": resolved_high,
        "identity_fetch_count": len(fetch_urls) + len(direct_fetch_urls),
        "identity_fetch_errors": fetch_errors,
        "identity_budget_exhausted": max(0, len(aggregator_urls) - len(fetch_urls))
        + max(0, len(direct_urls) - len(direct_fetch_urls)),
    }


def _resolve_directory_profile_seed_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_fetches: int,
    timeout_seconds: int = IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
    concurrency: int = IDENTITY_RESOLUTION_CONCURRENCY,
) -> dict[str, Any]:
    if max_fetches <= 0:
        return {
            "candidates_considered": 0,
            "candidates_selected": 0,
            "identity_resolved_count": 0,
            "identity_fetch_count": 0,
            "identity_fetch_errors": {},
            "identity_budget_exhausted": 0,
        }

    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        origins = candidate.get("_origins") if isinstance(candidate.get("_origins"), list) else []
        if not any(str(origin.get("origin_type") or "") == "directory_seed" for origin in origins if isinstance(origin, dict)):
            continue
        official_website = str(candidate.get("official_website_url") or candidate.get("website") or "").strip()
        if official_website and not _is_non_first_party_profile_domain(normalize_domain(official_website)):
            continue
        profile_url = str(candidate.get("profile_url") or candidate.get("discovery_url") or "").strip()
        if not profile_url:
            continue
        profile_domain = normalize_domain(profile_url)
        if not profile_domain or not _is_non_first_party_profile_domain(profile_domain):
            continue
        eligible.append(candidate)

    if not eligible:
        return {
            "candidates_considered": 0,
            "candidates_selected": 0,
            "identity_resolved_count": 0,
            "identity_fetch_count": 0,
            "identity_fetch_errors": {},
            "identity_budget_exhausted": 0,
        }

    ranked = sorted(eligible, key=_directory_seed_priority, reverse=True)
    selected = ranked[:max_fetches]
    stats = _resolve_identities_for_candidates(
        selected,
        max_fetches=max_fetches,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
    )
    stats["candidates_considered"] = len(eligible)
    stats["candidates_selected"] = len(selected)
    return stats


def _registry_identifier(candidate: dict[str, Any]) -> Optional[str]:
    for key in ("registry_id", "legal_entity_id", "company_number", "siren"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return None


def _canonical_country(candidate: dict[str, Any]) -> Optional[str]:
    direct = normalize_country(str(candidate.get("hq_country") or "").strip())
    if direct:
        return direct
    identity = candidate.get("identity") if isinstance(candidate.get("identity"), dict) else {}
    inferred = normalize_country(_infer_country_from_domain(identity.get("canonical_domain")))
    return inferred


def _merge_candidate_into_entity(
    entity: dict[str, Any],
    candidate: dict[str, Any],
    merge_reason: str,
    merge_confidence: float,
) -> None:
    identity = candidate.get("identity") if isinstance(candidate.get("identity"), dict) else {}
    website = str(candidate.get("website") or "").strip() or None
    domain = normalize_domain(website)
    candidate_name = str(candidate.get("name") or "").strip()
    discovery_url = str(candidate.get("discovery_url") or candidate.get("profile_url") or "").strip() or None
    entity_type = str(candidate.get("entity_type") or "company").strip().lower() or "company"
    solution_name = str(candidate.get("solution_name") or "").strip() or None

    entity["alias_names"] = _dedupe_strings(entity.get("alias_names", []) + ([candidate_name] if candidate_name else []))
    entity["alias_websites"] = _dedupe_strings(entity.get("alias_websites", []) + ([website] if website else []))
    entity["merge_reasons"] = _dedupe_strings(entity.get("merge_reasons", []) + [merge_reason])

    existing_origins = entity.get("origins", [])
    entity["origins"] = existing_origins + _origin_entries(candidate)
    entity["why_relevant"] = _normalize_reasons((entity.get("why_relevant") or []) + (candidate.get("why_relevant") or []))
    entity["capability_signals"] = _dedupe_strings(
        (entity.get("capability_signals") or [])
        + _extract_capability_signals(candidate)
    )
    entity["likely_verticals"] = _dedupe_strings(
        [str(v).strip() for v in (entity.get("likely_verticals") or []) + (candidate.get("likely_verticals") or []) if str(v).strip()]
    )
    entity["brand_names"] = _dedupe_strings(
        [str(v).strip() for v in (entity.get("brand_names") or []) + (candidate.get("brand_names") or []) if str(v).strip()]
    )
    entity["short_description"] = (
        str(candidate.get("short_description") or "").strip()
        or str(entity.get("short_description") or "").strip()
        or None
    )
    entity["legal_name"] = entity.get("legal_name") or (str(candidate.get("legal_name") or "").strip() or None)
    if not str(entity.get("display_name") or "").strip():
        entity["display_name"] = (
            str(candidate.get("display_name") or "").strip()
            or str(candidate.get("brand_name") or "").strip()
            or entity.get("canonical_name")
        )
    entity["registry_fields"] = {
        **(entity.get("registry_fields") if isinstance(entity.get("registry_fields"), dict) else {}),
        **(candidate.get("registry_fields") if isinstance(candidate.get("registry_fields"), dict) else {}),
    }
    entity["reference_input"] = bool(entity.get("reference_input")) or bool(candidate.get("reference_input"))
    entity["merged_candidates_count"] = int(entity.get("merged_candidates_count") or 1) + 1
    entity["merge_confidence"] = max(float(entity.get("merge_confidence") or 0.0), merge_confidence)
    entity["precomputed_discovery_score"] = max(
        float(entity.get("precomputed_discovery_score") or 0.0),
        float(candidate.get("precomputed_discovery_score") or candidate.get("discovery_score") or 0.0),
    )
    entity["node_fit_summary"] = _merge_node_fit_summary(
        entity.get("node_fit_summary") if isinstance(entity.get("node_fit_summary"), dict) else {},
        candidate.get("node_fit_summary") if isinstance(candidate.get("node_fit_summary"), dict) else {},
    )
    entity["directness"] = _directness_from_node_fit_summary(
        entity.get("node_fit_summary") if isinstance(entity.get("node_fit_summary"), dict) else {},
        fallback=str(candidate.get("directness") or entity.get("directness") or "").strip().lower() or None,
    )
    if not entity.get("discovery_primary_url") and discovery_url:
        entity["discovery_primary_url"] = discovery_url

    first_party_domains = _normalize_domain_list(entity.get("first_party_domains") or [])
    for item in _normalize_domain_list(candidate.get("first_party_domains") or []):
        if item not in first_party_domains:
            first_party_domains.append(item)
    if domain and not _is_non_first_party_profile_domain(domain) and domain not in first_party_domains:
        first_party_domains.append(domain)
    entity["first_party_domains"] = first_party_domains

    if entity.get("entity_type") != "company" and entity_type == "company":
        entity["entity_type"] = "company"
    elif not entity.get("entity_type"):
        entity["entity_type"] = entity_type

    if solution_name:
        current_solutions = entity.get("solutions") if isinstance(entity.get("solutions"), list) else []
        candidate_solution = {
            "name": solution_name,
            "solution_slug": str(candidate.get("solution_slug") or "").strip() or None,
            "profile_url": str(candidate.get("profile_url") or "").strip() or None,
            "listing_url": str(candidate.get("listing_url") or "").strip() or None,
        }
        dedupe_key = (
            str(candidate_solution.get("name") or "").lower(),
            str(candidate_solution.get("solution_slug") or "").lower(),
        )
        deduped_solutions: list[dict[str, Any]] = []
        seen_solution_keys: set[tuple[str, str]] = set()
        for existing in current_solutions + [candidate_solution]:
            if not isinstance(existing, dict):
                continue
            key = (
                str(existing.get("name") or "").lower(),
                str(existing.get("solution_slug") or "").lower(),
            )
            if key in seen_solution_keys:
                continue
            seen_solution_keys.add(key)
            deduped_solutions.append(existing)
        if dedupe_key not in seen_solution_keys:
            deduped_solutions.append(candidate_solution)
        entity["solutions"] = deduped_solutions[:50]

    entity_registry_id = str(entity.get("registry_id") or "").strip()
    candidate_registry_id = _registry_identifier(candidate)
    if not entity_registry_id and candidate_registry_id:
        entity["registry_id"] = candidate_registry_id
        entity["registry_country"] = normalize_country(candidate.get("registry_country") or entity.get("country"))
        entity["registry_source"] = candidate.get("registry_source")
        entity["registry_ids"] = _dedupe_strings((entity.get("registry_ids") or []) + [candidate_registry_id])
    elif candidate_registry_id:
        entity["registry_ids"] = _dedupe_strings((entity.get("registry_ids") or []) + [candidate_registry_id])

    candidate_registry_identity = candidate.get("registry_identity") if isinstance(candidate.get("registry_identity"), dict) else {}
    if candidate_registry_identity:
        existing_identity = entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {}
        existing_conf = float(existing_identity.get("match_confidence") or 0.0)
        candidate_conf = float(candidate_registry_identity.get("match_confidence") or 0.0)
        if candidate_conf >= existing_conf:
            entity["registry_identity"] = candidate_registry_identity
        elif not existing_identity:
            entity["registry_identity"] = candidate_registry_identity

    entity["industry_signature"] = _merge_industry_signatures(
        entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
        candidate.get("industry_signature") if isinstance(candidate.get("industry_signature"), dict) else {},
    )

    current_conf = str(entity.get("identity_confidence") or "low")
    candidate_conf = str(identity.get("identity_confidence") or "low")
    if current_conf != "high" and candidate_conf == "high":
        entity["identity_confidence"] = "high"
        entity["identity_error"] = identity.get("error")

    current_domain = normalize_domain(entity.get("canonical_website"))
    if domain and not _is_non_first_party_profile_domain(domain):
        if not current_domain or _is_non_first_party_profile_domain(current_domain):
            entity["canonical_website"] = website
            entity["canonical_domain"] = domain
            if candidate_name:
                entity["canonical_name"] = candidate_name
                entity["display_name"] = str(candidate.get("display_name") or candidate_name).strip() or candidate_name

    if not entity.get("country"):
        entity["country"] = _canonical_country(candidate)


def _collapse_candidates_to_entities(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    domain_to_entity: dict[str, int] = {}
    registry_to_entity: dict[str, int] = {}
    slug_to_entity: dict[str, int] = {}
    suspected_duplicates: list[dict[str, Any]] = []

    for candidate in candidates:
        name = str(candidate.get("name") or "").strip()
        if not name:
            continue
        website = str(candidate.get("website") or "").strip() or None
        domain = normalize_domain(website)
        company_slug = str(candidate.get("company_slug") or "").strip().lower() or None
        registry_id = _registry_identifier(candidate)
        country = _canonical_country(candidate)
        identity = candidate.get("identity") if isinstance(candidate.get("identity"), dict) else {}
        name_norm = _normalize_name_for_matching(name)

        matched_index: Optional[int] = None
        merge_reason = ""
        merge_confidence = 0.0

        if (
            domain
            and not _is_aggregator_domain(domain)
            and not _is_registry_profile_domain(domain)
            and domain in domain_to_entity
        ):
            matched_index = domain_to_entity[domain]
            merge_reason = "canonical_domain_exact"
            merge_confidence = 1.0
        elif registry_id and registry_id in registry_to_entity:
            matched_index = registry_to_entity[registry_id]
            merge_reason = "registry_id_exact"
            merge_confidence = 1.0
        elif company_slug and company_slug in slug_to_entity:
            matched_index = slug_to_entity[company_slug]
            merge_reason = "company_slug_exact"
            merge_confidence = 0.98
        else:
            best_ratio = 0.0
            for idx, entity in enumerate(entities):
                entity_country = normalize_country(entity.get("country"))
                if country and entity_country and country != entity_country:
                    continue
                ratio = _name_similarity(name_norm, str(entity.get("normalized_name") or ""))
                if ratio < 0.86:
                    continue
                if _domains_conflict(domain, normalize_domain(entity.get("canonical_website"))):
                    if _can_bridge_registry_profile(candidate, entity, ratio):
                        if ratio > best_ratio:
                            best_ratio = ratio
                            matched_index = idx
                            merge_reason = "registry_profile_domain_bridge"
                            merge_confidence = round(ratio, 4)
                        continue
                    suspected_duplicates.append(
                        {
                            "left": entity.get("canonical_name"),
                            "right": name,
                            "confidence": round(ratio, 4),
                        }
                    )
                    continue
                if ratio > best_ratio:
                    best_ratio = ratio
                    matched_index = idx
            if matched_index is not None and best_ratio >= 0.86 and not merge_reason:
                merge_reason = "name_similarity_country_match"
                merge_confidence = round(best_ratio, 4)

        if matched_index is not None:
            _merge_candidate_into_entity(
                entities[matched_index],
                candidate,
                merge_reason=merge_reason or "merge",
                merge_confidence=merge_confidence or 0.86,
            )
            continue

        new_entity = {
            "temp_entity_id": f"entity_{len(entities) + 1}",
            "canonical_name": name,
            "display_name": str(candidate.get("display_name") or candidate.get("brand_name") or name).strip() or name,
            "legal_name": str(candidate.get("legal_name") or "").strip() or None,
            "brand_names": _dedupe_strings(
                [
                    str(v).strip()
                    for v in ([candidate.get("brand_name")] if str(candidate.get("brand_name") or "").strip() else [])
                    + (candidate.get("brand_names") or [])
                    if str(v).strip()
                ]
            ),
            "canonical_website": website,
            "canonical_domain": domain,
            "discovery_primary_url": str(candidate.get("discovery_url") or candidate.get("profile_url") or "")[:1000] or None,
            "entity_type": str(candidate.get("entity_type") or "company").strip().lower() or "company",
            "company_slug": company_slug,
            "country": country,
            "identity_confidence": identity.get("identity_confidence", "low"),
            "identity_error": identity.get("error"),
            "registry_country": normalize_country(candidate.get("registry_country") or country),
            "registry_id": registry_id,
            "registry_source": candidate.get("registry_source"),
            "registry_ids": [registry_id] if registry_id else [],
            "alias_names": [name] if name else [],
            "alias_websites": [website] if website else [],
            "merge_reasons": [],
            "origins": _origin_entries(candidate),
            "why_relevant": _normalize_reasons(candidate.get("why_relevant") or []),
            "capability_signals": _extract_capability_signals(candidate),
            "likely_verticals": _dedupe_strings(
                [str(v).strip() for v in (candidate.get("likely_verticals") or []) if str(v).strip()]
            ),
            "employee_estimate": _extract_employee_estimate(candidate),
            "qualification": candidate.get("qualification") if isinstance(candidate.get("qualification"), dict) else {},
            "reference_input": bool(candidate.get("reference_input")),
            "normalized_name": name_norm,
            "merged_candidates_count": 1,
            "merge_confidence": 1.0,
            "precomputed_discovery_score": float(candidate.get("precomputed_discovery_score") or candidate.get("discovery_score") or 0.0),
            "directness": _directness_from_node_fit_summary(
                candidate.get("node_fit_summary") if isinstance(candidate.get("node_fit_summary"), dict) else {},
                fallback=str(candidate.get("directness") or "").strip().lower() or None,
            ),
            "node_fit_summary": candidate.get("node_fit_summary") if isinstance(candidate.get("node_fit_summary"), dict) else {},
            "short_description": str(candidate.get("short_description") or "").strip() or None,
            "first_party_domains": _normalize_domain_list(candidate.get("first_party_domains") or []),
            "solutions": (
                [
                    {
                        "name": str(candidate.get("solution_name") or "").strip(),
                        "solution_slug": str(candidate.get("solution_slug") or "").strip() or None,
                        "profile_url": str(candidate.get("profile_url") or "").strip() or None,
                        "listing_url": str(candidate.get("listing_url") or "").strip() or None,
                    }
                ]
                if str(candidate.get("solution_name") or "").strip()
                else []
            ),
            "registry_identity": candidate.get("registry_identity") if isinstance(candidate.get("registry_identity"), dict) else {},
            "industry_signature": candidate.get("industry_signature") if isinstance(candidate.get("industry_signature"), dict) else {},
            "registry_fields": candidate.get("registry_fields") if isinstance(candidate.get("registry_fields"), dict) else {},
        }
        entities.append(new_entity)
        entity_idx = len(entities) - 1
        if domain and not _is_aggregator_domain(domain) and not _is_registry_profile_domain(domain):
            domain_to_entity[domain] = entity_idx
        if company_slug:
            slug_to_entity[company_slug] = entity_idx
        if registry_id:
            registry_to_entity[registry_id] = entity_idx

    for entity in entities:
        entity["origins"] = entity.get("origins", [])
        entity["origin_types"] = _dedupe_strings(
            [str(origin.get("origin_type")) for origin in entity["origins"] if origin.get("origin_type")]
        )
        entity["registry_ids"] = _dedupe_strings(entity.get("registry_ids", []))
        if not isinstance(entity.get("registry_identity"), dict):
            entity["registry_identity"] = {}
        entity["first_party_domains"] = _normalize_domain_list(entity.get("first_party_domains") or [])
        if not isinstance(entity.get("solutions"), list):
            entity["solutions"] = []
        entity["entity_type"] = str(entity.get("entity_type") or "company").strip().lower() or "company"
        entity["industry_signature"] = _merge_industry_signatures(
            entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
            {},
        )
        entity["node_fit_summary"] = _merge_node_fit_summary(
            entity.get("node_fit_summary") if isinstance(entity.get("node_fit_summary"), dict) else {},
            {},
        )
        entity["directness"] = _directness_from_node_fit_summary(
            entity.get("node_fit_summary") if isinstance(entity.get("node_fit_summary"), dict) else {},
            fallback=str(entity.get("directness") or "").strip().lower() or None,
        )

    raw_count = len(candidates)
    collapsed_count = max(0, raw_count - len(entities))
    alias_clusters_count = len([e for e in entities if len(e.get("alias_names", [])) > 1])
    return entities, {
        "raw_candidate_count": raw_count,
        "canonical_entity_count": len(entities),
        "duplicates_collapsed_count": collapsed_count,
        "alias_clusters_count": alias_clusters_count,
        "suspected_duplicate_count": len(suspected_duplicates),
        "suspected_duplicates": suspected_duplicates[:100],
    }


def _name_key_token(name: str) -> Optional[str]:
    normalized = _normalize_name_for_matching(name)
    tokens = [token for token in normalized.split() if len(token) >= 4 and token not in NAME_STOPWORDS]
    if not tokens:
        return None
    tokens.sort(key=len, reverse=True)
    return tokens[0]


def _registry_neighbor_signals(name: str, context: str) -> list[str]:
    combined = f"{name} {context}".lower()
    signals: list[str] = []
    for token in REGISTRY_RELEVANCE_TOKENS:
        if token in combined:
            signals.append(token)
    for token in INSTITUTIONAL_TOKENS:
        if token in combined:
            signals.append(token)
    return _dedupe_strings(signals)


def _registry_request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    retries: int = 2,
    **kwargs: Any,
) -> tuple[Optional[httpx.Response], Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(retries):
        try:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response, None
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep(0.2 * (attempt + 1))
    return None, last_error


def _extract_de_registry_identifier(court_line: str) -> Optional[str]:
    pattern = re.compile(r"\b(?:HRB|HRA|PR|GNR|VR)\s*\d+[A-Za-z]*\b", re.IGNORECASE)
    match = pattern.search(court_line or "")
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(0).upper())


def _search_de_registry_neighbors(
    query: str,
    max_hits: int = REGISTRY_MAX_RAW_HITS_PER_QUERY,
    timeout_seconds: int = 4,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return [], None

    gleif_results, gleif_error = _search_de_gleif_neighbors(
        normalized_query,
        max_hits=max_hits,
        timeout_seconds=timeout_seconds,
    )
    if gleif_results:
        return gleif_results, None

    base_url = "https://www.handelsregister.de/rp_web"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers, http2=False) as client:
            welcome_response, error = _registry_request_with_retries(client, "GET", f"{base_url}/welcome.xhtml")
            if error or welcome_response is None:
                merged = [f"de_registry_welcome_failed:{error}" if error else "de_registry_welcome_failed", gleif_error]
                return [], ";".join(item for item in merged if item)

            welcome_tree = HTMLParser(welcome_response.text)
            view_state_node = welcome_tree.css_first("input[name='javax.faces.ViewState']")
            if view_state_node is None:
                return [], "de_registry_welcome_viewstate_missing"
            view_state = str(view_state_node.attributes.get("value") or "")

            _, nav_error = _registry_request_with_retries(
                client,
                "POST",
                f"{base_url}/welcome.xhtml",
                data={
                    "naviForm": "naviForm",
                    "naviForm:normaleSucheLink": "naviForm:normaleSucheLink",
                    "javax.faces.ViewState": view_state,
                },
            )
            if nav_error:
                merged = [f"de_registry_navigation_failed:{nav_error}" if nav_error else "de_registry_navigation_failed", gleif_error]
                return [], ";".join(item for item in merged if item)

            search_page_response, search_page_error = _registry_request_with_retries(
                client,
                "GET",
                f"{base_url}/normalesuche/welcome.xhtml",
            )
            if search_page_error or search_page_response is None:
                merged = [f"de_registry_search_page_failed:{search_page_error}" if search_page_error else "de_registry_search_page_failed", gleif_error]
                return [], ";".join(item for item in merged if item)

            search_tree = HTMLParser(search_page_response.text)
            search_view_state_node = search_tree.css_first("input[name='javax.faces.ViewState']")
            if search_view_state_node is None:
                merged = ["de_registry_search_viewstate_missing", gleif_error]
                return [], ";".join(item for item in merged if item)
            search_view_state = str(search_view_state_node.attributes.get("value") or "")

            result_response, result_error = _registry_request_with_retries(
                client,
                "POST",
                f"{base_url}/normalesuche/welcome.xhtml",
                data={
                    "form": "form",
                    "form:schlagwoerter": normalized_query,
                    "form:schlagwortOptionen": "1",
                    "form:registerart": "0",
                    "form:btnSuche": "Suche starten",
                    "javax.faces.ViewState": search_view_state,
                },
            )
            if result_error or result_response is None:
                merged = [f"de_registry_search_failed:{result_error}" if result_error else "de_registry_search_failed", gleif_error]
                return [], ";".join(item for item in merged if item)

            result_tree = HTMLParser(result_response.text)
            results: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            seen_names: set[str] = set()
            citation_url = str(result_response.url)

            for row in result_tree.css("tr"):
                cells = [
                    re.sub(r"\s+", " ", cell.text(separator=" ", strip=True)).strip()
                    for cell in row.css("td")
                ]
                if len(cells) < 5:
                    continue

                court_line = cells[1] if len(cells) > 1 else ""
                name = cells[2] if len(cells) > 2 else ""
                city = cells[3] if len(cells) > 3 else ""
                status = cells[4] if len(cells) > 4 else ""

                if not name or name.lower() == "historie":
                    continue
                if "amtsgericht" not in court_line.lower():
                    continue

                registry_id = _extract_de_registry_identifier(court_line)
                dedupe_key = registry_id or _normalize_name_for_matching(name)
                if not dedupe_key:
                    continue
                if dedupe_key in seen_ids:
                    continue
                if _normalize_name_for_matching(name) in seen_names:
                    continue

                seen_ids.add(dedupe_key)
                seen_names.add(_normalize_name_for_matching(name))

                context_text = " ".join(
                    token for token in [court_line, city, status, normalized_query] if token
                ).strip()
                register_type = registry_id[:3] if registry_id else ""
                industry_codes = [register_type] if register_type else []
                record = {
                    "name": name,
                    "website": None,
                    "country": "DE",
                    "registry_id": registry_id,
                    "registry_source": "de_handelsregister_html",
                    "registry_url": citation_url,
                    "status": status,
                    "is_active": _looks_active_status(status),
                    "context_text": context_text,
                    "industry_codes": industry_codes,
                }
                record["industry_keywords"] = _industry_keywords_from_record(record)
                results.append(record)
                if len(results) >= max_hits:
                    break

            if results:
                return results, None
            return [], gleif_error
    except Exception as exc:
        merged = [f"de_registry_search_failed:{exc}", gleif_error]
        return [], ";".join(item for item in merged if item)


def _search_de_gleif_neighbors(
    query: str,
    max_hits: int = REGISTRY_MAX_RAW_HITS_PER_QUERY,
    timeout_seconds: int = 6,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return [], None

    url = "https://api.gleif.org/api/v1/lei-records"
    params = {
        "filter[entity.legalAddress.country]": "DE",
        "filter[entity.legalName]": normalized_query,
        "page[size]": min(25, max(1, max_hits)),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return [], f"de_gleif_search_failed:{exc}"

    results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in (payload.get("data") or [])[: max_hits]:
        if not isinstance(row, dict):
            continue
        attributes = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        entity = attributes.get("entity") if isinstance(attributes.get("entity"), dict) else {}
        legal_name_obj = entity.get("legalName") if isinstance(entity.get("legalName"), dict) else {}
        legal_name = str(legal_name_obj.get("name") or "").strip()
        if not legal_name:
            continue

        legal_country = str((entity.get("legalAddress") or {}).get("country") or "").strip().upper()
        if legal_country and legal_country != "DE":
            continue

        registry_id = str(entity.get("registeredAs") or "").strip() or None
        lei_id = str(row.get("id") or "").strip()
        dedupe_key = registry_id or lei_id or _normalize_name_for_matching(legal_name)
        if not dedupe_key or dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        entity_status = str(entity.get("status") or "").strip().upper()
        is_active = entity_status in {"ACTIVE", "ISSUED"} or _looks_active_status(entity_status)
        legal_form_obj = entity.get("legalForm") if isinstance(entity.get("legalForm"), dict) else {}
        legal_form_id = str(legal_form_obj.get("id") or "").strip().upper()
        jurisdiction = str(entity.get("jurisdiction") or "").strip().upper()
        context_text = " ".join(
            token
            for token in [
                legal_name,
                jurisdiction,
                legal_form_id,
                entity_status,
                normalized_query,
                "LEI registry",
            ]
            if token
        ).strip()
        citation_url = f"https://search.gleif.org/#/record/{lei_id}" if lei_id else str(response.url)
        industry_codes = [code for code in [legal_form_id] if code]
        record = {
            "name": legal_name,
            "website": None,
            "country": "DE",
            "registry_id": registry_id or lei_id or None,
            "registry_source": "de_gleif_lei",
            "registry_url": citation_url,
            "status": entity_status.lower() if entity_status else "",
            "is_active": bool(is_active),
            "context_text": context_text,
            "industry_codes": industry_codes,
        }
        record["industry_keywords"] = _industry_keywords_from_record(record)
        results.append(record)
        if len(results) >= max_hits:
            break
    return results, None


def _search_gleif_registry_neighbors(
    country: str,
    query: str,
    max_hits: int = REGISTRY_MAX_RAW_HITS_PER_QUERY,
    timeout_seconds: int = 6,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    normalized_country = normalize_country(country)
    normalized_query = str(query or "").strip()
    if not normalized_query or normalized_country not in REGISTRY_EXPANSION_COUNTRIES:
        return [], None

    url = "https://api.gleif.org/api/v1/lei-records"
    params = {
        "filter[entity.legalAddress.country]": normalized_country,
        "filter[entity.legalName]": normalized_query,
        "page[size]": min(25, max(1, max_hits)),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return [], f"{str(normalized_country).lower()}_gleif_search_failed:{exc}"

    results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in (payload.get("data") or [])[: max_hits]:
        if not isinstance(row, dict):
            continue
        attributes = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        entity = attributes.get("entity") if isinstance(attributes.get("entity"), dict) else {}
        legal_name_obj = entity.get("legalName") if isinstance(entity.get("legalName"), dict) else {}
        legal_name = str(legal_name_obj.get("name") or "").strip()
        if not legal_name:
            continue

        legal_country = str((entity.get("legalAddress") or {}).get("country") or "").strip().upper()
        if legal_country and legal_country != normalized_country:
            continue

        registry_id = str(entity.get("registeredAs") or "").strip() or None
        lei_id = str(row.get("id") or "").strip()
        dedupe_key = registry_id or lei_id or _normalize_name_for_matching(legal_name)
        if not dedupe_key or dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        entity_status = str(entity.get("status") or "").strip().upper()
        is_active = entity_status in {"ACTIVE", "ISSUED"} or _looks_active_status(entity_status)
        legal_form_obj = entity.get("legalForm") if isinstance(entity.get("legalForm"), dict) else {}
        legal_form_id = str(legal_form_obj.get("id") or "").strip().upper()
        jurisdiction = str(entity.get("jurisdiction") or "").strip().upper()
        context_text = " ".join(
            token
            for token in [
                legal_name,
                jurisdiction,
                legal_form_id,
                entity_status,
                normalized_query,
                f"{normalized_country} LEI registry",
            ]
            if token
        ).strip()
        citation_url = f"https://search.gleif.org/#/record/{lei_id}" if lei_id else str(response.url)
        industry_codes = [code for code in [legal_form_id] if code]
        record = {
            "name": legal_name,
            "website": None,
            "country": normalized_country,
            "registry_id": registry_id or lei_id or None,
            "registry_source": f"{str(normalized_country).lower()}_gleif_lei",
            "registry_url": citation_url,
            "status": entity_status.lower() if entity_status else "",
            "is_active": bool(is_active),
            "context_text": context_text,
            "industry_codes": industry_codes,
        }
        record["industry_keywords"] = _industry_keywords_from_record(record)
        results.append(record)
        if len(results) >= max_hits:
            break
    return results, None


def _search_fr_registry_neighbors(
    query: str,
    max_hits: int = REGISTRY_MAX_RAW_HITS_PER_QUERY,
    timeout_seconds: int = 6,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not query.strip():
        return [], None
    rows, error = _fr_registry_search(
        query=query.strip(),
        page=1,
        per_page=max_hits,
        only_active=False,
        timeout_seconds=timeout_seconds,
    )
    if error:
        return [], error
    results: list[dict[str, Any]] = []
    for row in rows[:max_hits]:
        record = _fr_registry_source_record(row, query_hint=query)
        if not str(record.get("name") or "").strip():
            continue
        results.append(record)
    return results, None


def _fetch_uk_company_profile(
    company_number: str,
    api_key: str,
    timeout_seconds: int = 6,
) -> tuple[dict[str, Any], Optional[str]]:
    profile_url = f"https://api.company-information.service.gov.uk/company/{company_number}"
    try:
        with httpx.Client(timeout=timeout_seconds, auth=(api_key, "")) as client:
            response = client.get(profile_url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return {}, f"uk_registry_profile_failed:{company_number}:{exc}"

    sic_codes = payload.get("sic_codes") if isinstance(payload.get("sic_codes"), list) else []
    company_status = str(payload.get("company_status") or "").strip().lower() or "unknown"
    jurisdiction = str(payload.get("jurisdiction") or "").strip().upper() or "UK"
    return {
        "sic_codes": [str(code).strip() for code in sic_codes if str(code).strip()],
        "company_status": company_status,
        "jurisdiction": jurisdiction,
    }, None


def _search_uk_registry_neighbors(
    query: str,
    max_hits: int = REGISTRY_MAX_RAW_HITS_PER_QUERY,
    timeout_seconds: int = 6,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not query.strip():
        return [], None
    api_key = str(settings.companies_house_api_key or "").strip()
    api_errors: list[str] = []
    if api_key:
        search_url = "https://api.company-information.service.gov.uk/search/companies"
        try:
            with httpx.Client(timeout=timeout_seconds, auth=(api_key, "")) as client:
                response = client.get(search_url, params={"q": query.strip(), "items_per_page": max_hits})
                response.raise_for_status()
                payload = response.json()

            results: list[dict[str, Any]] = []
            for item in (payload.get("items") or [])[:max_hits]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("title") or item.get("company_name") or "").strip()
                company_number = str(item.get("company_number") or "").strip().upper()
                if not name or not company_number:
                    continue
                profile, profile_error = _fetch_uk_company_profile(company_number, api_key, timeout_seconds=timeout_seconds)
                if profile_error:
                    api_errors.append(profile_error)

                status = str(profile.get("company_status") or item.get("company_status") or "unknown").strip().lower()
                sic_codes = [str(code).strip().upper() for code in (profile.get("sic_codes") or []) if str(code).strip()]
                context_tokens = [
                    str(item.get("description") or ""),
                    " ".join(sic_codes),
                    str(item.get("kind") or ""),
                ]
                context_text = " ".join(token for token in context_tokens if token).strip()
                citation_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
                record = {
                    "name": name,
                    "website": None,
                    "country": "UK",
                    "registry_id": company_number,
                    "registry_source": "uk_companies_house_api",
                    "registry_url": citation_url,
                    "status": status,
                    "is_active": _looks_active_status(status),
                    "context_text": context_text,
                    "industry_codes": sic_codes,
                }
                record["industry_keywords"] = _industry_keywords_from_record(record)
                results.append(record)

            if results:
                if api_errors:
                    return results, ";".join(api_errors[:3])
                return results, None
        except Exception as exc:
            api_errors.append(f"uk_registry_api_search_failed:{exc}")

    url = "https://find-and-update.company-information.service.gov.uk/search/companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
            response = client.get(url, params={"q": query.strip()})
            response.raise_for_status()
            tree = HTMLParser(response.text)
    except Exception as exc:
        merged_error = ";".join(api_errors + [f"uk_registry_search_failed:{exc}"])
        return [], merged_error

    if tree is None:
        merged_error = ";".join(api_errors + ["uk_registry_parse_failed"])
        return [], merged_error

    results: list[dict[str, Any]] = []
    seen_numbers: set[str] = set()
    for anchor in tree.css("a[href]"):
        href = str(anchor.attributes.get("href") or "").strip()
        if "/company/" not in href:
            continue
        match = re.search(r"/company/([A-Za-z0-9]+)", href)
        if not match:
            continue
        company_number = match.group(1).upper()
        if company_number in seen_numbers:
            continue
        name = re.sub(r"\s+", " ", anchor.text(separator=" ", strip=True)).strip()
        if not name:
            continue
        seen_numbers.add(company_number)
        citation_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
        record = {
            "name": name,
            "website": None,
            "country": "UK",
            "registry_id": company_number,
            "registry_source": "uk_companies_house_html",
            "registry_url": citation_url,
            "status": "",
            "is_active": True,
            "context_text": query.strip(),
            "industry_codes": [],
        }
        record["industry_keywords"] = _industry_keywords_from_record(record)
        results.append(record)
        if len(results) >= max_hits:
            break
    if api_errors:
        return results, ";".join(api_errors[:3])
    return results, None


def _registry_country_candidates_for_entity(entity: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    explicit_country = normalize_country(entity.get("country"))
    if explicit_country in REGISTRY_EXPANSION_COUNTRIES:
        candidates.append(explicit_country)

    reason_hints = _country_hints_from_reasons(entity.get("why_relevant") or [])
    for hint in reason_hints:
        if hint in REGISTRY_EXPANSION_COUNTRIES:
            candidates.append(hint)

    inferred_country = normalize_country(_infer_country_from_domain(normalize_domain(entity.get("canonical_website"))))
    if inferred_country in REGISTRY_EXPANSION_COUNTRIES:
        candidates.append(inferred_country)

    name_hint = _infer_country_from_text(str(entity.get("canonical_name") or ""))
    if name_hint in REGISTRY_EXPANSION_COUNTRIES:
        candidates.append(name_hint)

    if not candidates:
        candidates.extend(["FR", "UK", "DE", "BE", "NL", "LU", "CH", "MC"])
    return _dedupe_strings(candidates)


def _registry_queries_for_entity(entity_name: str, industry_signature: dict[str, Any] | None = None) -> list[str]:
    queries: list[str] = []
    seen_queries: set[str] = set()
    canonical_name = str(entity_name or "").strip()
    if canonical_name:
        lowered = canonical_name.lower()
        queries.append(canonical_name)
        seen_queries.add(lowered)
    token = _name_key_token(canonical_name)
    if token and token.lower() not in canonical_name.lower():
        lowered = token.lower()
        if lowered not in seen_queries:
            queries.append(token)
            seen_queries.add(lowered)

    if industry_signature:
        keywords = industry_signature.get("industry_keywords") or []
        if isinstance(keywords, list):
            for keyword in keywords[:2]:
                normalized = str(keyword or "").strip()
                if len(normalized) >= 4 and normalized.lower() not in seen_queries:
                    queries.append(normalized)
                    seen_queries.add(normalized.lower())
    return queries[:3]


def _industry_code_similarity(seed_codes: list[str], candidate_codes: list[str]) -> float:
    seed_norm = [str(code or "").upper().strip() for code in seed_codes if str(code or "").strip()]
    candidate_norm = [str(code or "").upper().strip() for code in candidate_codes if str(code or "").strip()]
    if not seed_norm or not candidate_norm:
        return 0.0

    best = 0.0
    for left in seed_norm:
        for right in candidate_norm:
            if left == right:
                best = max(best, 1.0)
            elif left[:2] and right[:2] and left[:2] == right[:2]:
                best = max(best, 0.8)
            elif left[:1] and right[:1] and left[:1] == right[:1]:
                best = max(best, 0.4)
    return best


def _institutional_language_score(record: dict[str, Any]) -> float:
    text = f"{record.get('name') or ''} {record.get('context_text') or ''}".lower()
    if not text.strip():
        return 0.0
    hit_count = 0
    for token in INSTITUTIONAL_TOKENS:
        if token in text:
            hit_count += 1
    return min(1.0, hit_count / 3.0)


def _software_signal_score(record: dict[str, Any]) -> float:
    codes = record.get("industry_codes") or []
    keywords = record.get("industry_keywords") or []
    code_score = _software_signal_score_from_codes(codes)
    keyword_score = 1.0 if any(token in keywords for token in SOFTWARE_SIGNAL_TOKENS) else 0.0
    text = f"{record.get('name') or ''} {record.get('context_text') or ''}".lower()
    text_score = 1.0 if any(token in text for token in SOFTWARE_SIGNAL_TOKENS) else 0.0
    return max(code_score, keyword_score, text_score * 0.6)


def _name_brand_proximity(seed_name: str, record_name: str) -> float:
    return _name_similarity(
        _normalize_name_for_matching(seed_name),
        _normalize_name_for_matching(record_name),
    )


def _score_registry_neighbor(
    seed_entity: dict[str, Any],
    record: dict[str, Any],
) -> tuple[float, dict[str, float], list[str]]:
    seed_signature = seed_entity.get("industry_signature") if isinstance(seed_entity.get("industry_signature"), dict) else {}
    seed_codes = seed_signature.get("industry_codes") or []

    industry_score = _industry_code_similarity(seed_codes, record.get("industry_codes") or [])
    institutional_score = _institutional_language_score(record)
    software_score = _software_signal_score(record)
    name_score = _name_brand_proximity(str(seed_entity.get("canonical_name") or ""), str(record.get("name") or ""))

    validity_score = 1.0 if (record.get("registry_id") and bool(record.get("is_active"))) else 0.0

    weighted = (
        40.0 * industry_score
        + 25.0 * institutional_score
        + 20.0 * software_score
        + 15.0 * name_score
        + 8.0 * validity_score
    )
    signals = _industry_keywords_from_record(record)
    return round(weighted, 2), {
        "industry_code_match": round(industry_score, 4),
        "institutional_language_match": round(institutional_score, 4),
        "software_signal_match": round(software_score, 4),
        "name_brand_proximity": round(name_score, 4),
        "validity_bonus": round(validity_score, 4),
    }, signals


def _select_registry_identity_seed_entities(
    entities: list[dict[str, Any]],
    top_n: int = REGISTRY_IDENTITY_TOP_SEEDS,
) -> list[dict[str, Any]]:
    ranked_seeds = [
        entity for entity in entities
        if set(entity.get("origin_types") or []).intersection(
            {"directory_seed", "reference_seed", "benchmark_seed"}
        )
        and str(entity.get("entity_type") or "company").strip().lower() == "company"
    ]
    ranked_seeds.sort(key=_candidate_priority_score, reverse=True)
    return ranked_seeds[:top_n]


def _run_registry_search(country: str, query: str) -> tuple[list[dict[str, Any]], Optional[str], str]:
    if country == "FR":
        records, error = _search_fr_registry_neighbors(query)
        return records, error, "fr_recherche_entreprises"
    if country == "DE":
        records, error = _search_de_registry_neighbors(query)
        source_name = str(records[0].get("registry_source") or "de_registry") if records else "de_registry"
        return records, error, source_name
    if country == "UK":
        records, error = _search_uk_registry_neighbors(query)
        return records, error, "uk_companies_house"
    if country in {"BE", "NL", "LU", "CH", "MC"}:
        records, error = _search_gleif_registry_neighbors(country, query)
        source_name = str(records[0].get("registry_source") or f"{country.lower()}_gleif_lei") if records else f"{country.lower()}_gleif_lei"
        return records, error, source_name
    return [], f"registry_country_unsupported:{country}", "registry_unsupported"


def _choose_identity_match(
    seed_entity: dict[str, Any],
    country: str,
    records: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], float, list[str]]:
    best_record: Optional[dict[str, Any]] = None
    best_score = 0.0
    best_reasons: list[str] = []

    seed_name = str(seed_entity.get("canonical_name") or "").strip()
    seed_name_norm = _normalize_name_for_matching(seed_name)
    seed_domain = normalize_domain(seed_entity.get("canonical_website"))
    seed_registry_id = str(seed_entity.get("registry_id") or "").strip()
    key_token = _name_key_token(seed_name)
    country_hints = _country_hints_from_reasons(seed_entity.get("why_relevant") or [])

    for record in records:
        record_name = str(record.get("name") or "").strip()
        if not record_name:
            continue
        record_name_norm = _normalize_name_for_matching(record_name)
        name_score = _name_similarity(
            seed_name_norm,
            record_name_norm,
        )
        if name_score < 0.60:
            continue

        reasons: list[str] = []
        score = name_score * 0.65
        reasons.append("name_similarity")

        if key_token and key_token in record_name_norm.split():
            score += 0.15
            reasons.append("brand_token_match")

        if seed_name_norm and record_name_norm and (
            seed_name_norm in record_name_norm or record_name_norm in seed_name_norm
        ):
            score += 0.08
            reasons.append("name_containment")

        record_website = str(record.get("website") or "").strip()
        record_domain = normalize_domain(record_website)
        website_match = bool(seed_domain and record_domain and seed_domain == record_domain)
        if website_match:
            score += 0.2
            reasons.append("website_domain_match")

        record_registry_id = str(record.get("registry_id") or "").strip()
        if seed_registry_id and record_registry_id and seed_registry_id == record_registry_id:
            score += 0.2
            reasons.append("registry_id_match")

        if country in country_hints:
            score += 0.1
            reasons.append("country_hint_match")

        if score > best_score:
            best_score = score
            best_record = record
            best_reasons = reasons

    if best_record is None:
        return None, 0.0, []
    return best_record, round(min(1.0, best_score), 4), best_reasons


def _apply_registry_identity_map(
    entities: list[dict[str, Any]],
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mapped_count = 0
    candidates_count = 0
    country_breakdown: dict[str, int] = {}
    queries_by_country: dict[str, int] = {}
    raw_hits_by_country: dict[str, int] = {}
    errors: list[str] = []
    query_logs: list[dict[str, Any]] = []
    de_query_count = 0
    de_error_count = 0
    de_disabled = False
    identity_top_n = max(10, int(getattr(settings, "registry_identity_top_seeds", REGISTRY_IDENTITY_TOP_SEEDS)))
    query_budget = max(1, int(getattr(settings, "registry_max_queries", REGISTRY_MAX_QUERIES)))
    de_query_limit = max(1, int(getattr(settings, "registry_max_de_queries", REGISTRY_MAX_DE_QUERIES)))
    identity_deadline_seconds = max(
        30,
        int(getattr(settings, "registry_identity_max_seconds", REGISTRY_IDENTITY_MAX_SECONDS)),
    )
    deadline = time.monotonic() + identity_deadline_seconds
    timed_out = False

    identity_seed_entities = _select_registry_identity_seed_entities(entities, top_n=identity_top_n)
    candidates_count = len(identity_seed_entities)

    for entity in identity_seed_entities:
        if time.monotonic() > deadline:
            timed_out = True
            errors.append("registry_identity_timeout")
            break
        if query_budget <= 0:
            break
        entity_name = str(entity.get("canonical_name") or "").strip()
        if not entity_name:
            continue

        candidate_countries = _registry_country_candidates_for_entity(entity)
        best_record: Optional[dict[str, Any]] = None
        best_score = 0.0
        best_reasons: list[str] = []
        best_country: Optional[str] = None
        matched_query: Optional[str] = None

        for country in candidate_countries:
            if time.monotonic() > deadline:
                timed_out = True
                errors.append("registry_identity_timeout")
                break
            if query_budget <= 0:
                break
            if country == "DE" and de_disabled:
                continue
            if country == "DE" and de_query_count >= de_query_limit:
                continue
            queries = _registry_queries_for_entity(entity_name, industry_signature=None)
            if country == "DE":
                queries = queries[:1]
            for query in queries:
                if time.monotonic() > deadline:
                    timed_out = True
                    errors.append("registry_identity_timeout")
                    break
                if query_budget <= 0:
                    break
                if country == "DE" and de_query_count >= de_query_limit:
                    break
                query_budget -= 1
                if country == "DE":
                    de_query_count += 1
                queries_by_country[country] = queries_by_country.get(country, 0) + 1
                records, error, source_name = _run_registry_search(country, query)
                if error:
                    errors.append(error)
                    if country == "DE":
                        de_error_count += 1
                        if de_error_count >= REGISTRY_DE_ERROR_BREAKER:
                            de_disabled = True
                raw_hits_by_country[country] = raw_hits_by_country.get(country, 0) + len(records)

                selected_record, confidence, reasons = _choose_identity_match(entity, country, records)
                query_logs.append(
                    {
                        "run_id": run_id,
                        "seed_entity_name": entity_name,
                        "query_type": "identity_map",
                        "country": country,
                        "source_name": source_name,
                        "query": query,
                        "raw_hits": len(records),
                        "kept_hits": 1 if selected_record else 0,
                        "reject_reasons_json": {} if selected_record else {"no_match": len(records)},
                        "metadata_json": {
                            "selected_registry_id": selected_record.get("registry_id") if selected_record else None,
                            "match_confidence": confidence,
                            "match_reasons": reasons,
                        },
                    }
                )
                if selected_record and confidence > best_score:
                    best_record = selected_record
                    best_score = confidence
                    best_reasons = reasons
                    best_country = country
                    matched_query = query

            if timed_out:
                break

        if timed_out:
            break

        has_brand_match = "brand_token_match" in best_reasons
        if best_record and best_country and (
            best_score >= REGISTRY_IDENTITY_MIN_SCORE
            or (has_brand_match and best_score >= REGISTRY_IDENTITY_BRAND_MATCH_MIN_SCORE)
        ):
            registry_id = str(best_record.get("registry_id") or "").strip() or None
            registry_source = str(best_record.get("registry_source") or "").strip() or None
            entity["registry_id"] = registry_id
            entity["registry_source"] = registry_source
            entity["registry_country"] = best_country
            entity["country"] = entity.get("country") or best_country
            entity["registry_identity"] = {
                "country": best_country,
                "id": registry_id,
                "source": registry_source,
                "match_confidence": best_score,
                "match_reasons": best_reasons,
            }
            entity["industry_signature"] = _normalize_industry_signature(
                best_record.get("industry_codes") or [],
                best_record.get("industry_keywords") or [],
            )
            entity["why_relevant"] = _normalize_reasons(
                (entity.get("why_relevant") or []) + [
                    {
                        "text": (
                            f"Registry identity mapped via {registry_source} ({best_country}) "
                            f"with confidence {best_score:.2f} using query '{matched_query}'."
                        ),
                        "citation_url": best_record.get("registry_url"),
                        "dimension": "registry_identity",
                    }
                ]
            )
            entity["origins"] = (entity.get("origins") or []) + [
                {
                    "origin_type": "registry_identity",
                    "origin_url": best_record.get("registry_url"),
                    "source_name": registry_source,
                    "source_run_id": None,
                    "metadata": {
                        "query_type": "identity_map",
                        "country": best_country,
                        "query": matched_query,
                        "match_confidence": best_score,
                    },
                }
            ]
            mapped_count += 1
            country_breakdown[best_country] = country_breakdown.get(best_country, 0) + 1

    diagnostics = {
        "registry_identity_candidates_count": candidates_count,
        "registry_identity_mapped_count": mapped_count,
        "registry_identity_country_breakdown": country_breakdown,
        "registry_queries_by_country": queries_by_country,
        "registry_raw_hits_by_country": raw_hits_by_country,
        "registry_de_queries_count": de_query_count,
        "registry_de_disabled": de_disabled,
        "registry_identity_timed_out": timed_out,
        "registry_identity_errors": errors[:100],
    }
    return entities, diagnostics, query_logs


def _expand_registry_neighbors(
    entities: list[dict[str, Any]],
    run_id: str,
    max_queries: int = REGISTRY_MAX_QUERIES,
    max_neighbors: int = REGISTRY_MAX_ACCEPTED_NEIGHBORS,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    query_count = 0
    raw_hits = 0
    errors: list[str] = []
    query_logs: list[dict[str, Any]] = []
    reject_reason_counts: dict[str, int] = {}
    queries_by_country: dict[str, int] = {}
    raw_hits_by_country: dict[str, int] = {}
    kept_pre_dedupe = 0
    seen_neighbor_keys: set[tuple[str, str, str]] = set()
    de_query_count = 0
    de_error_count = 0
    de_disabled = False
    effective_max_queries = max(1, int(getattr(settings, "registry_max_queries", max_queries)))
    effective_max_neighbors = max(1, int(getattr(settings, "registry_max_accepted_neighbors", max_neighbors)))
    de_query_limit = max(1, int(getattr(settings, "registry_max_de_queries", REGISTRY_MAX_DE_QUERIES)))
    neighbor_deadline_seconds = max(
        30,
        int(getattr(settings, "registry_neighbor_max_seconds", REGISTRY_NEIGHBOR_MAX_SECONDS)),
    )
    deadline = time.monotonic() + neighbor_deadline_seconds
    timed_out = False
    neighbors_with_first_party_website_count = 0
    neighbors_dropped_missing_official_website_count = 0
    origin_screening_counts: dict[str, int] = {
        "seed_entities_considered": 0,
        "records_screened": 0,
        "records_accepted": 0,
        "records_rejected": 0,
    }

    ranked_entities = sorted(entities, key=_candidate_priority_score, reverse=True)

    for seed_entity in ranked_entities:
        if time.monotonic() > deadline:
            timed_out = True
            errors.append("registry_neighbor_timeout")
            break
        if len(accepted) >= effective_max_neighbors or query_count >= effective_max_queries:
            break
        if str(seed_entity.get("entity_type") or "company").strip().lower() != "company":
            continue

        seed_identity = seed_entity.get("registry_identity") if isinstance(seed_entity.get("registry_identity"), dict) else {}
        seed_country = normalize_country(seed_identity.get("country") or seed_entity.get("registry_country") or seed_entity.get("country"))
        if seed_country not in REGISTRY_EXPANSION_COUNTRIES:
            continue
        if seed_country == "DE" and de_disabled:
            continue
        origin_screening_counts["seed_entities_considered"] = origin_screening_counts.get("seed_entities_considered", 0) + 1

        seed_name = str(seed_entity.get("canonical_name") or "").strip()
        if not seed_name:
            continue
        seed_signature = seed_entity.get("industry_signature") if isinstance(seed_entity.get("industry_signature"), dict) else {}
        if not (seed_signature.get("industry_keywords") or []):
            seed_reason_text = " ".join(
                str(reason.get("text") or "")
                for reason in (seed_entity.get("why_relevant") or [])
                if isinstance(reason, dict)
            )
            seed_context_text = " ".join(
                part for part in [
                    seed_reason_text,
                    " ".join(str(item) for item in (seed_entity.get("capability_signals") or [])),
                    " ".join(str(item) for item in (seed_entity.get("likely_verticals") or [])),
                ] if part
            ).strip()
            inferred_keywords = _industry_keywords_from_record(
                {
                    "name": seed_name,
                    "context_text": seed_context_text,
                    "industry_codes": [],
                }
            )
            if inferred_keywords:
                seed_signature = _merge_industry_signatures(
                    seed_signature,
                    _normalize_industry_signature([], inferred_keywords),
                )
        queries = _registry_queries_for_entity(seed_name, industry_signature=seed_signature)
        if seed_country == "DE":
            queries = queries[:2]

        candidates_for_seed: list[dict[str, Any]] = []
        for query in queries:
            if time.monotonic() > deadline:
                timed_out = True
                errors.append("registry_neighbor_timeout")
                break
            if len(accepted) >= effective_max_neighbors or query_count >= effective_max_queries:
                break
            if seed_country == "DE" and de_query_count >= de_query_limit:
                break
            query_count += 1
            if seed_country == "DE":
                de_query_count += 1
            queries_by_country[seed_country] = queries_by_country.get(seed_country, 0) + 1

            records, error, source_name = _run_registry_search(seed_country, query)
            if error:
                errors.append(error)
                if seed_country == "DE":
                    de_error_count += 1
                    if de_error_count >= REGISTRY_DE_ERROR_BREAKER:
                        de_disabled = True
                        break
            raw_hits += len(records)
            raw_hits_by_country[seed_country] = raw_hits_by_country.get(seed_country, 0) + len(records)

            query_kept = 0
            query_rejects: dict[str, int] = {}
            for record in records:
                if len(accepted) >= effective_max_neighbors:
                    break
                origin_screening_counts["records_screened"] = origin_screening_counts.get("records_screened", 0) + 1
                candidate_name = str(record.get("name") or "").strip()
                registry_id = str(record.get("registry_id") or "").strip()
                website = str(record.get("website") or "").strip()
                normalized_website = website if website.startswith(("http://", "https://")) else (f"https://{website}" if website else "")
                website_domain = normalize_domain(normalized_website) if normalized_website else None
                key = (
                    seed_country,
                    registry_id or _normalize_name_for_matching(candidate_name),
                    normalized_website.lower(),
                )
                if key in seen_neighbor_keys:
                    reject_reason_counts["duplicate_candidate"] = reject_reason_counts.get("duplicate_candidate", 0) + 1
                    query_rejects["duplicate_candidate"] = query_rejects.get("duplicate_candidate", 0) + 1
                    origin_screening_counts["records_rejected"] = origin_screening_counts.get("records_rejected", 0) + 1
                    continue

                if not candidate_name:
                    reject_reason_counts["missing_identity"] = reject_reason_counts.get("missing_identity", 0) + 1
                    query_rejects["missing_identity"] = query_rejects.get("missing_identity", 0) + 1
                    origin_screening_counts["records_rejected"] = origin_screening_counts.get("records_rejected", 0) + 1
                    continue
                if not normalized_website or not website_domain or _is_non_first_party_profile_domain(website_domain):
                    reject_reason_counts["missing_official_website"] = reject_reason_counts.get("missing_official_website", 0) + 1
                    query_rejects["missing_official_website"] = query_rejects.get("missing_official_website", 0) + 1
                    neighbors_dropped_missing_official_website_count += 1
                    origin_screening_counts["records_rejected"] = origin_screening_counts.get("records_rejected", 0) + 1
                    continue
                record_is_active_value = record.get("is_active")
                if isinstance(record_is_active_value, bool):
                    record_is_active = record_is_active_value
                else:
                    record_is_active = _looks_active_status(record.get("status"))
                if not record_is_active:
                    reject_reason_counts["inactive_status"] = reject_reason_counts.get("inactive_status", 0) + 1
                    query_rejects["inactive_status"] = query_rejects.get("inactive_status", 0) + 1
                    origin_screening_counts["records_rejected"] = origin_screening_counts.get("records_rejected", 0) + 1
                    continue

                score, score_breakdown, signals = _score_registry_neighbor(seed_entity, record)
                if score < REGISTRY_NEIGHBOR_MIN_SCORE:
                    reject_reason_counts["score_below_threshold"] = reject_reason_counts.get("score_below_threshold", 0) + 1
                    query_rejects["score_below_threshold"] = query_rejects.get("score_below_threshold", 0) + 1
                    origin_screening_counts["records_rejected"] = origin_screening_counts.get("records_rejected", 0) + 1
                    continue

                seen_neighbor_keys.add(key)
                kept_pre_dedupe += 1
                query_kept += 1
                neighbors_with_first_party_website_count += 1
                origin_screening_counts["records_accepted"] = origin_screening_counts.get("records_accepted", 0) + 1
                citation_url = str(record.get("registry_url") or "").strip()
                industry_signature = _normalize_industry_signature(
                    record.get("industry_codes") or [],
                    record.get("industry_keywords") or [],
                )
                candidates_for_seed.append(
                    {
                        "name": candidate_name,
                        "website": normalized_website,
                        "official_website_url": normalized_website,
                        "discovery_url": citation_url,
                        "first_party_domains": [website_domain] if website_domain else [],
                        "hq_country": seed_country,
                        "likely_verticals": [],
                        "employee_estimate": None,
                        "capability_signals": [],
                        "qualification": {},
                        "registry_country": seed_country,
                        "registry_id": registry_id or None,
                        "registry_source": record.get("registry_source"),
                        "industry_signature": industry_signature,
                        "registry_identity": {
                            "country": seed_country,
                            "id": registry_id or None,
                            "source": record.get("registry_source"),
                            "match_confidence": None,
                            "match_reasons": ["neighbor_expansion"],
                        },
                        "why_relevant": [
                            {
                                "text": (
                                    f"Registry neighbor surfaced from {record.get('registry_source')} query '{query}'. "
                                    f"Industry/fit score: {score:.2f}."
                                ),
                                "citation_url": citation_url,
                                "dimension": "registry_neighbor",
                            }
                        ],
                        "_origins": [
                            {
                                "origin_type": "registry_neighbor",
                                "origin_url": citation_url,
                                "source_name": record.get("registry_source"),
                                "source_run_id": None,
                                "metadata": {
                                    "query_type": "neighbor_expand",
                                    "seed_entity_name": seed_name,
                                    "country": seed_country,
                                    "query": query,
                                    "relevance_signals": signals[:8],
                                    "score": score,
                                    "score_breakdown": score_breakdown,
                                },
                            }
                        ],
                    }
                )

            query_logs.append(
                {
                    "run_id": run_id,
                    "seed_entity_name": seed_name,
                    "query_type": "neighbor_expand",
                    "country": seed_country,
                    "source_name": source_name,
                    "query": query,
                    "raw_hits": len(records),
                    "kept_hits": query_kept,
                    "reject_reasons_json": query_rejects,
                    "metadata_json": {
                        "seed_registry_country": seed_country,
                        "industry_signature": seed_signature,
                    },
                    }
                )

            if timed_out:
                break

        if candidates_for_seed:
            candidates_for_seed.sort(
                key=lambda item: float(
                    (
                        ((item.get("_origins") or [{}])[0].get("metadata") or {}).get("score")
                    )
                    or 0.0
                ),
                reverse=True,
            )
            accepted.extend(candidates_for_seed[:REGISTRY_MAX_NEIGHBORS_PER_ENTITY])

        if timed_out:
            break

    diagnostics = {
        "registry_queries_count": query_count,
        "registry_queries_by_country": queries_by_country,
        "registry_raw_hits_count": raw_hits,
        "registry_raw_hits_by_country": raw_hits_by_country,
        "registry_de_queries_count": de_query_count,
        "registry_de_disabled": de_disabled,
        "registry_neighbors_kept_pre_dedupe": kept_pre_dedupe,
        "registry_neighbors_kept_count": len(accepted),
        "registry_neighbors_with_first_party_website_count": neighbors_with_first_party_website_count,
        "registry_neighbors_dropped_missing_official_website_count": neighbors_dropped_missing_official_website_count,
        "registry_origin_screening_counts": origin_screening_counts,
        "registry_reject_reason_breakdown": reject_reason_counts,
        "registry_neighbors_timed_out": timed_out,
        "registry_errors": errors[:100],
    }
    return accepted, diagnostics, query_logs


def _candidate_priority_score(entity: dict[str, Any]) -> float:
    explicit_score = float(
        entity.get("precomputed_discovery_score")
        or entity.get("discovery_score")
        or 0.0
    )
    lane_keys, query_families, source_families = _entity_scoring_dimensions(entity)
    origin_types = set(entity.get("origin_types") or [])
    node_fit_summary = entity.get("node_fit_summary") if isinstance(entity.get("node_fit_summary"), dict) else {}
    node_fit_score = float(node_fit_summary.get("node_fit_score") or 0.0)
    explicit_directness = str(entity.get("directness") or "").strip().lower()
    if explicit_directness:
        directness = explicit_directness
    elif {"competitor_direct", "alternatives"} & set(query_families):
        directness = "direct"
    elif lane_keys and lane_keys != ["unscoped"]:
        directness = "adjacent"
    else:
        directness = "broad_market"
    geo_signals = _dedupe_strings(
        [
            value
            for value in [
                normalize_country(entity.get("country")),
                normalize_country(entity.get("registry_country")),
            ]
            if value
        ]
    )
    score = compute_discovery_score(
        source_families=source_families,
        query_families=query_families,
        lane_ids=lane_keys,
        geo_signals=geo_signals,
        origin_types=list(origin_types),
        identity_confidence=str(entity.get("identity_confidence") or "low").lower(),
        directness=directness,
    )
    entity_type = str(entity.get("entity_type") or "company").strip().lower()
    if entity_type == "solution":
        score -= 18.0
    if "reference_seed" in origin_types:
        score += 14.0
    if "benchmark_seed" in origin_types:
        score += 12.0
    if "registry_neighbor" in origin_types:
        score += 6.0
    if "registry_fr_seed_lookup" in origin_types:
        score += 12.0
    canonical_website = str(entity.get("canonical_website") or "").strip()
    canonical_domain = normalize_domain(canonical_website)
    if canonical_domain and not _is_non_first_party_profile_domain(canonical_domain):
        score += 6.0
    elif not canonical_domain:
        score -= 4.0
    registry_identity = entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {}
    if registry_identity.get("id"):
        score += 4.0
    if float(registry_identity.get("match_confidence") or 0.0) >= 0.8:
        score += 3.0
    if "competitor_direct" in query_families:
        score += 6.0
    if "alternatives" in query_families:
        score += 5.0
    if "local_market" in query_families:
        score += 4.0
    if "comparative_source" in query_families:
        score += 3.0
    if "peer_expansion" in query_families:
        score += 2.0
    if "traffic_affinity" in query_families:
        score += 1.0
    if directness == "direct":
        score += 4.0
    elif directness == "adjacent":
        score += 2.0
    elif directness == "out_of_scope":
        score -= 40.0
    score += min(20.0, node_fit_score)
    if explicit_score > 0:
        score = max(score, explicit_score)
    return score


def _entity_scoring_dimensions(entity: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    lane_keys: list[str] = []
    query_families: list[str] = []
    source_families: list[str] = []
    for origin in entity.get("origins") or []:
        if not isinstance(origin, dict):
            continue
        origin_type = str(origin.get("origin_type") or "").strip().lower()
        metadata = origin.get("metadata") if isinstance(origin.get("metadata"), dict) else {}
        lane_keys.extend(
            _dedupe_strings(
                metadata.get("fit_to_adjacency_box_ids")
                or metadata.get("fit_to_adjacency_box_labels")
                or metadata.get("source_capability_matches")
                or ([metadata.get("brick_name")] if str(metadata.get("brick_name") or "").strip() else [])
                or ([str(metadata.get("scope_bucket") or "").strip().lower()] if str(metadata.get("scope_bucket") or "").strip() else [])
            )
        )
        provider = str(metadata.get("provider") or "").strip().lower()
        query_type = str(metadata.get("query_type") or "").strip().lower()
        query_family = normalize_discovery_query_family(
            metadata.get("query_family") or metadata.get("query_intent") or query_type,
            query_text=metadata.get("query_text"),
            provider=provider,
        )
        brick_name = str(metadata.get("brick_name") or "").strip().lower()
        if query_family:
            query_families.append(query_family)
        elif brick_name:
            query_families.append(normalize_discovery_query_family(brick_name, query_text=brick_name))
        elif origin_type:
            query_families.append(normalize_discovery_query_family(origin_type, query_text=origin_type))
        source_families.append(
            normalize_discovery_source_family(
                origin_type,
                metadata,
                origin.get("origin_url"),
            )
        )
    return (
        _dedupe_strings(lane_keys or ["unscoped"]),
        _dedupe_strings(query_families or ["unknown"]),
        _dedupe_strings(source_families or ["unknown"]),
    )


def _trim_entities_for_universe(
    entities: list[dict[str, Any]],
    cap: int = CANDIDATE_ENTITY_CAP_DEFAULT,
) -> tuple[list[dict[str, Any]], int]:
    if len(entities) <= cap:
        return entities, 0
    ranked = sorted(entities, key=lambda entity: _candidate_priority_score(entity), reverse=True)
    trimmed_out = max(0, len(ranked) - cap)
    return ranked[:cap], trimmed_out


def _select_scoring_entities(
    entities: list[dict[str, Any]],
    cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ranked = sorted(entities, key=lambda entity: _candidate_priority_score(entity), reverse=True)
    if len(ranked) <= cap:
        preferred_count = 0
        directory_backfill_count = 0
        for entity in ranked:
            origin_types = set(entity.get("origin_types") or [])
            canonical_domain = normalize_domain(str(entity.get("canonical_website") or "").strip())
            has_first_party = bool(canonical_domain and not _is_non_first_party_profile_domain(canonical_domain))
            if "external_search_seed" in origin_types or "llm_seed" in origin_types or has_first_party:
                preferred_count += 1
            else:
                directory_backfill_count += 1
        return ranked, {
            "reserved_external_or_first_party": preferred_count,
            "reserved_directory_backfill": directory_backfill_count,
        }

    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for entity in ranked:
        origin_types = set(entity.get("origin_types") or [])
        canonical_domain = normalize_domain(str(entity.get("canonical_website") or "").strip())
        has_first_party = bool(canonical_domain and not _is_non_first_party_profile_domain(canonical_domain))
        if "external_search_seed" in origin_types or "llm_seed" in origin_types or has_first_party:
            preferred.append(entity)
        else:
            fallback.append(entity)

    lane_cap = max(1, int(getattr(settings, "discovery_scoring_lane_cap", 18)))
    query_family_cap = max(1, int(getattr(settings, "discovery_scoring_query_family_cap", 10)))
    source_family_cap = max(1, int(getattr(settings, "discovery_scoring_source_family_cap", 48)))
    lane_counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def _entity_id(entity: dict[str, Any]) -> str:
        return str(entity.get("temp_entity_id") or entity.get("canonical_name") or id(entity))

    def _can_add(entity: dict[str, Any], *, relax: bool) -> bool:
        lane_keys, query_families, source_families = _entity_scoring_dimensions(entity)
        if not relax and all(lane_counts.get(key, 0) >= lane_cap for key in lane_keys):
            return False
        if all(query_counts.get(key, 0) >= query_family_cap for key in query_families):
            return False
        if all(source_counts.get(key, 0) >= source_family_cap for key in source_families):
            return False
        return True

    def _mark(entity: dict[str, Any]) -> None:
        lane_keys, query_families, source_families = _entity_scoring_dimensions(entity)
        for key in lane_keys:
            lane_counts[key] = lane_counts.get(key, 0) + 1
        for key in query_families:
            query_counts[key] = query_counts.get(key, 0) + 1
        for key in source_families:
            source_counts[key] = source_counts.get(key, 0) + 1

    for relax in (False, True):
        for pool_name, pool in (("preferred", preferred), ("fallback", fallback)):
            for entity in pool:
                entity_id = _entity_id(entity)
                if entity_id in selected_ids:
                    continue
                if not _can_add(entity, relax=relax):
                    continue
                selected.append(entity)
                selected_ids.add(entity_id)
                _mark(entity)
                if len(selected) >= cap:
                    break
            if len(selected) >= cap:
                break
        if len(selected) >= cap:
            break

    if len(selected) < cap:
        for entity in ranked:
            entity_id = _entity_id(entity)
            if entity_id in selected_ids:
                continue
            selected.append(entity)
            selected_ids.add(entity_id)
            if len(selected) >= cap:
                break

    preferred_selected_count = 0
    for entity in selected:
        origin_types = set(entity.get("origin_types") or [])
        canonical_domain = normalize_domain(str(entity.get("canonical_website") or "").strip())
        has_first_party = bool(canonical_domain and not _is_non_first_party_profile_domain(canonical_domain))
        if "external_search_seed" in origin_types or "llm_seed" in origin_types or has_first_party:
            preferred_selected_count += 1
    fallback_selected_count = max(0, len(selected) - preferred_selected_count)

    return selected, {
        "reserved_external_or_first_party": preferred_selected_count,
        "reserved_directory_backfill": fallback_selected_count,
    }


def _dedupe_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _extract_capability_signals(candidate: dict[str, Any]) -> list[str]:
    capabilities: list[str] = []
    raw_caps = candidate.get("capability_signals") or candidate.get("capabilities") or []
    if isinstance(raw_caps, list):
        for entry in raw_caps:
            if isinstance(entry, str) and entry.strip():
                capabilities.append(entry.strip())

    if not capabilities:
        for item in candidate.get("why_relevant", []) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if text:
                capabilities.append(text)

    return _dedupe_strings(capabilities[:10])


def _sanitize_employee_estimate(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    parsed: Optional[int] = None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        digits = re.sub(r"[^\d]", "", value.strip())
        if digits:
            parsed = int(digits)
    if parsed is None:
        return None
    if parsed <= 0 or parsed > 500000:
        return None
    return parsed


def _extract_employee_estimate_from_text(text: str) -> int | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    ranged = re.search(
        r"(?:between\s+)?(\d{1,5})\s*(?:-|to|and|–)\s*(\d{1,5})\s*"
        r"(?:employees?|staff|people|headcount|team|effectif)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if ranged:
        low = _sanitize_employee_estimate(ranged.group(1))
        high = _sanitize_employee_estimate(ranged.group(2))
        if low is not None and high is not None and high >= low:
            return int(round((low + high) / 2.0))

    suffix = re.search(
        r"(\d{1,5})\s*(?:employees?|staff|people|headcount|team|effectif)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if suffix:
        parsed = _sanitize_employee_estimate(suffix.group(1))
        if parsed is not None:
            return parsed

    prefix = re.search(
        r"(?:employees?|staff|people|headcount|team|effectif)[^\d]{0,16}(\d{1,5})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if prefix:
        parsed = _sanitize_employee_estimate(prefix.group(1))
        if parsed is not None:
            return parsed
    return None


def _extract_employee_estimate_from_blob(payload: Any, max_nodes: int = 250) -> int | None:
    stack: list[Any] = [payload]
    visited = 0
    while stack and visited < max_nodes:
        visited += 1
        current = stack.pop()
        direct = _sanitize_employee_estimate(current)
        if direct is not None:
            return direct
        if isinstance(current, str):
            parsed = _extract_employee_estimate_from_text(current)
            if parsed is not None:
                return parsed
            continue
        if isinstance(current, dict):
            for key in (
                "employee_estimate",
                "team_size_estimate",
                "employees",
                "headcount",
                "staff_count",
            ):
                if key in current:
                    parsed = _sanitize_employee_estimate(current.get(key))
                    if parsed is not None:
                        return parsed
                    if isinstance(current.get(key), str):
                        parsed_text = _extract_employee_estimate_from_text(str(current.get(key)))
                        if parsed_text is not None:
                            return parsed_text
            stack.extend(current.values())
            continue
        if isinstance(current, (list, tuple, set)):
            stack.extend(list(current))
    return None


def _resolve_buyer_employee_estimate(workspace: Workspace, profile: CompanyProfile) -> int | None:
    policy = workspace.decision_policy_json if isinstance(workspace.decision_policy_json, dict) else {}
    policy_candidates = [
        policy.get("buyer_employee_estimate"),
        policy.get("buyer_headcount"),
        policy.get("buyer_company_employee_estimate"),
    ]
    buyer_size = policy.get("buyer_size")
    if isinstance(buyer_size, dict):
        policy_candidates.extend(
            [
                buyer_size.get("employee_estimate"),
                buyer_size.get("headcount"),
            ]
        )

    for candidate in policy_candidates:
        parsed = _sanitize_employee_estimate(candidate)
        if parsed is not None:
            return parsed
        if isinstance(candidate, str):
            parsed_text = _extract_employee_estimate_from_text(candidate)
            if parsed_text is not None:
                return parsed_text

    text_candidates = [profile.context_pack_markdown]
    comparator_seed_summaries = (
        profile.comparator_seed_summaries if isinstance(profile.comparator_seed_summaries, dict) else {}
    )
    text_candidates.extend([str(value) for value in comparator_seed_summaries.values()])
    for text in text_candidates:
        if not isinstance(text, str):
            continue
        parsed = _extract_employee_estimate_from_text(text)
        if parsed is not None:
            return parsed

    context_pack_json = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    parsed_blob = _extract_employee_estimate_from_blob(context_pack_json)
    if parsed_blob is not None:
        return parsed_blob

    return None


def _extract_employee_estimate(candidate: dict[str, Any]) -> int | None:
    direct = candidate.get("employee_estimate") or candidate.get("team_size_estimate")
    direct_value = _sanitize_employee_estimate(direct)
    if direct_value is not None:
        return direct_value

    possible_texts: list[str] = []
    if isinstance(direct, str):
        possible_texts.append(direct)

    for item in candidate.get("why_relevant", []) or []:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                possible_texts.append(text)

    for text in possible_texts:
        parsed = _extract_employee_estimate_from_text(text)
        if parsed is not None:
            return parsed
    return None


def _map_capabilities_to_modules(
    capabilities: list[str],
    taxonomy_bricks: list[dict[str, Any]],
    trusted_urls: list[str],
) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    if not capabilities:
        return modules

    for capability in capabilities:
        cap_lower = capability.lower()
        cap_tokens = set(re.findall(r"[a-z0-9]+", cap_lower))
        best_brick = None
        best_score = 0
        for brick in taxonomy_bricks or []:
            brick_name = str(brick.get("name") or "").strip()
            if not brick_name:
                continue
            brick_lower = brick_name.lower()
            brick_tokens = set(re.findall(r"[a-z0-9]+", brick_lower))
            token_overlap = len(cap_tokens & brick_tokens)
            score = token_overlap
            if brick_lower in cap_lower or cap_lower in brick_lower:
                score += 2
            if score > best_score:
                best_score = score
                best_brick = brick

        modules.append(
            {
                "name": capability[:200],
                "brick_id": best_brick.get("id") if best_brick and best_score > 0 else None,
                "brick_name": best_brick.get("name") if best_brick and best_score > 0 else None,
                "description": capability[:400],
                "evidence_urls": trusted_urls[:2],
            }
        )

    return modules


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _tag_value(tags: list[str], prefix: str) -> str | None:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            return tag.split(":", 1)[1]
    return None


def _company_passes_enterprise_screen(tags_custom: list[str] | None) -> bool:
    tags = [t for t in (tags_custom or []) if isinstance(t, str)]
    gtm = (_tag_value(tags, "gtm:") or "").lower()
    if gtm == "b2c":
        return False

    public_price = _parse_float(_tag_value(tags, "public_price_floor_usd:"))
    if public_price is not None and public_price < MIN_PUBLIC_PRICE_USD:
        return False

    heaviness = _parse_float(_tag_value(tags, "software_heaviness:"))
    if heaviness is not None and heaviness < MIN_SOFTWARE_HEAVINESS:
        return False

    return True


def _normalize_reasons(raw_reasons: list[Any] | None) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    for item in raw_reasons or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        citation_url = str(item.get("citation_url") or "").strip()
        if not text or not citation_url:
            continue
        normalized: dict[str, str] = {
            "text": text[:800],
            "citation_url": citation_url,
            "dimension": str(item.get("dimension") or "evidence")[:64],
        }
        source_kind = str(item.get("source_kind") or "").strip().lower()
        if source_kind:
            normalized["source_kind"] = source_kind[:32]
        reasons.append(normalized)
    return reasons


PRICE_CONTEXT_TOKENS = {
    "pricing",
    "price",
    "plan",
    "plans",
    "tier",
    "tiers",
    "subscription",
    "monthly",
    "annually",
    "annual",
    "per month",
    "per year",
    "per seat",
    "per user",
    "/month",
    "/mo",
    "/year",
    "/yr",
    "starts at",
    "starting at",
    "from",
    "license",
    "licenses",
}

NON_PRICING_CURRENCY_TOKENS = {
    "fundraising",
    "funding",
    "funding round",
    "seed round",
    "series a",
    "series b",
    "series c",
    "series d",
    "raised",
    "raise",
    "valuation",
    "capital raise",
    "investor",
    "investors",
}

CURRENCY_MARKERS = ("$", "usd", "€", "eur", "£", "gbp")

CURRENCY_AMOUNT_PREFIX_PATTERN = re.compile(
    r"(?:\$|usd|€|eur|£|gbp)\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>k|m|b|bn|mn|thousand|million|billion)?\b",
    flags=re.IGNORECASE,
)
CURRENCY_AMOUNT_SUFFIX_PATTERN = re.compile(
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>k|m|b|bn|mn|thousand|million|billion)?\s*(?:\$|usd|€|eur|£|gbp)\b",
    flags=re.IGNORECASE,
)

AMOUNT_SUFFIX_MULTIPLIERS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "mn": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}


def _contains_any_token(text: str, tokens: set[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _iter_pricing_text_chunks(text: str) -> list[str]:
    chunks = [text]
    for part in re.split(r"[\n\r;.!?]+", text):
        part = part.strip()
        if part:
            chunks.append(part)
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = chunk.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _parse_currency_amount(text: str) -> float | None:
    for pattern in (CURRENCY_AMOUNT_PREFIX_PATTERN, CURRENCY_AMOUNT_SUFFIX_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        amount_value = _parse_float(str(match.group("amount") or "").replace(",", ""))
        if amount_value is None:
            continue
        suffix = str(match.group("suffix") or "").strip().lower()
        multiplier = AMOUNT_SUFFIX_MULTIPLIERS.get(suffix, 1.0)
        return float(amount_value) * multiplier
    return None


def _has_pricing_amount_evidence(reasons: list[dict[str, str]]) -> bool:
    for reason in reasons:
        text = str(reason.get("text") or "").strip()
        if not text:
            continue
        reason_dimension = str(reason.get("dimension") or "").strip().lower()
        for chunk in _iter_pricing_text_chunks(text):
            lowered = chunk.lower()
            if not any(marker in lowered for marker in CURRENCY_MARKERS):
                continue
            has_pricing_context = reason_dimension == "pricing_gtm" or _contains_any_token(lowered, PRICE_CONTEXT_TOKENS)
            if not has_pricing_context:
                continue
            if reason_dimension != "pricing_gtm" and _contains_any_token(lowered, NON_PRICING_CURRENCY_TOKENS):
                continue
            if _parse_currency_amount(lowered) is not None:
                return True
    return False


def _extract_price_floor_usd(candidate: dict[str, Any], reasons: list[dict[str, str]]) -> float | None:
    qualification = candidate.get("qualification") if isinstance(candidate.get("qualification"), dict) else {}
    direct = _parse_float(qualification.get("public_price_floor_usd_month"))
    if direct is not None:
        if direct < MIN_PUBLIC_PRICE_USD and not _has_pricing_amount_evidence(reasons):
            return None
        return direct

    for reason in reasons:
        text = str(reason.get("text") or "").strip()
        if not text:
            continue
        reason_dimension = str(reason.get("dimension") or "").strip().lower()
        for chunk in _iter_pricing_text_chunks(text):
            lowered = chunk.lower()
            if not any(marker in lowered for marker in CURRENCY_MARKERS):
                continue
            has_pricing_context = reason_dimension == "pricing_gtm" or _contains_any_token(lowered, PRICE_CONTEXT_TOKENS)
            if not has_pricing_context:
                continue
            if reason_dimension != "pricing_gtm" and _contains_any_token(lowered, NON_PRICING_CURRENCY_TOKENS):
                continue
            parsed = _parse_currency_amount(lowered)
            if parsed is not None:
                # rough parity across USD/EUR/GBP is sufficient for low-ticket gating.
                return parsed
    return None


def _has_token(text: str, tokens: set[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _evaluate_enterprise_b2b_fit(candidate: dict[str, Any], reasons: list[dict[str, str]]) -> tuple[bool, list[str], dict[str, Any]]:
    qualification = candidate.get("qualification") if isinstance(candidate.get("qualification"), dict) else {}
    go_to_market = str(qualification.get("go_to_market") or "unknown").strip().lower()
    pricing_model = str(qualification.get("pricing_model") or "unknown").strip().lower()
    target_customer = str(qualification.get("target_customer") or "unknown").strip().lower()
    software_heaviness = int(_parse_float(qualification.get("software_heaviness")) or 0)
    price_floor = _extract_price_floor_usd(candidate, reasons)

    combined_text = " ".join(
        [
            str(candidate.get("name") or ""),
            " ".join(str(v) for v in (candidate.get("capability_signals") or []) if isinstance(v, str)),
            " ".join(r.get("text", "") for r in reasons),
        ]
    )
    has_institutional_signal = _has_token(combined_text, INSTITUTIONAL_TOKENS)
    has_b2c_signal = _has_token(combined_text, B2C_TOKENS)

    reject_reasons: list[str] = []

    if go_to_market == "b2c":
        reject_reasons.append("go_to_market_b2c")

    if target_customer == "retail_investors" and not has_institutional_signal:
        reject_reasons.append("retail_only_icp")

    if has_b2c_signal and not has_institutional_signal:
        reject_reasons.append("consumer_language_without_institutional_icp")

    if software_heaviness and software_heaviness < MIN_SOFTWARE_HEAVINESS:
        reject_reasons.append("low_software_heaviness")

    if price_floor is not None and price_floor < MIN_PUBLIC_PRICE_USD:
        reject_reasons.append("low_ticket_public_pricing")

    if pricing_model == "public_tiered" and price_floor is not None and price_floor < MIN_PUBLIC_PRICE_USD:
        reject_reasons.append("public_self_serve_pricing")

    meta = {
        "go_to_market": go_to_market,
        "pricing_model": pricing_model,
        "target_customer": target_customer,
        "software_heaviness": software_heaviness if software_heaviness else None,
        "public_price_floor_usd_month": price_floor,
        "has_institutional_signal": has_institutional_signal,
        "has_b2c_signal": has_b2c_signal,
        "hard_fail": any(reason in HARD_FAIL_REASONS for reason in reject_reasons),
    }
    return len(reject_reasons) == 0, reject_reasons, meta


def _extract_period(text: str) -> Optional[str]:
    match = re.search(r"(20\d{2})", text)
    if not match:
        return None
    return f"FY{match.group(1)}"


def _classify_first_party_dimension(text: str, page_type: Optional[str] = None) -> Optional[str]:
    lowered = str(text or "").lower()
    if not lowered.strip():
        return None

    icp_tokens = INSTITUTIONAL_TOKENS.union(
        {
            "private wealth",
            "institutional investor",
            "asset owner",
            "family office",
            "wealth advisory",
        }
    )
    services_tokens = {
        "implementation",
        "integration",
        "onboarding",
        "consulting",
        "migration",
        "deployment",
        "professional services",
        "managed service",
        "delivery team",
    }
    customer_tokens = {
        "customer",
        "client",
        "case study",
        "trusted by",
        "reference",
        "asset manager",
        "private bank",
    }
    pricing_tokens = {
        "pricing",
        "plans",
        "request demo",
        "book a demo",
        "contact sales",
        "enterprise",
        "per user",
        "per month",
    }
    moat_tokens = {
        "regulatory",
        "compliance",
        "iso 27001",
        "soc 2",
        "licensed",
        "patent",
        "proprietary",
        "certified",
    }
    product_tokens = {
        "portfolio",
        "order management",
        "pms",
        "oms",
        "reporting",
        "analytics",
        "reconciliation",
        "trading",
        "platform",
        "module",
        "workflow",
        "custody",
    }

    if page_type in {"customers"} or any(token in lowered for token in customer_tokens):
        return "customer"
    if page_type in {"services"} or any(token in lowered for token in services_tokens):
        return "services"
    if page_type in {"pricing"} or any(token in lowered for token in pricing_tokens):
        return "pricing_gtm"
    if any(token in lowered for token in icp_tokens):
        return "icp"
    if any(token in lowered for token in moat_tokens):
        return "moat"
    if page_type in {"product", "features", "solutions", "integrations"} or any(token in lowered for token in product_tokens):
        return "product"
    return None


def _dedupe_reason_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        text = str(item.get("text") or "").strip()
        citation_url = str(item.get("citation_url") or "").strip()
        dimension = str(item.get("dimension") or "evidence").strip()
        if not text or not citation_url:
            continue
        key = (text.lower(), citation_url.lower(), dimension.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized: dict[str, str] = {
            "text": text[:700],
            "citation_url": citation_url[:1000],
            "dimension": dimension[:64],
        }
        source_kind = str(item.get("source_kind") or "").strip().lower()
        if source_kind:
            normalized["source_kind"] = source_kind[:32]
        deduped.append(normalized)
    return deduped


def _prioritize_reason_items(
    items: list[dict[str, str]],
    max_items: int = FIRST_PARTY_CRAWL_MAX_REASONS,
) -> list[dict[str, str]]:
    if not items:
        return []
    priority = {
        "customer": 0,
        "customers": 0,
        "case_study": 0,
        "icp": 1,
        "services": 2,
        "pricing_gtm": 3,
        "moat": 4,
        "product": 5,
        "company_profile": 6,
    }
    ranked: list[tuple[int, int, dict[str, str]]] = []
    for index, item in enumerate(items):
        dimension = str(item.get("dimension") or "evidence").strip().lower()
        ranked.append((priority.get(dimension, 7), index, item))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in ranked[: max(1, int(max_items))]]


def _extract_reasons_from_rendered_text(
    content: str,
    source_url: str,
    *,
    max_items: int = 8,
) -> tuple[list[dict[str, str]], list[str]]:
    lines = [line.strip() for line in re.split(r"[\n\r]+", str(content or "")) if str(line or "").strip()]
    reasons: list[dict[str, str]] = []
    capabilities: list[str] = []
    for line in lines:
        normalized = " ".join(line.split())
        if len(normalized) < 50:
            continue
        dimension = _classify_first_party_dimension(normalized)
        if not dimension:
            continue
        reason = {
            "text": normalized[:700],
            "citation_url": source_url[:1000],
            "dimension": dimension,
            "source_kind": "rendered_browser",
        }
        reasons.append(reason)
        if dimension in {"product", "services"} and len(normalized) <= 220:
            capabilities.append(normalized[:180])
        if len(reasons) >= max(1, int(max_items)):
            break
    return reasons, _dedupe_strings(capabilities)[:12]


def _extract_first_party_signals_via_rendered_browser(
    website: str,
    candidate_name: str,
    *,
    hint_urls: Optional[list[str]] = None,
    max_pages: int = 2,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any], Optional[str]]:
    normalized = _normalize_hint_url(website)
    if not normalized:
        return [], [], {"method": "chrome_devtools_mcp", "pages_crawled": 0}, "invalid_website"
    domain = normalize_domain(normalized)
    if not domain or _is_non_first_party_profile_domain(domain):
        return [], [], {"method": "chrome_devtools_mcp", "pages_crawled": 0}, "invalid_domain"
    if not bool(getattr(settings, "chrome_mcp_enabled", False)):
        return [], [], {"method": "chrome_devtools_mcp", "pages_crawled": 0}, "chrome_mcp_disabled"

    targets = _dedupe_strings(
        [
            *[
                value
                for value in (hint_urls or [])
                if normalize_domain(_normalize_hint_url(value) or "") == domain
            ],
            normalized,
        ]
    )
    if not targets:
        targets = [normalized]

    max_pages_cap = max(
        1,
        min(
            int(getattr(settings, "chrome_mcp_max_pages_per_domain", 2)),
            int(max_pages),
        ),
    )
    min_chars = max(120, int(getattr(settings, "chrome_mcp_min_text_chars", 700)))

    reasons: list[dict[str, str]] = []
    capabilities: list[str] = []
    providers: list[str] = []
    errors: list[str] = []
    pages_crawled = 0
    hit_urls: list[str] = []
    for target in targets[:max_pages_cap]:
        rendered = render_page_via_chrome_devtools_mcp(
            target,
            timeout_seconds=int(getattr(settings, "chrome_mcp_timeout_seconds", 25)),
        )
        content = str(rendered.get("content") or "").strip()
        provider = str(rendered.get("provider") or "").strip()
        if provider:
            providers.append(provider)
        if not content:
            error = str(rendered.get("error") or "empty_rendered_content").strip()
            if error:
                errors.append(error)
            continue
        pages_crawled += 1
        final_url = _normalize_hint_url(str(rendered.get("final_url") or target)) or target
        if len(content) < min_chars:
            snippet = " ".join(content.split())[:700]
            if snippet:
                reasons.append(
                    {
                        "text": snippet,
                        "citation_url": final_url,
                        "dimension": _classify_first_party_dimension(snippet) or "company_profile",
                        "source_kind": "rendered_browser",
                    }
                )
                hit_urls.append(final_url)
            continue

        extracted_reasons, extracted_capabilities = _extract_reasons_from_rendered_text(
            content,
            final_url,
            max_items=max(3, FIRST_PARTY_CRAWL_MAX_REASONS // 2),
        )
        if extracted_reasons:
            reasons.extend(extracted_reasons)
            hit_urls.append(final_url)
        capabilities.extend(extracted_capabilities)

    deduped_reasons = _prioritize_reason_items(
        _dedupe_reason_items(reasons),
        FIRST_PARTY_CRAWL_MAX_REASONS,
    )
    deduped_capabilities = _dedupe_strings(capabilities)[:12]
    if not deduped_reasons and pages_crawled > 0:
        deduped_reasons = [
            {
                "text": f"Rendered first-party website content for {candidate_name}: {normalized}"[:700],
                "citation_url": normalized,
                "dimension": "company_profile",
                "source_kind": "rendered_browser",
            }
        ]

    meta = {
        "method": "chrome_devtools_mcp",
        "provider": providers[0] if providers else None,
        "providers": _dedupe_strings(providers),
        "pages_crawled": pages_crawled,
        "signals_extracted": len(deduped_reasons),
        "customer_evidence_count": len([row for row in deduped_reasons if row.get("dimension") == "customer"]),
        "hint_urls_used": targets[:20],
        "hint_urls_used_count": len(targets[:20]),
        "hint_pages_crawled": len(hit_urls),
        "hint_hit_urls": hit_urls[:20],
        "errors": errors[:8],
    }
    if deduped_reasons:
        return deduped_reasons, deduped_capabilities, meta, None
    return [], [], meta, ";".join(errors[:3]) if errors else "rendered_browser_empty"


def _extract_first_party_signals_from_crawl(
    website: str,
    candidate_name: str,
    max_pages: int,
    hint_urls: Optional[list[str]] = None,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any], Optional[str]]:
    """Crawl first-party pages and extract richer buy-side signals."""
    normalized = website.strip()
    if not normalized:
        return [], [], {"method": "crawler", "pages_crawled": 0}, "missing_website"
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"

    domain = normalize_domain(normalized)
    if not domain or _is_non_first_party_profile_domain(domain):
        return [], [], {"method": "crawler", "pages_crawled": 0}, "invalid_domain"

    normalized_hint_urls: list[str] = []
    hint_seen: set[str] = set()
    for raw_hint in hint_urls or []:
        normalized_hint = _normalize_hint_url(raw_hint)
        if not normalized_hint:
            continue
        hint_domain = normalize_domain(normalized_hint)
        if hint_domain != domain:
            continue
        key = normalized_hint.lower()
        if key in hint_seen:
            continue
        hint_seen.add(key)
        normalized_hint_urls.append(normalized_hint)

    def _browser_fallback() -> tuple[list[dict[str, str]], list[str], dict[str, Any], Optional[str]]:
        return _extract_first_party_signals_via_rendered_browser(
            website=normalized,
            candidate_name=candidate_name,
            hint_urls=normalized_hint_urls,
            max_pages=max(
                1,
                min(
                    int(max_pages),
                    int(getattr(settings, "chrome_mcp_max_pages_per_domain", 2)),
                ),
            ),
        )

    try:
        from app.services.crawler import UnifiedCrawler
    except Exception as exc:
        return [], [], {"method": "crawler", "pages_crawled": 0}, f"crawler_import_failed:{exc}"

    loop = asyncio.new_event_loop()
    context_pack = None
    try:
        asyncio.set_event_loop(loop)
        crawler = UnifiedCrawler(
            max_pages=max_pages,
            timeout=max(FIRST_PARTY_FETCH_TIMEOUT_SECONDS, 10),
        )
        context_pack = loop.run_until_complete(
            crawler.crawl_for_context(
                normalized,
                start_urls=normalized_hint_urls,
            )
        )
    except Exception as exc:
        browser_reasons, browser_capabilities, browser_meta, browser_error = _browser_fallback()
        if browser_reasons:
            browser_meta = {
                **(browser_meta if isinstance(browser_meta, dict) else {}),
                "method": "chrome_devtools_mcp_fallback",
                "hint_urls_used": normalized_hint_urls[:20],
                "hint_urls_used_count": len(normalized_hint_urls),
            }
            return browser_reasons, browser_capabilities, browser_meta, None
        fast = fetch_page_fast(normalized)
        fast_content = str(fast.get("content") or "").strip()
        if fast_content:
            snippet = re.sub(r"\s+", " ", fast_content)[:700]
            dimension = _classify_first_party_dimension(snippet) or "company_profile"
            return (
                [
                    {
                        "text": snippet,
                        "citation_url": normalized,
                        "dimension": dimension,
                    }
                ],
                [],
                {
                    "method": "external_fast_fetch",
                    "provider": fast.get("provider"),
                    "pages_crawled": 1,
                    "signals_extracted": 1,
                    "customer_evidence_count": 0,
                    "page_types": {"external_fast_fetch": 1},
                    "max_pages_requested": max_pages,
                    "hint_urls_used": normalized_hint_urls[:20],
                    "hint_urls_used_count": len(normalized_hint_urls),
                    "hint_pages_crawled": 0,
                    "hint_hit_urls": [],
                    "browser_fallback_attempted": bool(getattr(settings, "chrome_mcp_enabled", False)),
                    "browser_fallback_error": browser_error,
                },
                None,
            )
        return [], [], {"method": "crawler", "pages_crawled": 0}, f"first_party_crawl_failed:{exc};browser:{browser_error}"
    finally:
        try:
            loop.close()
        except Exception:
            pass
        asyncio.set_event_loop(None)

    if context_pack is None:
        browser_reasons, browser_capabilities, browser_meta, browser_error = _browser_fallback()
        if browser_reasons:
            browser_meta = {
                **(browser_meta if isinstance(browser_meta, dict) else {}),
                "method": "chrome_devtools_mcp_fallback",
                "hint_urls_used": normalized_hint_urls[:20],
                "hint_urls_used_count": len(normalized_hint_urls),
            }
            return browser_reasons, browser_capabilities, browser_meta, None
        fast = fetch_page_fast(normalized)
        fast_content = str(fast.get("content") or "").strip()
        if fast_content:
            snippet = re.sub(r"\s+", " ", fast_content)[:700]
            return (
                [
                    {
                        "text": snippet,
                        "citation_url": normalized,
                        "dimension": _classify_first_party_dimension(snippet) or "company_profile",
                    }
                ],
                [],
                {
                    "method": "external_fast_fetch",
                    "provider": fast.get("provider"),
                    "pages_crawled": 1,
                    "signals_extracted": 1,
                    "customer_evidence_count": 0,
                    "page_types": {"external_fast_fetch": 1},
                    "max_pages_requested": max_pages,
                    "hint_urls_used": normalized_hint_urls[:20],
                    "hint_urls_used_count": len(normalized_hint_urls),
                    "hint_pages_crawled": 0,
                    "hint_hit_urls": [],
                    "browser_fallback_attempted": bool(getattr(settings, "chrome_mcp_enabled", False)),
                    "browser_fallback_error": browser_error,
                },
                None,
            )
        return [], [], {"method": "crawler", "pages_crawled": 0}, f"first_party_crawl_empty;browser:{browser_error}"

    reasons: list[dict[str, str]] = []
    capability_signals: list[str] = []
    page_types: dict[str, int] = {}

    for signal in context_pack.signals or []:
        signal_type = str(getattr(signal, "type", "") or "").lower()
        signal_value = str(getattr(signal, "value", "") or "").strip()
        evidence = getattr(signal, "evidence", None)
        snippet = str(getattr(evidence, "snippet", "") or "").strip()
        source_url = str(getattr(evidence, "source_url", "") or "").strip() or normalized
        if not signal_value and not snippet:
            continue
        text = snippet if len(snippet) >= 20 else (signal_value or snippet)
        if len(text) < 20:
            continue
        if signal_type == "customer":
            dimension = "customer"
        elif signal_type == "service":
            dimension = "services"
        elif signal_type in {"integration", "capability"}:
            dimension = "product"
        else:
            dimension = _classify_first_party_dimension(text)
        if not dimension:
            continue
        reasons.append(
            {
                "text": text[:700],
                "citation_url": source_url,
                "dimension": dimension,
            }
        )
        if dimension in {"product", "services"} and signal_value:
            capability_signals.append(signal_value[:180])

    for page in (context_pack.pages or [])[:max_pages]:
        page_url = str(getattr(page, "url", "") or "").strip() or normalized
        page_type = str(getattr(page, "page_type", "") or "other").strip().lower()
        page_types[page_type] = page_types.get(page_type, 0) + 1

        title = str(getattr(page, "title", "") or "").strip()
        if title:
            title_dimension = _classify_first_party_dimension(title, page_type=page_type)
            if title_dimension:
                reasons.append(
                    {
                        "text": title[:700],
                        "citation_url": page_url,
                        "dimension": title_dimension,
                    }
                )

        blocks = getattr(page, "blocks", []) or []
        for block in blocks[:10]:
            content = str(getattr(block, "content", "") or "").strip()
            if len(content) < 45:
                continue
            dimension = _classify_first_party_dimension(content, page_type=page_type)
            if not dimension:
                continue
            reasons.append(
                {
                    "text": content[:700],
                    "citation_url": page_url,
                    "dimension": dimension,
                }
            )
            if dimension in {"product", "services"} and len(content) <= 220:
                capability_signals.append(content[:180])
            if len(reasons) >= FIRST_PARTY_CRAWL_MAX_REASONS:
                break
        if len(reasons) >= FIRST_PARTY_CRAWL_MAX_REASONS:
            break

    for customer_evidence in (context_pack.customer_evidence or [])[:8]:
        customer_name = str(getattr(customer_evidence, "name", "") or "").strip()
        if not customer_name:
            continue
        context = str(getattr(customer_evidence, "context", "") or "").strip()
        source_url = str(getattr(customer_evidence, "source_url", "") or "").strip() or normalized
        text = f"Named customer evidence: {customer_name}"
        if context:
            text = f"{text} ({context})"
        reasons.append(
            {
                "text": text[:700],
                "citation_url": source_url,
                "dimension": "customer",
            }
        )

    deduped_reasons = _prioritize_reason_items(
        _dedupe_reason_items(reasons),
        FIRST_PARTY_CRAWL_MAX_REASONS,
    )
    deduped_capabilities = _dedupe_strings(capability_signals)[:12]

    hint_pages_crawled = 0
    hint_url_keys = [url.lower() for url in normalized_hint_urls]
    hint_hit_urls: list[str] = []
    for page in context_pack.pages or []:
        page_url_raw = str(getattr(page, "url", "") or "").strip()
        page_url = page_url_raw.lower()
        if not page_url_raw:
            continue
        if any(page_url == hint_url or page_url.startswith(f"{hint_url}/") for hint_url in hint_url_keys):
            hint_pages_crawled += 1
            hint_hit_urls.append(page_url_raw)

    browser_fallback_attempted = False
    browser_fallback_error: Optional[str] = None
    browser_meta_applied: dict[str, Any] = {}
    should_try_browser = bool(getattr(settings, "chrome_mcp_enabled", False)) and (
        (normalized_hint_urls and hint_pages_crawled == 0)
        or len(deduped_reasons) <= 1
        or len(context_pack.pages or []) <= 1
    )
    if should_try_browser:
        browser_fallback_attempted = True
        browser_reasons, browser_capabilities, browser_meta, browser_error = _browser_fallback()
        if browser_reasons:
            deduped_reasons = _prioritize_reason_items(
                _dedupe_reason_items(deduped_reasons + browser_reasons),
                FIRST_PARTY_CRAWL_MAX_REASONS,
            )
            deduped_capabilities = _dedupe_strings(deduped_capabilities + browser_capabilities)[:12]
            hint_pages_crawled += int(browser_meta.get("hint_pages_crawled") or 0)
            hint_hit_urls = _dedupe_strings(
                hint_hit_urls
                + [
                    str(value)
                    for value in (browser_meta.get("hint_hit_urls") or [])
                    if str(value).strip()
                ]
            )
            browser_meta_applied = browser_meta if isinstance(browser_meta, dict) else {}
        else:
            browser_fallback_error = browser_error

    if not deduped_reasons:
        fallback_text = f"First-party website crawled for {candidate_name}: {normalized}"
        deduped_reasons = [
            {
                "text": fallback_text[:700],
                "citation_url": normalized,
                "dimension": "company_profile",
            }
        ]

    meta: dict[str, Any] = {
        "method": "crawler",
        "pages_crawled": len(context_pack.pages or []),
        "signals_extracted": len(context_pack.signals or []),
        "customer_evidence_count": len(context_pack.customer_evidence or []),
        "page_types": page_types,
        "max_pages_requested": max_pages,
        "hint_urls_used": normalized_hint_urls[:20],
        "hint_urls_used_count": len(normalized_hint_urls),
        "hint_pages_crawled": hint_pages_crawled,
        "hint_hit_urls": hint_hit_urls[:20],
        "browser_fallback_attempted": browser_fallback_attempted,
    }
    if browser_meta_applied:
        meta["method"] = "crawler_plus_rendered_browser"
        meta["browser_provider"] = browser_meta_applied.get("provider")
        meta["browser_pages_crawled"] = int(browser_meta_applied.get("pages_crawled") or 0)
    if browser_fallback_error:
        meta["browser_fallback_error"] = browser_fallback_error
    return deduped_reasons, deduped_capabilities, meta, None


def _extract_first_party_signals(
    website: str,
    candidate_name: str,
) -> tuple[list[dict[str, str]], Optional[str]]:
    """Fetch lightweight first-party signals from the company website."""
    normalized = website.strip()
    if not normalized:
        return [], "missing_website"
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"

    domain = normalize_domain(normalized)
    if not domain or _is_non_first_party_profile_domain(domain):
        return [], "invalid_domain"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0; +https://example.com/bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        with httpx.Client(
            timeout=FIRST_PARTY_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(normalized)
            response.raise_for_status()
            final_url = str(response.url)
            tree = HTMLParser(response.text)
            if tree is None:
                return [], "parse_failed"

            snippets: list[str] = []
            title = tree.css_first("title")
            title_text = title.text(strip=True) if title and title.text(strip=True) else ""
            if title_text:
                snippets.append(title_text)
            site_name = ""
            for selector, attr in (
                ('meta[property="og:site_name"]', "content"),
                ('meta[name="application-name"]', "content"),
                ('meta[property="og:title"]', "content"),
            ):
                node = tree.css_first(selector)
                if node:
                    site_name = str(node.attributes.get(attr) or "").strip()
                if site_name:
                    snippets.append(site_name)
                    break
            meta_desc = tree.css_first('meta[name="description"]')
            if meta_desc:
                desc = str(meta_desc.attributes.get("content") or "").strip()
                if desc:
                    snippets.append(desc)

            for selector in ("h1", "h2", "h3", "p", "li"):
                for node in tree.css(selector)[:20]:
                    text = node.text(separator=" ", strip=True)
                    if not text:
                        continue
                    if len(text) < 30:
                        continue
                    snippets.append(text)
                    if len(snippets) >= 60:
                        break
                if len(snippets) >= 60:
                    break

            seen_snippets: set[str] = set()
            deduped_snippets: list[str] = []
            for snippet in snippets:
                normalized_snippet = " ".join(snippet.split())
                key = normalized_snippet.lower()
                if not normalized_snippet or key in seen_snippets:
                    continue
                seen_snippets.add(key)
                deduped_snippets.append(normalized_snippet)

            icp_tokens = {
                "asset manager",
                "wealth manager",
                "private bank",
                "institutional",
                "fund manager",
                "adviser",
                "advisor",
                "bank",
            }
            product_tokens = {
                "portfolio management",
                "portfolio",
                "oms",
                "pms",
                "compliance",
                "risk",
                "attribution",
                "reporting",
                "reconciliation",
                "platform",
            }
            services_tokens = {
                "implementation",
                "integration",
                "onboarding",
                "consulting",
                "migration",
                "deployment",
                "professional services",
            }
            customer_tokens = {
                "customer",
                "client",
                "case study",
                "trusted by",
                "users",
            }

            reasons: list[dict[str, str]] = []
            brand_hint = _extract_brand_name_hint_from_reasons(
                [{"text": value, "citation_url": final_url} for value in deduped_snippets[:6]],
                final_url,
            )
            if brand_hint:
                reasons.append(
                    {
                        "text": f"Brand identity: {brand_hint}",
                        "citation_url": final_url,
                        "dimension": "company_profile",
                    }
                )
            for snippet in deduped_snippets:
                lowered = snippet.lower()
                dimension = "product"
                if any(token in lowered for token in icp_tokens):
                    dimension = "icp"
                elif any(token in lowered for token in services_tokens):
                    dimension = "services"
                elif any(token in lowered for token in customer_tokens):
                    dimension = "customer"
                elif any(token in lowered for token in product_tokens):
                    dimension = "product"
                else:
                    continue

                reasons.append(
                    {
                        "text": snippet[:700],
                        "citation_url": final_url,
                        "dimension": dimension,
                    }
                )
                if len(reasons) >= FIRST_PARTY_MAX_SIGNALS:
                    break

            if not reasons:
                fallback = f"First-party website available for {candidate_name}: {final_url}"
                reasons.append(
                    {
                        "text": fallback[:700],
                        "citation_url": final_url,
                        "dimension": "company_profile",
                    }
                )

            return reasons, None
    except Exception as exc:
        return [], f"first_party_fetch_failed:{exc}"
    return [], "first_party_fetch_failed:unknown"


def _normalize_domain_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        host = normalize_domain(str(value or "").strip())
        if not host or host in seen:
            continue
        seen.add(host)
        normalized.append(host)
    return normalized


def _source_type_for_url(
    url: str,
    candidate_domain: Optional[str],
    first_party_domains: Optional[list[str]] = None,
    provenance_hint: Optional[str] = None,
) -> str:
    if str(provenance_hint or "").strip().lower() == "rendered_browser":
        return "rendered_browser"
    if str(provenance_hint or "").strip().lower() == "external_search_snippet":
        return "external_search_snippet"
    lowered = url.lower()
    host = normalize_domain(url) or ""
    if any(token in lowered for token in ("thewealthmosaic.com", "crunchbase.com", "g2.com", "capterra.com")):
        return "directory_comparator"
    if any(
        pattern in lowered
        for pattern in (
            "companieshouse.gov.uk",
            "find-and-update.company-information.service.gov.uk",
            "pappers.fr",
            "infogreffe.fr",
            "inpi.fr",
            "annuaire-entreprises.data.gouv.fr",
            "gleif.org",
            "handelsregister.de",
            "unternehmensregister.de",
        )
    ):
        return "official_registry_filing"
    normalized_first_party = _normalize_domain_list(first_party_domains or [])
    if normalized_first_party and any(host == domain or host.endswith(f".{domain}") for domain in normalized_first_party):
        return "first_party_website"
    if candidate_domain and (host == candidate_domain or host.endswith(f".{candidate_domain}")):
        return "first_party_website"
    return "trusted_third_party"


def _numeric_from_claim_text(text: str) -> tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    lowered = text.lower()
    period = _extract_period(text)

    employee_match = re.search(
        r"(?:employees|employee|staff|team|headcount|effectif)[^\d]{0,14}(\d{1,4})",
        lowered,
    )
    if employee_match:
        try:
            return "employees", float(employee_match.group(1)), "people", period
        except Exception:
            return "employees", None, "people", period

    employee_ranged = re.search(
        r"(\d{1,4})\s*(?:-|to|and|–)\s*(\d{1,4})\s*(?:employees|employee|staff|team|people|headcount|effectif)",
        lowered,
    )
    if employee_ranged:
        low = float(employee_ranged.group(1))
        high = float(employee_ranged.group(2))
        if high >= low:
            return "employees", round((low + high) / 2.0, 2), "people", period

    revenue_match = re.search(
        r"(?:revenue|turnover|arr|chiffre\s*d['’]?affaires)[^\d]{0,20}(\d[\d\s.,]{0,16})\s*(m|million|k|thousand)?\s*(eur|€|gbp|£|usd|\$)?",
        lowered,
    )
    if revenue_match:
        raw_num = str(revenue_match.group(1) or "").replace(" ", "").replace(",", ".")
        try:
            value = float(raw_num)
        except Exception:
            value = None
        mag = str(revenue_match.group(2) or "").lower()
        currency = str(revenue_match.group(3) or "").upper().replace("€", "EUR").replace("£", "GBP").replace("$", "USD")
        unit = currency or None
        if mag in {"m", "million"}:
            unit = f"{unit or ''}m".strip()
        elif mag in {"k", "thousand"}:
            unit = f"{unit or ''}k".strip()
        return "revenue", value, unit, period

    return None, None, None, period


def _evidence_signal_counts(
    reasons: list[dict[str, str]],
    capability_signals: list[str],
) -> dict[str, int]:
    reason_text = " ".join(r.get("text", "") for r in reasons).lower()
    capability_text = " ".join(capability_signals).lower()
    combined_text = f"{reason_text} {capability_text}"

    services_tokens = {
        "implementation",
        "integration",
        "migration",
        "onboarding",
        "customization",
        "consulting",
        "professional services",
        "deployment",
    }
    moat_tokens = {
        "regulatory",
        "compliance",
        "workflow",
        "audit",
        "integration",
        "data hub",
        "risk",
        "attribution",
        "oms",
        "pms",
        "custodian",
    }
    customer_tokens = {
        "customer",
        "client",
        "case study",
        "private bank",
        "asset manager",
        "wealth manager",
        "insurer",
        "logo",
    }

    services_count = sum(1 for token in services_tokens if token in combined_text)
    moat_count = sum(1 for token in moat_tokens if token in combined_text)
    customer_count = sum(1 for token in customer_tokens if token in combined_text)
    named_customer_reasons = sum(
        1
        for reason in reasons
        if str(reason.get("dimension", "")).lower() in {"customer", "customers", "case_study"}
        or any(token in str(reason.get("text", "")).lower() for token in {"case study", "customer", "client"})
    )
    return {
        "services": services_count,
        "moat": moat_count,
        "customer_tokens": customer_count,
        "named_customer_reasons": named_customer_reasons,
        "capability_count": len(capability_signals),
    }


def _scale_count(count: int, max_count: int) -> float:
    if max_count <= 0:
        return 0.0
    return min(1.0, float(count) / float(max_count))


def _score_buy_side_candidate(
    candidate: dict[str, Any],
    reasons: list[dict[str, str]],
    capability_signals: list[str],
    gate_meta: dict[str, Any],
    reject_reasons: list[str],
    candidate_employee_estimate: int | None = None,
    buyer_employee_estimate: int | None = None,
) -> tuple[float, dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    counts = _evidence_signal_counts(reasons, capability_signals)

    institutional_icp_fit = 0.0
    if gate_meta.get("has_institutional_signal"):
        institutional_icp_fit += 0.6
    target_customer = (gate_meta.get("target_customer") or "").lower()
    if target_customer in {"asset_managers", "wealth_managers", "banks", "fund_admins", "mixed"}:
        institutional_icp_fit += 0.25
    if counts["customer_tokens"] > 0:
        institutional_icp_fit += 0.15
    institutional_icp_fit = min(1.0, institutional_icp_fit)

    platform_product_depth = _scale_count(counts["capability_count"], 6)
    if any("platform" in signal.lower() or "suite" in signal.lower() for signal in capability_signals):
        platform_product_depth = min(1.0, platform_product_depth + 0.1)

    services_complexity = _scale_count(counts["services"], 3)
    if any("workflow" in signal.lower() or "compliance" in signal.lower() for signal in capability_signals):
        services_complexity = min(1.0, services_complexity + 0.1)

    named_customer_credibility = _scale_count(counts["named_customer_reasons"], 3)
    if counts["named_customer_reasons"] == 0 and counts["customer_tokens"] > 0:
        named_customer_credibility = max(named_customer_credibility, 0.2)

    enterprise_gtm = 0.0
    gtm = (gate_meta.get("go_to_market") or "").lower()
    pricing_model = (gate_meta.get("pricing_model") or "").lower()
    price_floor = gate_meta.get("public_price_floor_usd_month")
    if gtm == "b2b_enterprise":
        enterprise_gtm += 0.6
    elif gtm == "b2b_mixed":
        enterprise_gtm += 0.45
    elif gtm == "unknown":
        enterprise_gtm += 0.2

    if pricing_model == "enterprise_quote":
        enterprise_gtm += 0.25
    elif pricing_model == "usage":
        enterprise_gtm += 0.15
    elif pricing_model == "unknown":
        enterprise_gtm += 0.1

    if price_floor is None:
        enterprise_gtm += 0.1
    elif float(price_floor) >= MIN_PUBLIC_PRICE_USD:
        enterprise_gtm += 0.15
    enterprise_gtm = min(1.0, enterprise_gtm)

    defensibility_moat = _scale_count(counts["moat"], 5)
    if counts["capability_count"] >= 4:
        defensibility_moat = min(1.0, defensibility_moat + 0.1)

    components = {
        "institutional_icp_fit": round(institutional_icp_fit, 4),
        "platform_product_depth": round(platform_product_depth, 4),
        "services_implementation_complexity": round(services_complexity, 4),
        "named_customer_credibility": round(named_customer_credibility, 4),
        "enterprise_gtm": round(enterprise_gtm, 4),
        "defensibility_moat": round(defensibility_moat, 4),
    }

    weighted_score = 0.0
    for key, weight in SCREEN_WEIGHTS.items():
        weighted_score += components.get(key, 0.0) * weight

    penalties: list[dict[str, Any]] = []
    size_window_ratio = max(
        0.0,
        min(0.9, float(getattr(settings, "size_fit_window_ratio", SIZE_FIT_WINDOW_RATIO))),
    )
    size_boost_points = max(
        0.0, float(getattr(settings, "size_fit_boost_points", SIZE_FIT_BOOST_POINTS))
    )
    size_large_threshold = max(
        1, int(getattr(settings, "size_large_company_threshold", SIZE_LARGE_COMPANY_THRESHOLD))
    )
    size_large_penalty_points = max(
        0.0,
        float(
            getattr(
                settings,
                "size_large_company_penalty_points",
                SIZE_LARGE_COMPANY_PENALTY_POINTS,
            )
        ),
    )

    normalized_candidate_size = _sanitize_employee_estimate(candidate_employee_estimate)
    normalized_buyer_size = _sanitize_employee_estimate(buyer_employee_estimate)
    if normalized_candidate_size is not None:
        if normalized_candidate_size > size_large_threshold and size_large_penalty_points > 0:
            penalties.append(
                {
                    "reason": "oversized_employee_count",
                    "points": size_large_penalty_points,
                    "candidate_employee_estimate": normalized_candidate_size,
                    "size_large_company_threshold": size_large_threshold,
                }
            )
        elif normalized_buyer_size and size_boost_points > 0:
            lower_bound = int(round(normalized_buyer_size * (1.0 - size_window_ratio)))
            upper_bound = int(round(normalized_buyer_size * (1.0 + size_window_ratio)))
            if lower_bound <= normalized_candidate_size <= upper_bound:
                penalties.append(
                    {
                        "reason": "buyer_size_similarity_boost",
                        "points": -size_boost_points,
                        "candidate_employee_estimate": normalized_candidate_size,
                        "buyer_employee_estimate": normalized_buyer_size,
                        "size_window_ratio": size_window_ratio,
                    }
                )

    for reason in reject_reasons:
        if reason in {"low_ticket_public_pricing", "public_self_serve_pricing"}:
            penalties.append({"reason": reason, "points": 25.0})
        elif reason in {"go_to_market_b2c", "retail_only_icp", "consumer_language_without_institutional_icp"}:
            penalties.append({"reason": reason, "points": 20.0})
        elif reason == "low_software_heaviness":
            penalties.append({"reason": reason, "points": 12.0})

    trusted_count = len(reasons)
    if trusted_count < MIN_TRUSTED_EVIDENCE_FOR_KEEP:
        penalties.append({"reason": "thin_evidence_mix", "points": 12.0})

    if len(capability_signals) < 2:
        penalties.append({"reason": "limited_capability_depth", "points": 8.0})

    penalty_total = sum(float(p.get("points", 0.0)) for p in penalties)
    penalty_total = min(MAX_PENALTY_POINTS, penalty_total)
    final_score = max(0.0, weighted_score - penalty_total)
    candidate_domain = normalize_domain(str(candidate.get("website") or ""))
    first_party_domains = _normalize_domain_list(candidate.get("first_party_domains") or [])
    source_type_counts: dict[str, int] = {}
    for reason in reasons:
        source_url = str(reason.get("citation_url") or "")
        source_type = (
            _source_type_for_url(
                source_url,
                candidate_domain,
                first_party_domains=first_party_domains,
            )
            if source_url
            else "unknown"
        )
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

    has_first_party_evidence = source_type_counts.get("first_party_website", 0) > 0
    first_party_evidence_count = int(source_type_counts.get("first_party_website", 0) or 0)
    has_non_directory_corroboration = (
        source_type_counts.get("official_registry_filing", 0) > 0
        or source_type_counts.get("trusted_third_party", 0) > 0
    )
    has_first_party_depth = first_party_evidence_count >= 3
    hard_fail = bool(gate_meta.get("hard_fail")) or any(reason in HARD_FAIL_REASONS for reason in reject_reasons)
    reference_input = bool(candidate.get("reference_input"))

    if hard_fail:
        screening_status = "rejected"
    elif (
        final_score >= KEEP_SCORE_THRESHOLD
        and trusted_count >= MIN_TRUSTED_EVIDENCE_FOR_KEEP
        and has_first_party_evidence
        and (has_non_directory_corroboration or has_first_party_depth)
    ):
        screening_status = "kept"
    elif final_score >= REVIEW_SCORE_THRESHOLD and trusted_count >= MIN_TRUSTED_EVIDENCE_FOR_REVIEW:
        screening_status = "review"
    else:
        screening_status = "rejected"

    if reference_input and screening_status == "rejected" and trusted_count >= 1 and not hard_fail:
        screening_status = "review"

    score_meta = {
        "screening_status": screening_status,
        "trusted_count": trusted_count,
        "hard_fail": hard_fail,
        "reference_input": reference_input,
        "has_first_party_evidence": has_first_party_evidence,
        "has_first_party_depth": has_first_party_depth,
        "has_non_directory_corroboration": has_non_directory_corroboration,
        "source_type_counts": source_type_counts,
        "candidate_employee_estimate": normalized_candidate_size,
        "buyer_employee_estimate": normalized_buyer_size,
        "size_window_ratio": size_window_ratio,
        "size_large_company_threshold": size_large_threshold,
    }
    return round(final_score, 2), components, penalties, score_meta


def _merge_candidates_by_domain(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    aggregator_domains = {
        "thewealthmosaic.com",
        "thewealthmosaic.co.uk",
    }
    for candidate in candidates:
        website = candidate.get("website")
        domain = normalize_domain(website) if website else None
        company_slug = str(candidate.get("company_slug") or "").strip().lower()
        name_key = str(candidate.get("name", "")).strip().lower()
        if company_slug:
            key = f"slug:{company_slug}"
        elif domain and any(domain == agg or domain.endswith(f".{agg}") for agg in aggregator_domains):
            key = f"name:{name_key}"
        else:
            key = domain or f"name:{name_key}"
        if not key:
            continue
        if key not in merged:
            merged[key] = candidate
            continue
        existing = merged[key]
        existing_reasons = existing.get("why_relevant") if isinstance(existing.get("why_relevant"), list) else []
        new_reasons = candidate.get("why_relevant") if isinstance(candidate.get("why_relevant"), list) else []
        existing_caps = existing.get("capability_signals") if isinstance(existing.get("capability_signals"), list) else []
        new_caps = candidate.get("capability_signals") if isinstance(candidate.get("capability_signals"), list) else []
        existing["why_relevant"] = existing_reasons + new_reasons
        existing["capability_signals"] = _dedupe_strings(
            [str(c) for c in existing_caps + new_caps if isinstance(c, str)]
        )
        existing["first_party_domains"] = _dedupe_strings(
            [str(d) for d in (existing.get("first_party_domains") or []) + (candidate.get("first_party_domains") or []) if str(d).strip()]
        )
        if not existing.get("official_website_url") and candidate.get("official_website_url"):
            existing["official_website_url"] = candidate.get("official_website_url")
            existing["website"] = candidate.get("official_website_url")
        if not existing.get("discovery_url") and candidate.get("discovery_url"):
            existing["discovery_url"] = candidate.get("discovery_url")
        if existing.get("entity_type") != "company" and str(candidate.get("entity_type") or "") == "company":
            existing["entity_type"] = "company"
        if not existing.get("solution_name") and candidate.get("solution_name"):
            existing["solution_name"] = candidate.get("solution_name")
        if not existing.get("hq_country") and candidate.get("hq_country"):
            existing["hq_country"] = candidate.get("hq_country")
    return list(merged.values())


def _scope_match_text(text: str, phrase: str) -> bool:
    haystack = _normalize_name_phrase(text)
    needle = _normalize_name_phrase(phrase)
    if not haystack or not needle:
        return False
    if needle in haystack:
        return True
    needle_tokens = [token for token in needle.split() if len(token) >= 4]
    if len(needle_tokens) >= 2:
        return sum(1 for token in needle_tokens if token in haystack) >= 2
    return False


def _directory_scope_metadata_for_mention(
    mention: dict[str, Any],
    normalized_scope: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    scope = normalized_scope if isinstance(normalized_scope, dict) else {}
    listing_url = str(mention.get("listing_url") or "").strip()
    category_tags = [str(tag).strip() for tag in (mention.get("category_tags") or []) if str(tag).strip()]
    snippets = [str(item).strip() for item in (mention.get("listing_text_snippets") or []) if str(item).strip()]
    search_text = " | ".join(
        item
        for item in [
            str(mention.get("company_name") or "").strip(),
            listing_url,
            " ".join(category_tags),
            " ".join(snippets[:6]),
        ]
        if item
    )

    matched_source_capabilities: list[str] = []
    for capability in _normalize_string_list(scope.get("source_capabilities"), max_items=12, max_len=140):
        if _scope_match_text(search_text, capability):
            matched_source_capabilities.append(capability)

    matched_boxes: list[dict[str, str]] = []
    for raw_box in (scope.get("adjacency_boxes") or []):
        if not isinstance(raw_box, dict):
            continue
        label = str(raw_box.get("label") or "").strip()
        if not label:
            continue
        candidate_terms = [label]
        candidate_terms.extend(
            _normalize_string_list(raw_box.get("likely_workflows"), max_items=4, max_len=120)
        )
        candidate_terms.extend(
            _normalize_string_list(raw_box.get("retrieval_query_seeds"), max_items=4, max_len=180)
        )
        if any(_scope_match_text(search_text, term) for term in candidate_terms):
            matched_boxes.append(
                {
                    "id": str(raw_box.get("id") or "").strip(),
                    "label": label,
                }
            )

    fit_to_adjacency_box_ids = [
        item["id"]
        for item in matched_boxes
        if item.get("id")
    ]
    fit_to_adjacency_box_labels = _dedupe_strings(
        [item["label"] for item in matched_boxes if item.get("label")]
    )[:8]

    if fit_to_adjacency_box_ids or fit_to_adjacency_box_labels:
        scope_bucket = "adjacent"
    elif matched_source_capabilities:
        scope_bucket = "core"
    else:
        scope_bucket = None

    return {
        "scope_bucket": scope_bucket,
        "source_capability_matches": matched_source_capabilities[:8],
        "fit_to_adjacency_box_ids": fit_to_adjacency_box_ids[:8],
        "fit_to_adjacency_box_labels": fit_to_adjacency_box_labels,
    }


def _directory_seed_identity_key(mention: dict[str, Any]) -> Optional[str]:
    official_domain = normalize_domain(str(mention.get("official_website_url") or "").strip())
    if official_domain and not _is_non_first_party_profile_domain(official_domain):
        return f"domain:{official_domain}"
    company_slug = str(mention.get("company_slug") or "").strip().lower()
    if company_slug:
        return f"slug:{company_slug}"
    company_name = str(mention.get("company_name") or "").strip()
    normalized_name = _normalize_name_for_matching(company_name)
    if normalized_name:
        return f"name:{normalized_name}"
    return None


def _seed_candidates_from_mentions(
    mentions: list[dict[str, Any]],
    *,
    normalized_scope: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for mention in mentions:
        company_name = str(mention.get("company_name") or "").strip()
        if not company_name:
            continue
        identity_key = _directory_seed_identity_key(mention)
        if not identity_key:
            continue
        profile_url = str(mention.get("profile_url") or mention.get("company_url") or mention.get("listing_url") or "").strip() or None
        listing_url = str(mention.get("listing_url") or profile_url or "").strip() or None
        official_website_url = str(mention.get("official_website_url") or "").strip() or None
        if official_website_url and not official_website_url.startswith(("http://", "https://")):
            official_website_url = f"https://{official_website_url}"
        if official_website_url and _is_non_first_party_profile_domain(normalize_domain(official_website_url)):
            official_website_url = None
        entity_type = str(mention.get("entity_type") or "company").strip().lower() or "company"
        company_slug = str(mention.get("company_slug") or "").strip() or None
        solution_slug = str(mention.get("solution_slug") or "").strip() or None
        solution_title = str((mention.get("provenance") or {}).get("solution_title") or "").strip() or None
        solution_name = solution_title or (
            solution_slug.replace("-", " ").replace("_", " ").title() if solution_slug else None
        )
        snippets = mention.get("listing_text_snippets") or []
        snippet_text = str(snippets[0]) if snippets else "Listed as comparator in domain directory."
        combined_listing_text = " ".join(
            str(item) for item in (mention.get("listing_text_snippets") or []) if str(item).strip()
        )
        inferred_country = (
            _infer_country_from_text(combined_listing_text)
            or _infer_country_from_domain(normalize_domain(official_website_url or profile_url))
            or "Unknown"
        )
        first_party_domains: list[str] = []
        official_domain = normalize_domain(official_website_url)
        if official_domain and not _is_non_first_party_profile_domain(official_domain):
            first_party_domains = [official_domain]
        scope_metadata = _directory_scope_metadata_for_mention(
            mention,
            normalized_scope=normalized_scope,
        )
        origin = {
            "origin_type": "directory_seed",
            "origin_url": str(listing_url or profile_url or ""),
            "source_name": str(mention.get("source_name") or "wealth_mosaic"),
            "source_run_id": mention.get("source_run_id"),
            "metadata": {
                "category_tags": mention.get("category_tags") or [],
                "profile_url": profile_url,
                "official_website_url": official_website_url,
                "company_slug": company_slug,
                "solution_slug": solution_slug,
                "entity_type": entity_type,
                "solution_name": solution_name,
                "listing_url": listing_url,
                "source_capability_matches": scope_metadata.get("source_capability_matches") or [],
                "fit_to_adjacency_box_ids": scope_metadata.get("fit_to_adjacency_box_ids") or [],
                "fit_to_adjacency_box_labels": scope_metadata.get("fit_to_adjacency_box_labels") or [],
                "scope_bucket": scope_metadata.get("scope_bucket"),
            },
        }
        reason = {
            "text": snippet_text[:700],
            "citation_url": str(listing_url or profile_url or ""),
            "dimension": "directory_context",
            "scope_bucket": scope_metadata.get("scope_bucket"),
        }

        existing = aggregated.get(identity_key)
        if existing is None:
            aggregated[identity_key] = {
                "name": company_name,
                "website": official_website_url,
                "official_website_url": official_website_url,
                "discovery_url": profile_url or listing_url,
                "profile_url": profile_url,
                "listing_url": listing_url,
                "entity_type": entity_type,
                "company_slug": company_slug,
                "solution_slug": solution_slug,
                "solution_name": solution_name,
                "first_party_domains": first_party_domains,
                "hq_country": inferred_country,
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
                "_origins": [origin],
                "why_relevant": [reason],
            }
            continue

        if official_website_url and not existing.get("official_website_url"):
            existing["official_website_url"] = official_website_url
            existing["website"] = official_website_url
        if profile_url and not existing.get("profile_url"):
            existing["profile_url"] = profile_url
        if listing_url and not existing.get("listing_url"):
            existing["listing_url"] = listing_url
        if (profile_url or listing_url) and not existing.get("discovery_url"):
            existing["discovery_url"] = profile_url or listing_url
        if existing.get("entity_type") != "company" and entity_type == "company":
            existing["entity_type"] = "company"
        if not existing.get("solution_name") and solution_name:
            existing["solution_name"] = solution_name
        if not existing.get("hq_country") or existing.get("hq_country") == "Unknown":
            existing["hq_country"] = inferred_country
        existing["first_party_domains"] = _dedupe_strings(
            [str(item) for item in (existing.get("first_party_domains") or []) + first_party_domains if str(item).strip()]
        )
        existing_origins = existing.get("_origins") if isinstance(existing.get("_origins"), list) else []
        existing_origin_keys = {
            (
                str(item.get("source_name") or "").strip().lower(),
                str(item.get("origin_url") or "").strip(),
            )
            for item in existing_origins
            if isinstance(item, dict)
        }
        origin_key = (
            str(origin.get("source_name") or "").strip().lower(),
            str(origin.get("origin_url") or "").strip(),
        )
        if origin_key not in existing_origin_keys:
            existing_origins.append(origin)
        existing["_origins"] = existing_origins
        existing_reasons = existing.get("why_relevant") if isinstance(existing.get("why_relevant"), list) else []
        existing["why_relevant"] = _normalize_reasons(existing_reasons + [reason])

    return list(aggregated.values())


def _directory_seed_priority(candidate: dict[str, Any]) -> tuple[float, str]:
    website = str(candidate.get("official_website_url") or candidate.get("website") or "").strip()
    domain = normalize_domain(website)
    entity_type = str(candidate.get("entity_type") or "company").strip().lower()
    solution_name = str(candidate.get("solution_name") or "").strip()
    name = str(candidate.get("name") or "").strip()
    score = 0.0
    if domain and not _is_non_first_party_profile_domain(domain):
        score += 40.0
    elif domain:
        score -= 20.0
    if entity_type == "company":
        score += 15.0
    elif entity_type == "solution":
        score -= 10.0
    if solution_name:
        score += 3.0
    why_relevant = candidate.get("why_relevant") if isinstance(candidate.get("why_relevant"), list) else []
    score += min(6.0, float(len(why_relevant)))
    return score, name.lower()


def _limit_directory_seed_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_total: int,
    max_per_listing: int,
    max_per_source: int,
    max_without_website: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not candidates:
        return [], {
            "input_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
            "dropped_total_cap": 0,
            "dropped_per_listing_cap": 0,
            "dropped_per_source_cap": 0,
            "dropped_without_website_cap": 0,
        }

    listing_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    kept_without_website = 0
    kept: list[dict[str, Any]] = []
    stats = {
        "input_count": len(candidates),
        "kept_count": 0,
        "dropped_count": 0,
        "dropped_total_cap": 0,
        "dropped_per_listing_cap": 0,
        "dropped_per_source_cap": 0,
        "dropped_without_website_cap": 0,
    }

    ranked = sorted(candidates, key=_directory_seed_priority, reverse=True)
    for candidate in ranked:
        if len(kept) >= max_total:
            stats["dropped_count"] += 1
            stats["dropped_total_cap"] += 1
            continue

        origins = candidate.get("_origins") if isinstance(candidate.get("_origins"), list) else []
        origin = origins[0] if origins and isinstance(origins[0], dict) else {}
        source_name = str(origin.get("source_name") or "unknown").strip().lower() or "unknown"
        metadata = origin.get("metadata") if isinstance(origin.get("metadata"), dict) else {}
        listing_url = str(origin.get("origin_url") or metadata.get("listing_url") or "").strip()
        listing_key = normalize_url(listing_url or "") or source_name

        if source_counts.get(source_name, 0) >= max_per_source:
            stats["dropped_count"] += 1
            stats["dropped_per_source_cap"] += 1
            continue
        if listing_counts.get(listing_key, 0) >= max_per_listing:
            stats["dropped_count"] += 1
            stats["dropped_per_listing_cap"] += 1
            continue

        website = str(candidate.get("official_website_url") or candidate.get("website") or "").strip()
        domain = normalize_domain(website)
        has_first_party_website = bool(domain and not _is_non_first_party_profile_domain(domain))
        if not has_first_party_website and kept_without_website >= max_without_website:
            stats["dropped_count"] += 1
            stats["dropped_without_website_cap"] += 1
            continue

        kept.append(candidate)
        listing_counts[listing_key] = listing_counts.get(listing_key, 0) + 1
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        if not has_first_party_website:
            kept_without_website += 1

    stats["kept_count"] = len(kept)
    return kept, stats


def _seed_candidates_from_reference_urls(reference_urls: list[str] | None) -> list[dict[str, Any]]:
    seeded: list[dict[str, Any]] = []
    for url in reference_urls or []:
        if not isinstance(url, str) or not url.strip():
            continue
        website = url.strip()
        domain = normalize_domain(website)
        if not domain:
            continue
        normalized_website = website if website.startswith(("http://", "https://")) else f"https://{website}"
        parsed = urlparse(normalized_website)
        canonical_website = f"{parsed.scheme or 'https'}://{parsed.netloc}" if parsed.netloc else normalized_website
        name_token = domain.split(".")[0].replace("-", " ").replace("_", " ").strip().title()
        seeded.append(
            {
                "name": name_token or domain,
                "website": canonical_website,
                "official_website_url": canonical_website,
                "discovery_url": normalized_website,
                "entity_type": "company",
                "first_party_domains": [domain],
                "hq_country": _infer_country_from_domain(domain) or "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "reference_input": True,
                "qualification": {},
                "_origins": [
                    {
                        "origin_type": "reference_seed",
                        "origin_url": normalized_website,
                        "source_name": "user_reference",
                        "source_run_id": None,
                        "metadata": {},
                    }
                ],
                "why_relevant": [
                    {
                        "text": "Reference comparator provided as explicit user input.",
                        "citation_url": normalized_website,
                        "dimension": "reference_input",
                    }
                ],
            }
        )
    return seeded


def _should_add_wealth_benchmark_seeds(
    profile: CompanyProfile,
    segment_hints: list[str],
    mentions: list[dict[str, Any]],
) -> bool:
    text_chunks: list[str] = []
    for item in [profile.context_pack_markdown]:
        normalized = str(item or "").strip()
        if normalized:
            text_chunks.append(normalized[:4000])

    for value in (segment_hints or []):
        normalized = str(value or "").strip()
        if normalized:
            text_chunks.append(normalized)

    for mention in mentions[:80]:
        listing_url = str(mention.get("listing_url") or "").strip().lower()
        if "portfolio-wealth-management-systems" in listing_url:
            return True
        for snippet in (mention.get("listing_text_snippets") or [])[:2]:
            normalized = str(snippet or "").strip()
            if normalized:
                text_chunks.append(normalized[:300])

    combined_text = " ".join(text_chunks).lower()
    return any(token in combined_text for token in WEALTH_CONTEXT_TOKENS)


def _discovery_source_names_for_workspace(
    profile: CompanyProfile,
    segment_hints: list[str],
    normalized_scope: Optional[dict[str, Any]] = None,
) -> list[str]:
    scope = normalized_scope if isinstance(normalized_scope, dict) else {}
    text_chunks: list[str] = []
    for item in [
        profile.buyer_company_url,
        profile.context_pack_markdown,
        " ".join(segment_hints or []),
        " ".join(_normalize_string_list(scope.get("source_capabilities"), max_items=12, max_len=140)),
        " ".join(_normalize_string_list(scope.get("source_customer_segments"), max_items=12, max_len=120)),
        " ".join(_normalize_string_list(scope.get("adjacency_box_labels"), max_items=12, max_len=140)),
    ]:
        normalized = str(item or "").strip()
        if normalized:
            text_chunks.append(normalized[:6000])

    combined_text = " ".join(text_chunks).lower()
    source_names: list[str] = []
    wealth_context = any(token in combined_text for token in WEALTH_CONTEXT_TOKENS)
    healthcare_context = any(token in combined_text for token in HEALTHCARE_CONTEXT_TOKENS)

    if wealth_context:
        for source_name in ["wealth_mosaic", "partner_graph_seed", "conference_exhibitors_seed"]:
            if source_name in SOURCE_REGISTRY:
                source_names.append(source_name)
    if healthcare_context:
        for source_name in ["healthcare_vendor_showcase_seed"]:
            if source_name in SOURCE_REGISTRY and source_name not in source_names:
                source_names.append(source_name)

    return source_names


def _seed_candidates_from_benchmark_list() -> list[dict[str, Any]]:
    seeded: list[dict[str, Any]] = []
    for benchmark in WEALTH_BENCHMARK_SEEDS:
        name = str(benchmark.get("name") or "").strip()
        website = str(benchmark.get("website") or "").strip()
        if not name or not website:
            continue
        normalized_website = website if website.startswith(("http://", "https://")) else f"https://{website}"
        seeded.append(
            {
                "name": name,
                "website": normalized_website,
                "official_website_url": normalized_website,
                "discovery_url": normalized_website,
                "entity_type": "company",
                "first_party_domains": [normalize_domain(normalized_website)] if normalize_domain(normalized_website) else [],
                "hq_country": str(benchmark.get("hq_country") or _infer_country_from_domain(normalize_domain(normalized_website)) or "Unknown"),
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "reference_input": True,
                "qualification": {},
                "_origins": [
                    {
                        "origin_type": "benchmark_seed",
                        "origin_url": normalized_website,
                        "source_name": "wealth_benchmark_pack",
                        "source_run_id": None,
                        "metadata": {
                            "benchmark_group": "wealth_platforms",
                        },
                    }
                ],
                "why_relevant": [
                    {
                        "text": "Benchmark comparator seeded for wealth-platform coverage to prevent directory omissions.",
                        "citation_url": normalized_website,
                        "dimension": "benchmark_seed",
                    }
                ],
            }
        )
    return seeded


def _build_discovery_prompt_for_orchestrator(
    context_pack: str,
    taxonomy_bricks: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    vertical_focus: list[str],
    comparator_mentions: list[dict[str, Any]],
) -> str:
    brick_names = [str(b.get("name") or "").strip() for b in (taxonomy_bricks or []) if str(b.get("name") or "").strip()]
    region = str((geo_scope or {}).get("region") or "EU+UK")
    include_countries = [str(c).strip() for c in ((geo_scope or {}).get("include_countries") or []) if str(c).strip()]
    verticals = [str(v).strip() for v in (vertical_focus or []) if str(v).strip()]

    seed_lines: list[str] = []
    for mention in (comparator_mentions or [])[:30]:
        name = str(mention.get("company_name") or "").strip()
        url = str(mention.get("company_url") or mention.get("official_website_url") or "").strip()
        if not name:
            continue
        if url:
            seed_lines.append(f"- {name} ({url})")
        else:
            seed_lines.append(f"- {name}")
    seeds_text = "\n".join(seed_lines) if seed_lines else "- No directory seeds."

    return f"""You are an M&A research analyst. Return ONLY a JSON array of B2B financial software companies.

Buyer context:
{context_pack[:3500] if context_pack else "No buyer context provided."}

Target filters:
- Region: {region}
- Include countries: {", ".join(include_countries) if include_countries else "auto"}
- Verticals: {", ".join(verticals) if verticals else "wealth, asset management, securities, investment tech"}
- Capability bricks: {", ".join(brick_names[:15]) if brick_names else "n/a"}

Comparator seeds:
{seeds_text}

Output schema:
[
  {{
    "name": "Company Name",
    "website": "https://example.com or null if not known",
    "discovery_url": "https://retrieval-context-url-that-mentioned-the-company",
    "hq_country": "DE",
    "likely_verticals": ["wealth_manager"],
    "employee_estimate": 120,
    "capability_signals": ["Portfolio management"],
    "qualification": {{"go_to_market": "b2b_enterprise", "target_customer": "asset_managers"}},
    "why_relevant": [
      {{"text":"Evidence text", "citation_url":"https://...", "dimension":"capability"}}
    ]
  }}
]

Constraints:
- 15-25 companies.
- Every company must have at least one why_relevant item with a citation_url.
- Prefer first-party pages or official filings/regulatory sources.
- Exclude mega incumbents (Bloomberg, FIS, Fiserv, Broadridge, SS&C, Refinitiv).
- Return JSON only.
    """


def _normalize_string_list(values: Any, max_items: int = 12, max_len: int = 120) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text not in normalized:
            normalized.append(text[:max_len])
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_query_entries(values: Any, max_items: int = 6) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(values, list):
        return entries
    for item in values:
        if isinstance(item, str):
            query_text = item.strip()
            if not query_text:
                continue
            entries.append(
                {
                    "query_text": query_text[:220],
                    "query_intent": None,
                    "query_family": normalize_discovery_query_family(query_text, query_text=query_text),
                    "brick_name": None,
                    "scope_bucket": None,
                }
            )
        elif isinstance(item, dict):
            query_text = str(item.get("query") or item.get("query_text") or item.get("text") or "").strip()
            if not query_text:
                continue
            query_family = normalize_discovery_query_family(
                item.get("query_family") or item.get("query_intent") or item.get("intent"),
                query_text=query_text,
                provider=item.get("provider"),
            )
            entries.append(
                {
                    "query_text": query_text[:220],
                    "query_intent": str(item.get("query_intent") or item.get("intent") or query_family).strip() or None,
                    "query_family": query_family,
                    "brick_name": str(item.get("brick_name") or "").strip() or None,
                    "scope_bucket": str(item.get("scope_bucket") or "").strip().lower() or None,
                }
            )
        if len(entries) >= max_items:
            break
    return entries


def _normalize_labeled_values(values: Any, *, max_items: int = 12, max_len: int = 120) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        text = ""
        if isinstance(value, dict):
            text = str(value.get("label") or value.get("name") or value.get("id") or "").strip()
        else:
            text = str(value or "").strip()
        if not text:
            continue
        if text not in normalized:
            normalized.append(text[:max_len])
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_scope_hints(scope_hints: Any) -> dict[str, Any]:
    payload = scope_hints if isinstance(scope_hints, dict) else {}
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    adjacency_boxes: list[dict[str, Any]] = []
    if isinstance(payload.get("adjacency_boxes"), list):
        for raw in payload.get("adjacency_boxes") or []:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label") or "").strip()
            if not label:
                continue
            kind = str(raw.get("adjacency_kind") or "adjacent_capability").strip().lower() or "adjacent_capability"
            if kind not in {"adjacent_capability", "adjacent_customer_segment", "adjacent_workflow"}:
                kind = "adjacent_capability"
            status = str(raw.get("status") or "hypothesis").strip().lower() or "hypothesis"
            priority_tier = str(raw.get("priority_tier") or "meaningful_adjacent").strip().lower() or "meaningful_adjacent"
            adjacency_boxes.append(
                {
                    "id": str(raw.get("id") or "").strip() or None,
                    "label": label[:140],
                    "adjacency_kind": kind,
                    "status": status,
                    "priority_tier": priority_tier,
                    "confidence": _safe_float(raw.get("confidence"), default=0.0),
                    "likely_customer_segments": _normalize_string_list(raw.get("likely_customer_segments"), max_items=4, max_len=120),
                    "likely_workflows": _normalize_string_list(raw.get("likely_workflows"), max_items=4, max_len=120),
                    "retrieval_query_seeds": _normalize_string_list(raw.get("retrieval_query_seeds"), max_items=4, max_len=180),
                }
            )
            if len(adjacency_boxes) >= 10:
                break

    company_seeds: list[dict[str, Any]] = []
    if isinstance(payload.get("company_seeds"), list):
        for raw in payload.get("company_seeds") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            website = str(raw.get("website") or "").strip() or None
            evidence_urls: list[str] = []
            if isinstance(raw.get("evidence"), list):
                for item in raw.get("evidence") or []:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url") or "").strip()
                    if url:
                        evidence_urls.append(url[:220])
            company_seeds.append(
                {
                    "id": str(raw.get("id") or "").strip() or None,
                    "name": name[:140],
                    "website": website[:220] if website else None,
                    "status": str(raw.get("status") or "").strip().lower() or None,
                    "seed_type": str(raw.get("seed_type") or "").strip().lower() or None,
                    "seed_role": str(raw.get("seed_role") or "").strip().lower() or None,
                    "confidence": _safe_float(raw.get("confidence"), default=0.0),
                    "evidence": [{"url": value} for value in _dedupe_strings(evidence_urls)[:6]],
                    "fit_to_adjacency_box_ids": _normalize_string_list(raw.get("fit_to_adjacency_box_ids"), max_items=8, max_len=120),
                }
            )
            if len(company_seeds) >= 16:
                break

    company_seed_urls = _normalize_string_list(payload.get("company_seed_urls"), max_items=16, max_len=220)
    for seed in company_seeds:
        website = str(seed.get("website") or "").strip()
        if website:
            company_seed_urls.append(website)

    adjacency_box_labels = _normalize_string_list(
        payload.get("adjacency_box_labels")
        or [box.get("label") for box in adjacency_boxes if isinstance(box, dict)],
        max_items=10,
        max_len=140,
    )
    adjacent_lanes = _normalize_string_list(
        payload.get("adjacent_lanes")
        or adjacency_box_labels,
        max_items=12,
        max_len=140,
    )
    return {
        "source_capabilities": _normalize_string_list(payload.get("source_capabilities"), max_items=10, max_len=140),
        "source_customer_segments": _normalize_string_list(payload.get("source_customer_segments"), max_items=8, max_len=120),
        "named_account_anchors": _normalize_labeled_values(payload.get("named_account_anchors"), max_items=8, max_len=120),
        "geography_expansions": _normalize_labeled_values(payload.get("geography_expansions"), max_items=8, max_len=96),
        "adjacency_boxes": adjacency_boxes,
        "adjacency_box_labels": adjacency_box_labels,
        "adjacent_lanes": adjacent_lanes,
        "company_seeds": company_seeds,
        "company_seed_urls": _dedupe_strings(company_seed_urls)[:16],
        "comparator_seed_urls": _normalize_string_list(payload.get("comparator_seed_urls"), max_items=8, max_len=220),
        "confirmed": bool(payload.get("confirmed")),
    }


def _taxonomy_compatible_hints_from_scope_hints(
    scope_hints: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized_scope = _normalize_scope_hints(scope_hints)
    box_capability_hints = [
        str(box.get("label") or "").strip()
        for box in (normalized_scope.get("adjacency_boxes") or [])
        if isinstance(box, dict)
        and str(box.get("adjacency_kind") or "").strip().lower() in {"adjacent_capability", "adjacent_workflow"}
        and str(box.get("label") or "").strip()
    ]
    box_customer_hints = [
        hint
        for box in (normalized_scope.get("adjacency_boxes") or [])
        if isinstance(box, dict)
        for hint in _normalize_string_list(box.get("likely_customer_segments"), max_items=4, max_len=120)
    ]
    capability_hints = _dedupe_strings(
        (normalized_scope.get("source_capabilities") or [])
        + box_capability_hints
    )
    segment_hints = _dedupe_strings(
        (normalized_scope.get("source_customer_segments") or [])
        + (normalized_scope.get("named_account_anchors") or [])
        + box_customer_hints
    )
    return ([{"name": name} for name in capability_hints], segment_hints)


def _adjacency_lane_candidates(normalized_scope: dict[str, Any]) -> list[dict[str, Any]]:
    priority_rank = {"core_adjacent": 0, "meaningful_adjacent": 1, "edge_case": 2}
    status_rank = {
        "user_kept": 0,
        "source_grounded": 1,
        "corroborated_expansion": 2,
        "hypothesis": 3,
        "user_deprioritized": 4,
        "user_removed": 5,
    }

    boxes = [
        box
        for box in (normalized_scope.get("adjacency_boxes") or [])
        if isinstance(box, dict) and str(box.get("label") or "").strip()
    ]

    def _is_active(box: dict[str, Any]) -> bool:
        status = str(box.get("status") or "").strip().lower()
        if status == "user_kept":
            return True
        if status not in {"source_grounded", "corroborated_expansion"}:
            return False
        return str(box.get("priority_tier") or "").strip().lower() != "edge_case"

    selected = [box for box in boxes if _is_active(box)]
    if not selected:
        selected = [
            box
            for box in boxes
            if str(box.get("status") or "").strip().lower() == "hypothesis"
            and str(box.get("priority_tier") or "").strip().lower() != "edge_case"
        ]

    selected = sorted(
        selected,
        key=lambda box: (
            priority_rank.get(str(box.get("priority_tier") or "").strip().lower(), 9),
            status_rank.get(str(box.get("status") or "").strip().lower(), 9),
            -float(box.get("confidence") or 0.0),
            str(box.get("label") or ""),
        ),
    )

    lanes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for box in selected:
        label = str(box.get("label") or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        lanes.append(
            {
                "label": label,
                "adjacency_kind": str(box.get("adjacency_kind") or "").strip().lower() or "adjacent_capability",
                "likely_customer_segments": _normalize_string_list(box.get("likely_customer_segments"), max_items=4, max_len=120),
                "likely_workflows": _normalize_string_list(box.get("likely_workflows"), max_items=4, max_len=120),
                "retrieval_query_seeds": _normalize_string_list(box.get("retrieval_query_seeds"), max_items=4, max_len=180),
            }
        )
        if len(lanes) >= 6:
            break

    if not lanes:
        for label in _normalize_string_list(
            normalized_scope.get("adjacent_lanes"),
            max_items=6,
            max_len=140,
        ):
            lanes.append(
                {
                    "label": label,
                    "adjacency_kind": "adjacent_capability",
                    "likely_customer_segments": [],
                    "likely_workflows": [],
                    "retrieval_query_seeds": [],
                }
            )
    return lanes


def _market_scan_hint(
    normalized_scope: dict[str, Any],
    vertical_focus: list[str],
    brick_names: list[str],
    lane_candidates: list[dict[str, Any]],
) -> str:
    corpus = " ".join(
        _dedupe_strings(
            (normalized_scope.get("source_capabilities") or [])
            + (normalized_scope.get("source_customer_segments") or [])
            + [str(item) for item in (vertical_focus or []) if str(item).strip()]
            + [str(item) for item in (brick_names or []) if str(item).strip()]
            + [
                str(box.get("label") or "").strip()
                for box in (lane_candidates or [])
                if isinstance(box, dict) and str(box.get("label") or "").strip()
            ]
            + [
                item
                for box in (lane_candidates or [])
                if isinstance(box, dict)
                for item in _normalize_string_list(box.get("retrieval_query_seeds"), max_items=4, max_len=180)
            ]
        )
    ).lower()
    if any(token in corpus for token in ("wealth", "private bank", "asset management", "portfolio", "fund admin", "investment")):
        return "wealthtech"
    if any(token in corpus for token in ("hospital", "healthcare", "clinic", "care provider")):
        if any(token in corpus for token in ("staff", "staffing", "shift", "rostering", "workforce", "mobility", "pool")):
            return "healthcare staffing"
        return "healthcare software"
    if any(token in corpus for token in ("staff", "staffing", "shift", "rostering", "workforce", "mobility", "recruit")):
        return "workforce software"
    if any(token in corpus for token in ("treasury", "post-trade", "settlement", "clearing", "securities")):
        return "capital markets software"
    vertical_hint = next((str(v).strip() for v in (vertical_focus or []) if str(v).strip()), "")
    if vertical_hint:
        return vertical_hint[:80]
    customer_hint = next((str(v).strip() for v in (normalized_scope.get("source_customer_segments") or []) if str(v).strip()), "")
    if customer_hint:
        return customer_hint[:80]
    return "b2b software"


def _default_exclude_terms_for_scope(
    normalized_scope: dict[str, Any],
    vertical_focus: list[str],
    brick_names: list[str],
    lane_candidates: list[dict[str, Any]],
) -> list[str]:
    market_hint = _market_scan_hint(normalized_scope, vertical_focus, brick_names, lane_candidates).lower()
    base = ["careers", "jobs", "pdf"]
    if market_hint == "wealthtech":
        base.extend(sorted(WEALTH_DISCOVERY_EXCLUDE_TERMS))
    return _dedupe_strings(base)


def _filter_discovery_seed_urls(urls: list[str]) -> list[str]:
    filtered: list[str] = []
    for raw_url in urls or []:
        normalized = normalize_url(raw_url or "")
        domain = normalize_domain(normalized)
        if not normalized or not domain:
            continue
        if domain in PLACEHOLDER_SEED_DOMAINS or any(domain.endswith(f".{suffix}") for suffix in PLACEHOLDER_SEED_DOMAINS):
            continue
        filtered.append(normalized)
    return _dedupe_strings(filtered)


def _company_label_from_url(raw_url: Any) -> Optional[str]:
    domain = normalize_domain(normalize_url(raw_url or ""))
    if not domain:
        return None
    label = domain.split(".")[0].replace("-", " ").replace("_", " ").strip()
    normalized = " ".join(part.capitalize() for part in label.split() if part)
    return normalized or None


def _has_non_placeholder_evidence_urls(seed: dict[str, Any]) -> bool:
    for item in (seed.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        url = normalize_url(item.get("url") or "")
        domain = normalize_domain(url)
        if not url or not domain:
            continue
        if domain in PLACEHOLDER_SEED_DOMAINS or any(domain.endswith(f".{suffix}") for suffix in PLACEHOLDER_SEED_DOMAINS):
            continue
        return True
    return False


def _high_confidence_company_seed_names(company_seeds: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for seed in company_seeds or []:
        if not isinstance(seed, dict):
            continue
        name = str(seed.get("name") or "").strip()
        if not name:
            continue
        website = normalize_url(seed.get("website") or "")
        website_domain = normalize_domain(website)
        status = str(seed.get("status") or "").strip().lower()
        try:
            confidence = float(seed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        has_real_website = bool(
            website_domain
            and website_domain not in PLACEHOLDER_SEED_DOMAINS
            and not any(website_domain.endswith(f".{suffix}") for suffix in PLACEHOLDER_SEED_DOMAINS)
            and not _is_non_first_party_profile_domain(website_domain)
        )
        strong_status = status in {"confirmed", "approved", "reference", "keep", "strong_signal", "validated"}
        if has_real_website and (strong_status or confidence >= 0.75 or _has_non_placeholder_evidence_urls(seed)):
            selected.append(name[:140])
    return _dedupe_strings(selected)[:6]


def _backfill_comparative_retrieval_context(
    retrieval_results: list[dict[str, Any]],
    *,
    cap: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not retrieval_results:
        return retrieval_results, {"attempted": 0, "updated": 0, "errors": []}

    updated: list[dict[str, Any]] = []
    attempted = 0
    changed = 0
    errors: list[str] = []

    for row in retrieval_results:
        item = dict(row)
        query_family = normalize_discovery_query_family(
            item.get("query_family") or item.get("query_intent") or item.get("query_type"),
            query_text=item.get("query_text"),
            provider=item.get("provider"),
        )
        snippet = str(item.get("snippet") or "").strip()
        url = normalize_url(item.get("normalized_url") or item.get("url") or "")
        needs_backfill = query_family in {
            "competitor_direct",
            "alternatives",
            "comparative_source",
            "traffic_affinity",
            "peer_expansion",
            "local_market",
        } and len(snippet) < 80 and bool(url)
        if needs_backfill and attempted < max(0, int(cap)):
            attempted += 1
            try:
                fast = fetch_page_fast(url)
                content = re.sub(r"\s+", " ", str(fast.get("content") or "").strip())
                if content:
                    item["snippet"] = content[:900]
                    item["snippet_backfilled"] = True
                    item["snippet_backfill_provider"] = fast.get("provider")
                    changed += 1
            except Exception as exc:
                errors.append(f"{url}:{exc}")
        updated.append(item)

    return updated, {
        "attempted": attempted,
        "updated": changed,
        "errors": errors[:12],
    }


def _planner_scope_hints(scope_hints: Optional[dict[str, Any]]) -> dict[str, Any]:
    normalized_scope = _normalize_scope_hints(scope_hints)
    if not normalized_scope:
        return {}
    trusted_seed_names = set(
        _high_confidence_company_seed_names(
            [seed for seed in (normalized_scope.get("company_seeds") or []) if isinstance(seed, dict)]
        )
    )
    if trusted_seed_names:
        normalized_scope["company_seeds"] = [
            seed
            for seed in (normalized_scope.get("company_seeds") or [])
            if isinstance(seed, dict) and str(seed.get("name") or "").strip() in trusted_seed_names
        ]
    else:
        normalized_scope["company_seeds"] = []
    return normalized_scope


def _default_discovery_query_plan(
    taxonomy_bricks: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    vertical_focus: list[str],
    scope_hints: Optional[dict[str, Any]] = None,
    source_company_url: Optional[str] = None,
) -> dict[str, Any]:
    normalized_scope = _normalize_scope_hints(scope_hints)
    brick_names = [str(b.get("name") or "").strip() for b in (taxonomy_bricks or []) if str(b.get("name") or "").strip()]
    region = str((geo_scope or {}).get("region") or "EU+UK")
    seed_urls = _dedupe_strings(
        (normalized_scope.get("comparator_seed_urls") or [])
        + (normalized_scope.get("company_seed_urls") or [])
    )
    seed_urls = _filter_discovery_seed_urls(seed_urls)[:12]
    source_company_name = _company_label_from_url(source_company_url)
    lane_candidates = _adjacency_lane_candidates(normalized_scope)
    market_scan_hint = _market_scan_hint(normalized_scope, vertical_focus, brick_names, lane_candidates)
    company_seed_names = _high_confidence_company_seed_names(
        [seed for seed in (normalized_scope.get("company_seeds") or []) if isinstance(seed, dict)]
    )

    precision_queries: list[dict[str, Any]] = []
    recall_queries: list[dict[str, Any]] = []
    source_company_precision_queries: list[dict[str, Any]] = []
    source_company_recall_queries: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    geography_terms = _dedupe_strings(
        [
            str(item.get("label") or item.get("name") or item.get("id") or "").strip()
            for item in (normalized_scope.get("geography_expansions") or [])
            if isinstance(item, dict) and str(item.get("label") or item.get("name") or item.get("id") or "").strip()
        ]
        + _normalize_string_list(
            [item for item in (normalized_scope.get("geography_expansions") or []) if not isinstance(item, dict)],
            max_items=2,
            max_len=32,
        )
    )
    geography_terms = [term for term in geography_terms if "{" not in term and "}" not in term][:4]
    include_country_terms = _dedupe_strings(
        [
            str(item.get("label") or item.get("name") or item.get("id") or "").strip()
            for item in ((geo_scope or {}).get("include_countries") or [])
            if isinstance(item, dict) and str(item.get("label") or item.get("name") or item.get("id") or "").strip()
        ]
        + _normalize_string_list(
            [item for item in ((geo_scope or {}).get("include_countries") or []) if not isinstance(item, dict)],
            max_items=2,
            max_len=32,
        )
    )
    include_country_terms = [term for term in include_country_terms if "{" not in term and "}" not in term][:4]
    language_terms = _dedupe_strings(
        [
            str(item.get("label") or item.get("name") or item.get("id") or "").strip()
            for item in ((geo_scope or {}).get("languages") or [])
            if isinstance(item, dict) and str(item.get("label") or item.get("name") or item.get("id") or "").strip()
        ]
        + _normalize_string_list(
            [item for item in ((geo_scope or {}).get("languages") or []) if not isinstance(item, dict)],
            max_items=2,
            max_len=16,
        )
    )
    language_terms = [term for term in language_terms if "{" not in term and "}" not in term][:4]
    local_market_hint = " ".join(
        _dedupe_strings(
            include_country_terms[:2]
            + language_terms[:2]
            + geography_terms[:2]
        )[:3]
    ).strip() or region

    def _add_query(
        bucket: list[dict[str, Any]],
        query_text: str,
        query_family: str,
        brick_name: Optional[str],
        scope_bucket: str,
    ) -> None:
        normalized_query = " ".join(str(query_text or "").split()).strip()
        if not normalized_query:
            return
        query_key = normalized_query.lower()
        if query_key in seen_queries:
            return
        seen_queries.add(query_key)
        bucket.append(
            {
                "query_text": normalized_query[:220],
                "query_intent": query_family,
                "query_family": normalize_discovery_query_family(query_family, query_text=normalized_query),
                "brick_name": brick_name,
                "scope_bucket": scope_bucket,
            }
        )

    if normalized_scope.get("source_capabilities") or lane_candidates:
        lane_customer_terms = _dedupe_strings(
            [
                term
                for box in lane_candidates
                for term in _normalize_string_list(box.get("likely_customer_segments"), max_items=4, max_len=120)
            ]
        )
        customer_terms = _dedupe_strings(
            (normalized_scope.get("source_customer_segments") or [])
            + lane_customer_terms
        )
        customer_hint = " ".join(customer_terms[:2]) or "B2B software"
        core_capabilities = normalized_scope.get("source_capabilities") or brick_names or ["core workflow"]
        for capability in core_capabilities[:4]:
            _add_query(
                precision_queries,
                f"{capability} software vendor {customer_hint} {region}",
                "category_vendor",
                capability,
                "core",
            )
            _add_query(
                precision_queries,
                f"{capability} platform provider {customer_hint} {region}",
                "buyer_substitute",
                capability,
                "core",
            )
            _add_query(
                precision_queries,
                f"{capability} competitors {customer_hint} {local_market_hint}",
                "competitor_direct",
                capability,
                "core",
            )
            _add_query(
                precision_queries,
                f"{capability} alternatives {customer_hint} {region}",
                "alternatives",
                capability,
                "core",
            )

        for lane in lane_candidates[:4]:
            lane_label = str(lane.get("label") or "").strip()
            if not lane_label:
                continue
            lane_customer_terms = _dedupe_strings(
                (lane.get("likely_customer_segments") or []) + customer_terms
            )
            lane_workflow = next(
                (
                    text
                    for text in _normalize_string_list(lane.get("likely_workflows"), max_items=4, max_len=120)
                    if text.lower() != lane_label.lower()
                ),
                None,
            )
            lane_customer_hint = " ".join(lane_customer_terms[:2]) or customer_hint
            retrieval_query_seed = next(
                (
                    seed
                    for seed in _normalize_string_list(lane.get("retrieval_query_seeds"), max_items=4, max_len=180)
                    if seed
                ),
                None,
            )
            lane_context_hint = " ".join(
                _dedupe_strings(
                    [
                        retrieval_query_seed or "",
                        lane_workflow or "",
                        market_scan_hint or "",
                    ]
                )[:2]
            ).strip()
            if retrieval_query_seed:
                _add_query(
                    recall_queries,
                    f"{retrieval_query_seed} software vendor {region}",
                    "adjacent_substitute",
                    lane_label,
                    "adjacent",
                )
            lane_query = f"{lane_label} {lane_context_hint} software vendor {lane_customer_hint} {region}".strip()
            _add_query(
                recall_queries,
                lane_query,
                "adjacent_substitute",
                lane_label,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"{lane_label} {lane_context_hint} platform provider {lane_customer_hint} {region}".strip(),
                "buyer_substitute",
                lane_label,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"{lane_label} {lane_context_hint} software competitors {lane_customer_hint} {local_market_hint}".strip(),
                "competitor_direct",
                lane_label,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"{lane_label} {lane_context_hint} software alternatives {lane_customer_hint} {region}".strip(),
                "alternatives",
                lane_label,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"best {lane_label} {lane_context_hint} software {local_market_hint}".strip(),
                "comparative_source",
                lane_label,
                "adjacent",
            )

        for seed_name in company_seed_names[:4]:
            _add_query(
                recall_queries,
                f"{seed_name} alternatives {customer_hint} software {region}",
                "alternatives",
                seed_name,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"{seed_name} competitors {market_scan_hint} {local_market_hint}",
                "competitor_direct",
                seed_name,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"companies like {seed_name} {market_scan_hint} {region}",
                "peer_expansion",
                seed_name,
                "adjacent",
            )
            _add_query(
                recall_queries,
                f"{seed_name} similarweb competitors",
                "traffic_affinity",
                seed_name,
                "adjacent",
            )

        if source_company_name:
            _add_query(
                source_company_precision_queries,
                f"{source_company_name} competitors",
                "competitor_direct",
                source_company_name,
                "core",
            )
            _add_query(
                source_company_precision_queries,
                f"{source_company_name} alternatives",
                "alternatives",
                source_company_name,
                "core",
            )
            _add_query(
                source_company_precision_queries,
                f"companies like {source_company_name}",
                "peer_expansion",
                source_company_name,
                "core",
            )
            _add_query(
                source_company_precision_queries,
                f"{source_company_name} competitors {market_scan_hint} {local_market_hint}",
                "competitor_direct",
                source_company_name,
                "core",
            )
            _add_query(
                source_company_recall_queries,
                f"best alternatives to {source_company_name}",
                "comparative_source",
                source_company_name,
                "core",
            )
            _add_query(
                source_company_recall_queries,
                f"{source_company_name} alternatives {market_scan_hint} {region}",
                "alternatives",
                source_company_name,
                "core",
            )
            _add_query(
                source_company_recall_queries,
                f"{source_company_name} similarweb competitors",
                "traffic_affinity",
                source_company_name,
                "core",
            )

        if not recall_queries:
            _add_query(
                recall_queries,
                f"adjacent workflow software vendor {customer_hint} {region}",
                "adjacent_substitute",
                None,
                "adjacent",
            )
        return {
            "precision_queries": _merge_query_entries(
                source_company_precision_queries,
                precision_queries,
                max_items=10,
            ),
            "recall_queries": _merge_query_entries(
                source_company_recall_queries,
                recall_queries,
                max_items=12,
            ),
            "seed_urls": seed_urls,
            "must_include_terms": _dedupe_strings(
                customer_terms[:2]
                + (normalized_scope.get("geography_expansions") or [])[:2]
            )[:4],
            "must_exclude_terms": _default_exclude_terms_for_scope(
                normalized_scope,
                vertical_focus,
                brick_names,
                lane_candidates,
            )[:8],
            "preferred_countries": [],
            "preferred_languages": [],
            "domain_allowlist": [],
            "domain_blocklist": [],
        }

    vertical_hint = ", ".join([str(v) for v in (vertical_focus or [])[:2] if str(v).strip()]) or "B2B software"
    brick_hint = ", ".join(brick_names[:2]) if brick_names else "core workflow"

    _add_query(
        precision_queries,
        f"{brick_hint} software vendor {vertical_hint} {region}",
        "category_vendor",
        brick_names[0] if brick_names else None,
        "core",
    )
    _add_query(
        precision_queries,
        f"{brick_hint} platform provider {vertical_hint} {region}",
        "buyer_substitute",
        brick_names[1] if len(brick_names) > 1 else (brick_names[0] if brick_names else None),
        "core",
    )
    _add_query(
        precision_queries,
        f"{brick_hint} competitors {vertical_hint} {local_market_hint}",
        "competitor_direct",
        brick_names[0] if brick_names else None,
        "core",
    )
    _add_query(
        recall_queries,
        f"{vertical_hint} SaaS vendor {region}",
        "category_vendor",
        None,
        "core",
    )
    _add_query(
        recall_queries,
        f"{vertical_hint} adjacent workflow software vendor {region}",
        "adjacent_substitute",
        None,
        "adjacent",
    )
    if source_company_name:
        _add_query(
            source_company_precision_queries,
            f"{source_company_name} competitors",
            "competitor_direct",
            source_company_name,
            "core",
        )
        _add_query(
            source_company_precision_queries,
            f"{source_company_name} alternatives",
            "alternatives",
            source_company_name,
            "core",
        )
        _add_query(
            source_company_precision_queries,
            f"{source_company_name} competitors {vertical_hint} {local_market_hint}",
            "competitor_direct",
            source_company_name,
            "core",
        )
        _add_query(
            source_company_recall_queries,
            f"best alternatives to {source_company_name}",
            "comparative_source",
            source_company_name,
            "core",
        )
        _add_query(
            source_company_recall_queries,
            f"{source_company_name} alternatives {vertical_hint} {region}",
            "alternatives",
            source_company_name,
            "core",
        )
        _add_query(
            source_company_recall_queries,
            f"companies like {source_company_name}",
            "peer_expansion",
            source_company_name,
            "core",
        )
    _add_query(
        recall_queries,
        f"{vertical_hint} alternatives {local_market_hint}",
        "alternatives",
        None,
        "core",
    )
    _add_query(
        recall_queries,
        f"best {vertical_hint} software {local_market_hint}",
        "comparative_source",
        None,
        "core",
    )
    return {
        "precision_queries": _merge_query_entries(
            source_company_precision_queries,
            precision_queries,
            max_items=10,
        ),
        "recall_queries": _merge_query_entries(
            source_company_recall_queries,
            recall_queries,
            max_items=12,
        ),
        "seed_urls": seed_urls,
        "must_include_terms": [],
        "must_exclude_terms": _default_exclude_terms_for_scope(
            normalized_scope,
            vertical_focus,
            brick_names,
            lane_candidates,
        )[:8],
        "preferred_countries": [],
        "preferred_languages": [],
        "domain_allowlist": [],
        "domain_blocklist": [],
    }


def _build_discovery_query_plan_prompt(
    context_pack: str,
    taxonomy_bricks: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    vertical_focus: list[str],
    comparator_mentions: list[dict[str, Any]],
    scope_hints: Optional[dict[str, Any]] = None,
    source_company_url: Optional[str] = None,
) -> str:
    normalized_scope = _planner_scope_hints(scope_hints)
    brick_names = [str(b.get("name") or "").strip() for b in (taxonomy_bricks or []) if str(b.get("name") or "").strip()]
    region = str((geo_scope or {}).get("region") or "EU+UK")
    include_countries = [str(c).strip() for c in ((geo_scope or {}).get("include_countries") or []) if str(c).strip()]
    verticals = [str(v).strip() for v in (vertical_focus or []) if str(v).strip()]
    source_company_name = _company_label_from_url(source_company_url) or "target company"

    seed_lines: list[str] = []
    for mention in (comparator_mentions or [])[:20]:
        name = str(mention.get("company_name") or "").strip()
        url = str(mention.get("company_url") or mention.get("official_website_url") or "").strip()
        if not name:
            continue
        seed_lines.append(f"- {name} ({url})" if url else f"- {name}")
    seeds_text = "\n".join(seed_lines) if seed_lines else "- No directory seeds."

    return f"""You are an M&A research analyst building a discovery query plan.
Return ONLY valid JSON with this schema:
{{
  "precision_queries": [
    {{"query": "string", "query_family": "category_vendor|competitor_direct|alternatives|buyer_substitute|adjacent_substitute|peer_expansion|local_market|comparative_source|traffic_affinity", "brick_name": "optional capability name", "scope_bucket": "core|adjacent"}}
  ],
  "recall_queries": [
    {{"query": "string", "query_family": "category_vendor|competitor_direct|alternatives|buyer_substitute|adjacent_substitute|peer_expansion|local_market|comparative_source|traffic_affinity", "brick_name": "optional capability name", "scope_bucket": "core|adjacent"}}
  ],
  "seed_urls": ["https://..."],
  "must_include_terms": ["string"],
  "must_exclude_terms": ["string"],
  "preferred_countries": ["FR","UK"],
  "preferred_languages": ["en","fr"],
  "domain_allowlist": ["example.com"],
  "domain_blocklist": ["crunchbase.com"]
}}

Buyer context:
{context_pack[:2800] if context_pack else "No buyer context provided."}

Target filters:
- Region: {region}
- Source company: {source_company_name}
- Include countries: {", ".join(include_countries) if include_countries else "auto"}
- Approved scope hints: {json.dumps(normalized_scope, ensure_ascii=False) if normalized_scope else "none confirmed yet"}
- Legacy vertical hints: {", ".join(verticals) if verticals else "generic software"}
- Legacy capability hints: {", ".join(brick_names[:12]) if brick_names else "n/a"}

Comparator seeds:
{seeds_text}

Constraints:
- 6-10 precision queries, 6-12 recall queries.
- Keep queries short, seller-oriented, and evidence-friendly.
- Prefer software vendors, platform providers, SaaS companies, and official company or product pages.
- Include a diversified mix across direct competitors, alternatives, local-market phrasing, adjacent substitutes, comparative/list-source discovery, and buyer-substitute wording when relevant.
- Avoid buyer institutions, bank homepages, consultants, market-research reports, listicles, generic news pages, and PDFs unless no stronger company evidence exists.
- Seed URLs should be high-confidence competitor/company homepages only, and should not point back to the seed company itself or its subdomains when searching for similar companies.
- Include must_exclude_terms only when they are clearly relevant to the market and query family. Always exclude generic low-signal pages like careers/jobs when useful.
- Return JSON only.
"""


def _parse_discovery_query_plan(
    raw_text: str,
    fallback_plan: dict[str, Any],
) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    payload: Any = None
    if text:
        try:
            payload = json.loads(text)
        except Exception:
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    payload = json.loads(text[start_idx:end_idx])
                except Exception:
                    payload = None
    if not isinstance(payload, dict):
        return fallback_plan

    precision_queries = _normalize_query_entries(payload.get("precision_queries"), max_items=8)
    recall_queries = _normalize_query_entries(payload.get("recall_queries"), max_items=8)
    plan = {
        "precision_queries": precision_queries,
        "recall_queries": recall_queries,
        "seed_urls": _normalize_string_list(payload.get("seed_urls"), max_items=8, max_len=220),
        "must_include_terms": _normalize_string_list(payload.get("must_include_terms"), max_items=8, max_len=60),
        "must_exclude_terms": _normalize_string_list(payload.get("must_exclude_terms"), max_items=8, max_len=60),
        "preferred_countries": _normalize_string_list(payload.get("preferred_countries"), max_items=10, max_len=8),
        "preferred_languages": _normalize_string_list(payload.get("preferred_languages"), max_items=6, max_len=8),
        "domain_allowlist": _normalize_string_list(payload.get("domain_allowlist"), max_items=8, max_len=120),
        "domain_blocklist": _normalize_string_list(payload.get("domain_blocklist"), max_items=12, max_len=120),
    }
    if not plan["precision_queries"]:
        return fallback_plan
    return plan


def _query_is_seller_oriented(query_text: Any) -> bool:
    text = str(query_text or "").strip().lower()
    if not text:
        return False
    markers = (
        "software vendor",
        "platform provider",
        "software company",
        "saas vendor",
        "alternatives",
        "companies like",
        "competitor",
        "competitors",
        "best ",
        "wealthtech",
    )
    return any(marker in text for marker in markers)


def _scope_metadata_for_query_entry(
    entry: dict[str, Any],
    *,
    normalized_scope: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    scope = normalized_scope if isinstance(normalized_scope, dict) else {}
    brick_name = str(entry.get("brick_name") or "").strip()
    query_text = str(entry.get("query_text") or "").strip()
    query_intent = str(entry.get("query_intent") or "").strip()
    scope_bucket = str(entry.get("scope_bucket") or "").strip().lower() or None
    search_text = " | ".join(item for item in [brick_name, query_text, query_intent] if item)

    fit_to_adjacency_box_ids: list[str] = []
    fit_to_adjacency_box_labels: list[str] = []
    source_capability_matches: list[str] = []
    if search_text:
        for raw_box in (scope.get("adjacency_boxes") or []):
            if not isinstance(raw_box, dict):
                continue
            box_label = str(raw_box.get("label") or "").strip()
            if not box_label:
                continue
            candidate_terms = [box_label]
            candidate_terms.extend(
                _normalize_string_list(raw_box.get("likely_workflows"), max_items=4, max_len=120)
            )
            candidate_terms.extend(
                _normalize_string_list(raw_box.get("retrieval_query_seeds"), max_items=4, max_len=180)
            )
            if any(_scope_match_text(search_text, term) for term in candidate_terms if str(term).strip()):
                box_id = str(raw_box.get("id") or "").strip()
                if box_id:
                    fit_to_adjacency_box_ids.append(box_id)
                fit_to_adjacency_box_labels.append(box_label)
        if not fit_to_adjacency_box_labels:
            for capability in _normalize_string_list(scope.get("source_capabilities"), max_items=12, max_len=140):
                if _scope_match_text(search_text, capability):
                    source_capability_matches.append(capability)

    fit_to_adjacency_box_ids = _dedupe_strings(fit_to_adjacency_box_ids)[:8]
    fit_to_adjacency_box_labels = _dedupe_strings(fit_to_adjacency_box_labels)[:8]
    source_capability_matches = _dedupe_strings(source_capability_matches)[:8]
    if not scope_bucket:
        scope_bucket = "adjacent" if fit_to_adjacency_box_ids or fit_to_adjacency_box_labels else ("core" if source_capability_matches else None)
    return {
        "scope_bucket": scope_bucket,
        "fit_to_adjacency_box_ids": fit_to_adjacency_box_ids,
        "fit_to_adjacency_box_labels": fit_to_adjacency_box_labels,
        "source_capability_matches": source_capability_matches,
    }


def _merge_query_entries(*entry_sets: list[dict[str, Any]], max_items: int = 8) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entries in entry_sets:
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            query_text = str(entry.get("query_text") or "").strip()
            if not query_text:
                continue
            key = query_text.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "query_text": query_text[:220],
                    "query_intent": str(entry.get("query_intent") or "").strip() or None,
                    "brick_name": str(entry.get("brick_name") or "").strip() or None,
                    "scope_bucket": str(entry.get("scope_bucket") or "").strip().lower() or None,
                }
            )
            if len(merged) >= max_items:
                return merged
    return merged


def _stabilize_discovery_query_plan(
    plan: dict[str, Any],
    fallback_plan: dict[str, Any],
    *,
    normalized_scope: Optional[dict[str, Any]] = None,
    vertical_focus: Optional[list[str]] = None,
    brick_names: Optional[list[str]] = None,
    source_company_url: Optional[str] = None,
) -> tuple[dict[str, Any], list[str]]:
    stabilized = {
        "precision_queries": _normalize_query_entries(plan.get("precision_queries"), max_items=8),
        "recall_queries": _normalize_query_entries(plan.get("recall_queries"), max_items=8),
        "seed_urls": _normalize_string_list(plan.get("seed_urls"), max_items=12, max_len=220),
        "must_include_terms": _normalize_string_list(plan.get("must_include_terms"), max_items=4, max_len=60),
        "must_exclude_terms": _normalize_string_list(plan.get("must_exclude_terms"), max_items=8, max_len=60),
        "preferred_countries": _normalize_string_list(plan.get("preferred_countries"), max_items=10, max_len=8),
        "preferred_languages": _normalize_string_list(plan.get("preferred_languages"), max_items=6, max_len=8),
        "domain_allowlist": _normalize_string_list(plan.get("domain_allowlist"), max_items=8, max_len=120),
        "domain_blocklist": _normalize_string_list(plan.get("domain_blocklist"), max_items=12, max_len=120),
    }
    fallback = {
        "precision_queries": _normalize_query_entries(fallback_plan.get("precision_queries"), max_items=8),
        "recall_queries": _normalize_query_entries(fallback_plan.get("recall_queries"), max_items=8),
        "seed_urls": _normalize_string_list(fallback_plan.get("seed_urls"), max_items=12, max_len=220),
        "must_include_terms": _normalize_string_list(fallback_plan.get("must_include_terms"), max_items=4, max_len=60),
        "must_exclude_terms": _normalize_string_list(fallback_plan.get("must_exclude_terms"), max_items=8, max_len=60),
        "preferred_countries": _normalize_string_list(fallback_plan.get("preferred_countries"), max_items=10, max_len=8),
        "preferred_languages": _normalize_string_list(fallback_plan.get("preferred_languages"), max_items=6, max_len=8),
        "domain_allowlist": _normalize_string_list(fallback_plan.get("domain_allowlist"), max_items=8, max_len=120),
        "domain_blocklist": _normalize_string_list(fallback_plan.get("domain_blocklist"), max_items=12, max_len=120),
    }
    adjustments: list[str] = []
    market_hint = _market_scan_hint(
        normalized_scope or {},
        [str(v).strip() for v in (vertical_focus or []) if str(v).strip()],
        [str(v).strip() for v in (brick_names or []) if str(v).strip()],
        _adjacency_lane_candidates(normalized_scope or {}),
    ).lower()
    source_company_name = str(_company_label_from_url(source_company_url) or "").strip()

    if stabilized["domain_allowlist"]:
        stabilized["domain_allowlist"] = []
        adjustments.append("dropped_domain_allowlist")

    precision_seller_count = sum(
        1 for entry in stabilized["precision_queries"] if _query_is_seller_oriented(entry.get("query_text"))
    )
    recall_seller_count = sum(
        1 for entry in stabilized["recall_queries"] if _query_is_seller_oriented(entry.get("query_text"))
    )

    if len(stabilized["precision_queries"]) < 4 or precision_seller_count < 2:
        stabilized["precision_queries"] = _merge_query_entries(
            fallback["precision_queries"],
            stabilized["precision_queries"],
            max_items=8,
        )
        adjustments.append("fallback_precision_queries")

    if len(stabilized["recall_queries"]) < 4 or recall_seller_count < 2:
        stabilized["recall_queries"] = _merge_query_entries(
            fallback["recall_queries"],
            stabilized["recall_queries"],
            max_items=8,
        )
        adjustments.append("fallback_recall_queries")

    required_precision_families = {"category_vendor", "competitor_direct"}
    required_recall_families = {"alternatives", "local_market", "comparative_source"}

    present_precision_families = {
        normalize_discovery_query_family(
            entry.get("query_family") or entry.get("query_intent"),
            query_text=entry.get("query_text"),
        )
        for entry in stabilized["precision_queries"]
    }
    present_recall_families = {
        normalize_discovery_query_family(
            entry.get("query_family") or entry.get("query_intent"),
            query_text=entry.get("query_text"),
        )
        for entry in stabilized["recall_queries"]
    }

    if not required_precision_families.issubset(present_precision_families):
        stabilized["precision_queries"] = _merge_query_entries(
            [entry for entry in fallback["precision_queries"] if normalize_discovery_query_family(entry.get("query_family") or entry.get("query_intent"), query_text=entry.get("query_text")) in required_precision_families],
            stabilized["precision_queries"],
            max_items=10,
        )
        adjustments.append("restore_precision_family_mix")

    if not required_recall_families.issubset(present_recall_families):
        stabilized["recall_queries"] = _merge_query_entries(
            [entry for entry in fallback["recall_queries"] if normalize_discovery_query_family(entry.get("query_family") or entry.get("query_intent"), query_text=entry.get("query_text")) in required_recall_families],
            stabilized["recall_queries"],
            max_items=12,
        )
        adjustments.append("restore_recall_family_mix")

    if source_company_name:
        source_company_query_present = any(
            source_company_name.lower() in str(entry.get("query_text") or "").strip().lower()
            for entry in stabilized["precision_queries"] + stabilized["recall_queries"]
        )
        if not source_company_query_present:
            stabilized["precision_queries"] = _merge_query_entries(
                [
                    entry
                    for entry in fallback["precision_queries"]
                    if source_company_name.lower() in str(entry.get("query_text") or "").strip().lower()
                ],
                stabilized["precision_queries"],
                max_items=10,
            )
            stabilized["recall_queries"] = _merge_query_entries(
                [
                    entry
                    for entry in fallback["recall_queries"]
                    if source_company_name.lower() in str(entry.get("query_text") or "").strip().lower()
                ],
                stabilized["recall_queries"],
                max_items=12,
            )
            adjustments.append("restore_source_company_queries")

    stabilized["seed_urls"] = _filter_discovery_seed_urls(
        _dedupe_strings(stabilized["seed_urls"] + fallback["seed_urls"])
    )[:12]
    stabilized["must_include_terms"] = _dedupe_strings(
        stabilized["must_include_terms"] + fallback["must_include_terms"]
    )[:4]
    stabilized["must_exclude_terms"] = _dedupe_strings(
        stabilized["must_exclude_terms"] + fallback["must_exclude_terms"]
    )[:8]
    if market_hint != "wealthtech":
        filtered_excludes = [
            term
            for term in stabilized["must_exclude_terms"]
            if str(term).strip().lower() not in WEALTH_DISCOVERY_EXCLUDE_TERMS
        ]
        if len(filtered_excludes) != len(stabilized["must_exclude_terms"]):
            stabilized["must_exclude_terms"] = filtered_excludes
            adjustments.append("dropped_wealth_excludes")
    stabilized["preferred_countries"] = _dedupe_strings(
        stabilized["preferred_countries"] + fallback["preferred_countries"]
    )[:10]
    stabilized["preferred_languages"] = _dedupe_strings(
        stabilized["preferred_languages"] + fallback["preferred_languages"]
    )[:6]
    stabilized["domain_blocklist"] = _dedupe_strings(
        stabilized["domain_blocklist"] + fallback["domain_blocklist"]
    )[:12]

    return stabilized, _dedupe_strings(adjustments)


def _build_external_search_queries_from_plan(
    plan: dict[str, Any],
    *,
    preferred_countries: Optional[list[str]] = None,
    preferred_languages: Optional[list[str]] = None,
    normalized_scope: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    precision_entries = plan.get("precision_queries") or []
    recall_entries = plan.get("recall_queries") or []
    include_terms = plan.get("must_include_terms") or []
    exclude_terms = plan.get("must_exclude_terms") or []
    allowlist = plan.get("domain_allowlist") or []
    blocklist = plan.get("domain_blocklist") or []

    country_hint = " ".join([str(c) for c in (preferred_countries or []) if str(c).strip()])
    language_hint = " ".join([str(c) for c in (preferred_languages or []) if str(c).strip()])
    hint_suffix = " ".join([country_hint, language_hint]).strip()
    lightweight_query_families = {
        "competitor_direct",
        "alternatives",
        "comparative_source",
        "traffic_affinity",
        "peer_expansion",
        "local_market",
    }
    lightweight_exclude_terms = [
        term
        for term in exclude_terms
        if str(term).strip().lower() in {"career", "careers", "job", "jobs"}
    ][:4]

    def _add_entries(entries: list[dict[str, Any]], query_type: str) -> None:
        for idx, entry in enumerate(entries, start=1):
            query_text = str(entry.get("query_text") or "").strip()
            query_family = normalize_discovery_query_family(
                entry.get("query_family") or entry.get("query_intent"),
                query_text=query_text,
            )
            if hint_suffix and query_family not in lightweight_query_families:
                query_text = f"{query_text} {hint_suffix}".strip()
            scope_metadata = _scope_metadata_for_query_entry(entry, normalized_scope=normalized_scope)
            family_include_terms = [] if query_family in lightweight_query_families else include_terms
            family_exclude_terms = lightweight_exclude_terms if query_family in lightweight_query_families else exclude_terms
            queries.append(
                {
                    "query_id": f"{query_type}_{idx}",
                    "query_text": query_text,
                    "query_type": query_type,
                    "query_intent": entry.get("query_intent") or query_family,
                    "query_family": query_family,
                    "brick_name": entry.get("brick_name"),
                    "scope_bucket": scope_metadata.get("scope_bucket"),
                    "fit_to_adjacency_box_ids": scope_metadata.get("fit_to_adjacency_box_ids") or [],
                    "fit_to_adjacency_box_labels": scope_metadata.get("fit_to_adjacency_box_labels") or [],
                    "source_capability_matches": scope_metadata.get("source_capability_matches") or [],
                    "must_include_terms": family_include_terms,
                    "must_exclude_terms": family_exclude_terms,
                    "domain_allowlist": allowlist,
                    "domain_blocklist": blocklist,
                }
            )

    _add_entries(precision_entries, "precision")
    _add_entries(recall_entries, "recall")

    summary = {
        "precision_queries": len(precision_entries),
        "recall_queries": len(recall_entries),
        "scope_buckets": _dedupe_strings(
            [
                str(entry.get("scope_bucket") or "")
                for entry in precision_entries + recall_entries
                if str(entry.get("scope_bucket") or "").strip()
            ]
        ),
        "query_families": _dedupe_strings(
            [
                normalize_discovery_query_family(
                    entry.get("query_family") or entry.get("query_intent"),
                    query_text=entry.get("query_text"),
                )
                for entry in precision_entries + recall_entries
            ]
        ),
        "must_include_terms": include_terms,
        "must_exclude_terms": exclude_terms,
        "domain_allowlist": allowlist,
        "domain_blocklist": blocklist,
    }
    return queries, summary


def _known_discovery_blocked_domains(
    profile: CompanyProfile,
    seed_urls: list[str],
) -> list[str]:
    blocked: list[str] = []

    def register(raw_url: Any) -> None:
        normalized = normalize_url(raw_url or "")
        domain = normalize_domain(normalized)
        if not domain or _is_non_first_party_profile_domain(domain):
            return
        if _is_known_third_party_context_domain(domain):
            return
        if domain in blocked:
            return
        blocked.append(domain)

    register(profile.buyer_company_url)
    for url in profile.comparator_seed_urls or []:
        register(url)
    for url in seed_urls or []:
        register(url)

    return blocked[:12]


def _known_entity_suppression_profile(
    profile: CompanyProfile | None,
    *,
    existing_companies: list[Any] | None = None,
) -> dict[str, Any]:
    blocked_domains: set[str] = set()
    blocked_registry_ids: set[str] = set()
    exact_name_norms: set[str] = set()
    phrase_names: set[str] = set()

    def _read(item: Any, key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    def register_name(raw_name: Any) -> None:
        name = str(raw_name or "").strip()
        if not name:
            return
        norm = _normalize_name_for_matching(name)
        if norm:
            exact_name_norms.add(norm)
        phrase = _normalize_name_phrase(name)
        if len(phrase) >= 4:
            phrase_names.add(phrase)

    def register_registry_id(raw_value: Any) -> None:
        value = str(raw_value or "").strip()
        if value:
            blocked_registry_ids.add(value)

    def register_domain(raw_domain: Any) -> None:
        domain = normalize_domain(raw_domain or "")
        if not domain:
            return
        if (
            _is_non_first_party_profile_domain(domain)
            or _is_registry_profile_domain(domain)
            or _is_aggregator_domain(domain)
            or _is_known_third_party_context_domain(domain)
        ):
            return
        blocked_domains.add(domain)
        first_label = domain.split(".")[0].replace("-", " ").replace("_", " ").strip()
        if first_label:
            register_name(first_label)

    def register_url(raw_url: Any) -> None:
        normalized = normalize_url(raw_url or "")
        if not normalized:
            return
        register_domain(normalized)

    if profile:
        register_url(profile.buyer_company_url)
        for url in profile.comparator_seed_urls or []:
            register_url(url)

    for company in existing_companies or []:
        register_name(_read(company, "name"))
        register_url(_read(company, "website"))

    return {
        "blocked_domains": sorted(blocked_domains),
        "blocked_registry_ids": sorted(blocked_registry_ids),
        "exact_name_norms": sorted(exact_name_norms),
        "phrase_names": sorted(phrase_names, key=len, reverse=True),
    }


def _known_entity_suppression_reason(candidate: dict[str, Any], profile: dict[str, Any]) -> Optional[str]:
    blocked_domains = set(profile.get("blocked_domains") or [])
    blocked_registry_ids = set(profile.get("blocked_registry_ids") or [])
    exact_name_norms = set(profile.get("exact_name_norms") or [])
    phrase_names = [str(item) for item in (profile.get("phrase_names") or []) if str(item).strip()]

    candidate_registry_id = _registry_identifier(candidate)
    if candidate_registry_id and candidate_registry_id in blocked_registry_ids:
        return "known_entity_registry_id"

    candidate_domains = _normalize_domain_list(
        [candidate.get("website"), candidate.get("official_website_url"), candidate.get("discovery_url"), candidate.get("profile_url")]
        + list(candidate.get("first_party_domains") or [])
    )
    if any(domain in blocked_domains for domain in candidate_domains):
        return "known_entity_domain"

    candidate_name = str(candidate.get("name") or "").strip()
    candidate_name_norm = _normalize_name_for_matching(candidate_name)
    if candidate_name_norm and candidate_name_norm in exact_name_norms:
        return "known_entity_name_exact"

    if not candidate_name:
        return None

    candidate_name_phrase = _normalize_name_phrase(candidate_name)
    if not candidate_name_phrase:
        return None

    first_party_candidate_domains = [
        domain
        for domain in candidate_domains
        if not _is_non_first_party_profile_domain(domain)
        and not _is_registry_profile_domain(domain)
        and not _is_aggregator_domain(domain)
        and not _is_known_third_party_context_domain(domain)
    ]
    if first_party_candidate_domains:
        return None

    for phrase in phrase_names:
        if len(phrase) < 4:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", candidate_name_phrase):
            return "known_entity_third_party_page"
    return None


def _suppress_known_entity_candidates(
    candidates: list[dict[str, Any]],
    suppression_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    stats = {
        "dropped_count": 0,
        "known_entity_domain": 0,
        "known_entity_registry_id": 0,
        "known_entity_name_exact": 0,
        "known_entity_third_party_page": 0,
        "examples": [],
    }
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        reason = _known_entity_suppression_reason(candidate, suppression_profile)
        if not reason:
            kept.append(candidate)
            continue
        stats["dropped_count"] += 1
        stats[reason] = int(stats.get(reason) or 0) + 1
        if len(stats["examples"]) < 10:
            stats["examples"].append(
                {
                    "name": str(candidate.get("name") or "").strip(),
                    "website": str(candidate.get("website") or candidate.get("discovery_url") or "").strip() or None,
                    "reason": reason,
                }
            )
    return kept, stats


def _domain_label_for_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return "Unknown"
        return host.split(".")[0].replace("-", " ").title()
    except Exception:
        return "Unknown"


def _brand_key(text: str) -> str:
    lowered = str(text or "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", lowered)


DISCOVERY_GENERIC_TITLE_TOKENS = {
    "advanced",
    "best",
    "care",
    "clinic",
    "company",
    "ehr",
    "employee",
    "eu",
    "europe",
    "france",
    "belgium",
    "germany",
    "health",
    "healthcare",
    "hospital",
    "hospitals",
    "hr",
    "internal",
    "management",
    "mobility",
    "patient",
    "patients",
    "payroll",
    "perfect",
    "planning",
    "platform",
    "pool",
    "provider",
    "providers",
    "replacement",
    "roster",
    "rostering",
    "schedule",
    "scheduling",
    "shift",
    "software",
    "solution",
    "solutions",
    "staff",
    "staffing",
    "system",
    "systems",
    "time",
    "timesheet",
    "vendor",
    "vms",
    "workforce",
}

DISCOVERY_DOMAIN_BRAND_PREFIXES = ("get", "go", "use", "try", "meet", "hello", "my", "whatis")


def _brand_key_variants(text: str) -> set[str]:
    variants = {_brand_key(text)}
    base = next(iter(variants))
    for prefix in DISCOVERY_DOMAIN_BRAND_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix) + 3:
            variants.add(base[len(prefix):])
    return {item for item in variants if item}


def _looks_like_generic_solution_title(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    informative = [token for token in tokens if token not in {"the", "and", "for", "in", "of", "to", "a", "an"}]
    if len(informative) < 4:
        return False
    generic_hits = sum(1 for token in informative if token in DISCOVERY_GENERIC_TITLE_TOKENS)
    return generic_hits >= max(3, len(informative) // 2 + 1)


def _looks_like_reasonable_brand_name(text: str) -> bool:
    normalized = str(text or "").strip(" -|:")
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) > 5:
        return False
    lowered = normalized.lower()
    if _looks_like_generic_solution_title(normalized):
        return False
    if lowered in {"home", "homepage", "platform", "software", "solution", "solutions", "company"}:
        return False
    return any(char.isalpha() for char in normalized)


def _extract_brand_name_hint_from_reasons(
    reasons: list[dict[str, Any]],
    website_url: str,
) -> Optional[str]:
    explicit_brand_prefix = "Brand identity:"
    domain_label = _domain_label_for_url(website_url)
    domain_variants = _brand_key_variants(domain_label)
    for reason in reasons:
        text = str(reason.get("text") or "").strip()
        if text.startswith(explicit_brand_prefix):
            brand = text.split(":", 1)[1].strip()
            if _looks_like_reasonable_brand_name(brand):
                return brand[:300]
    for reason in reasons:
        text = str(reason.get("text") or "").strip()
        for segment in [part.strip(" -|:") for part in re.split(r"[|:-]", text) if part.strip(" -|:")]:
            if not _looks_like_reasonable_brand_name(segment):
                continue
            segment_variants = _brand_key_variants(segment)
            if domain_variants & segment_variants:
                return segment[:300]
            if any(segment_key in domain_variants or any(domain_key in segment_key for domain_key in domain_variants) for segment_key in segment_variants):
                return segment[:300]
    return None


def _should_replace_candidate_name_with_brand_hint(
    current_name: str,
    brand_hint: Optional[str],
    website_url: str,
) -> bool:
    brand = str(brand_hint or "").strip()
    current = str(current_name or "").strip()
    if not brand or not current:
        return False
    if _brand_key(current) == _brand_key(brand):
        return False
    domain_label = _domain_label_for_url(website_url)
    current_variants = _brand_key_variants(current)
    domain_variants = _brand_key_variants(domain_label)
    if _looks_like_generic_solution_title(current):
        return True
    if current_variants & domain_variants:
        return True
    return False


def _external_candidate_name(title: Any, url: str) -> str:
    domain_label = _domain_label_for_url(url)
    domain_variants = _brand_key_variants(domain_label)
    raw_title = str(title or "").strip()
    if not raw_title:
        return domain_label
    normalized = raw_title.replace("\n", " ").strip()
    lower = normalized.lower()
    profile_match = re.match(r"^\s*([^|:\-]{2,80}?)\s*[-|:]\s*(?:\d{4}\s+)?company profile\b", normalized, flags=re.I)
    if profile_match:
        leading_name = str(profile_match.group(1) or "").strip()
        if leading_name and _brand_key(leading_name) not in domain_variants:
            return leading_name[:300]
    domain_lower = domain_label.lower()
    generic_segment_tokens = (
        "software",
        "platform",
        "solution",
        "solutions",
        "report",
        "review",
        "asset management",
        "wealth management",
        "portfolio management",
        "service",
        "services",
    )
    article_markers = (
        " review ",
        " report",
        " matrix",
        " best ",
        " page ",
        " introduces ",
        " breaks new ground ",
        " pioneers ",
        " disclosure",
    )
    if any(marker in f" {lower} " for marker in article_markers):
        return domain_label
    if _looks_like_generic_solution_title(normalized) and all(variant not in _brand_key(normalized) for variant in domain_variants):
        return domain_label
    segments = [segment.strip(" -|:") for segment in re.split(r"[|:-]", normalized) if segment.strip(" -|:")]
    for segment in segments:
        segment_variants = _brand_key_variants(segment)
        segment_lower = segment.lower()
        if segment_lower.endswith(" home") and any(domain_key in segment_key for domain_key in domain_variants for segment_key in segment_variants):
            return domain_label
        if any(domain_key in segment_key for domain_key in domain_variants for segment_key in segment_variants):
            if len(segment.split()) > 5:
                return domain_label
            return segment[:300]
    for segment in segments:
        segment_lower = segment.lower()
        if len(segment.split()) <= 4 and not any(token in segment_lower for token in generic_segment_tokens):
            return segment[:300]
    normalized_key = _brand_key(normalized)
    if domain_lower and lower.startswith(f"{domain_lower} ") and len(normalized.split()) >= 5:
        return domain_label
    if any(variant in normalized_key for variant in domain_variants) and len(normalized.split()) >= 6:
        return domain_label
    if domain_lower and domain_lower not in lower and any(marker in lower for marker in (" for ", " with ", " by ")) and len(normalized.split()) > 4:
        return domain_label
    if domain_lower and domain_lower not in lower and any(token in lower for token in (" software", " platform", " solution")):
        return domain_label
    if "|" in normalized:
        parts = [part.strip() for part in normalized.split("|") if part.strip()]
        for part in parts:
            if domain_lower and domain_lower in part.lower():
                return part[:300]
        if parts and len(parts[0].split()) > 8:
            return domain_label
    if domain_lower and domain_lower not in lower and len(normalized.split()) > 8:
        return domain_label
    return normalized[:300]


def _is_third_party_profile_result(row: dict[str, Any]) -> bool:
    title = str(row.get("title") or "").strip()
    if not title:
        return False
    lowered = title.lower()
    if "company profile" in lowered or "company overview" in lowered or "organization profile" in lowered:
        return True
    url = normalize_url(row.get("normalized_url") or row.get("url") or "")
    path = str(urlparse(url).path or "").lower()
    return any(token in path for token in ("/companies/", "/company/", "/organization/")) and any(
        marker in lowered for marker in ("profile", "overview", "company")
    )


def _looks_like_vendor_candidate_result(row: dict[str, Any]) -> bool:
    url = str(row.get("normalized_url") or row.get("url") or "").strip()
    if not url:
        return False
    normalized_url = normalize_url(url) or url
    domain = normalize_domain(normalized_url)
    if not domain:
        return False
    if _is_non_first_party_profile_domain(domain):
        return False
    if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in DISCOVERY_NON_VENDOR_HOSTS):
        return False

    title = str(row.get("title") or "").strip().lower()
    snippet = str(row.get("snippet") or "").strip().lower()
    parsed = urlparse(normalized_url)
    path = str(parsed.path or "").strip().lower()
    host = str(parsed.netloc or "").strip().lower()
    if any(path.endswith(suffix) for suffix in DISCOVERY_NON_VENDOR_FILE_SUFFIXES):
        return False
    if any(token in path for token in DISCOVERY_NON_VENDOR_PATH_TOKENS):
        return False
    if domain.endswith((".gov", ".gov.uk", ".gouv.fr", ".edu", ".edu.au", ".ac.uk", ".nhs.uk")):
        return False
    combined = " ".join(part for part in (title, snippet, path) if part).strip()
    domain_label = _domain_label_for_url(normalized_url).lower()
    article_like_path = any(token in path for token in DISCOVERY_ARTICLE_PATH_TOKENS)
    domain_label_tokens = {token for token in re.split(r"[^a-z0-9]+", domain_label) if token}
    host_tokens = {token for token in re.split(r"[^a-z0-9]+", host) if token}

    has_vendor_signal = any(token in combined for token in DISCOVERY_VENDORISH_TOKENS)
    has_strong_vendor_signal = any(token in combined for token in DISCOVERY_STRONG_VENDOR_TOKENS)
    has_specific_strong_vendor_signal = any(
        token in combined for token in (DISCOVERY_STRONG_VENDOR_TOKENS - DISCOVERY_GENERIC_VENDOR_TOKENS)
    )
    has_editorial_signal = any(token in f" {combined} " for token in DISCOVERY_EDITORIAL_TOKENS)
    has_institution_signal = any(token in f" {combined} " for token in DISCOVERY_INSTITUTION_TOKENS)
    has_service_provider_signal = any(token in f" {combined} " for token in DISCOVERY_SERVICE_PROVIDER_TOKENS)
    has_investor_signal = any(token in f" {combined} " for token in DISCOVERY_INVESTOR_TOKENS)
    has_non_vendor_domain_label = bool(
        (domain_label_tokens | host_tokens) & DISCOVERY_NON_VENDOR_DOMAIN_LABEL_TOKENS
    ) or any(token in domain_label or token in host for token in DISCOVERY_NON_VENDOR_DOMAIN_LABEL_TOKENS)
    path_depth = len([segment for segment in path.split("/") if segment])

    if has_editorial_signal and not has_strong_vendor_signal:
        return False
    if has_institution_signal and not has_strong_vendor_signal:
        return False
    if has_investor_signal and not has_strong_vendor_signal:
        return False
    if has_service_provider_signal and not has_specific_strong_vendor_signal:
        return False
    if (
        article_like_path
        and any(token in combined for token in ("bootcamp", "academy", "course", "tutorial", "community"))
        and not has_specific_strong_vendor_signal
    ):
        return False
    if has_non_vendor_domain_label and (article_like_path or has_editorial_signal):
        return False
    if (
        article_like_path
        and not has_specific_strong_vendor_signal
        and domain_label
        and domain_label not in title
        and len(title.split()) >= 5
    ):
        return False
    if not has_strong_vendor_signal and any(token in combined for token in ("journal", "study", "report", "conference", "abstract", "article", "pdf")):
        return False
    if path_depth >= 2 and not has_vendor_signal and domain_label and domain_label not in title:
        return False
    if not has_vendor_signal and domain_label and domain_label not in title and len(title.split()) > 8:
        return False
    return True


def _adaptive_hint_discovery_enabled(
    *,
    first_party_enrichment_enabled: bool,
    first_party_hint_crawl_budget_default: int,
    first_party_crawl_budget_default: int,
) -> bool:
    if not first_party_enrichment_enabled:
        return False
    return first_party_hint_crawl_budget_default > 0 or first_party_crawl_budget_default > 0


def _candidates_from_retrieval_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in results:
        url = str(row.get("normalized_url") or row.get("url") or "").strip()
        if not url:
            continue
        if not _looks_like_vendor_candidate_result(row):
            continue
        normalized_url = normalize_url(url)
        parsed = urlparse(normalized_url or url)
        domain = normalize_domain(normalized_url or url)
        if domain and _is_non_first_party_profile_domain(domain):
            continue
        if domain and any(domain == blocked or domain.endswith(f".{blocked}") for blocked in DISCOVERY_NON_VENDOR_HOSTS):
            continue
        third_party_profile = _is_third_party_profile_result(row)
        homepage_url = (
            f"{parsed.scheme or 'https'}://{parsed.netloc}"
            if parsed.netloc
            else (normalized_url or url)
        )
        official_website_url = None if third_party_profile else homepage_url
        discovery_url = normalized_url or url
        candidates.append(
            {
                "name": _external_candidate_name(row.get("title"), url),
                "website": official_website_url,
                "official_website_url": official_website_url,
                "discovery_url": discovery_url,
                "entity_type": "company",
                "first_party_domains": [domain] if domain and not third_party_profile else [],
                "hq_country": "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
                "why_relevant": [
                    {
                        "text": str(row.get("snippet") or "Externally discovered candidate signal.")[:400],
                        "citation_url": discovery_url,
                        "dimension": "external_search_seed",
                        "source_kind": "external_search_snippet",
                        "provider": row.get("provider"),
                        "query_id": row.get("query_id"),
                        "query_intent": row.get("query_intent"),
                        "query_family": row.get("query_family"),
                        "scope_bucket": row.get("scope_bucket"),
                        "fit_to_adjacency_box_ids": row.get("fit_to_adjacency_box_ids") or [],
                        "fit_to_adjacency_box_labels": row.get("fit_to_adjacency_box_labels") or [],
                        "source_capability_matches": row.get("source_capability_matches") or [],
                        "rank": row.get("rank"),
                    }
                ],
                "_origins": [
                    {
                        "origin_type": "external_search_seed",
                        "origin_url": discovery_url,
                        "source_name": row.get("provider"),
                        "source_run_id": None,
                        "metadata": {
                            "query_id": row.get("query_id"),
                            "query_type": row.get("query_type"),
                            "query_intent": row.get("query_intent"),
                            "query_family": row.get("query_family"),
                            "query_text": row.get("query_text"),
                            "brick_name": row.get("brick_name"),
                            "scope_bucket": row.get("scope_bucket"),
                            "fit_to_adjacency_box_ids": row.get("fit_to_adjacency_box_ids") or [],
                            "fit_to_adjacency_box_labels": row.get("fit_to_adjacency_box_labels") or [],
                            "source_capability_matches": row.get("source_capability_matches") or [],
                            "rank": row.get("rank"),
                            "provider": row.get("provider"),
                            "source_family": normalize_discovery_source_family(
                                "external_search_seed",
                                {
                                    "query_family": row.get("query_family"),
                                    "query_type": row.get("query_type"),
                                    "provider": row.get("provider"),
                                },
                                discovery_url,
                            ),
                        },
                    }
                ],
            }
        )
    return candidates


def _candidate_validation_lane_metadata(
    candidate: dict[str, Any],
    *,
    scope_buckets: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    origin_items = candidate.get("origins")
    if not isinstance(origin_items, list):
        origin_items = _origin_entries(candidate)
    lane_ids: list[str] = []
    lane_labels: list[str] = []
    source_caps: list[str] = []
    brick_labels: list[str] = []

    for origin in origin_items:
        if not isinstance(origin, dict):
            continue
        metadata = origin.get("metadata") if isinstance(origin.get("metadata"), dict) else {}
        lane_ids.extend(_dedupe_strings(metadata.get("fit_to_adjacency_box_ids") or []))
        lane_labels.extend(_dedupe_strings(metadata.get("fit_to_adjacency_box_labels") or []))
        source_caps.extend(_dedupe_strings(metadata.get("source_capability_matches") or []))
        brick_name = str(metadata.get("brick_name") or "").strip()
        if brick_name:
            brick_labels.append(brick_name)

    lane_ids = _dedupe_strings(lane_ids)[:8]
    lane_labels = _dedupe_strings(lane_labels)[:8]
    source_caps = _dedupe_strings(source_caps)[:8]
    brick_labels = _dedupe_strings(brick_labels)[:8]
    scope_bucket_labels = [
        str(bucket).strip().title()
        for bucket in _dedupe_strings(scope_buckets or [])
        if str(bucket).strip()
    ]

    brick_lane_ids = [
        _normalize_name_phrase(label).replace(" ", "_")
        for label in brick_labels
        if _normalize_name_phrase(label)
    ]
    final_lane_ids = lane_ids or source_caps or brick_lane_ids or _dedupe_strings(scope_buckets or [])[:8]
    final_lane_labels = lane_labels or source_caps or brick_labels or scope_bucket_labels
    return _dedupe_strings(final_lane_ids)[:8], _dedupe_strings(final_lane_labels)[:8]


def _candidate_origin_discovery_families(origin_items: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    query_families: list[str] = []
    source_families: list[str] = []
    for origin in origin_items or []:
        if not isinstance(origin, dict):
            continue
        metadata = origin.get("metadata") if isinstance(origin.get("metadata"), dict) else {}
        query_families.append(
            normalize_discovery_query_family(
                metadata.get("query_family") or metadata.get("query_intent") or metadata.get("query_type"),
                query_text=metadata.get("query_text"),
                provider=metadata.get("provider"),
            )
        )
        source_families.append(
            normalize_discovery_source_family(
                origin.get("origin_type"),
                metadata,
                origin.get("origin_url"),
            )
        )
    return _dedupe_strings(query_families)[:8], _dedupe_strings(source_families)[:8]


def _is_persistable_vendor_entity(entity: dict[str, Any]) -> bool:
    canonical_website = str(entity.get("canonical_website") or "").strip()
    discovery_url = str(entity.get("discovery_primary_url") or "").strip()
    canonical_domain = normalize_domain(canonical_website)
    discovery_domain = normalize_domain(discovery_url)
    origin_types = {
        str(origin_type or "").strip().lower()
        for origin_type in (entity.get("origin_types") or [])
        if str(origin_type or "").strip()
    }
    registry_id = str(entity.get("registry_id") or "").strip()
    registry_source = str(entity.get("registry_source") or "").strip().lower()
    registry_fields = entity.get("registry_fields") if isinstance(entity.get("registry_fields"), dict) else {}
    is_fr_registry_candidate = bool(
        registry_id
        and (
            "registry_fr_base" in origin_types
            or registry_source.startswith("fr_")
            or normalize_country(entity.get("registry_country")) == "FR"
            or bool(registry_fields)
        )
    )

    for domain in [canonical_domain]:
        if not domain:
            continue
        if _is_non_first_party_profile_domain(domain):
            return False
        if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in DISCOVERY_NON_VENDOR_HOSTS):
            return False
    if discovery_domain:
        if _is_non_first_party_profile_domain(discovery_domain):
            if not is_fr_registry_candidate:
                return False
        elif any(
            discovery_domain == blocked or discovery_domain.endswith(f".{blocked}")
            for blocked in DISCOVERY_NON_VENDOR_HOSTS
        ):
            return False

    parsed_discovery = urlparse(normalize_url(discovery_url) or discovery_url) if discovery_url else None
    discovery_path = str(parsed_discovery.path or "").strip().lower() if parsed_discovery else ""
    article_like_path = any(token in discovery_path for token in DISCOVERY_ARTICLE_PATH_TOKENS)

    evidence_text = " ".join(
        str(reason.get("text") or "").strip().lower()
        for reason in (entity.get("why_relevant") or [])
        if isinstance(reason, dict)
    )
    combined = " ".join(
        item
        for item in [
            str(entity.get("canonical_name") or "").strip().lower(),
            canonical_website.lower(),
            evidence_text,
        ]
        if item
    )
    has_specific_strong_vendor_signal = any(
        token in combined for token in (DISCOVERY_STRONG_VENDOR_TOKENS - DISCOVERY_GENERIC_VENDOR_TOKENS)
    )
    has_training_or_community_signal = any(
        token in combined for token in ("bootcamp", "academy", "course", "training", "tutorial", "community")
    )
    has_service_provider_signal = any(token in f" {combined} " for token in DISCOVERY_SERVICE_PROVIDER_TOKENS)
    has_institution_signal = any(token in f" {combined} " for token in DISCOVERY_INSTITUTION_TOKENS)
    has_investor_signal = any(token in f" {combined} " for token in DISCOVERY_INVESTOR_TOKENS)
    if article_like_path and has_training_or_community_signal and not has_specific_strong_vendor_signal:
        return False
    if has_service_provider_signal and not has_specific_strong_vendor_signal:
        return False
    if has_institution_signal and not has_specific_strong_vendor_signal:
        return False
    if has_investor_signal and not has_specific_strong_vendor_signal:
        return False
    return True


def _build_candidate_synthesis_prompt(
    retrieval_results: list[dict[str, Any]],
    context_pack: str,
    taxonomy_bricks: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    vertical_focus: list[str],
    scope_hints: Optional[dict[str, Any]] = None,
) -> str:
    normalized_scope = _normalize_scope_hints(scope_hints)
    region = str((geo_scope or {}).get("region") or "EU+UK")
    brick_names = [str(b.get("name") or "").strip() for b in (taxonomy_bricks or []) if str(b.get("name") or "").strip()]
    verticals = [str(v).strip() for v in (vertical_focus or []) if str(v).strip()]

    retrieval_context: list[dict[str, Any]] = []
    for row in retrieval_results[:80]:
        retrieval_context.append(
            {
                "url": row.get("normalized_url") or row.get("url"),
                "title": row.get("title"),
                "snippet": str(row.get("snippet") or "")[:300],
                "provider": row.get("provider"),
                "query_id": row.get("query_id"),
                "query_type": row.get("query_type"),
                "query_family": row.get("query_family") or row.get("query_intent"),
                "rank": row.get("rank"),
            }
        )

    return f"""You are an M&A research analyst. Use ONLY the retrieval_context URLs below.
Return ONLY a JSON array of B2B software companies.

Buyer context:
{context_pack[:1800] if context_pack else "No buyer context provided."}

Target filters:
- Region: {region}
- Approved scope hints: {json.dumps(normalized_scope, ensure_ascii=False) if normalized_scope else "none confirmed yet"}
- Legacy vertical hints: {", ".join(verticals) if verticals else "generic software"}
- Legacy capability hints: {", ".join(brick_names[:12]) if brick_names else "n/a"}

retrieval_context:
{json.dumps(retrieval_context, ensure_ascii=False)}

Output schema:
[
  {{
    "name": "Company Name",
    "website": "https://example.com",
    "discovery_url": "https://example.com/competitors-page",
    "hq_country": "DE",
    "likely_verticals": ["wealth_manager"],
    "employee_estimate": 120,
    "capability_signals": ["Portfolio management"],
    "qualification": {{"go_to_market": "b2b_enterprise", "target_customer": "asset_managers"}},
    "why_relevant": [
      {{
        "text":"Evidence text",
        "citation_url":"https://...",
        "dimension":"external_search_seed",
        "source_kind":"external_search_snippet",
        "provider":"exa",
        "query_id":"precision_1",
        "rank":1
      }}
    ]
  }}
]

Constraints:
- Every company must have at least one citation_url from retrieval_context.
- `website` is optional. Use it only if the official company site is clearly present in retrieval_context. Otherwise leave it null and keep the company as a discovery candidate.
- `discovery_url` should point to the retrieval-context page that mentioned or surfaced the company.
- Do not invent companies or URLs not present above.
- Prefer plausible software vendors and market-map candidates even if the evidence is comparative/list-source rather than first-party.
- It is acceptable to extract a company from a comparative/list article snippet when the company is clearly named there, even if the official company site is not in retrieval_context yet.
- Return JSON only.
"""


def _validate_closed_world_candidates(
    candidates: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    allowed_urls = {
        normalize_url(str(row.get("normalized_url") or row.get("url") or "").strip())
        for row in retrieval_results
        if normalize_url(str(row.get("normalized_url") or row.get("url") or "").strip())
    }
    retrieval_by_url: dict[str, dict[str, Any]] = {}
    for row in retrieval_results:
        url = normalize_url(str(row.get("normalized_url") or row.get("url") or "").strip())
        if not url:
            continue
        retrieval_by_url[url] = row

    stats = {
        "dropped_missing_url": 0,
        "dropped_missing_evidence": 0,
        "dropped_non_first_party_website": 0,
        "dropped_non_vendor_result": 0,
    }
    validated: list[dict[str, Any]] = []

    for candidate in candidates:
        website_raw = str(candidate.get("website") or "").strip()
        website = normalize_url(website_raw) if website_raw else ""
        discovery_url = normalize_url(
            str(candidate.get("discovery_url") or candidate.get("profile_url") or "").strip()
        ) if str(candidate.get("discovery_url") or candidate.get("profile_url") or "").strip() else ""
        provenance_url = discovery_url or website
        if not provenance_url or provenance_url not in allowed_urls:
            stats["dropped_missing_url"] += 1
            continue
        website_domain = normalize_domain(website)
        if website_domain and _is_non_first_party_profile_domain(website_domain):
            stats["dropped_non_first_party_website"] += 1
            website = ""
            website_domain = None

        reasons: list[dict[str, Any]] = []
        for item in candidate.get("why_relevant", []) or []:
            if not isinstance(item, dict):
                continue
            citation_url = str(item.get("citation_url") or "").strip()
            citation_url = normalize_url(citation_url) if citation_url else ""
            if not citation_url or citation_url not in allowed_urls:
                continue
            reason = dict(item)
            reason["citation_url"] = citation_url
            reason.setdefault("source_kind", "external_search_snippet")
            provenance = retrieval_by_url.get(citation_url)
            if provenance:
                reason.setdefault("provider", provenance.get("provider"))
                reason.setdefault("query_id", provenance.get("query_id"))
                reason.setdefault("rank", provenance.get("rank"))
            reasons.append(reason)

        if not reasons:
            provenance = retrieval_by_url.get(provenance_url)
            if provenance:
                reasons.append(
                    {
                        "text": str(provenance.get("snippet") or "Externally discovered candidate signal.")[:400],
                        "citation_url": provenance_url,
                        "dimension": "external_search_seed",
                        "source_kind": "external_search_snippet",
                        "provider": provenance.get("provider"),
                        "query_id": provenance.get("query_id"),
                        "query_intent": provenance.get("query_intent"),
                        "query_family": provenance.get("query_family"),
                        "rank": provenance.get("rank"),
                    }
                )

        if not reasons:
            stats["dropped_missing_evidence"] += 1
            continue

        candidate["website"] = website or None
        candidate["official_website_url"] = website or None
        candidate["discovery_url"] = provenance_url
        candidate["why_relevant"] = reasons
        candidate.setdefault("_origins", [])
        provenance = retrieval_by_url.get(provenance_url)
        if provenance and not _looks_like_vendor_candidate_result(provenance):
            provenance_query_family = normalize_discovery_query_family(
                provenance.get("query_family") or provenance.get("query_intent") or provenance.get("query_type"),
                query_text=provenance.get("query_text"),
                provider=provenance.get("provider"),
            )
            if provenance_query_family not in {
                "competitor_direct",
                "alternatives",
                "comparative_source",
                "traffic_affinity",
                "peer_expansion",
                "local_market",
            }:
                stats["dropped_non_vendor_result"] += 1
                continue
        if provenance:
            candidate["_origins"].append(
                {
                    "origin_type": "external_search_seed",
                    "origin_url": provenance_url,
                    "source_name": provenance.get("provider"),
                    "source_run_id": None,
                    "metadata": {
                        "query_id": provenance.get("query_id"),
                        "query_type": provenance.get("query_type"),
                        "query_intent": provenance.get("query_intent"),
                        "query_family": provenance.get("query_family"),
                        "query_text": provenance.get("query_text"),
                        "brick_name": provenance.get("brick_name"),
                        "scope_bucket": provenance.get("scope_bucket"),
                        "rank": provenance.get("rank"),
                        "provider": provenance.get("provider"),
                        "source_family": normalize_discovery_source_family(
                            "external_search_seed",
                            {
                                "query_family": provenance.get("query_family"),
                                "query_type": provenance.get("query_type"),
                                "provider": provenance.get("provider"),
                            },
                            provenance_url,
                        ),
                    },
                }
            )
        validated.append(candidate)

    return validated, stats


def _capability_signals_from_reason_items(reasons: list[dict[str, Any]], max_items: int = 6) -> list[str]:
    extracted: list[str] = []
    for item in reasons or []:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "").strip().lower()
        if dimension not in {"product", "services", "customer"}:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        short = " ".join(text.split())[:180]
        if short:
            extracted.append(short)
        if len(extracted) >= max(1, int(max_items)):
            break
    return _dedupe_strings(extracted)[: max(1, int(max_items))]


def _parse_discovery_candidates_from_text(raw_text: str) -> list[dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return []
    payload: Any = None
    try:
        payload = json.loads(text)
    except Exception:
        start_idx = text.find("[")
        end_idx = text.rfind("]") + 1
        if start_idx != -1 and end_idx > start_idx:
            try:
                payload = json.loads(text[start_idx:end_idx])
            except Exception:
                payload = None
    if isinstance(payload, dict):
        for key in ("candidates", "companies", "items", "results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload.get(key)
                break
    if not isinstance(payload, list):
        return []

    validated: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        website = str(row.get("website") or "").strip()
        discovery_url = str(row.get("discovery_url") or row.get("profile_url") or "").strip()
        why_relevant = row.get("why_relevant") if isinstance(row.get("why_relevant"), list) else []
        has_citation = any(
            isinstance(item, dict) and str(item.get("citation_url") or "").strip()
            for item in why_relevant
        )
        if not name or not (website or discovery_url or has_citation):
            continue
        validated.append(
            {
                "name": name,
                "website": website or None,
                "discovery_url": discovery_url or None,
                "hq_country": str(row.get("hq_country") or "Unknown").strip() or "Unknown",
                "likely_verticals": row.get("likely_verticals") if isinstance(row.get("likely_verticals"), list) else [],
                "employee_estimate": row.get("employee_estimate"),
                "capability_signals": row.get("capability_signals") if isinstance(row.get("capability_signals"), list) else [],
                "qualification": row.get("qualification") if isinstance(row.get("qualification"), dict) else {},
                "why_relevant": why_relevant,
            }
        )
    return validated


def _normalize_hint_url(url: str | None) -> Optional[str]:
    raw = str(url or "").strip()
    if not raw:
        return None
    normalized = raw if raw.startswith(("http://", "https://")) else f"https://{raw}"
    try:
        parsed = urlparse(normalized)
    except Exception:
        return None
    host = str(parsed.netloc or "").strip().lower()
    if not host:
        return None
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme or 'https'}://{host}{path}{query}"


def _is_path_level_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = str(parsed.path or "").strip()
    return bool(path and path not in {"", "/"})


def _hint_url_score(url: str) -> int:
    normalized = str(url or "").strip().lower()
    if not normalized:
        return -999
    try:
        parsed = urlparse(normalized)
    except Exception:
        return -999
    path = str(parsed.path or "").lower()
    if not path or path in {"", "/"}:
        return -20
    if path.endswith((".pdf", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".zip", ".mp4")):
        return -500

    score = 0
    for token in FIRST_PARTY_HINT_URL_TOKENS:
        if token in path:
            score += 7
    if is_career_page_url(normalized):
        score += 5
        career_target_hits = career_target_keyword_hits(normalized)
        career_excluded_hits = career_excluded_keyword_hits(normalized)
        if career_target_hits:
            score += 18 + min(24, len(career_target_hits) * 4)
        elif career_excluded_hits:
            score -= 24 + min(18, len(career_excluded_hits) * 4)
    if "/blog/" in path or "/news/" in path or "/press/" in path:
        score += 6
    if "/customer" in path or "/client" in path or "/case" in path:
        score += 10
    if "?" in normalized:
        score -= 2
    depth = len([segment for segment in path.split("/") if segment])
    if depth >= 2:
        score += 2
    return score


def _fetch_hint_document(url: str, timeout_seconds: int = 6) -> tuple[str, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0; +https://example.com/bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    with httpx.Client(timeout=max(2, timeout_seconds), follow_redirects=True, headers=headers, http2=False) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "").lower()
        return str(response.text or ""), content_type, str(response.url or url)


def _discover_homepage_hint_urls_for_domain(
    domain: str,
    timeout_seconds: int = 6,
) -> list[str]:
    homepage_url = f"https://{domain}/"
    try:
        html, _, final_url = _fetch_hint_document(homepage_url, timeout_seconds=timeout_seconds)
    except Exception:
        return []
    tree = HTMLParser(html)
    if tree is None:
        return []

    collected: list[str] = []
    seen: set[str] = set()
    base_url = final_url or homepage_url

    def register(raw_href: Optional[str]) -> None:
        href = str(raw_href or "").strip()
        if not href:
            return
        absolute = _normalize_hint_url(urljoin(base_url, href))
        if not absolute or not _is_path_level_url(absolute):
            return
        if normalize_domain(absolute) != domain:
            return
        if _hint_url_score(absolute) < 4:
            return
        key = absolute.lower()
        if key in seen:
            return
        seen.add(key)
        collected.append(absolute)

    for selector in ("nav a", "footer a", "header a", "a"):
        for node in tree.css(selector)[:220]:
            register(node.attributes.get("href"))

    for node in tree.css("link[rel='alternate']")[:40]:
        node_type = str(node.attributes.get("type") or "").lower()
        if "rss" in node_type or "atom" in node_type:
            register(node.attributes.get("href"))

    return collected


def _discover_sitemap_hint_urls_for_domain(
    domain: str,
    timeout_seconds: int = 6,
    max_documents: int = 8,
    max_urls: int = 300,
) -> list[str]:
    sitemap_queue: list[str] = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
    ]
    seen_sitemaps: set[str] = set()
    collected: list[str] = []
    seen_urls: set[str] = set()

    try:
        robots_text, _, _ = _fetch_hint_document(f"https://{domain}/robots.txt", timeout_seconds=timeout_seconds)
        for line in robots_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() != "sitemap":
                continue
            sitemap_url = _normalize_hint_url(value.strip())
            if sitemap_url:
                sitemap_queue.append(sitemap_url)
    except Exception:
        pass

    while sitemap_queue and len(seen_sitemaps) < max(1, int(max_documents)) and len(collected) < max(1, int(max_urls)):
        current = _normalize_hint_url(sitemap_queue.pop(0))
        if not current:
            continue
        key = current.lower()
        if key in seen_sitemaps:
            continue
        seen_sitemaps.add(key)
        try:
            xml_text, _, _ = _fetch_hint_document(current, timeout_seconds=timeout_seconds)
        except Exception:
            continue

        for match in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml_text, flags=re.IGNORECASE):
            loc = _normalize_hint_url(unescape(str(match or "").strip()))
            if not loc:
                continue
            loc_domain = normalize_domain(loc)
            if loc_domain != domain:
                continue
            if loc.lower().endswith(".xml") or "/sitemap" in loc.lower():
                if loc.lower() not in seen_sitemaps:
                    sitemap_queue.append(loc)
                continue
            if not _is_path_level_url(loc):
                continue
            if _hint_url_score(loc) < 5:
                continue
            loc_key = loc.lower()
            if loc_key in seen_urls:
                continue
            seen_urls.add(loc_key)
            collected.append(loc)

    return collected


def _discover_adaptive_hint_urls_for_domain(
    domain: str,
    timeout_seconds: int = 6,
    max_urls: int = 40,
) -> list[str]:
    if not domain or _is_non_first_party_profile_domain(domain):
        return []

    urls = _auto_first_party_hint_urls_for_domains([domain])
    urls.extend(_discover_homepage_hint_urls_for_domain(domain, timeout_seconds=timeout_seconds))
    urls.extend(_discover_sitemap_hint_urls_for_domain(domain, timeout_seconds=timeout_seconds))

    deduped = _dedupe_strings([url for url in urls if _normalize_hint_url(url)])
    ranked = sorted(
        deduped,
        key=lambda value: (-_hint_url_score(value), len(value)),
    )
    return ranked[: max(1, int(max_urls))]


def _build_first_party_hint_url_map(
    profile: Optional[CompanyProfile],
    include_benchmark_hints: bool,
) -> dict[str, list[str]]:
    hint_map: dict[str, list[str]] = {}

    def register(raw_url: str | None, require_path: bool = False) -> None:
        normalized = _normalize_hint_url(raw_url)
        if not normalized:
            return
        if require_path and not _is_path_level_url(normalized):
            return
        domain = normalize_domain(normalized)
        if not domain or _is_non_first_party_profile_domain(domain):
            return
        existing = hint_map.setdefault(domain, [])
        if normalized.lower() in {item.lower() for item in existing}:
            return
        existing.append(normalized)

    if profile:
        for url in (profile.supporting_evidence_urls or []):
            register(url)
        for url in (profile.comparator_seed_urls or []):
            register(url, require_path=True)

    if include_benchmark_hints:
        for urls in WEALTH_BENCHMARK_EVIDENCE_URLS.values():
            for url in urls:
                register(url)

    return {domain: urls[:20] for domain, urls in hint_map.items()}


def _collect_hint_urls_for_domains(
    hint_map: dict[str, list[str]],
    domains: list[str],
) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for domain in _normalize_domain_list(domains):
        for url in hint_map.get(domain, []):
            normalized = _normalize_hint_url(url)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            collected.append(normalized)
    return collected


def _context_pack_start_urls_for_site(profile: CompanyProfile | None, site_url: str | None) -> list[str]:
    normalized_site_url = normalize_url(site_url or "")
    site_domain = normalize_domain(normalized_site_url)
    if not site_domain or not profile:
        return []

    buyer_url = normalize_url(profile.buyer_company_url or "")
    is_buyer_site = bool(buyer_url and normalized_site_url == buyer_url)
    max_hint_urls = 18 if is_buyer_site else 12

    manual_urls: list[str] = []
    adaptive_urls: list[str] = []
    seen: set[str] = set()

    def register(raw_url: str | None, *, source: str = "manual") -> None:
        normalized = normalize_url(raw_url or "")
        if not normalized:
            return
        if normalize_domain(normalized) != site_domain:
            return
        if normalized.lower() == normalized_site_url.lower():
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        if source == "adaptive":
            adaptive_urls.append(normalized)
        else:
            manual_urls.append(normalized)

    if is_buyer_site:
        for url in profile.supporting_evidence_urls or []:
            register(url)
    for url in profile.comparator_seed_urls or []:
        register(url)

    adaptive_hint_urls = _discover_adaptive_hint_urls_for_domain(
        site_domain,
        timeout_seconds=5,
        max_urls=max_hint_urls,
    )
    for url in adaptive_hint_urls:
        register(url, source="adaptive")

    ranked_manual_urls = sorted(
        manual_urls,
        key=lambda value: (-_hint_url_score(value), len(value)),
    )
    ranked_adaptive_urls = sorted(
        adaptive_urls,
        key=lambda value: (-_hint_url_score(value), len(value)),
    )
    return (ranked_manual_urls + ranked_adaptive_urls)[:max_hint_urls]


def _auto_first_party_hint_urls_for_domains(domains: list[str]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for domain in _normalize_domain_list(domains):
        if _is_non_first_party_profile_domain(domain):
            continue
        for path in FIRST_PARTY_AUTO_HINT_PATHS:
            hint = f"https://{domain}{path}"
            key = hint.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(hint)
    return hints


def _build_mention_indexes(mentions: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for mention in mentions:
        domain = normalize_domain(
            str(
                mention.get("official_website_url")
                or mention.get("company_url")
                or ""
            )
        )
        if domain:
            by_domain.setdefault(domain, []).append(mention)
        name = str(mention.get("company_name") or "").strip().lower()
        if name:
            by_name.setdefault(name, []).append(mention)
    return by_domain, by_name


def _match_mentions_for_candidate(
    candidate: dict[str, Any],
    mentions_by_domain: dict[str, list[dict[str, Any]]],
    mentions_by_name: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    domain = normalize_domain(str(candidate.get("website") or ""))
    aggregator_domains = {
        "thewealthmosaic.com",
        "thewealthmosaic.co.uk",
    }
    if domain and domain in mentions_by_domain and not any(
        domain == agg or domain.endswith(f".{agg}") for agg in aggregator_domains
    ):
        matches.extend(mentions_by_domain[domain])
    candidate_url = str(candidate.get("website") or "").strip().lower()
    if candidate_url:
        for bucket in mentions_by_domain.values():
            for mention in bucket:
                mention_url = str(
                    mention.get("official_website_url")
                    or mention.get("company_url")
                    or mention.get("profile_url")
                    or ""
                ).strip().lower()
                if mention_url and mention_url == candidate_url:
                    matches.append(mention)
    name_key = str(candidate.get("name") or "").strip().lower()
    if name_key and name_key in mentions_by_name:
        matches.extend(mentions_by_name[name_key])
    # Deduplicate
    deduped: list[dict[str, Any]] = []
    seen = set()
    for mention in matches:
        key = (
            str(mention.get("company_name") or "").lower(),
            str(mention.get("listing_url") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mention)
    return deduped


def _build_claim_records(
    workspace_id: int,
    company_id: Optional[int],
    company_screening_id: int,
    candidate: dict[str, Any],
    trusted_reasons: list[dict[str, str]],
    matched_mentions: list[dict[str, Any]],
    policy: Optional[dict[str, Any]] = None,
    source_evidence_ids: Optional[dict[str, int]] = None,
    first_party_domains: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate_domain = normalize_domain(str(candidate.get("website") or ""))
    effective_policy = normalize_policy(policy or DEFAULT_EVIDENCE_POLICY)
    evidence_id_by_url = {
        str(key).strip().lower(): value
        for key, value in (source_evidence_ids or {}).items()
        if str(key).strip() and value is not None
    }

    def infer_claim_group(
        dimension: str,
        claim_key: Optional[str],
        claim_text: str,
    ) -> str:
        group = claim_group_for_dimension(dimension, claim_key)
        dim = str(dimension or "").strip().lower()
        if group != "product_depth" or dim not in {"product", "services", "customer", "capability"}:
            return group
        lowered = str(claim_text or "").strip().lower()
        if not lowered:
            return group
        has_institutional = _has_token(lowered, INSTITUTIONAL_TOKENS)
        has_workflow_hint = _has_token(lowered, VERTICAL_WORKFLOW_HINT_TOKENS)
        if has_institutional and has_workflow_hint:
            return "vertical_workflow"
        return group

    def add_claim(
        dimension: str,
        text: str,
        source_url: str,
        confidence: str = "medium",
        claim_key: Optional[str] = None,
        provenance_hint: Optional[str] = None,
    ):
        source_url = str(source_url or "").strip()
        claim_text = str(text or "").strip()
        if not claim_text or not source_url:
            return
        if not is_trusted_source_url(source_url):
            return
        source_type = _source_type_for_url(
            source_url,
            candidate_domain,
            first_party_domains=first_party_domains,
            provenance_hint=provenance_hint,
        )
        source_tier = infer_source_tier(source_url, source_type, candidate_domain)
        claim_group = infer_claim_group(dimension, claim_key, claim_text)
        ttl_days, valid_through = valid_through_from_claim_group(
            claim_group=claim_group,
            policy=effective_policy,
        )
        parsed_key, numeric_value, numeric_unit, period = _numeric_from_claim_text(claim_text)
        records.append(
            {
                "workspace_id": workspace_id,
                "company_id": company_id,
                "company_screening_id": company_screening_id,
                "dimension": (dimension or "evidence")[:64],
                "claim_group": claim_group,
                "claim_status": "fact",
                "claim_key": (claim_key or parsed_key),
                "claim_text": claim_text[:3000],
                "source_url": source_url[:1000],
                "source_type": source_type,
                "source_tier": source_tier,
                "source_evidence_id": evidence_id_by_url.get(source_url.lower()),
                "confidence": confidence,
                "contradiction_group_id": None,
                "freshness_ttl_days": ttl_days,
                "valid_through": valid_through,
                "numeric_value": numeric_value,
                "numeric_unit": numeric_unit,
                "period": period,
                "is_conflicting": False,
            }
        )

    for reason in trusted_reasons:
        add_claim(
            dimension=str(reason.get("dimension") or "evidence"),
            text=str(reason.get("text") or ""),
            source_url=str(reason.get("citation_url") or ""),
            confidence="high" if "filing" in str(reason.get("dimension", "")).lower() else "medium",
            provenance_hint=str(reason.get("source_kind") or "").strip().lower() or None,
        )

    for mention in matched_mentions:
        listing_url = str(mention.get("listing_url") or "")
        for snippet in (mention.get("listing_text_snippets") or [])[:2]:
            if not isinstance(snippet, str) or not snippet.strip():
                continue
            add_claim(
                dimension="directory_context",
                text=snippet,
                source_url=listing_url,
                confidence="medium",
            )

    website = str(candidate.get("website") or "").strip()
    website_source_type = _source_type_for_url(
        website,
        candidate_domain,
        first_party_domains=first_party_domains,
    ) if website else "unknown"
    if website and is_trusted_source_url(website) and website_source_type == "first_party_website":
        add_claim(
            dimension="company_profile",
            text=f"First-party company website: {website}",
            source_url=website,
            confidence="high",
        )

    # Flag conflicts for numeric values in same claim key.
    values_by_key: dict[str, set[float]] = {}
    for record in records:
        if not record.get("claim_key") or record.get("numeric_value") is None:
            continue
        key = str(record["claim_key"])
        values_by_key.setdefault(key, set()).add(float(record["numeric_value"]))
    conflicting_keys = {k for k, values in values_by_key.items() if len(values) > 1}
    for record in records:
        key = record.get("claim_key")
        if key and key in conflicting_keys:
            record["is_conflicting"] = True
            record["claim_status"] = "contradicted"
            record["contradiction_group_id"] = f"{key}:numeric_conflict"

    return records


CLAIM_TIER_ORDER = {
    "tier0_registry": 0,
    "tier1_vendor": 1,
    "tier2_partner_customer": 2,
    "tier3_third_party": 3,
    "tier4_discovery": 4,
}

CLAIM_GROUP_PRIORITY = {
    "vertical_workflow": 0,
    "product_depth": 1,
    "traction": 2,
    "ecosystem_defensibility": 3,
    "identity_scope": 4,
}


def _select_top_claim(claim_records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_claims = [
        claim
        for claim in claim_records
        if isinstance(claim, dict)
        and str(claim.get("source_url") or "").strip()
        and str(claim.get("source_tier") or "").strip()
    ]
    if not valid_claims:
        return {}

    def rank_key(claim: dict[str, Any]) -> tuple[int, int, int]:
        tier = str(claim.get("source_tier") or "tier4_discovery")
        group = str(claim.get("claim_group") or "product_depth")
        claim_text_len = len(str(claim.get("claim_text") or ""))
        return (
            CLAIM_TIER_ORDER.get(tier, 99),
            CLAIM_GROUP_PRIORITY.get(group, 99),
            -claim_text_len,
        )

    best = sorted(valid_claims, key=rank_key)[0]
    return {
        "text": str(best.get("claim_text") or "")[:600],
        "claim_type": str(best.get("dimension") or "evidence"),
        "source_url": str(best.get("source_url") or "")[:1000],
        "source_tier": str(best.get("source_tier") or "tier4_discovery"),
        "source_kind": infer_source_kind(
            str(best.get("source_url") or ""),
            str(best.get("source_type") or ""),
            None,
        ),
        "captured_at": datetime.utcnow().isoformat(),
    }


def _normalize_source_url_for_citation(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    normalized = raw if raw.startswith(("http://", "https://")) else f"https://{raw}"
    try:
        parsed = urlparse(normalized)
    except Exception:
        return normalized.lower()
    host = str(parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    path = str(parsed.path or "").strip()
    if not path:
        path = "/"
    elif path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme or 'https'}://{host}{path}{query}".lower()


def _build_citation_summary_v1(
    claim_records: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    trusted_claims = [
        claim
        for claim in claim_records
        if isinstance(claim, dict)
        and str(claim.get("claim_text") or "").strip()
        and str(claim.get("source_url") or "").strip()
        and is_trusted_source_url(str(claim.get("source_url") or "").strip())
    ]
    if not trusted_claims:
        return None

    source_pills: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    pill_id_by_key: dict[tuple[str, str], str] = {}
    sentence_idx_by_text: dict[str, int] = {}
    now_iso = datetime.utcnow().isoformat()

    for claim in trusted_claims:
        source_url = str(claim.get("source_url") or "").strip()[:1000]
        claim_text = str(claim.get("claim_text") or "").strip()[:700]
        if not source_url or not claim_text:
            continue
        claim_group = str(
            claim.get("claim_group")
            or claim_group_for_dimension(claim.get("dimension"), claim.get("claim_key"))
            or "product_depth"
        )
        normalized_url = _normalize_source_url_for_citation(source_url)
        pill_key = (normalized_url, claim_group)
        pill_id = pill_id_by_key.get(pill_key)
        if not pill_id:
            pill_id = f"p{len(source_pills) + 1}"
            pill_id_by_key[pill_key] = pill_id
            source_type = str(claim.get("source_type") or "")
            source_tier = str(claim.get("source_tier") or infer_source_tier(source_url, source_type, normalize_domain(source_url)))
            source_kind = infer_source_kind(source_url, source_type, normalize_domain(source_url))
            source_pills.append(
                {
                    "pill_id": pill_id,
                    "label": source_label_for_url(source_url),
                    "url": source_url,
                    "source_tier": source_tier,
                    "source_kind": source_kind,
                    "captured_at": str(claim.get("captured_at") or now_iso),
                    "claim_group": claim_group,
                }
            )

        sentence_key = claim_text.lower()
        if sentence_key in sentence_idx_by_text:
            sentence_idx = sentence_idx_by_text[sentence_key]
            pills = sentence_rows[sentence_idx]["citation_pill_ids"]
            if pill_id not in pills:
                pills.append(pill_id)
            continue

        sentence_idx_by_text[sentence_key] = len(sentence_rows)
        sentence_rows.append(
            {
                "id": f"s{len(sentence_rows) + 1}",
                "text": claim_text,
                "citation_pill_ids": [pill_id],
            }
        )

    if not source_pills or not sentence_rows:
        return None

    valid_pill_ids = {pill["pill_id"] for pill in source_pills}
    cited_sentences = [
        {
            "id": sentence["id"],
            "text": sentence["text"],
            "citation_pill_ids": [
                pill_id for pill_id in sentence["citation_pill_ids"]
                if pill_id in valid_pill_ids
            ],
        }
        for sentence in sentence_rows
    ]
    cited_sentences = [
        sentence for sentence in cited_sentences if sentence["citation_pill_ids"]
    ]
    if not cited_sentences:
        return None

    return {
        "version": "v1",
        "sentences": cited_sentences,
        "source_pills": source_pills,
    }


def _is_ranking_eligible_candidate(
    candidate: dict[str, Any],
    decision_classification: str,
    claim_records: list[dict[str, Any]],
) -> bool:
    entity_type = str(candidate.get("entity_type") or "company").strip().lower()
    official_website = str(candidate.get("official_website_url") or candidate.get("website") or "").strip()
    if entity_type != "company":
        return False
    if not official_website:
        return False
    official_domain = normalize_domain(official_website)
    if not official_domain or _is_non_first_party_profile_domain(official_domain):
        return False
    if decision_classification == "not_good_target":
        return False
    has_tier1 = any(
        str(claim.get("source_tier") or "") == "tier1_vendor"
        for claim in claim_records
        if isinstance(claim, dict)
    )
    return has_tier1


class ContextPackJobInterrupted(RuntimeError):
    """Raised when a context-pack job is cancelled or superseded mid-run."""


def _fail_superseded_context_pack_jobs(db, workspace_id: int, current_job_id: int) -> int:
    superseded_jobs = (
        db.query(Job)
        .filter(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.context_pack,
            Job.id < current_job_id,
            Job.state.in_(DB_ACTIVE_JOB_STATES),
        )
        .all()
    )

    for stale_job in superseded_jobs:
        stale_job.state = JobState.failed
        stale_job.error_message = "Superseded by newer sourcing brief run"
        stale_job.progress_message = "Superseded by newer sourcing brief run"
        stale_job.finished_at = datetime.utcnow()

    if superseded_jobs:
        db.commit()

    return len(superseded_jobs)


def _ensure_context_pack_job_active(db, job: Job) -> None:
    db.refresh(job)
    if job.state == JobState.failed and job.finished_at:
        raise ContextPackJobInterrupted(job.error_message or "Context-pack job stopped")
    if job.state == JobState.completed and job.finished_at:
        raise ContextPackJobInterrupted("Context-pack job already completed")


def _append_job_live_event(job: Job, message: str) -> None:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    live_events = payload.get("live_events") if isinstance(payload.get("live_events"), list) else []
    event_message = str(message or "").strip()
    if not event_message:
        return
    if live_events and isinstance(live_events[-1], dict) and live_events[-1].get("message") == event_message:
        return
    live_events.append(
        {
            "message": event_message[:280],
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    payload["live_events"] = live_events[-10:]
    job.result_json = payload


def _estimate_context_pack_site_progress(message: str) -> Optional[float]:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return None

    preview_match = re.search(r"previewed\s+(\d+)/(\d+)\s+pages", normalized)
    if preview_match:
        seen = int(preview_match.group(1))
        total = max(1, int(preview_match.group(2)))
        return min(0.58, 0.18 + (0.40 * (seen / total)))

    triage_match = re.search(r"triaging pages with llm\s+\((\d+)/(\d+)\)", normalized)
    if triage_match:
        batch = int(triage_match.group(1))
        total = max(1, int(triage_match.group(2)))
        return min(0.78, 0.60 + (0.18 * (batch / total)))

    if "checking robots.txt" in normalized:
        return 0.04
    if "parsing sitemaps" in normalized:
        return 0.08
    if "extracting navigation links" in normalized:
        return 0.12
    if "expanding from hub pages" in normalized:
        return 0.16
    if "fetching previews and scoring" in normalized or normalized.startswith("preview batch"):
        return 0.2
    if "triaging pages with llm" in normalized or "llm classification error" in normalized:
        return 0.62
    if normalized.startswith("selected page for extraction:"):
        return 0.8
    if "deep content extraction" in normalized or normalized.startswith("extraction batch"):
        return 0.84
    if re.search(r"extracted\s+\d+/\d+\s+pages", normalized):
        return 0.9
    if "generating context pack" in normalized:
        return 0.96
    if "context pack complete" in normalized:
        return 1.0

    return None


@celery_app.task(name="app.workers.workspace_tasks.generate_context_pack_v2")
def generate_context_pack_v2(job_id: int):
    """Generate context pack by crawling buyer and reference URLs."""
    from app.services.crawler import UnifiedCrawler
    from app.services.company_context import build_company_context_artifacts, build_context_pack_v2
    
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        
        job.state = JobState.running
        job.started_at = datetime.utcnow()
        job.progress = 0.1
        job.progress_message = "Starting crawl..."
        db.commit()
        
        # Get workspace and profile
        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        if not workspace:
            job.state = JobState.failed
            job.error_message = "Workspace not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Workspace not found"}

        _fail_superseded_context_pack_jobs(db, workspace.id, job.id)
        _ensure_context_pack_job_active(db, job)
        
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == workspace.id).first()
        if not profile:
            job.state = JobState.failed
            job.error_message = "Company profile not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Company profile not found"}
        
        try:
            # Crawl buyer URL
            all_urls = []
            
            if profile.buyer_company_url:
                all_urls.append(profile.buyer_company_url)
            
            if profile.comparator_seed_urls:
                all_urls.extend(profile.comparator_seed_urls[:3])
            
            if not all_urls:
                job.state = JobState.failed
                job.error_message = "No URLs to crawl"
                job.finished_at = datetime.utcnow()
                db.commit()
                return {"error": "No URLs to crawl"}
            
            job.progress = 0.2
            job.progress_message = f"Crawling {len(all_urls)} sites..."
            db.commit()
            
            # Run async crawler in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            combined_markdown = []
            product_pages_total = 0
            comparator_seed_summaries = {}
            all_context_packs = []  # Store full context packs for JSON export
            buyer_context_pack = None

            def update_progress(message: str, site_start: float, site_end: float):
                """Update job progress message."""
                _ensure_context_pack_job_active(db, job)
                site_progress = _estimate_context_pack_site_progress(message)
                if site_progress is not None:
                    next_progress = site_start + ((site_end - site_start) * site_progress)
                    job.progress = max(float(job.progress or 0.0), round(next_progress, 4))
                job.progress_message = message
                _append_job_live_event(job, message)
                db.commit()
            
            for i, url in enumerate(all_urls):
                try:
                    _ensure_context_pack_job_active(db, job)
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc or url
                    site_start = 0.2 + (0.5 * i / len(all_urls))
                    site_end = 0.2 + (0.5 * (i + 1) / len(all_urls))
                    
                    # Create unified crawler with progress callback for this URL
                    # Use default argument to capture domain in closure
                    crawler = UnifiedCrawler(
                        max_pages=30,
                        progress_callback=lambda msg, d=domain, start=site_start, end=site_end: update_progress(f"[{d}] {msg}", start, end)
                    )
                    start_urls = _context_pack_start_urls_for_site(profile, url)
                    
                    job.progress = site_start
                    job.progress_message = f"Starting crawl of {domain}..."
                    _append_job_live_event(job, f"[{domain}] Starting crawl")
                    db.commit()
                    
                    context_pack = loop.run_until_complete(
                        crawler.crawl_for_context(
                            url,
                            start_urls=start_urls,
                        )
                    )
                    combined_markdown.append(context_pack.raw_markdown)
                    product_pages_total += context_pack.product_pages_count
                    all_context_packs.append((url, context_pack))  # Store for later
                    if url == profile.buyer_company_url:
                        buyer_context_pack = context_pack
                    
                    # #region agent log
                    import json
                    log_path = "/app/debug.log"
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "K", "location": "workspace_tasks.py:123", "message": "context_pack result", "data": {"url": url, "product_pages_count": context_pack.product_pages_count, "total_pages": len(context_pack.pages), "product_pages_total": product_pages_total}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
                    
                    if url != profile.buyer_company_url:
                        comparator_seed_summaries[url] = context_pack.summary
                    
                    job.progress = site_end
                    job.progress_message = f"Completed {domain} ({len(context_pack.pages)} pages)"
                    _append_job_live_event(job, f"[{domain}] Completed crawl with {len(context_pack.pages)} pages")
                    db.commit()
                except ContextPackJobInterrupted:
                    raise
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
                    job.progress_message = f"Error crawling {url}: {str(e)[:100]}"
                    _append_job_live_event(job, f"[{urlparse(url).netloc or url}] Crawl error: {str(e)[:100]}")
                    db.commit()
                    continue
            
            loop.close()
            
            raw_markdown = "\n\n---\n\n".join(combined_markdown)
            buyer_raw_markdown = (
                buyer_context_pack.raw_markdown
                if buyer_context_pack is not None and str(getattr(buyer_context_pack, "raw_markdown", "") or "").strip()
                else ""
            )
            
            # Build combined context pack JSON for export from stored data
            combined_context_pack_json = {
                "generated_at": datetime.utcnow().isoformat(),
                "urls_crawled": all_urls,
                "sites": []
            }
            
            for url, context_pack in all_context_packs:
                try:
                    # Serialize context pack to JSON-compatible dict
                    site_data = {
                        "url": url,
                        "company_name": context_pack.company_name,
                        "website": context_pack.website,
                        "product_pages_count": context_pack.product_pages_count,
                        "summary": context_pack.summary,
                        "pages": [
                            {
                                "url": page.url,
                                "title": page.title,
                                "page_type": page.page_type,
                                "blocks": [
                                    {"type": b.type, "content": b.content, "level": b.level}
                                    for b in page.blocks
                                ],
                                "signals": [
                                    {"type": s.type, "value": s.value, "source_url": s.evidence.source_url, "snippet": s.evidence.snippet}
                                    for s in page.signals
                                ],
                                "customer_evidence": [
                                    {"name": c.name, "source_url": c.source_url, "evidence_type": c.evidence_type, "context": c.context}
                                    for c in page.customer_evidence
                                ],
                                "raw_content": page.raw_content[:10000] if page.raw_content else ""  # Truncate for storage
                            }
                            for page in context_pack.pages
                        ],
                        "signals": [
                            {"type": s.type, "value": s.value, "source_url": s.evidence.source_url, "snippet": s.evidence.snippet}
                            for s in context_pack.signals
                        ],
                        "customer_evidence": [
                            {"name": c.name, "source_url": c.source_url, "evidence_type": c.evidence_type, "context": c.context}
                            for c in context_pack.customer_evidence
                        ]
                    }
                    combined_context_pack_json["sites"].append(site_data)
                except Exception as e:
                    print(f"Error building JSON for {url}: {e}")
            
            # Update profile
            profile.context_pack_markdown = raw_markdown
            profile.context_pack_json = build_context_pack_v2(combined_context_pack_json)
            profile.context_pack_generated_at = datetime.utcnow()
            profile.product_pages_found = product_pages_total
            profile.comparator_seed_summaries = comparator_seed_summaries

            job.progress = 0.8
            job.progress_message = "Preparing company context..."
            db.commit()
            
            # #region agent log
            import json
            log_path = "/app/debug.log"
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "L", "location": "workspace_tasks.py:168", "message": "before db commit", "data": {"product_pages_found": profile.product_pages_found, "product_pages_total": product_pages_total}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            job.state = JobState.completed
            job.progress = 1.0
            job.progress_message = "Complete"
            result_payload = job.result_json if isinstance(job.result_json, dict) else {}
            result_payload.update({
                "urls_crawled": len(all_urls),
                "sites_crawled": len(all_context_packs),
                "pages_crawled": sum(len(context_pack.pages) for _, context_pack in all_context_packs),
                "product_pages_found": product_pages_total,
                "markdown_length": len(raw_markdown),
                "duration_seconds": max(
                    0,
                    int(
                        (
                            (datetime.utcnow() - (job.started_at or job.created_at or datetime.utcnow())).total_seconds()
                        )
                    ),
                ),
            })
            job.result_json = result_payload
            job.finished_at = datetime.utcnow()
            db.commit()
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "L", "location": "workspace_tasks.py:181", "message": "after db commit", "data": {"product_pages_found": profile.product_pages_found}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            return {"success": True, "product_pages": product_pages_total}
        except ContextPackJobInterrupted as e:
            db.rollback()
            interrupted_job = db.query(Job).filter(Job.id == job_id).first()
            if interrupted_job and interrupted_job.state != JobState.failed:
                interrupted_job.state = JobState.failed
                interrupted_job.error_message = str(e)
                interrupted_job.progress_message = str(e)
                interrupted_job.finished_at = datetime.utcnow()
                db.commit()
            return {"error": str(e)}
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            failed_job = db.query(Job).filter(Job.id == job_id).first()
            if failed_job:
                failed_job.state = JobState.failed
                failed_job.error_message = str(e)
                failed_job.finished_at = datetime.utcnow()
                try:
                    db.commit()
                except Exception:
                    db.rollback()
            return {"error": str(e)}
            
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.run_monitoring_delta")
def run_monitoring_delta(job_id: int):
    """Refresh watchlist candidates based on stale evidence or material claim deltas."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.state = JobState.running
        job.started_at = datetime.utcnow()
        job.progress = 0.05
        job.progress_message = "Evaluating watchlist monitoring triggers..."
        db.commit()

        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        if not workspace:
            job.state = JobState.failed
            job.error_message = "Workspace not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Workspace not found"}

        monitor_cfg = job.result_json if isinstance(job.result_json, dict) else {}
        effective_policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)
        max_companies = int(monitor_cfg.get("max_companies", monitor_cfg.get("max_vendors", 80)) or 80)
        stale_only = bool(monitor_cfg.get("stale_only", False))
        tracked_classes = set(
            monitor_cfg.get("classifications")
            or ["borderline_watchlist", "insufficient_evidence"]
        )

        screenings = (
            db.query(CompanyScreening)
            .filter(CompanyScreening.workspace_id == workspace.id)
            .order_by(CompanyScreening.created_at.desc())
            .all()
        )
        latest_by_company: dict[int, CompanyScreening] = {}
        for screening in screenings:
            if screening.company_id and screening.company_id not in latest_by_company:
                latest_by_company[screening.company_id] = screening

        candidate_rows = [
            screening
            for screening in latest_by_company.values()
            if str(screening.decision_classification or "insufficient_evidence") in tracked_classes
        ][:max_companies]

        now = datetime.utcnow()
        triggered: list[dict[str, Any]] = []
        enrich_job_ids: list[int] = []

        total = max(1, len(candidate_rows))
        for index, screening in enumerate(candidate_rows):
            company_id = screening.company_id
            if not company_id:
                continue

            evidence_rows = (
                db.query(SourceEvidence)
                .filter(SourceEvidence.workspace_id == workspace.id, SourceEvidence.company_id == company_id)
                .order_by(SourceEvidence.captured_at.desc())
                .all()
            )
            claim_rows = (
                db.query(CompanyClaim)
                .filter(CompanyClaim.workspace_id == workspace.id, CompanyClaim.company_id == company_id)
                .all()
            )

            stale_evidence = [
                row for row in evidence_rows if row.valid_through and not is_fresh(row.valid_through, as_of=now)
            ]
            stale_claims = [
                row for row in claim_rows if row.valid_through and not is_fresh(row.valid_through, as_of=now)
            ]
            missing_ttl_evidence = [
                row
                for row in evidence_rows
                if row.valid_through is None and row.captured_at and (now - row.captured_at).days > 365
            ]
            new_evidence_since_screen = len(
                [
                    row
                    for row in evidence_rows
                    if row.captured_at and screening.created_at and row.captured_at > screening.created_at
                ]
            )
            new_claims_since_screen = len(
                [
                    row
                    for row in claim_rows
                    if row.created_at and screening.created_at and row.created_at > screening.created_at
                ]
            )
            contradiction_count = int(screening.unresolved_contradictions_count or 0)

            trigger_reasons: list[str] = []
            if stale_evidence or stale_claims:
                trigger_reasons.append("ttl_expired")
            if missing_ttl_evidence:
                trigger_reasons.append("legacy_missing_ttl")
            if new_evidence_since_screen > 0 or new_claims_since_screen > 0:
                trigger_reasons.append("material_delta")
            if contradiction_count > 0:
                trigger_reasons.append("unresolved_contradictions")
            if screening.evidence_sufficiency == "insufficient" and not evidence_rows:
                trigger_reasons.append("insufficient_no_evidence")

            if stale_only:
                should_refresh = bool(stale_evidence or stale_claims or missing_ttl_evidence)
            else:
                should_refresh = bool(trigger_reasons)

            if not should_refresh:
                job.progress = 0.05 + (0.7 * (index + 1) / total)
                job.progress_message = f"Checked {index + 1}/{total} watchlist companies"
                db.commit()
                continue

            enrich_job = Job(
                workspace_id=workspace.id,
                company_id=company_id,
                job_type=JobType.enrich_full,
                state=JobState.queued,
                provider=JobProvider.gemini_flash,
                result_json={
                    "triggered_by": "monitoring_delta",
                    "monitoring_job_id": job.id,
                    "trigger_reasons": trigger_reasons,
                },
            )
            db.add(enrich_job)
            db.flush()
            enrich_job_ids.append(enrich_job.id)
            triggered.append(
                {
                    "company_id": company_id,
                    "classification": screening.decision_classification,
                    "trigger_reasons": trigger_reasons,
                    "stale_evidence_count": len(stale_evidence) + len(stale_claims) + len(missing_ttl_evidence),
                    "new_signal_count": new_evidence_since_screen + new_claims_since_screen,
                }
            )

            job.progress = 0.05 + (0.7 * (index + 1) / total)
            job.progress_message = f"Checked {index + 1}/{total} watchlist companies"
            db.commit()

        for enrich_job_id in enrich_job_ids:
            run_enrich_company.delay(enrich_job_id)

        claims_graph_metrics = rebuild_workspace_claims_graph(db, workspace.id)
        job.state = JobState.completed
        job.progress = 1.0
        job.progress_message = "Monitoring delta complete"
        job.result_json = {
            "watchlist_candidates_checked": len(candidate_rows),
            "triggered_company_count": len(triggered),
            "triggered_company_ids": [entry["company_id"] for entry in triggered],
            "triggered": triggered[:200],
            "enrichment_job_ids": enrich_job_ids,
            "policy_version": effective_policy.get("version"),
            "claims_graph_refresh": claims_graph_metrics,
        }
        job.finished_at = datetime.utcnow()
        db.commit()
        return {"success": True, "triggered_company_count": len(triggered)}
    except Exception as exc:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.state = JobState.failed
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(exc)}
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.run_company_context_refresh")
def run_company_context_refresh(workspace_id: int):
    """Refresh sourcing graph/brief artifacts for a workspace."""
    from app.services.company_context import build_company_context_artifacts

    db = SessionLocal()
    company_context_pack = None
    try:
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == workspace_id).first()
        if not profile:
            logger.warning("company_context_refresh_task_missing_profile workspace_id=%s", workspace_id)
            return {"error": "Company profile not found", "workspace_id": workspace_id}

        company_context_pack = (
            db.query(CompanyContextPack)
            .filter(CompanyContextPack.workspace_id == workspace_id)
            .first()
        )
        refreshed = build_company_context_artifacts(profile)

        if company_context_pack is None:
            company_context_pack = CompanyContextPack(workspace_id=workspace_id)
            db.add(company_context_pack)
            db.flush()

        company_context_pack.sourcing_brief_json = refreshed.get("sourcing_brief") or {}
        company_context_pack.expansion_brief_json = {}
        company_context_pack.expansion_status = "not_generated"
        company_context_pack.expansion_error = None
        company_context_pack.expansion_generated_at = None
        company_context_pack.taxonomy_nodes_json = refreshed.get("taxonomy_nodes") or []
        company_context_pack.taxonomy_edges_json = refreshed.get("taxonomy_edges") or []
        company_context_pack.lens_seeds_json = refreshed.get("lens_seeds") or []
        company_context_pack.generated_at = refreshed.get("generated_at")
        company_context_pack.confirmed_at = None
        company_context_pack.updated_at = datetime.utcnow()
        sync_company_context_pack_graph(
            company_context_pack,
            profile,
            payload_override=refreshed,
        )
        company_context_pack.updated_at = datetime.utcnow()
        db.commit()
        logger.info(
            "company_context_refresh_task_completed workspace_id=%s graph_status=%s",
            workspace_id,
            company_context_pack.graph_sync_status,
        )
        return {
            "workspace_id": workspace_id,
            "status": company_context_pack.graph_sync_status,
            "graph_ref": company_context_pack.company_context_graph_ref,
        }
    except Exception as exc:
        logger.exception("company_context_refresh_task_failed workspace_id=%s", workspace_id)
        if company_context_pack is not None:
            company_context_pack.graph_sync_status = "failed"
            company_context_pack.graph_sync_error = "Worker sourcing refresh failed"
            company_context_pack.graph_synced_at = datetime.utcnow()
            company_context_pack.updated_at = datetime.utcnow()
            db.commit()
        return {"error": str(exc), "workspace_id": workspace_id}
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.run_expansion_refresh")
def run_expansion_refresh(workspace_id: int):
    """Refresh expansion brief artifacts for a workspace."""
    from app.services.company_context import build_expansion_artifacts

    db = SessionLocal()
    company_context_pack = None
    try:
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == workspace_id).first()
        if not profile:
            logger.warning("expansion_refresh_task_missing_profile workspace_id=%s", workspace_id)
            return {"error": "Company profile not found", "workspace_id": workspace_id}

        company_context_pack = (
            db.query(CompanyContextPack)
            .filter(CompanyContextPack.workspace_id == workspace_id)
            .first()
        )
        if company_context_pack is None:
            logger.warning("expansion_refresh_task_missing_pack workspace_id=%s", workspace_id)
            return {"error": "Company context pack not found", "workspace_id": workspace_id}

        sourcing_brief = company_context_pack.sourcing_brief_json or {}
        taxonomy_nodes = company_context_pack.taxonomy_nodes_json or []
        if not isinstance(sourcing_brief, dict) or not sourcing_brief:
            raise ValueError("Sourcing brief must exist before expansion generation")

        refreshed = build_expansion_artifacts(
            profile,
            sourcing_brief=sourcing_brief,
            taxonomy_nodes=taxonomy_nodes,
        )
        company_context_pack.expansion_brief_json = refreshed.get("expansion_brief") or {}
        company_context_pack.expansion_status = "ready"
        company_context_pack.expansion_error = None
        company_context_pack.expansion_generated_at = refreshed.get("generated_at") or datetime.utcnow()
        company_context_pack.updated_at = datetime.utcnow()
        db.commit()
        logger.info(
            "expansion_refresh_task_completed workspace_id=%s expansion_status=%s",
            workspace_id,
            company_context_pack.expansion_status,
        )
        return {
            "workspace_id": workspace_id,
            "status": company_context_pack.expansion_status,
        }
    except Exception as exc:
        logger.exception("expansion_refresh_task_failed workspace_id=%s", workspace_id)
        if company_context_pack is not None:
            company_context_pack.expansion_status = "failed"
            company_context_pack.expansion_error = str(exc)[:500] or "Expansion generation failed"
            company_context_pack.updated_at = datetime.utcnow()
            db.commit()
        return {"error": str(exc), "workspace_id": workspace_id}
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.run_claims_graph_refresh")
def run_claims_graph_refresh(workspace_id: int):
    """Refresh claims graph snapshot for a workspace."""
    db = SessionLocal()
    try:
        metrics = rebuild_workspace_claims_graph(db, workspace_id)
        db.commit()
        return {"success": True, "workspace_id": workspace_id, "metrics": metrics}
    except Exception as exc:
        db.rollback()
        return {"error": str(exc)}
    finally:
        db.close()


def _run_discovery_universe_fixture(job_id: int):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == job.workspace_id).first()
        company_context_pack = (
            db.query(CompanyContextPack)
            .filter(CompanyContextPack.workspace_id == job.workspace_id)
            .first()
        )
        if not workspace or not profile or not company_context_pack:
            return {"error": "Missing workspace/profile/company-context data"}

        from app.services.company_context import normalize_expansion_brief

        expansion_brief = normalize_expansion_brief(company_context_pack.expansion_brief_json or {})
        adjacency_boxes = [
            box for box in (expansion_brief.get("adjacency_boxes") or [])
            if str(box.get("status") or "").strip().lower() != "user_removed"
        ]
        company_seeds = [
            seed for seed in (expansion_brief.get("company_seeds") or [])
            if str(seed.get("status") or "").strip().lower() != "user_removed"
        ]
        if not adjacency_boxes:
            return {"error": "No approved adjacency boxes available for fixture discovery"}
        if not company_seeds:
            return {"error": "No company seeds available for fixture discovery"}

        screening_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        ctx = _load_discovery_context(job_id)
        ctx["screening_run_id"] = screening_run_id
        ctx["stage_execution_mode"] = "fixture"
        _save_discovery_context(job_id, ctx)

        box_by_id = {
            str(box.get("id") or ""): box
            for box in adjacency_boxes
            if str(box.get("id") or "").strip()
        }

        previous_screenings = db.query(CompanyScreening).filter(CompanyScreening.workspace_id == workspace.id).all()
        for row in previous_screenings:
            db.delete(row)
        previous_claims = db.query(CompanyClaim).filter(CompanyClaim.workspace_id == workspace.id).all()
        for row in previous_claims:
            db.delete(row)
        previous_registry_logs = db.query(RegistryQueryLog).filter(RegistryQueryLog.workspace_id == workspace.id).all()
        for row in previous_registry_logs:
            db.delete(row)
        previous_entities = db.query(CandidateEntity).filter(CandidateEntity.workspace_id == workspace.id).all()
        for entity in previous_entities:
            db.delete(entity)
        non_manual_companies = (
            db.query(Company)
            .filter(Company.workspace_id == workspace.id, Company.is_manual.is_(False))
            .all()
        )
        for company in non_manual_companies:
            db.delete(company)
        db.flush()

        created_companies = 0
        claims_created = 0
        kept_count = 0
        review_count = 0
        candidate_entities_count = 0
        decision_class_counts: dict[str, int] = {}
        evidence_sufficiency_counts: dict[str, int] = {}
        ranking_eligible_count = 0
        entity_row_by_id: dict[int, CandidateEntity] = {}
        run_screenings_for_quality_audit: list[CompanyScreening] = []

        for seed in company_seeds[:16]:
            seed_name = str(seed.get("name") or "").strip()
            if not seed_name:
                continue
            matched_boxes = [
                box_by_id.get(box_id)
                for box_id in (seed.get("fit_to_adjacency_box_ids") or [])
                if box_by_id.get(box_id)
            ]
            if not matched_boxes:
                matched_boxes = adjacency_boxes[:1]
            top_box = matched_boxes[0]
            top_priority = str(top_box.get("priority_tier") or "").strip().lower()
            decision_classification = "good_target" if top_priority == "core_adjacent" else "borderline_watchlist"
            screening_status = "kept" if decision_classification == "good_target" else "review"
            total_score = 68.0 if decision_classification == "good_target" else 52.0
            if screening_status == "kept":
                kept_count += 1
            else:
                review_count += 1
            decision_class_counts[decision_classification] = decision_class_counts.get(decision_classification, 0) + 1
            evidence_sufficiency_counts["sufficient"] = evidence_sufficiency_counts.get("sufficient", 0) + 1
            ranking_eligible_count += 1

            website = str(seed.get("website") or "").strip() or None
            candidate_domain = normalize_domain(website)
            hq_country = _infer_country_from_domain(candidate_domain) or "Unknown"
            entity = CandidateEntity(
                workspace_id=workspace.id,
                canonical_name=seed_name[:300],
                canonical_website=(website[:1000] if website else None),
                canonical_domain=candidate_domain,
                discovery_primary_url=(website[:1000] if website else None),
                entity_type="company",
                first_party_domains_json=[candidate_domain] if candidate_domain else [],
                solutions_json=[],
                country=hq_country,
                identity_confidence="medium",
                identity_error=None,
                registry_country=None,
                registry_id=None,
                registry_source=None,
                metadata_json={
                    "origin_types": ["expansion_company_seed"],
                    "fixture_mode": True,
                    "fit_to_adjacency_box_ids": seed.get("fit_to_adjacency_box_ids") or [],
                    "short_description": str(seed.get("why_relevant") or "Expansion company seed.")[:240],
                    "why_relevant": [
                        {
                            "text": str(seed.get("why_relevant") or "Expansion company seed.")[:240],
                            "citation_url": website,
                        }
                    ] if website or seed.get("why_relevant") else [],
                    "discovery_score": float(total_score),
                    "geo_signals": [hq_country] if hq_country else [],
                },
            )
            db.add(entity)
            db.flush()
            entity_row_by_id[int(entity.id)] = entity
            candidate_entities_count += 1
            db.add(
                CandidateOriginEdge(
                    entity_id=entity.id,
                    origin_type="expansion_company_seed",
                    origin_url=(website[:1000] if website else None),
                    source_run_id=None,
                    metadata_json={
                        "seed_role": seed.get("seed_role"),
                        "scope_bucket": str(top_box.get("adjacency_kind") or "").strip().lower() or None,
                        "query_type": "fixture_seed",
                        "brick_name": top_box.get("label"),
                    },
                )
            )
            validation_recommendation = (
                VALIDATION_STATUS_KEEP
                if decision_classification == "good_target"
                else VALIDATION_STATUS_WATCHLIST
            )
            set_validation_metadata(
                entity,
                {
                    "status": VALIDATION_STATUS_QUEUED,
                    "recommendation": validation_recommendation,
                    "promoted_to_cards": False,
                    "lane_ids": [
                        str(box.get("id") or "").strip()
                        for box in matched_boxes[:4]
                        if str(box.get("id") or "").strip()
                    ],
                    "lane_labels": [
                        str(box.get("label") or "").strip()
                        for box in matched_boxes[:4]
                        if str(box.get("label") or "").strip()
                    ],
                    "query_families": ["fixture_seed"],
                    "source_families": ["expansion_company_seed"],
                    "origin_types": ["expansion_company_seed"],
                    "discovery_sources": [website] if website else [],
                    "vendor_classification": "software_vendor",
                    "identity_confidence": "medium",
                    "official_website_confidence": "medium" if website else "low",
                    "priority_score": float(total_score),
                },
            )

            why_relevant = []
            for evidence in (seed.get("evidence") or [])[:2]:
                url = str(evidence.get("url") or "").strip()
                if not url:
                    continue
                why_relevant.append(
                    {
                        "text": str(evidence.get("claim") or seed.get("why_relevant") or "Expansion company seed.")[:400],
                        "citation_url": url,
                        "dimension": "adjacent_workflow_seed",
                    }
                )
            if not why_relevant and website:
                why_relevant.append(
                    {
                        "text": str(seed.get("why_relevant") or "Expansion company seed.")[:400],
                        "citation_url": website,
                        "dimension": "adjacent_workflow_seed",
                    }
                )

            company = Company(
                workspace_id=workspace.id,
                name=seed_name[:255],
                website=(website[:500] if website else None),
                hq_country=hq_country,
                tags_custom=_dedupe_strings(
                    [
                        "discovery_mode:fixture",
                        f"screening_run:{screening_run_id}",
                        f"screening_status:{screening_status}",
                        *(f"adjacency:{str(box.get('label') or '')}" for box in matched_boxes[:3]),
                    ]
                ),
                status=CompanyStatus.kept if screening_status == "kept" else CompanyStatus.candidate,
                why_relevant=why_relevant,
                is_manual=False,
            )
            db.add(company)
            db.flush()
            created_companies += 1

            for evidence in why_relevant[:2]:
                citation_url = str(evidence.get("citation_url") or "").strip()
                if not citation_url:
                    continue
                db.add(
                    SourceEvidence(
                        workspace_id=workspace.id,
                        company_id=company.id,
                        source_url=citation_url,
                        source_title=source_label_for_url(citation_url),
                        excerpt_text=str(evidence.get("text") or "")[:1200],
                        content_type="web",
                        source_tier="tier2_third_party" if citation_url != website else "tier1_first_party",
                        source_kind="expansion_seed",
                        freshness_ttl_days=90,
                        valid_through=None,
                        asserted_by="discovery_fixture_mode",
                    )
                )
                claims_created += 1

            screening = CompanyScreening(
                workspace_id=workspace.id,
                company_id=company.id,
                candidate_entity_id=entity.id,
                candidate_name=seed_name,
                candidate_website=website,
                candidate_discovery_url=website,
                candidate_official_website=website,
                screening_status=screening_status,
                total_score=total_score,
                component_scores_json={"fixture_seed_fit": total_score},
                penalties_json=[],
                reject_reasons_json=[],
                positive_reason_codes_json=["fixture_seed_match"],
                caution_reason_codes_json=[] if screening_status == "kept" else ["fixture_requires_live_validation"],
                reject_reason_codes_json=[],
                missing_claim_groups_json=[],
                unresolved_contradictions_count=0,
                decision_classification=decision_classification,
                evidence_sufficiency="sufficient",
                rationale_summary=str(seed.get("why_relevant") or f"Matched to {top_box.get('label')}.")[:500],
                rationale_markdown=str(seed.get("why_relevant") or f"Matched to {top_box.get('label')}.")[:1000],
                top_claim_json=why_relevant[0] if why_relevant else {},
                decision_engine_version="fixture-v1",
                gating_passed=True,
                ranking_eligible=True,
                screening_meta_json={
                    "job_id": job.id,
                    "screening_run_id": screening_run_id,
                    "fixture_mode": True,
                    "entity_type": "company",
                    "capability_signals": [str(box.get("label") or "") for box in matched_boxes[:4] if str(box.get("label") or "").strip()],
                    "likely_verticals": [],
                    "origin_types": ["expansion_company_seed"],
                    "scope_buckets": [str(box.get("adjacency_kind") or "").strip().lower() for box in matched_boxes[:4]],
                    "registry_identity": {},
                    "candidate_hq_country": hq_country,
                    "expansion_provenance": [
                        {
                            "query_id": seed.get("id"),
                            "query_type": "fixture_seed",
                            "query_text": seed.get("name"),
                            "provider": "expansion_brief_v3",
                            "brick_name": box.get("label"),
                            "scope_bucket": str(box.get("adjacency_kind") or "").strip().lower(),
                        }
                        for box in matched_boxes[:3]
                    ],
                },
                source_summary_json={
                    "expansion_provenance": [
                        {
                            "query_id": seed.get("id"),
                            "query_type": "fixture_seed",
                            "query_text": seed.get("name"),
                            "provider": "expansion_brief_v3",
                            "brick_name": box.get("label"),
                            "scope_bucket": str(box.get("adjacency_kind") or "").strip().lower(),
                        }
                        for box in matched_boxes[:3]
                    ]
                },
            )
            db.add(screening)
            db.flush()
            run_screenings_for_quality_audit.append(screening)

        validation_queue_candidates: list[dict[str, Any]] = []
        for screening in run_screenings_for_quality_audit:
            if not screening.ranking_eligible or not screening.candidate_entity_id:
                continue
            entity_row = entity_row_by_id.get(int(screening.candidate_entity_id))
            if entity_row is None:
                continue
            validation = validation_metadata(entity_row)
            validation_lane_ids = validation.get("lane_ids") or []
            validation_lane_labels = validation.get("lane_labels") or []
            validation_source_families = validation.get("source_families") or []
            if not (validation_lane_ids or validation_lane_labels) and set(validation_source_families) <= {"directory"}:
                continue
            validation_queue_candidates.append(
                {
                    "candidate_entity_id": int(entity_row.id),
                    "company_name": screening.candidate_name,
                    "canonical_name": entity_row.canonical_name,
                    "official_website_url": screening.candidate_official_website or entity_row.canonical_website,
                    "discovery_url": screening.candidate_discovery_url or entity_row.discovery_primary_url,
                    "entity_type": entity_row.entity_type,
                    "decision_classification": screening.decision_classification,
                    "priority_score": float(validation.get("priority_score") or screening.total_score or 0.0),
                    "multi_origin_count": len(validation.get("origin_types") or []),
                    "validation_status": validation.get("status") or VALIDATION_STATUS_QUEUED,
                    "promoted_to_cards": bool(validation.get("promoted_to_cards")),
                    "validation_lane_ids": validation_lane_ids,
                    "validation_lane_labels": validation_lane_labels,
                    "validation_query_families": validation.get("query_families") or [],
                    "validation_source_families": validation_source_families,
                }
            )

        validation_queue_ranked = build_diversified_validation_queue(
            validation_queue_candidates,
            limit=max(1, int(getattr(settings, "discovery_validation_queue_limit", 36))),
            lane_cap=max(1, int(getattr(settings, "discovery_validation_lane_cap", 6))),
            family_cap=max(1, int(getattr(settings, "discovery_validation_query_family_cap", 4))),
            source_family_cap=max(1, int(getattr(settings, "discovery_validation_source_family_cap", 18))),
        )
        queue_rank_by_entity_id = {
            int(item["candidate_entity_id"]): int(item.get("queue_rank") or 0)
            for item in validation_queue_ranked
        }
        for entity_id, entity_row in entity_row_by_id.items():
            validation = validation_metadata(entity_row)
            validation["queue_rank"] = int(queue_rank_by_entity_id.get(int(entity_id), 0))
            if validation["queue_rank"] > 0 and not str(validation.get("status") or "").strip():
                validation["status"] = VALIDATION_STATUS_QUEUED
            validation["validation_state"] = str(validation.get("status") or VALIDATION_STATUS_QUEUED)
            set_validation_metadata(entity_row, validation)

        discovery_candidate_graph_sync = {"status": "not_run"}
        try:
            graph_payload = build_discovery_candidate_graph_payload(
                workspace_id=workspace.id,
                candidates=[
                    {
                        **candidate,
                        "validation_queue_rank": queue_rank_by_entity_id.get(int(candidate["candidate_entity_id"])),
                    }
                    for candidate in validation_queue_candidates
                ],
            )
            discovery_candidate_graph_sync = Neo4jDiscoveryCandidateGraphStore().sync_graph(graph_payload)
        except Exception as discovery_graph_exc:
            discovery_candidate_graph_sync = {
                "status": "failed",
                "error": str(discovery_graph_exc),
            }

        claims_graph_metrics = rebuild_workspace_claims_graph(db, workspace.id)
        job.state = JobState.completed
        job.progress = 1.0
        job.progress_message = "Complete (fixture mode)"
        job.result_json = {
            "screening_run_id": screening_run_id,
            "companies_created": created_companies,
            "vendors_updated": 0,
            "kept_count": kept_count,
            "review_count": review_count,
            "rejected_count": 0,
            "claims_created": claims_created,
            "decision_class_counts": decision_class_counts,
            "evidence_sufficiency_counts": evidence_sufficiency_counts,
            "ranking_eligible_count": ranking_eligible_count,
            "final_universe_count": candidate_entities_count,
            "scoring_entities_count": candidate_entities_count,
            "seed_directory_count": 0,
            "seed_reference_count": 0,
            "seed_benchmark_count": 0,
            "seed_llm_count": 0,
            "seed_external_search_count": 0,
            "seed_mentions_count": 0,
            "llm_candidates_found": 0,
            "external_search_candidates_found": 0,
            "source_coverage": {"fixture": {"company_seed_count": len(company_seeds), "adjacency_box_count": len(adjacency_boxes)}},
            "run_quality_tier": "fixture",
            "quality_gate_passed": True,
            "quality_audit_passed": True,
            "quality_validation_ready": True,
            "quality_validation_blocked_reasons": [],
            "pre_rerun_quality_validation_ready": False,
            "pre_rerun_quality_validation_blocked_reasons": [],
            "degraded_reasons": [],
            "model_attempt_trace": [],
            "stage_time_ms": {},
            "timeout_events": [],
            "stage_execution_mode": "fixture",
            "fallback_mode": False,
            "validation_queue_count": len(validation_queue_ranked),
            "validation_queue_entity_ids": [int(item["candidate_entity_id"]) for item in validation_queue_ranked],
            "discovery_candidate_graph_sync": discovery_candidate_graph_sync,
            "claims_graph_refresh": claims_graph_metrics,
        }
        job.finished_at = datetime.utcnow()
        db.commit()
        return {
            "created": created_companies,
            "screening_run_id": screening_run_id,
            "stage_execution_mode": "fixture",
        }
    except Exception as exc:
        db.rollback()
        return {"error": str(exc)}
    finally:
        db.close()


def _run_discovery_universe_monolith(job_id: int):
    """Run discovery to find candidate universe."""

    db = SessionLocal()
    # This monolith mixes DB persistence with long-running network stages.
    # Keep loaded ORM state stable across commits so later attribute access
    # does not open a fresh lazy-load transaction and leave it idle while
    # search/LLM work is running.
    db.expire_on_commit = False
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.state = JobState.running
        job.started_at = datetime.utcnow()
        job.progress = 0.1
        job.progress_message = "Starting discovery..."
        db.commit()

        # Get workspace data
        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == job.workspace_id).first()
        if not workspace or not profile:
            job.state = JobState.failed
            job.error_message = "Missing workspace/profile data"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Missing workspace/profile data"}
        effective_policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)
        discovery_ctx_snapshot = _load_discovery_context(job_id)
        buyer_employee_estimate = _sanitize_employee_estimate(
            discovery_ctx_snapshot.get("buyer_employee_estimate")
            if isinstance(discovery_ctx_snapshot, dict)
            else None
        )
        if buyer_employee_estimate is None:
            buyer_employee_estimate = _resolve_buyer_employee_estimate(workspace, profile)
        from app.services.company_context import derive_discovery_scope_hints

        company_context_pack = (
            db.query(CompanyContextPack)
            .filter(CompanyContextPack.workspace_id == workspace.id)
            .first()
        )
        normalized_scope_hints = _normalize_scope_hints(
            derive_discovery_scope_hints(company_context_pack, profile)
            if company_context_pack and profile
            else {}
        )
        capability_hints, segment_hints = _taxonomy_compatible_hints_from_scope_hints(normalized_scope_hints)

        try:
            pipeline_started = time.perf_counter()
            run_ctx = _load_discovery_context(job_id)
            screening_run_id = str(run_ctx.get("screening_run_id") or "").strip()
            if not screening_run_id:
                screening_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                run_ctx["screening_run_id"] = screening_run_id
                _save_discovery_context(job_id, run_ctx)
            source_coverage: dict[str, Any] = {}
            stage_time_ms: dict[str, int] = {}
            timeout_events: list[dict[str, Any]] = []
            model_attempt_trace: list[dict[str, Any]] = []

            def _stage_started() -> float:
                return time.perf_counter()

            def _stage_finished(stage_name: str, started_at: float, timeout_seconds: int) -> None:
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                stage_time_ms[stage_name] = duration_ms
                if duration_ms > max(1, int(timeout_seconds)) * 1000:
                    timeout_events.append(
                        {
                            "stage": stage_name,
                            "duration_ms": duration_ms,
                            "timeout_seconds": int(timeout_seconds),
                        }
                    )

            france_registry_first = _should_use_france_registry_universe(profile)
            job.progress = 0.2
            job.progress_message = (
                "Building France registry universe..."
                if france_registry_first
                else "Ingesting comparator directories..."
            )
            db.commit()

            seed_stage_started = _stage_started()
            mention_records: list[dict[str, Any]] = []
            comparator_errors: list[str] = []
            source_names: list[str] = []
            if not france_registry_first:
                source_names = _discovery_source_names_for_workspace(
                    profile,
                    segment_hints=segment_hints,
                    normalized_scope=normalized_scope_hints,
                )
                for source_name in source_names:
                    source_result = ingest_source(source_name)
                    run_row = ComparatorSourceRun(
                        workspace_id=workspace.id,
                        source_name=source_result["source_name"],
                        source_url=source_result["source_url"],
                        status="completed" if not source_result.get("errors") else "completed_with_errors",
                        mentions_found=len(source_result.get("mentions", [])),
                        metadata_json={
                            "pages_crawled": source_result.get("pages_crawled", 0),
                            "errors": source_result.get("errors", []),
                            "source_tier": "tier4_discovery",
                        },
                    )
                    db.add(run_row)
                    db.flush()

                    for mention in source_result.get("mentions", []):
                        profile_url = str(
                            mention.get("profile_url")
                            or mention.get("company_url")
                            or mention.get("listing_url")
                            or source_result["source_url"]
                        ).strip()
                        official_website_url = str(mention.get("official_website_url") or "").strip() or None
                        if official_website_url and _is_non_first_party_profile_domain(normalize_domain(official_website_url)):
                            official_website_url = None
                        record = CompanyMention(
                            workspace_id=workspace.id,
                            source_run_id=run_row.id,
                            source_name=source_result["source_name"],
                            listing_url=str(mention.get("listing_url") or source_result["source_url"]),
                            company_name=str(mention.get("company_name") or "")[:300],
                            company_url=str(mention.get("company_url") or "")[:1000] or None,
                            profile_url=profile_url[:1000] if profile_url else None,
                            official_website_url=official_website_url[:1000] if official_website_url else None,
                            company_slug=str(mention.get("company_slug") or "")[:180] or None,
                            solution_slug=str(mention.get("solution_slug") or "")[:220] or None,
                            entity_type=str(mention.get("entity_type") or "company")[:32] or "company",
                            category_tags=mention.get("category_tags") or [],
                            listing_text_snippets=mention.get("listing_text_snippets") or [],
                            provenance_json={
                                **(mention.get("provenance") or {}),
                                "source_tier": "tier4_discovery",
                            },
                        )
                        db.add(record)
                        mention_records.append(
                            {
                                "company_name": record.company_name,
                                "company_url": record.company_url,
                                "profile_url": record.profile_url,
                                "official_website_url": record.official_website_url,
                                "company_slug": record.company_slug,
                                "solution_slug": record.solution_slug,
                                "entity_type": record.entity_type,
                                "listing_url": record.listing_url,
                                "category_tags": record.category_tags or [],
                                "listing_text_snippets": record.listing_text_snippets or [],
                                "source_name": record.source_name,
                                "source_run_id": run_row.id,
                            }
                        )
                    comparator_errors.extend(source_result.get("errors", []))
                    source_coverage[source_name] = {
                        "mentions": len(source_result.get("mentions", [])),
                        "pages_crawled": source_result.get("pages_crawled", 0),
                        "errors": source_result.get("errors", []),
                    }
                    db.commit()
            else:
                source_coverage["fr_registry"] = {"mode": "registry_first", "enabled": True}
            _stage_finished("stage_seed_ingest", seed_stage_started, settings.stage_seed_ingest_timeout_seconds)
            _save_stage_checkpoint(
                job_id,
                "stage_seed_ingest",
                {
                    "mentions_count": len(mention_records),
                    "source_coverage_keys": list(source_coverage.keys())[:20],
                    "comparator_errors_count": len(comparator_errors),
                    "france_registry_first": france_registry_first,
                },
            )

            mentions_by_domain, mentions_by_name = _build_mention_indexes(mention_records)

            job.progress = 0.35
            job.progress_message = "Searching and expanding candidate universe..."
            db.commit()

            llm_stage_started = _stage_started()
            llm_candidates: list[dict[str, Any]] = []
            external_search_candidates: list[dict[str, Any]] = []
            external_search_errors: list[str] = []
            llm_error: Optional[str] = None
            llm_plan_error: Optional[str] = None
            llm_synthesis_error: Optional[str] = None
            query_plan_summary: dict[str, Any] = {}
            external_search_provider_mix: dict[str, int] = {}
            external_search_dedupe_stats: dict[str, Any] = {}
            external_search_result_counts: dict[str, Any] = {}
            external_search_brick_yield: dict[str, int] = {}
            retrieval_results: list[dict[str, Any]] = []
            backfill_stats: dict[str, Any] = {"attempted": 0, "updated": 0, "errors": []}
            search_queries: list[dict[str, Any]] = []
            query_plan: dict[str, Any] = {}

            if france_registry_first:
                query_plan_summary = {
                    "france_registry_first": True,
                    "search_overlay_skipped": True,
                }
                source_coverage["external_search"] = {
                    "candidates": 0,
                    "synthesized_candidates": 0,
                    "provider_mix": {},
                    "query_plan_summary": query_plan_summary,
                    "query_plan_error": None,
                    "candidate_synthesis_enabled": False,
                    "brick_yield": {},
                    "dedupe_stats": {},
                    "result_counts": {},
                    "errors": [],
                }
                _stage_finished(
                    "stage_llm_discovery_fanout",
                    llm_stage_started,
                    settings.stage_llm_discovery_timeout_seconds,
                )
                _save_stage_checkpoint(
                    job_id,
                    "stage_llm_discovery_fanout",
                    {
                        "llm_candidates_count": 0,
                        "external_search_candidates_count": 0,
                        "external_search_results_count": 0,
                        "llm_error": None,
                        "external_search_errors_count": 0,
                    },
                )
            else:
                fallback_plan = _default_discovery_query_plan(
                    taxonomy_bricks=capability_hints,
                    geo_scope=profile.geo_scope or {},
                    vertical_focus=segment_hints,
                    scope_hints=normalized_scope_hints,
                    source_company_url=profile.buyer_company_url,
                )
                query_plan = fallback_plan
                query_plan_adjustments: list[str] = []
                preferred_countries = _normalize_string_list((profile.geo_scope or {}).get("include_countries"), max_items=8, max_len=8)
                preferred_languages = _normalize_string_list((profile.geo_scope or {}).get("languages"), max_items=6, max_len=8)
                try:
                    plan_prompt = _build_discovery_query_plan_prompt(
                        context_pack=profile.context_pack_markdown or "",
                        taxonomy_bricks=capability_hints,
                        geo_scope=profile.geo_scope or {},
                        vertical_focus=segment_hints,
                        comparator_mentions=mention_records[:120],
                        scope_hints=normalized_scope_hints,
                        source_company_url=profile.buyer_company_url,
                    )
                    plan_response = LLMOrchestrator().run_stage(
                        LLMRequest(
                            stage=LLMStage.discovery_query_planning,
                            prompt=plan_prompt,
                            timeout_seconds=max(45, int(settings.stage_llm_discovery_timeout_seconds // 2)),
                            use_web_search=False,
                            expect_json=True,
                            metadata={"workspace_id": workspace.id, "job_id": job.id},
                        )
                    )
                    model_attempt_trace.extend([asdict(attempt) for attempt in plan_response.attempts])
                    query_plan = _parse_discovery_query_plan(plan_response.text, fallback_plan)
                except Exception as exc:
                    llm_plan_error = str(exc)
                    if isinstance(exc, LLMOrchestrationError):
                        model_attempt_trace.extend([asdict(attempt) for attempt in exc.attempts])
                query_plan, query_plan_adjustments = _stabilize_discovery_query_plan(
                    query_plan,
                    fallback_plan,
                    normalized_scope=normalized_scope_hints,
                    vertical_focus=segment_hints,
                    brick_names=[str(item.get("name") or "").strip() for item in capability_hints if str(item.get("name") or "").strip()],
                    source_company_url=profile.buyer_company_url,
                )

                search_queries, query_plan_summary = _build_external_search_queries_from_plan(
                    query_plan,
                    preferred_countries=preferred_countries,
                    preferred_languages=preferred_languages,
                    normalized_scope=normalized_scope_hints,
                )
                query_plan_summary["plan_adjustments"] = query_plan_adjustments
                query_plan_summary["used_fallback_plan"] = bool(query_plan_adjustments)
            query_brick_map = {
                str(entry.get("query_id")): str(entry.get("brick_name") or "").strip()
                for entry in search_queries
                if str(entry.get("brick_name") or "").strip()
            }
            scope_bucket_map = {
                str(entry.get("query_id")): str(entry.get("scope_bucket") or "").strip().lower()
                for entry in search_queries
                if str(entry.get("scope_bucket") or "").strip()
            }

            seed_urls = _normalize_string_list(query_plan.get("seed_urls"), max_items=8, max_len=220)
            for url in (profile.comparator_seed_urls or [])[:6]:
                seed_urls.append(str(url))
            seed_urls = _dedupe_strings([normalize_url(url) for url in seed_urls if normalize_url(url)])
            known_blocked_domains = _known_discovery_blocked_domains(profile, seed_urls)
            if known_blocked_domains and not france_registry_first:
                query_plan["domain_blocklist"] = _dedupe_strings(
                    [str(item) for item in (query_plan.get("domain_blocklist") or [])] + known_blocked_domains
                )[:12]
                for query in search_queries:
                    query["domain_blocklist"] = _dedupe_strings(
                        [str(item) for item in (query.get("domain_blocklist") or [])] + known_blocked_domains
                    )[:12]
            high_confidence_seed_urls: list[str] = []
            for seed_url in seed_urls:
                seed_domain = normalize_domain(seed_url)
                if not seed_domain or _is_non_first_party_profile_domain(seed_domain):
                    continue
                if not is_trusted_source_url(seed_url):
                    continue
                high_confidence_seed_urls.append(seed_url)
            similar_seed_cap = max(0, int(getattr(settings, "discovery_retrieval_similar_seed_cap", 4)))
            if similar_seed_cap > 0 and not france_registry_first:
                for idx, seed_url in enumerate(high_confidence_seed_urls[:similar_seed_cap], start=1):
                    search_queries.append(
                        {
                            "query_id": f"seed_similar_{idx}",
                            "query_text": seed_url,
                            "query_type": "seed_similar",
                            "query_intent": "adjacent",
                            "seed_url": seed_url,
                            "scope_bucket": "adjacent",
                            "must_include_terms": query_plan.get("must_include_terms") or [],
                            "must_exclude_terms": query_plan.get("must_exclude_terms") or [],
                            "domain_allowlist": query_plan.get("domain_allowlist") or [],
                            "domain_blocklist": query_plan.get("domain_blocklist") or [],
                        }
                    )

            provider_order = [
                token.strip().lower()
                for token in str(getattr(settings, "discovery_retrieval_provider_order", "")).split(",")
                if str(token).strip()
            ]
            provider_keys = {
                "exa": bool(settings.exa_api_key),
                "brave": bool(settings.brave_api_key),
                "tavily": bool(settings.tavily_api_key),
                "serpapi": bool(settings.serpapi_api_key),
            }
            available_providers = [] if france_registry_first else [p for p in provider_order if provider_keys.get(p)]

            if not available_providers:
                external_search_errors = ["external_search_unavailable:no_provider_keys"]
            elif not search_queries:
                external_search_errors = ["external_search_unavailable:no_queries"]
            else:
                try:
                    search_output = run_external_search_queries(
                        search_queries,
                        provider_order=available_providers,
                        per_query_cap=max(1, int(getattr(settings, "discovery_retrieval_per_query_cap", 8))),
                        total_cap=max(5, int(getattr(settings, "discovery_retrieval_total_cap", 60))),
                        per_domain_cap=max(1, int(getattr(settings, "discovery_retrieval_per_domain_cap", 3))),
                    )
                    retrieval_results = [row for row in (search_output.get("results") or []) if isinstance(row, dict)]
                    retrieval_results, backfill_stats = _backfill_comparative_retrieval_context(
                        retrieval_results,
                        cap=max(0, int(getattr(settings, "discovery_retrieval_snippet_backfill_cap", 8))),
                    )
                    external_search_errors = [
                        str(item) for item in (search_output.get("errors") or []) if str(item).strip()
                    ]
                    external_search_provider_mix = search_output.get("provider_mix") or {}
                    external_search_dedupe_stats = search_output.get("dedupe_stats") or {}
                    external_search_result_counts = {
                        "raw_count": int(search_output.get("dedupe_stats", {}).get("total_in", 0)),
                        "deduped_count": len(retrieval_results),
                        "per_query_counts": search_output.get("query_counts") or {},
                        "backfill": backfill_stats,
                    }
                    for row in retrieval_results:
                        brick_name = query_brick_map.get(str(row.get("query_id") or ""))
                        if brick_name:
                            external_search_brick_yield[brick_name] = (
                                external_search_brick_yield.get(brick_name, 0) + 1
                            )
                except Exception as exc:
                    external_search_errors = [f"external_search_failed:{exc}"]

            # Persist external search run/results for auditability.
            if not france_registry_first:
                try:
                    run_key = f"job:{job_id}:run:{screening_run_id}"
                    query_plan_hash = hashlib.sha256(
                        json.dumps(query_plan, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                    run_row = ExternalSearchRun(
                        workspace_id=workspace.id,
                        job_id=job.id,
                        run_id=run_key,
                        provider_order=",".join(available_providers),
                        caps_json={
                            "per_query_cap": int(getattr(settings, "discovery_retrieval_per_query_cap", 8)),
                            "total_cap": int(getattr(settings, "discovery_retrieval_total_cap", 60)),
                            "per_domain_cap": int(getattr(settings, "discovery_retrieval_per_domain_cap", 3)),
                            "similar_seed_cap": int(getattr(settings, "discovery_retrieval_similar_seed_cap", 4)),
                        },
                        query_plan_json=query_plan if isinstance(query_plan, dict) else {},
                        query_plan_hash=query_plan_hash,
                    )
                    db.add(run_row)
                    db.flush()
                    for row in retrieval_results:
                        retrieved_at = row.get("retrieved_at")
                        parsed_retrieved_at = datetime.utcnow()
                        if isinstance(retrieved_at, str):
                            try:
                                parsed_retrieved_at = datetime.fromisoformat(retrieved_at)
                            except Exception:
                                parsed_retrieved_at = datetime.utcnow()
                        db.add(
                            ExternalSearchResult(
                                run_id=run_key,
                                provider=str(row.get("provider") or "")[:40],
                                query_id=str(row.get("query_id") or "")[:80],
                                query_type=str(row.get("query_type") or "precision")[:40],
                                query_text=str(row.get("query_text") or "")[:500],
                                rank=int(row.get("rank") or 0),
                                url=str(row.get("normalized_url") or row.get("url") or "")[:1000],
                                url_fingerprint=str(row.get("url_fingerprint") or "")[:40],
                                domain_fingerprint=str(row.get("domain_fingerprint") or "")[:40] or None,
                                title=str(row.get("title") or "")[:500] or None,
                                snippet=str(row.get("snippet") or "")[:2000] or None,
                                retrieved_at=parsed_retrieved_at,
                            )
                        )
                    db.flush()
                except Exception as exc:
                    external_search_errors.append(f"external_search_persist_failed:{exc}")

            candidate_synthesis_enabled = bool(
                retrieval_results and getattr(settings, "discovery_candidate_synthesis_enabled", False)
            )
            if candidate_synthesis_enabled:
                try:
                    synthesis_prompt = _build_candidate_synthesis_prompt(
                        retrieval_results,
                        context_pack=profile.context_pack_markdown or "",
                        taxonomy_bricks=capability_hints,
                        geo_scope=profile.geo_scope or {},
                        vertical_focus=segment_hints,
                        scope_hints=normalized_scope_hints,
                    )
                    synthesis_response = LLMOrchestrator().run_stage(
                        LLMRequest(
                            stage=LLMStage.discovery_candidate_synthesis,
                            prompt=synthesis_prompt,
                            timeout_seconds=max(45, int(settings.stage_llm_discovery_timeout_seconds // 2)),
                            use_web_search=False,
                            expect_json=True,
                            metadata={"workspace_id": workspace.id, "job_id": job.id},
                        )
                    )
                    model_attempt_trace.extend([asdict(attempt) for attempt in synthesis_response.attempts])
                    llm_candidates = _parse_discovery_candidates_from_text(synthesis_response.text)
                    llm_candidates, validation_stats = _validate_closed_world_candidates(
                        llm_candidates,
                        retrieval_results,
                    )
                    if validation_stats.get("dropped_missing_url") or validation_stats.get("dropped_missing_evidence"):
                        external_search_dedupe_stats["closed_world_drops"] = validation_stats
                except Exception as exc:
                    llm_synthesis_error = str(exc)
                    if isinstance(exc, LLMOrchestrationError):
                        model_attempt_trace.extend([asdict(attempt) for attempt in exc.attempts])
            elif retrieval_results:
                llm_synthesis_error = None

            if retrieval_results:
                external_search_candidates = _candidates_from_retrieval_results(retrieval_results)

            llm_error = llm_synthesis_error
            if not llm_error and llm_plan_error:
                llm_error = None

            source_coverage["external_search"] = {
                "candidates": len(external_search_candidates),
                "synthesized_candidates": len(llm_candidates),
                "provider_mix": external_search_provider_mix,
                "query_plan_summary": query_plan_summary,
                "query_plan_error": llm_plan_error,
                "candidate_synthesis_enabled": candidate_synthesis_enabled,
                "brick_yield": external_search_brick_yield,
                "dedupe_stats": external_search_dedupe_stats,
                "result_counts": external_search_result_counts,
                "errors": external_search_errors[:10],
            }
            _stage_finished(
                "stage_llm_discovery_fanout",
                llm_stage_started,
                settings.stage_llm_discovery_timeout_seconds,
            )
            _save_stage_checkpoint(
                job_id,
                "stage_llm_discovery_fanout",
                {
                    "llm_candidates_count": len([c for c in llm_candidates if isinstance(c, dict)]),
                    "external_search_candidates_count": len([c for c in external_search_candidates if isinstance(c, dict)]),
                    "external_search_results_count": len(retrieval_results),
                    "llm_error": llm_error,
                        "external_search_errors_count": len(external_search_errors),
                    },
                )

            if france_registry_first:
                fr_registry_candidates, fr_registry_metrics = _build_france_registry_universe_candidates(
                    profile,
                    normalized_scope_hints,
                )
                first_party_hint_urls_by_domain = {}
                source_coverage["fr_registry"] = {
                    **(source_coverage.get("fr_registry") or {}),
                    **fr_registry_metrics,
                    "candidate_count": len(fr_registry_candidates),
                }
                source_coverage["benchmark_seeds"] = {"seeded_candidates": 0, "enabled": False}
                source_coverage["directory_seeds"] = {
                    "seeded_candidates": 0,
                    "deduped_company_names": 0,
                    "limit_stats": {},
                    "profile_resolution": {"enabled": False},
                }
                raw_candidates = [candidate for candidate in fr_registry_candidates if isinstance(candidate, dict)]
                seed_llm_count = 0
                seed_external_search_count = 0
                seed_directory_count = 0
                seed_reference_count = 0
                seed_benchmark_count = 0
            else:
                seeded_candidates = _seed_candidates_from_mentions(
                    mention_records,
                    normalized_scope=normalized_scope_hints,
                )
                seeded_candidates, directory_seed_limit_stats = _limit_directory_seed_candidates(
                    seeded_candidates,
                    max_total=max(50, int(getattr(settings, "discovery_directory_seed_total_cap", 400))),
                    max_per_listing=max(25, int(getattr(settings, "discovery_directory_seed_per_listing_cap", 400))),
                    max_per_source=max(25, int(getattr(settings, "discovery_directory_seed_per_source_cap", 350))),
                    max_without_website=max(25, int(getattr(settings, "discovery_directory_seed_without_website_cap", 300))),
                )
                reference_seeded_candidates = _seed_candidates_from_reference_urls(
                    profile.comparator_seed_urls or []
                )
                benchmark_seeded_candidates: list[dict[str, Any]] = []
                if _should_add_wealth_benchmark_seeds(profile, segment_hints, mention_records):
                    benchmark_seeded_candidates = _seed_candidates_from_benchmark_list()
                first_party_hint_urls_by_domain = _build_first_party_hint_url_map(
                    profile=profile,
                    include_benchmark_hints=bool(benchmark_seeded_candidates),
                )
                source_coverage["benchmark_seeds"] = {
                    "seeded_candidates": len(benchmark_seeded_candidates),
                    "enabled": bool(benchmark_seeded_candidates),
                }
                source_coverage["directory_seeds"] = {
                    "seeded_candidates": len(seeded_candidates),
                    "deduped_company_names": len({str(candidate.get("name") or "").strip().lower() for candidate in seeded_candidates if str(candidate.get("name") or "").strip()}),
                    "limit_stats": directory_seed_limit_stats,
                }
                directory_profile_resolution_enabled = bool(
                    getattr(settings, "discovery_directory_profile_resolution_enabled", True)
                )
                directory_profile_resolution_stats: dict[str, Any] = {}
                if directory_profile_resolution_enabled and seeded_candidates:
                    directory_profile_resolution_stats = _resolve_directory_profile_seed_candidates(
                        seeded_candidates,
                        max_fetches=max(
                            0,
                            int(getattr(settings, "discovery_directory_profile_resolution_cap", 30)),
                        ),
                        timeout_seconds=IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
                        concurrency=IDENTITY_RESOLUTION_CONCURRENCY,
                    )
                source_coverage["directory_seeds"]["profile_resolution"] = {
                    "enabled": directory_profile_resolution_enabled,
                    **directory_profile_resolution_stats,
                }
                for candidate in llm_candidates:
                    if not isinstance(candidate, dict):
                        continue
                    website = str(candidate.get("website") or "").strip()
                    if website and not website.startswith(("http://", "https://")):
                        website = f"https://{website}"
                    website_domain = normalize_domain(website)
                    candidate["website"] = website or None
                    candidate["official_website_url"] = website or None
                    candidate["discovery_url"] = candidate.get("discovery_url") or website or None
                    candidate["entity_type"] = str(candidate.get("entity_type") or "company").strip().lower() or "company"
                    if website_domain and not _is_non_first_party_profile_domain(website_domain):
                        candidate["first_party_domains"] = _dedupe_strings(
                            [str(v) for v in (candidate.get("first_party_domains") or []) + [website_domain]]
                        )
                    else:
                        candidate["first_party_domains"] = _dedupe_strings(
                            [str(v) for v in (candidate.get("first_party_domains") or []) if str(v).strip()]
                        )
                    candidate.setdefault(
                        "_origins",
                        [
                            {
                                "origin_type": "llm_seed",
                                "origin_url": str(candidate.get("discovery_url") or candidate.get("website") or ""),
                                "source_name": "external_search_synthesis",
                                "source_run_id": None,
                                "metadata": {},
                            }
                        ],
                    )

                raw_candidates = []
                raw_candidates.extend([c for c in llm_candidates if isinstance(c, dict)])
                raw_candidates.extend([c for c in external_search_candidates if isinstance(c, dict)])
                raw_candidates.extend(seeded_candidates)
                raw_candidates.extend(reference_seeded_candidates)
                raw_candidates.extend(benchmark_seeded_candidates)
                seed_llm_count = len([c for c in llm_candidates if isinstance(c, dict)])
                seed_external_search_count = len([c for c in external_search_candidates if isinstance(c, dict)])
                seed_directory_count = len(seeded_candidates)
                seed_reference_count = len(reference_seeded_candidates)
                seed_benchmark_count = len(benchmark_seeded_candidates)

            registry_stage_started = _stage_started()
            identity_resolution_enabled = bool(
                getattr(settings, "discovery_identity_resolution_enabled", False)
            ) and not france_registry_first
            identity_seed_stats: dict[str, Any] = {}
            if identity_resolution_enabled:
                identity_seed_stats = _resolve_identities_for_candidates(
                    raw_candidates,
                    max_fetches=MAX_IDENTITY_FETCHES_PER_RUN,
                    timeout_seconds=IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
                    concurrency=IDENTITY_RESOLUTION_CONCURRENCY,
                )
            previous_entities = db.query(CandidateEntity).filter(CandidateEntity.workspace_id == workspace.id).all()
            previous_entity_ids = [int(entity.id) for entity in previous_entities if getattr(entity, "id", None) is not None]
            previous_aliases = []
            if previous_entity_ids:
                previous_aliases = (
                    db.query(CandidateEntityAlias)
                    .filter(CandidateEntityAlias.entity_id.in_(previous_entity_ids))
                    .all()
                )
            existing_companies = db.query(Company).filter(Company.workspace_id == workspace.id).all()
            known_entity_profile = _known_entity_suppression_profile(
                profile,
                existing_companies=existing_companies,
            )
            raw_candidates, known_entity_raw_stats = _suppress_known_entity_candidates(
                raw_candidates,
                known_entity_profile,
            )
            pre_registry_entities, pre_registry_metrics = _collapse_candidates_to_entities(raw_candidates)
            registry_expansion_enabled = bool(
                getattr(settings, "discovery_registry_expansion_enabled", False)
            ) and not france_registry_first
            registry_identity_metrics: dict[str, Any] = {}
            registry_identity_query_logs: list[dict[str, Any]] = []
            registry_neighbor_candidates: list[dict[str, Any]] = []
            registry_expansion_metrics: dict[str, Any] = {}
            registry_neighbor_query_logs: list[dict[str, Any]] = []
            identity_registry_stats: dict[str, Any] = {}

            if registry_expansion_enabled:
                pre_registry_entities, registry_identity_metrics, registry_identity_query_logs = _apply_registry_identity_map(
                    pre_registry_entities,
                    run_id=screening_run_id,
                )

                job.progress = 0.45
                job.progress_message = "Expanding with national registries..."
                db.commit()

                registry_neighbor_candidates, registry_expansion_metrics, registry_neighbor_query_logs = _expand_registry_neighbors(
                    pre_registry_entities,
                    run_id=screening_run_id,
                )
                if identity_resolution_enabled:
                    identity_registry_stats = _resolve_identities_for_candidates(
                        registry_neighbor_candidates,
                        max_fetches=MAX_IDENTITY_FETCHES_PER_RUN,
                        timeout_seconds=IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
                        concurrency=IDENTITY_RESOLUTION_CONCURRENCY,
                    )
            else:
                job.progress = 0.45
                job.progress_message = "Skipping registry expansion..."
                db.commit()
            _stage_finished(
                "stage_registry_identity_expand",
                registry_stage_started,
                settings.stage_registry_timeout_seconds,
            )
            registry_queries_count_checkpoint = int(
                sum(
                    int(value)
                    for value in (registry_identity_metrics.get("registry_queries_by_country", {}) or {}).values()
                )
            ) + int(
                sum(
                    int(value)
                    for value in (registry_expansion_metrics.get("registry_queries_by_country", {}) or {}).values()
                )
            )
            _save_stage_checkpoint(
                job_id,
                "stage_registry_identity_expand",
                {
                    "raw_candidates_count": len(raw_candidates),
                    "registry_neighbor_candidates_count": len(registry_neighbor_candidates),
                    "pre_registry_entities_count": len(pre_registry_entities),
                    "registry_queries_count": int(registry_queries_count_checkpoint),
                    "known_entity_raw_drops": int(known_entity_raw_stats.get("dropped_count") or 0),
                    "identity_resolution_enabled": identity_resolution_enabled,
                    "registry_expansion_enabled": registry_expansion_enabled,
                },
            )

            all_candidates = raw_candidates + registry_neighbor_candidates
            all_candidates, known_entity_final_stats = _suppress_known_entity_candidates(
                all_candidates,
                known_entity_profile,
            )
            canonical_entities, canonical_metrics = _collapse_candidates_to_entities(all_candidates)
            candidate_entity_cap = max(
                100,
                int(getattr(settings, "discovery_candidate_entity_cap", CANDIDATE_ENTITY_CAP_DEFAULT)),
            )
            canonical_entities, trimmed_out_count = _trim_entities_for_universe(
                canonical_entities,
                cap=candidate_entity_cap,
            )
            canonical_entities = [
                entity
                for entity in canonical_entities
                if _is_persistable_vendor_entity(entity)
            ]
            registry_neighbors_unique_post_dedupe = len(
                [
                    entity for entity in canonical_entities
                    if "registry_neighbor" in set(entity.get("origin_types") or [])
                ]
            )

            job.progress = 0.55
            job.progress_message = "Preparing canonical scoring pool..."
            db.commit()

            # Reset canonical entity tables for this workspace (latest run snapshot).
            for previous in previous_entities:
                db.delete(previous)
            db.flush()

            # Persist registry query diagnostics for this run.
            for query_log in registry_identity_query_logs + registry_neighbor_query_logs:
                db.add(
                    RegistryQueryLog(
                        workspace_id=workspace.id,
                        run_id=screening_run_id,
                        seed_entity_name=(str(query_log.get("seed_entity_name") or "")[:300] or None),
                        query_type=str(query_log.get("query_type") or "neighbor_expand")[:40],
                        country=str(query_log.get("country") or "UNKNOWN")[:16],
                        source_name=str(query_log.get("source_name") or "unknown")[:120],
                        query=str(query_log.get("query") or "")[:300],
                        raw_hits=int(query_log.get("raw_hits") or 0),
                        kept_hits=int(query_log.get("kept_hits") or 0),
                        reject_reasons_json=query_log.get("reject_reasons_json") if isinstance(query_log.get("reject_reasons_json"), dict) else {},
                        metadata_json=query_log.get("metadata_json") if isinstance(query_log.get("metadata_json"), dict) else {},
                    )
                )

            entity_id_map: dict[str, int] = {}
            entity_row_by_id: dict[int, CandidateEntity] = {}
            origin_mix_distribution: dict[str, int] = {}
            alias_clusters_count = 0
            for entity in canonical_entities:
                why_relevant = entity.get("why_relevant") if isinstance(entity.get("why_relevant"), list) else []
                short_description = str(entity.get("short_description") or "").strip()[:240] or None
                if not short_description:
                    for reason in why_relevant:
                        if isinstance(reason, dict) and str(reason.get("text") or "").strip():
                            short_description = str(reason.get("text") or "").strip()[:240]
                            break
                row = CandidateEntity(
                    workspace_id=workspace.id,
                    canonical_name=str(entity.get("canonical_name") or "")[:300],
                    canonical_website=str(entity.get("canonical_website") or "")[:1000] or None,
                    canonical_domain=normalize_domain(entity.get("canonical_website")),
                    discovery_primary_url=str(entity.get("discovery_primary_url") or "")[:1000] or None,
                    entity_type=str(entity.get("entity_type") or "company")[:32] or "company",
                    first_party_domains_json=entity.get("first_party_domains") if isinstance(entity.get("first_party_domains"), list) else [],
                    solutions_json=entity.get("solutions") if isinstance(entity.get("solutions"), list) else [],
                    country=normalize_country(entity.get("country")),
                    identity_confidence=str(entity.get("identity_confidence") or "low")[:20],
                    identity_error=(str(entity.get("identity_error") or "")[:255] or None),
                    registry_country=normalize_country(entity.get("registry_country")),
                    registry_id=(str(entity.get("registry_id") or "")[:128] or None),
                    registry_source=(str(entity.get("registry_source") or "")[:120] or None),
                    metadata_json={
                        "display_name": str(entity.get("display_name") or entity.get("canonical_name") or "")[:300] or None,
                        "legal_name": (str(entity.get("legal_name") or "")[:300] or None),
                        "brand_names": entity.get("brand_names") if isinstance(entity.get("brand_names"), list) else [],
                        "merge_rationale": entity.get("merge_reasons") or [],
                        "origin_types": entity.get("origin_types") or [],
                        "merged_candidates_count": int(entity.get("merged_candidates_count") or 1),
                        "merge_confidence": float(entity.get("merge_confidence") or 0.0),
                        "registry_identity": entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {},
                        "registry_fields": entity.get("registry_fields") if isinstance(entity.get("registry_fields"), dict) else {},
                        "industry_signature": entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
                        "entity_type": str(entity.get("entity_type") or "company"),
                        "first_party_domains": entity.get("first_party_domains") if isinstance(entity.get("first_party_domains"), list) else [],
                        "solutions": entity.get("solutions") if isinstance(entity.get("solutions"), list) else [],
                        "qualification": entity.get("qualification") if isinstance(entity.get("qualification"), dict) else {},
                        "why_relevant": why_relevant[:12],
                        "short_description": short_description,
                        "discovery_primary_url": entity.get("discovery_primary_url"),
                        "directness": str(entity.get("directness") or "").strip() or None,
                        "node_fit_summary": entity.get("node_fit_summary") if isinstance(entity.get("node_fit_summary"), dict) else {},
                        "priority_score": _candidate_priority_score(entity),
                        "discovery_score": _candidate_priority_score(entity),
                        "geo_signals": _dedupe_strings([normalize_country(entity.get("country")), normalize_country(entity.get("registry_country"))]),
                        "suspected_duplicates": canonical_metrics.get("suspected_duplicates", []),
                    },
                )
                db.add(row)
                db.flush()
                entity_id_map[str(entity.get("temp_entity_id"))] = row.id
                entity_row_by_id[int(row.id)] = row
                entity["db_entity_id"] = row.id

                if len(entity.get("alias_names") or []) > 1:
                    alias_clusters_count += 1

                merge_reason_text = "; ".join(entity.get("merge_reasons") or [])[:255] or None
                merge_confidence = float(entity.get("merge_confidence") or 0.0)
                for alias_name in entity.get("alias_names") or []:
                    alias_name = str(alias_name or "").strip()
                    if not alias_name:
                        continue
                    db.add(
                        CandidateEntityAlias(
                            entity_id=row.id,
                            alias_name=alias_name[:300],
                            alias_website=None,
                            source_name="alias_collapse",
                            merge_confidence=merge_confidence,
                            merge_reason=merge_reason_text,
                            metadata_json={},
                        )
                    )
                for alias_website in entity.get("alias_websites") or []:
                    alias_website = str(alias_website or "").strip()
                    if not alias_website:
                        continue
                    db.add(
                        CandidateEntityAlias(
                            entity_id=row.id,
                            alias_name=None,
                            alias_website=alias_website[:1000],
                            source_name="alias_collapse",
                            merge_confidence=merge_confidence,
                            merge_reason=merge_reason_text,
                            metadata_json={},
                        )
                    )

                for origin in entity.get("origins") or []:
                    if not isinstance(origin, dict):
                        continue
                    origin_type = str(origin.get("origin_type") or "").strip()
                    if not origin_type:
                        continue
                    origin_mix_distribution[origin_type] = origin_mix_distribution.get(origin_type, 0) + 1
                    source_run_id = origin.get("source_run_id")
                    source_run_id = source_run_id if isinstance(source_run_id, int) else None
                    origin_metadata = origin.get("metadata") if isinstance(origin.get("metadata"), dict) else {}
                    if origin_type == "registry_neighbor" and "query_type" not in origin_metadata:
                        origin_metadata["query_type"] = "neighbor_expand"
                    if origin_type == "registry_identity" and "query_type" not in origin_metadata:
                        origin_metadata["query_type"] = "identity_map"
                    db.add(
                        CandidateOriginEdge(
                            entity_id=row.id,
                            origin_type=origin_type[:40],
                            origin_url=(str(origin.get("origin_url") or "")[:1000] or None),
                            source_run_id=source_run_id,
                            metadata_json=origin_metadata,
                        )
                    )

                validation_lane_ids, validation_lane_labels = _candidate_validation_lane_metadata(
                    entity,
                    scope_buckets=_dedupe_strings(
                        [
                            str((origin.get("metadata") or {}).get("scope_bucket") or "").strip().lower()
                            for origin in (entity.get("origins") or [])
                            if isinstance(origin, dict) and isinstance(origin.get("metadata"), dict)
                        ]
                    ),
                )
                discovery_query_families, discovery_source_families = _candidate_origin_discovery_families(
                    [origin for origin in (entity.get("origins") or []) if isinstance(origin, dict)]
                )
                set_validation_metadata(
                    row,
                    {
                        "status": VALIDATION_STATUS_QUEUED,
                        "recommendation": VALIDATION_STATUS_QUEUED,
                        "promoted_to_cards": False,
                        "lane_ids": validation_lane_ids,
                        "lane_labels": validation_lane_labels,
                        "query_families": discovery_query_families,
                        "source_families": discovery_source_families,
                        "origin_types": _dedupe_strings(entity.get("origin_types") or [])[:8],
                        "identity_confidence": str(row.identity_confidence or "low"),
                        "short_description": short_description,
                        "validation_score": 0.0,
                        "priority_score": float(_candidate_priority_score(entity)),
                        "display_name": str(entity.get("display_name") or entity.get("canonical_name") or "").strip() or None,
                        "legal_name": str(entity.get("legal_name") or "").strip() or None,
                    },
                )

            # Existing companies index
            existing_by_domain: dict[str, Company] = {}
            existing_by_name: dict[str, Company] = {}
            for company in existing_companies:
                domain = normalize_domain(company.website)
                if domain:
                    existing_by_domain[domain] = company
                existing_by_name[company.name.strip().lower()] = company

            registry_neighbors_with_first_party_website_count = int(
                registry_expansion_metrics.get("registry_neighbors_with_first_party_website_count", 0)
            )
            registry_neighbors_dropped_missing_official_website_count = int(
                registry_expansion_metrics.get("registry_neighbors_dropped_missing_official_website_count", 0)
            )
            registry_origin_screening_counts = (
                registry_expansion_metrics.get("registry_origin_screening_counts", {})
                if isinstance(registry_expansion_metrics.get("registry_origin_screening_counts"), dict)
                else {}
            )

            created_companies = 0
            updated_existing = 0
            kept_count = 0
            review_count = 0
            rejected_count = 0
            external_search_scored_count = 0
            external_search_kept_count = 0
            external_search_review_count = 0
            untrusted_sources_skipped = 0
            filter_reason_counts: dict[str, int] = {}
            penalties_count: dict[str, int] = {}
            claims_created = 0
            decision_class_counts: dict[str, int] = {}
            evidence_sufficiency_counts: dict[str, int] = {}
            ranking_eligible_count = 0
            unresolved_directory_only_skipped_count = 0
            run_screenings_for_quality_audit: list[CompanyScreening] = []
            run_claims_by_screening_id: dict[int, list[dict[str, Any]]] = {}
            solution_entity_screening_count = 0
            first_party_reason_cache: dict[str, list[dict[str, str]]] = {}
            first_party_capability_cache: dict[str, list[str]] = {}
            first_party_meta_cache: dict[str, dict[str, Any]] = {}
            first_party_error_cache: dict[str, str] = {}
            adaptive_hint_cache_by_domain: dict[str, list[str]] = {}
            adaptive_hint_domain_stats: dict[str, dict[str, int]] = {}
            first_party_fetch_budget_default = max(0, int(getattr(settings, "first_party_fetch_budget", FIRST_PARTY_FETCH_BUDGET)))
            first_party_crawl_budget_default = max(0, int(getattr(settings, "first_party_crawl_budget", FIRST_PARTY_CRAWL_BUDGET)))
            first_party_crawl_deep_budget_default = max(
                0,
                int(getattr(settings, "first_party_crawl_deep_budget", FIRST_PARTY_CRAWL_DEEP_BUDGET)),
            )
            first_party_hint_crawl_budget_default = max(
                0,
                int(getattr(settings, "first_party_hint_crawl_budget", FIRST_PARTY_HINT_CRAWL_BUDGET)),
            )
            adaptive_hint_discovery_enabled = _adaptive_hint_discovery_enabled(
                first_party_enrichment_enabled=bool(getattr(settings, "discovery_first_party_enrichment_enabled", False)),
                first_party_hint_crawl_budget_default=first_party_hint_crawl_budget_default,
                first_party_crawl_budget_default=first_party_crawl_budget_default,
            )
            adaptive_hint_domain_budget_default = max(
                0,
                int(getattr(settings, "first_party_adaptive_hint_domain_budget", 25))
                if adaptive_hint_discovery_enabled
                else 0,
            )
            adaptive_hint_domain_budget = adaptive_hint_domain_budget_default
            first_party_crawl_light_max_pages = max(
                1,
                int(getattr(settings, "first_party_crawl_light_max_pages", FIRST_PARTY_CRAWL_LIGHT_MAX_PAGES)),
            )
            first_party_crawl_deep_max_pages = max(
                first_party_crawl_light_max_pages,
                int(getattr(settings, "first_party_crawl_deep_max_pages", FIRST_PARTY_CRAWL_DEEP_MAX_PAGES)),
            )
            first_party_min_priority_for_crawl = float(
                getattr(settings, "first_party_min_priority_for_crawl", 0.0)
            )
            first_party_fetch_budget = first_party_fetch_budget_default
            first_party_crawl_budget = first_party_crawl_budget_default
            first_party_crawl_deep_budget = min(first_party_crawl_budget, first_party_crawl_deep_budget_default)
            first_party_hint_crawl_budget = first_party_hint_crawl_budget_default
            first_party_crawl_attempted_count = 0
            first_party_crawl_success_count = 0
            first_party_crawl_failed_count = 0
            first_party_crawl_deep_count = 0
            first_party_crawl_light_count = 0
            first_party_crawl_fallback_count = 0
            first_party_crawl_pages_total = 0
            first_party_hint_urls_used_count = 0
            first_party_hint_pages_crawled_total = 0
            first_party_crawl_errors: list[str] = []

            max_scoring_entities = max(25, int(getattr(settings, "discovery_scoring_entities_cap", len(canonical_entities))))
            ranked_scoring_entities = sorted(canonical_entities, key=_candidate_priority_score, reverse=True)
            scoring_entities, scoring_selection_meta = _select_scoring_entities(
                canonical_entities,
                cap=max_scoring_entities,
            )
            scoring_entities_skipped_count = max(0, len(ranked_scoring_entities) - len(scoring_entities))
            scoring_write_batch_size = 5
            if scoring_entities_skipped_count > 0:
                job.progress_message = (
                    f"Scoring {len(scoring_entities)} canonical candidates "
                    f"(capped from {len(ranked_scoring_entities)})..."
                )
            else:
                job.progress_message = f"Scoring {len(scoring_entities)} canonical candidates..."
            db.commit()
            enrichment_stage_started = _stage_started()

            scoring_total = max(1, len(scoring_entities))
            for entity_index, entity in enumerate(scoring_entities, start=1):
                official_website = str(entity.get("canonical_website") or "").strip() or None
                official_domain = normalize_domain(official_website)
                first_party_domains = _normalize_domain_list(entity.get("first_party_domains") or [])
                if official_domain and official_domain not in first_party_domains and not _is_non_first_party_profile_domain(official_domain):
                    first_party_domains.append(official_domain)
                candidate = {
                    "name": str(entity.get("canonical_name") or "").strip(),
                    "website": official_website,
                    "official_website_url": official_website,
                    "discovery_url": entity.get("discovery_primary_url"),
                    "profile_url": entity.get("discovery_primary_url"),
                    "entity_type": str(entity.get("entity_type") or "company").strip().lower() or "company",
                    "first_party_domains": first_party_domains,
                    "solutions": entity.get("solutions") if isinstance(entity.get("solutions"), list) else [],
                    "hq_country": entity.get("country"),
                    "likely_verticals": entity.get("likely_verticals") or [],
                    "employee_estimate": entity.get("employee_estimate"),
                    "capability_signals": entity.get("capability_signals") or [],
                    "qualification": entity.get("qualification") or {},
                    "reference_input": bool(entity.get("reference_input")),
                    "why_relevant": entity.get("why_relevant") or [],
                    "registry_id": entity.get("registry_id"),
                    "registry_source": entity.get("registry_source"),
                    "registry_country": entity.get("registry_country"),
                    "registry_identity": entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {},
                    "industry_signature": entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
                    "original_website": entity.get("discovery_primary_url") or entity.get("canonical_website"),
                    "identity": {
                        "input_website": entity.get("discovery_primary_url") or entity.get("canonical_website"),
                        "official_website": entity.get("canonical_website"),
                        "canonical_domain": official_domain,
                        "identity_confidence": entity.get("identity_confidence"),
                        "identity_sources": [origin.get("origin_url") for origin in (entity.get("origins") or []) if origin.get("origin_url")][:3],
                        "captured_at": datetime.utcnow().isoformat(),
                        "error": entity.get("identity_error"),
                    },
                }
                candidate_name = str(candidate.get("name") or "").strip()
                if not candidate_name:
                    continue
                is_solution_entity = str(candidate.get("entity_type") or "company") == "solution"
                candidate_domain = normalize_domain(candidate.get("website"))
                candidate_first_party_domains = _normalize_domain_list(candidate.get("first_party_domains") or [])
                origin_types = set(entity.get("origin_types") or [])
                if origin_types == {"directory_seed"} and not candidate_domain:
                    unresolved_directory_only_skipped_count += 1
                    filter_reason_counts["directory_only_missing_first_party"] = (
                        filter_reason_counts.get("directory_only_missing_first_party", 0) + 1
                    )
                    continue
                if is_solution_entity:
                    solution_entity_screening_count += 1
                priority_score = _candidate_priority_score(entity)
                if not is_solution_entity and candidate_domain and not _is_non_first_party_profile_domain(candidate_domain):
                    existing_company = existing_by_domain.get(candidate_domain)
                elif not is_solution_entity:
                    existing_company = existing_by_name.get(candidate_name.lower())
                else:
                    existing_company = None

                capability_signals = _extract_capability_signals(candidate)
                employee_estimate = _extract_employee_estimate(candidate)
                likely_verticals = [
                    str(v).strip()
                    for v in (candidate.get("likely_verticals") or [])
                    if isinstance(v, str) and str(v).strip()
                ]
                why_relevant = [
                    item
                    for item in (candidate.get("why_relevant") or [])
                    if isinstance(item, dict)
                ]
                identity_meta = candidate.get("identity") if isinstance(candidate.get("identity"), dict) else {}
                if identity_meta.get("input_website") and identity_meta.get("official_website"):
                    input_website = str(identity_meta.get("input_website"))
                    official_website = str(identity_meta.get("official_website"))
                    if input_website and official_website and input_website != official_website:
                        why_relevant.append(
                            {
                                "text": f"Official website resolved from comparator profile: {official_website}",
                                "citation_url": input_website,
                                "dimension": "company_profile",
                            }
                        )

                if not candidate.get("hq_country"):
                    inferred_country = _infer_country_from_domain(candidate_domain)
                    if inferred_country:
                        candidate["hq_country"] = inferred_country

                matched_mentions = _match_mentions_for_candidate(candidate, mentions_by_domain, mentions_by_name)
                for mention in matched_mentions:
                    for snippet in (mention.get("listing_text_snippets") or [])[:1]:
                        if not isinstance(snippet, str) or not snippet.strip():
                            continue
                        why_relevant.append(
                            {
                                "text": snippet[:700],
                                "citation_url": mention.get("listing_url"),
                                "dimension": "directory_context",
                            }
                        )

                for registry_reason in _registry_lookup_reasons(candidate_name, candidate.get("hq_country")):
                    why_relevant.append(registry_reason)

                normalized_reasons = _normalize_reasons(why_relevant)
                # Add tiered first-party crawl signals (deep/light), fallback to lightweight fetch when needed.
                first_party_enrichment_meta: dict[str, Any] = {}
                first_party_enrichment_enabled = bool(
                    getattr(settings, "discovery_first_party_enrichment_enabled", False)
                )
                candidate_website = str(candidate.get("website") or "").strip()
                candidate_domain_for_fetch = normalize_domain(candidate_website)
                candidate_hint_domains = _normalize_domain_list(
                    ([candidate_domain_for_fetch] if candidate_domain_for_fetch else [])
                    + (candidate_first_party_domains or [])
                )
                candidate_hint_urls: list[str] = []
                if first_party_enrichment_enabled and adaptive_hint_discovery_enabled:
                    candidate_hint_urls = _collect_hint_urls_for_domains(
                        first_party_hint_urls_by_domain,
                        candidate_hint_domains,
                    )
                    auto_hint_urls = _auto_first_party_hint_urls_for_domains(candidate_hint_domains)
                    adaptive_hint_urls: list[str] = []
                    adaptive_hint_timeout = max(2, int(getattr(settings, "first_party_adaptive_hint_timeout_seconds", 6)))
                    adaptive_hint_max_urls = max(5, int(getattr(settings, "first_party_adaptive_hint_max_urls_per_domain", 40)))
                    for hint_domain in candidate_hint_domains:
                        cached_adaptive = adaptive_hint_cache_by_domain.get(hint_domain)
                        if cached_adaptive is None:
                            if adaptive_hint_domain_budget > 0:
                                try:
                                    cached_adaptive = _discover_adaptive_hint_urls_for_domain(
                                        hint_domain,
                                        timeout_seconds=adaptive_hint_timeout,
                                        max_urls=adaptive_hint_max_urls,
                                    )
                                except Exception:
                                    cached_adaptive = []
                                adaptive_hint_domain_budget -= 1
                            else:
                                cached_adaptive = []
                            adaptive_hint_cache_by_domain[hint_domain] = cached_adaptive
                        stats = adaptive_hint_domain_stats.setdefault(
                            hint_domain,
                            {
                                "discovered_count": 0,
                                "used_count": 0,
                                "hint_pages_crawled": 0,
                            },
                        )
                        stats["discovered_count"] = max(
                            int(stats.get("discovered_count", 0)),
                            len(cached_adaptive),
                        )
                        adaptive_hint_urls.extend(cached_adaptive)
                    candidate_hint_urls = _dedupe_strings(candidate_hint_urls + auto_hint_urls + adaptive_hint_urls)
                    for hint_url in candidate_hint_urls:
                        hint_domain = normalize_domain(hint_url)
                        if not hint_domain:
                            continue
                        stats = adaptive_hint_domain_stats.setdefault(
                            hint_domain,
                            {
                                "discovered_count": 0,
                                "used_count": 0,
                                "hint_pages_crawled": 0,
                            },
                        )
                        stats["used_count"] = int(stats.get("used_count", 0)) + 1
                if (
                    first_party_enrichment_enabled
                    and candidate_website
                    and candidate_domain_for_fetch
                    and not _is_non_first_party_profile_domain(candidate_domain_for_fetch)
                    and is_trusted_source_url(candidate_website)
                ):
                    cached_first_party = first_party_reason_cache.get(candidate_domain_for_fetch)
                    cached_capabilities = first_party_capability_cache.get(candidate_domain_for_fetch, [])
                    cached_meta = first_party_meta_cache.get(candidate_domain_for_fetch, {})
                    if cached_first_party is None:
                        if candidate_hint_urls and first_party_hint_crawl_budget > 0:
                            first_party_crawl_attempted_count += 1
                            first_party_crawl_deep_count += 1
                            first_party_hint_crawl_budget -= 1
                            first_party_hint_urls_used_count += len(candidate_hint_urls)
                            crawled_reasons, crawled_capabilities, crawl_meta, crawl_error = _extract_first_party_signals_from_crawl(
                                candidate_website,
                                candidate_name,
                                max_pages=first_party_crawl_deep_max_pages,
                                hint_urls=candidate_hint_urls,
                            )
                            crawl_meta = crawl_meta if isinstance(crawl_meta, dict) else {}
                            crawl_meta = {
                                **crawl_meta,
                                "tier": "hint",
                            }
                            if crawled_reasons:
                                first_party_crawl_success_count += 1
                                first_party_crawl_pages_total += int(crawl_meta.get("pages_crawled") or 0)
                                first_party_hint_pages_crawled_total += int(crawl_meta.get("hint_pages_crawled") or 0)
                                for hit_url in (crawl_meta.get("hint_hit_urls") or []):
                                    hit_domain = normalize_domain(hit_url)
                                    if not hit_domain:
                                        continue
                                    stats = adaptive_hint_domain_stats.setdefault(
                                        hit_domain,
                                        {"discovered_count": 0, "used_count": 0, "hint_pages_crawled": 0},
                                    )
                                    stats["hint_pages_crawled"] = int(stats.get("hint_pages_crawled", 0)) + 1
                                first_party_reason_cache[candidate_domain_for_fetch] = crawled_reasons
                                first_party_capability_cache[candidate_domain_for_fetch] = crawled_capabilities
                                first_party_meta_cache[candidate_domain_for_fetch] = crawl_meta
                                cached_first_party = crawled_reasons
                                cached_capabilities = crawled_capabilities
                                cached_meta = crawl_meta
                            else:
                                first_party_crawl_failed_count += 1
                                if crawl_error:
                                    first_party_error_cache[candidate_domain_for_fetch] = crawl_error
                                    first_party_crawl_errors.append(crawl_error)
                        if (
                            cached_first_party is None
                            and first_party_crawl_budget > 0
                            and (
                                bool(candidate_hint_urls)
                                or bool(candidate.get("reference_input"))
                                or "reference_seed" in origin_types
                                or "benchmark_seed" in origin_types
                                or priority_score >= first_party_min_priority_for_crawl
                            )
                        ):
                            first_party_crawl_attempted_count += 1
                            use_deep_crawl = (
                                first_party_crawl_deep_budget > 0
                                and (
                                    bool(candidate.get("reference_input"))
                                    or "reference_seed" in origin_types
                                    or "benchmark_seed" in origin_types
                                    or priority_score >= FIRST_PARTY_CRAWL_DEEP_PRIORITY_THRESHOLD
                                )
                            )
                            max_pages = first_party_crawl_deep_max_pages if use_deep_crawl else first_party_crawl_light_max_pages
                            if use_deep_crawl:
                                first_party_crawl_deep_budget -= 1
                                first_party_crawl_deep_count += 1
                            else:
                                first_party_crawl_light_count += 1

                            crawled_reasons, crawled_capabilities, crawl_meta, crawl_error = _extract_first_party_signals_from_crawl(
                                candidate_website,
                                candidate_name,
                                max_pages=max_pages,
                                hint_urls=candidate_hint_urls,
                            )
                            first_party_crawl_budget -= 1

                            crawl_meta = crawl_meta if isinstance(crawl_meta, dict) else {}
                            crawl_meta = {
                                **crawl_meta,
                                "tier": "deep" if use_deep_crawl else "light",
                            }

                            if crawled_reasons:
                                first_party_crawl_success_count += 1
                                first_party_crawl_pages_total += int(crawl_meta.get("pages_crawled") or 0)
                                first_party_hint_pages_crawled_total += int(crawl_meta.get("hint_pages_crawled") or 0)
                                first_party_hint_urls_used_count += int(crawl_meta.get("hint_urls_used_count") or 0)
                                for hit_url in (crawl_meta.get("hint_hit_urls") or []):
                                    hit_domain = normalize_domain(hit_url)
                                    if not hit_domain:
                                        continue
                                    stats = adaptive_hint_domain_stats.setdefault(
                                        hit_domain,
                                        {"discovered_count": 0, "used_count": 0, "hint_pages_crawled": 0},
                                    )
                                    stats["hint_pages_crawled"] = int(stats.get("hint_pages_crawled", 0)) + 1
                                first_party_reason_cache[candidate_domain_for_fetch] = crawled_reasons
                                first_party_capability_cache[candidate_domain_for_fetch] = crawled_capabilities
                                first_party_meta_cache[candidate_domain_for_fetch] = crawl_meta
                                cached_first_party = crawled_reasons
                                cached_capabilities = crawled_capabilities
                                cached_meta = crawl_meta
                            else:
                                first_party_crawl_failed_count += 1
                                if crawl_error:
                                    first_party_error_cache[candidate_domain_for_fetch] = crawl_error
                                    first_party_crawl_errors.append(crawl_error)
                        if cached_first_party is None and first_party_fetch_budget > 0:
                            fetched_reasons, fetch_error = _extract_first_party_signals(
                                candidate_website,
                                candidate_name,
                            )
                            first_party_reason_cache[candidate_domain_for_fetch] = fetched_reasons
                            first_party_capability_cache[candidate_domain_for_fetch] = []
                            first_party_meta_cache[candidate_domain_for_fetch] = {
                                "method": "homepage_fetch_fallback",
                                "tier": "fallback",
                                "pages_crawled": 1 if fetched_reasons else 0,
                                "signals_extracted": len(fetched_reasons),
                                "hint_urls_used": [],
                                "hint_urls_used_count": 0,
                                "hint_pages_crawled": 0,
                            }
                            if fetch_error:
                                first_party_error_cache[candidate_domain_for_fetch] = fetch_error
                            first_party_fetch_budget -= 1
                            first_party_crawl_fallback_count += 1
                            cached_first_party = fetched_reasons
                            cached_capabilities = []
                            cached_meta = first_party_meta_cache[candidate_domain_for_fetch]

                    if cached_first_party:
                        brand_hint = _extract_brand_name_hint_from_reasons(cached_first_party, candidate_website)
                        if _should_replace_candidate_name_with_brand_hint(candidate_name, brand_hint, candidate_website):
                            candidate["canonical_name"] = str(brand_hint).strip()[:300]
                            candidate_name = candidate["canonical_name"]

                    if cached_meta:
                        first_party_enrichment_meta = cached_meta
                    if cached_first_party:
                        normalized_reasons = _normalize_reasons(normalized_reasons + cached_first_party)
                    if cached_capabilities:
                        capability_signals = _dedupe_strings(capability_signals + cached_capabilities)
                    elif cached_first_party:
                        capability_signals = _dedupe_strings(
                            capability_signals + _capability_signals_from_reason_items(cached_first_party)
                        )

                passed_gate, reject_reasons, gate_meta = _evaluate_enterprise_b2b_fit(candidate, normalized_reasons)

                trusted_reason_items = [
                    reason
                    for reason in normalized_reasons
                    if is_trusted_source_url(reason.get("citation_url"))
                ]
                untrusted_sources_skipped += max(0, len(normalized_reasons) - len(trusted_reason_items))

                if candidate.get("website") and is_trusted_source_url(candidate.get("website")):
                    website_url = str(candidate.get("website"))
                    website_source_type = _source_type_for_url(
                        website_url,
                        candidate_domain,
                        first_party_domains=candidate_first_party_domains,
                    )
                    has_website_reason = any(
                        str(reason.get("citation_url") or "").strip().lower() == website_url.strip().lower()
                        for reason in trusted_reason_items
                    )
                    if not has_website_reason and website_source_type == "first_party_website":
                        trusted_reason_items.append(
                            {
                                "text": "Company first-party website used as baseline evidence.",
                                "citation_url": website_url,
                                "dimension": "company_profile",
                            }
                        )
                if not trusted_reason_items:
                    reject_reasons.append("no_trusted_evidence")
                    filter_reason_counts["no_trusted_evidence"] = filter_reason_counts.get("no_trusted_evidence", 0) + 1

                total_score, component_scores, penalties, score_meta = _score_buy_side_candidate(
                    candidate=candidate,
                    reasons=trusted_reason_items,
                    capability_signals=capability_signals,
                    gate_meta=gate_meta,
                    reject_reasons=reject_reasons,
                    candidate_employee_estimate=employee_estimate,
                    buyer_employee_estimate=buyer_employee_estimate,
                )
                for penalty in penalties:
                    reason_key = str(penalty.get("reason") or "unknown_penalty")
                    penalties_count[reason_key] = penalties_count.get(reason_key, 0) + 1

                screening_status = str(score_meta.get("screening_status") or "rejected")
                if not passed_gate and gate_meta.get("hard_fail"):
                    screening_status = "rejected"
                if is_solution_entity and screening_status == "kept":
                    screening_status = "review"
                if screening_status == "kept":
                    kept_count += 1
                elif screening_status == "review":
                    review_count += 1
                else:
                    rejected_count += 1
                    for reason in reject_reasons:
                        filter_reason_counts[reason] = filter_reason_counts.get(reason, 0) + 1

                if "external_search_seed" in origin_types:
                    external_search_scored_count += 1
                    if screening_status == "kept":
                        external_search_kept_count += 1
                    elif screening_status == "review":
                        external_search_review_count += 1

                tags_custom: list[str] = []
                for capability in capability_signals[:8]:
                    tags_custom.append(f"capability:{capability}")
                if employee_estimate is not None:
                    tags_custom.append(f"employee_estimate:{employee_estimate}")
                if gate_meta.get("go_to_market"):
                    tags_custom.append(f"gtm:{gate_meta['go_to_market']}")
                if gate_meta.get("target_customer"):
                    tags_custom.append(f"target_customer:{gate_meta['target_customer']}")
                if gate_meta.get("pricing_model"):
                    tags_custom.append(f"pricing_model:{gate_meta['pricing_model']}")
                if gate_meta.get("software_heaviness") is not None:
                    tags_custom.append(f"software_heaviness:{gate_meta['software_heaviness']}")
                if gate_meta.get("public_price_floor_usd_month") is not None:
                    tags_custom.append(f"public_price_floor_usd:{int(gate_meta['public_price_floor_usd_month'])}")
                tags_custom.append(f"screen_score:{total_score}")
                tags_custom.append(f"screening_status:{screening_status}")
                tags_custom.append(f"screening_run:{screening_run_id}")
                tags_custom.append(
                    "screen_result:pass_enterprise_b2b" if passed_gate else "screen_result:reject_enterprise_b2b"
                )
                tags_custom = _dedupe_strings(tags_custom)
                candidate_scope_buckets = _dedupe_strings(
                    [
                        str((origin.get("metadata") or {}).get("scope_bucket") or "").strip().lower()
                        for origin in (entity.get("origins") or [])
                        if isinstance(origin, dict)
                        and isinstance(origin.get("metadata"), dict)
                        and str((origin.get("metadata") or {}).get("scope_bucket") or "").strip()
                    ]
                    + [
                        str(scope_bucket_map.get(str((origin.get("metadata") or {}).get("query_id") or "")) or "").strip().lower()
                        for origin in (entity.get("origins") or [])
                        if isinstance(origin, dict)
                        and isinstance(origin.get("metadata"), dict)
                        and str(scope_bucket_map.get(str((origin.get("metadata") or {}).get("query_id") or "")) or "").strip()
                    ]
                    + [
                        str(reason.get("scope_bucket") or "").strip().lower()
                        for reason in normalized_reasons
                        if isinstance(reason, dict)
                        and str(reason.get("scope_bucket") or "").strip()
                    ]
                )
                validation_lane_ids, validation_lane_labels = _candidate_validation_lane_metadata(
                    entity,
                    scope_buckets=candidate_scope_buckets,
                )

                company_ref: Optional[Company] = existing_company
                can_persist_company = bool(candidate_domain and not _is_non_first_party_profile_domain(candidate_domain))

                if screening_status in {"kept", "review"} and not is_solution_entity and can_persist_company:
                    target_company_status = CompanyStatus.kept if screening_status == "kept" else CompanyStatus.candidate
                    if company_ref is None:
                        company_ref = Company(
                            workspace_id=workspace.id,
                            name=candidate_name,
                            website=candidate.get("website"),
                            hq_country=candidate.get("hq_country", "Unknown"),
                            tags_custom=tags_custom,
                            status=target_company_status,
                            why_relevant=trusted_reason_items[:10],
                            is_manual=False,
                        )
                        db.add(company_ref)
                        db.flush()
                        created_companies += 1
                        if candidate_domain and not _is_non_first_party_profile_domain(candidate_domain):
                            existing_by_domain[candidate_domain] = company_ref
                        existing_by_name[candidate_name.lower()] = company_ref
                    else:
                        merged_reasons = _normalize_reasons((company_ref.why_relevant or []) + trusted_reason_items)
                        company_ref.why_relevant = merged_reasons[:12]
                        company_ref.tags_custom = _dedupe_strings((company_ref.tags_custom or []) + tags_custom)
                        if not company_ref.hq_country or company_ref.hq_country == "Unknown":
                            company_ref.hq_country = candidate.get("hq_country", company_ref.hq_country)
                        if not company_ref.is_manual:
                            company_ref.status = target_company_status
                        updated_existing += 1
                else:
                    if company_ref and not company_ref.is_manual and company_ref.status == CompanyStatus.candidate:
                        company_ref.status = CompanyStatus.removed
                        updated_existing += 1

                if company_ref and trusted_reason_items:
                    trusted_evidence_urls: list[str] = []
                    source_evidence_ids: dict[str, int] = {}
                    for item in trusted_reason_items:
                        citation_url = str(item.get("citation_url") or "").strip()
                        reason_text = str(item.get("text") or "").strip()
                        if not citation_url or not is_trusted_source_url(citation_url):
                            continue
                        exists = (
                            db.query(SourceEvidence)
                            .filter(
                                SourceEvidence.workspace_id == workspace.id,
                                SourceEvidence.company_id == company_ref.id,
                                SourceEvidence.source_url == citation_url,
                                SourceEvidence.excerpt_text == reason_text,
                            )
                            .first()
                        )
                        if exists:
                            trusted_evidence_urls.append(citation_url)
                            source_evidence_ids[citation_url.lower()] = exists.id
                            continue
                        source_kind_hint = str(item.get("source_kind") or "").strip().lower() or None
                        evidence_source_type = _source_type_for_url(
                            citation_url,
                            candidate_domain,
                            first_party_domains=candidate_first_party_domains,
                            provenance_hint=source_kind_hint,
                        )
                        evidence_source_tier = infer_source_tier(citation_url, evidence_source_type, candidate_domain)
                        evidence_source_kind = (
                            "rendered_browser"
                            if source_kind_hint == "rendered_browser"
                            else infer_source_kind(citation_url, evidence_source_type, candidate_domain)
                        )
                        claim_group = claim_group_for_dimension(item.get("dimension"), None)
                        ttl_days, valid_through = valid_through_from_claim_group(
                            claim_group,
                            policy=effective_policy,
                        )
                        evidence_row = SourceEvidence(
                            workspace_id=workspace.id,
                            company_id=company_ref.id,
                            source_url=citation_url,
                            source_title=source_label_for_url(citation_url),
                            excerpt_text=reason_text[:1200],
                            content_type="web",
                            source_tier=evidence_source_tier,
                            source_kind=evidence_source_kind,
                            freshness_ttl_days=ttl_days,
                            valid_through=valid_through,
                            asserted_by="run_discovery_universe",
                        )
                        db.add(evidence_row)
                        db.flush()
                        source_evidence_ids[citation_url.lower()] = evidence_row.id
                        trusted_evidence_urls.append(citation_url)

                    if not trusted_evidence_urls and company_ref.website and is_trusted_source_url(company_ref.website):
                        fallback_source_type = _source_type_for_url(
                            company_ref.website,
                            candidate_domain,
                            first_party_domains=candidate_first_party_domains,
                        )
                        if fallback_source_type == "first_party_website":
                            fallback_source_tier = infer_source_tier(company_ref.website, fallback_source_type, candidate_domain)
                            fallback_source_kind = infer_source_kind(company_ref.website, fallback_source_type, candidate_domain)
                            ttl_days, valid_through = valid_through_from_claim_group(
                                "identity_scope",
                                policy=effective_policy,
                            )
                            fallback_row = SourceEvidence(
                                workspace_id=workspace.id,
                                company_id=company_ref.id,
                                source_url=company_ref.website,
                                source_title=f"{company_ref.name} website",
                                excerpt_text="First-party website baseline source.",
                                content_type="web",
                                source_tier=fallback_source_tier,
                                source_kind=fallback_source_kind,
                                freshness_ttl_days=ttl_days,
                                valid_through=valid_through,
                                asserted_by="run_discovery_universe",
                            )
                            db.add(fallback_row)
                            db.flush()
                            source_evidence_ids[company_ref.website.lower()] = fallback_row.id
                            trusted_evidence_urls.append(company_ref.website)

                    if screening_status in {"kept", "review"}:
                        existing_dossier = (
                            db.query(CompanyDossier)
                            .filter(CompanyDossier.company_id == company_ref.id)
                            .order_by(CompanyDossier.version.desc())
                            .first()
                        )
                        if not existing_dossier:
                            bootstrap_dossier = {
                                "modules": _map_capabilities_to_modules(
                                    capabilities=capability_signals,
                                    taxonomy_bricks=capability_hints,
                                    trusted_urls=trusted_evidence_urls,
                                ),
                                "customers": [],
                                "hiring": {
                                    "postings": [],
                                    "mix_summary": {
                                        "engineering_heavy": None,
                                        "team_size_estimate": employee_estimate,
                                        "notes": "bootstrapped from discovery signals",
                                    },
                                },
                                "integrations": [],
                            }
                            db.add(
                                CompanyDossier(
                                    company_id=company_ref.id,
                                    dossier_json=bootstrap_dossier,
                                    version=1,
                                )
                            )

                screening = CompanyScreening(
                    workspace_id=workspace.id,
                    company_id=company_ref.id if company_ref else None,
                    candidate_entity_id=entity.get("db_entity_id"),
                    candidate_name=candidate_name,
                    candidate_website=str(candidate.get("website") or "")[:1000] or None,
                    candidate_discovery_url=str(candidate.get("discovery_url") or "")[:1000] or None,
                    candidate_official_website=str(candidate.get("official_website_url") or candidate.get("website") or "")[:1000] or None,
                    screening_status=screening_status,
                    total_score=total_score,
                    component_scores_json=component_scores,
                    penalties_json=penalties,
                    reject_reasons_json=_dedupe_strings(reject_reasons),
                    positive_reason_codes_json=[],
                    caution_reason_codes_json=[],
                    reject_reason_codes_json=[],
                    missing_claim_groups_json=[],
                    unresolved_contradictions_count=0,
                    decision_classification="insufficient_evidence",
                    evidence_sufficiency="insufficient",
                    rationale_summary=None,
                    rationale_markdown=None,
                    top_claim_json={},
                    decision_engine_version=None,
                    gating_passed=False,
                    ranking_eligible=False,
                    screening_meta_json={
                        "job_id": job.id,
                        "screening_run_id": screening_run_id,
                        "passed_enterprise_gate": passed_gate,
                        "hard_fail_gate": bool(gate_meta.get("hard_fail")),
                        "qualification": candidate.get("qualification") or {},
                        "matched_mentions": len(matched_mentions),
                        "entity_type": candidate.get("entity_type"),
                        "first_party_domains": candidate_first_party_domains,
                        "discovery_url": candidate.get("discovery_url"),
                        "official_website_url": candidate.get("official_website_url") or candidate.get("website"),
                        "solutions": candidate.get("solutions") if isinstance(candidate.get("solutions"), list) else [],
                        "candidate_hq_country": candidate.get("hq_country"),
                        "candidate_employee_estimate": score_meta.get("candidate_employee_estimate"),
                        "buyer_employee_estimate": score_meta.get("buyer_employee_estimate"),
                        "size_window_ratio": score_meta.get("size_window_ratio"),
                        "size_large_company_threshold": score_meta.get("size_large_company_threshold"),
                        "identity": candidate.get("identity") or {},
                        "input_website": candidate.get("original_website"),
                        "resolved_website": candidate.get("website"),
                        "candidate_entity_id": entity.get("db_entity_id"),
                        "aliases": entity.get("alias_names") or [],
                        "merge_rationale": entity.get("merge_reasons") or [],
                        "origin_types": entity.get("origin_types") or [],
                        "scope_buckets": candidate_scope_buckets,
                        "capability_signals": capability_signals[:10],
                        "likely_verticals": likely_verticals[:8],
                        "registry_identity": entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {},
                        "industry_signature": entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
                        "registry_neighbors_with_first_party_website_count": registry_neighbors_with_first_party_website_count,
                        "registry_neighbors_dropped_missing_official_website_count": registry_neighbors_dropped_missing_official_website_count,
                        "registry_origin_screening_counts": registry_origin_screening_counts,
                        "first_party_hint_urls_used_count": int(first_party_enrichment_meta.get("hint_urls_used_count") or 0),
                        "first_party_hint_pages_crawled_total": int(first_party_enrichment_meta.get("hint_pages_crawled") or 0),
                        "first_party_enrichment": {
                            "method": first_party_enrichment_meta.get("method"),
                            "tier": first_party_enrichment_meta.get("tier"),
                            "pages_crawled": int(first_party_enrichment_meta.get("pages_crawled") or 0),
                            "signals_extracted": int(first_party_enrichment_meta.get("signals_extracted") or 0),
                            "customer_evidence_count": int(first_party_enrichment_meta.get("customer_evidence_count") or 0),
                            "page_types": first_party_enrichment_meta.get("page_types") if isinstance(first_party_enrichment_meta.get("page_types"), dict) else {},
                            "hint_urls_used": first_party_enrichment_meta.get("hint_urls_used") if isinstance(first_party_enrichment_meta.get("hint_urls_used"), list) else [],
                            "hint_urls_used_count": int(first_party_enrichment_meta.get("hint_urls_used_count") or 0),
                            "hint_pages_crawled": int(first_party_enrichment_meta.get("hint_pages_crawled") or 0),
                            "error": first_party_error_cache.get(candidate_domain_for_fetch) if candidate_domain_for_fetch else None,
                        },
                    },
                    source_summary_json={
                        "trusted_reason_count": len(trusted_reason_items),
                        "source_urls": _dedupe_strings(
                            [str(reason.get("citation_url") or "") for reason in trusted_reason_items if reason.get("citation_url")]
                        )[:12],
                        "mention_listing_urls": _dedupe_strings(
                            [str(m.get("listing_url") or "") for m in matched_mentions if m.get("listing_url")]
                        )[:8],
                        "source_type_counts": score_meta.get("source_type_counts") or {},
                        "has_first_party_evidence": bool(score_meta.get("has_first_party_evidence")),
                        "has_first_party_depth": bool(score_meta.get("has_first_party_depth")),
                        "has_non_directory_corroboration": bool(score_meta.get("has_non_directory_corroboration")),
                        "entity_type": candidate.get("entity_type"),
                        "discovery_url": candidate.get("discovery_url"),
                        "official_website_url": candidate.get("official_website_url") or candidate.get("website"),
                        "first_party_domains": candidate_first_party_domains,
                        "candidate_employee_estimate": score_meta.get("candidate_employee_estimate"),
                        "buyer_employee_estimate": score_meta.get("buyer_employee_estimate"),
                        "first_party_enrichment": {
                            "method": first_party_enrichment_meta.get("method"),
                            "tier": first_party_enrichment_meta.get("tier"),
                            "pages_crawled": int(first_party_enrichment_meta.get("pages_crawled") or 0),
                            "hint_urls_used_count": int(first_party_enrichment_meta.get("hint_urls_used_count") or 0),
                            "hint_pages_crawled": int(first_party_enrichment_meta.get("hint_pages_crawled") or 0),
                        },
                        "registry_neighbors_with_first_party_website_count": registry_neighbors_with_first_party_website_count,
                        "registry_neighbors_dropped_missing_official_website_count": registry_neighbors_dropped_missing_official_website_count,
                        "registry_origin_screening_counts": registry_origin_screening_counts,
                        "first_party_hint_urls_used_count": int(first_party_enrichment_meta.get("hint_urls_used_count") or 0),
                        "first_party_hint_pages_crawled_total": int(first_party_enrichment_meta.get("hint_pages_crawled") or 0),
                        "origin_types": entity.get("origin_types") or [],
                        "expansion_provenance": [
                            (origin.get("metadata") or {})
                            for origin in (entity.get("origins") or [])
                            if isinstance(origin, dict)
                            and isinstance(origin.get("metadata"), dict)
                            and (origin.get("metadata") or {}).get("query_type")
                        ][:6],
                    },
                )
                db.add(screening)
                db.flush()

                claim_records = _build_claim_records(
                    workspace_id=workspace.id,
                    company_id=company_ref.id if company_ref else None,
                    company_screening_id=screening.id,
                    candidate=candidate,
                    trusted_reasons=trusted_reason_items,
                    matched_mentions=matched_mentions,
                    policy=effective_policy,
                    source_evidence_ids=(source_evidence_ids if company_ref and trusted_reason_items else None),
                    first_party_domains=candidate_first_party_domains,
                )
                for record in claim_records:
                    db.add(CompanyClaim(**record))
                claims_created += len(claim_records)
                citation_summary_v1 = _build_citation_summary_v1(claim_records)

                decision = evaluate_decision(
                    screening_status=screening_status,
                    reject_reasons=_dedupe_strings(reject_reasons),
                    claims=claim_records,
                    component_scores=component_scores,
                    source_type_counts=score_meta.get("source_type_counts") or {},
                    policy=effective_policy,
                )
                screening.positive_reason_codes_json = decision.positive_reason_codes
                screening.caution_reason_codes_json = decision.caution_reason_codes
                screening.reject_reason_codes_json = decision.reject_reason_codes
                screening.missing_claim_groups_json = decision.missing_claim_groups
                screening.unresolved_contradictions_count = decision.unresolved_contradictions_count
                screening.decision_classification = decision.classification
                screening.evidence_sufficiency = decision.evidence_sufficiency
                screening.rationale_summary = decision.rationale_summary
                screening.rationale_markdown = decision.rationale_markdown
                top_claim = _select_top_claim(claim_records)
                if top_claim.get("source_url") and top_claim.get("source_tier"):
                    screening.top_claim_json = top_claim
                else:
                    screening.top_claim_json = {}
                screening.decision_engine_version = decision.decision_engine_version
                screening.gating_passed = bool(decision.gating_passed)
                screening.ranking_eligible = _is_ranking_eligible_candidate(
                    candidate=candidate,
                    decision_classification=decision.classification,
                    claim_records=claim_records,
                )
                entity_row = entity_row_by_id.get(int(entity.get("db_entity_id") or 0))
                if entity_row is not None:
                    discovery_query_families, discovery_source_families = _candidate_origin_discovery_families(
                        [origin for origin in (entity.get("origins") or []) if isinstance(origin, dict)]
                    )
                    set_validation_metadata(
                        entity_row,
                        {
                            "status": VALIDATION_STATUS_QUEUED,
                            "recommendation": (
                                VALIDATION_STATUS_KEEP
                                if decision.classification == "good_target"
                                else VALIDATION_STATUS_WATCHLIST
                                if decision.classification == "borderline_watchlist"
                                else VALIDATION_STATUS_REJECT
                                if decision.classification == "not_good_target"
                                else VALIDATION_STATUS_QUEUED
                            ),
                            "promoted_to_cards": False,
                            "identity_confidence": str(entity_row.identity_confidence or "low"),
                            "vendor_classification": (
                                "solution_profile"
                                if str(candidate.get("entity_type") or "").strip().lower() == "solution"
                                else "vendor_candidate"
                                if str(candidate.get("website") or "").strip()
                                else "directory_only_candidate"
                            ),
                            "official_website_confidence": (
                                "high"
                                if str(candidate.get("website") or "").strip()
                                else "low"
                            ),
                            "lane_ids": validation_lane_ids,
                            "lane_labels": validation_lane_labels,
                            "query_families": discovery_query_families,
                            "source_families": discovery_source_families,
                            "origin_types": _dedupe_strings(entity.get("origin_types") or [])[:8],
                            "priority_score": float(decision.total_score if hasattr(decision, "total_score") else total_score or 0.0),
                        },
                    )
                screening_meta = (
                    dict(screening.screening_meta_json)
                    if isinstance(screening.screening_meta_json, dict)
                    else {}
                )
                if citation_summary_v1:
                    screening_meta["citation_summary_v1"] = citation_summary_v1
                elif "citation_summary_v1" in screening_meta:
                    screening_meta.pop("citation_summary_v1", None)
                screening_meta["registry_neighbors_with_first_party_website_count"] = registry_neighbors_with_first_party_website_count
                screening_meta["registry_neighbors_dropped_missing_official_website_count"] = (
                    registry_neighbors_dropped_missing_official_website_count
                )
                screening_meta["registry_origin_screening_counts"] = registry_origin_screening_counts
                screening_meta["first_party_hint_urls_used_count"] = int(
                    first_party_enrichment_meta.get("hint_urls_used_count") or 0
                )
                screening_meta["first_party_hint_pages_crawled_total"] = int(
                    first_party_enrichment_meta.get("hint_pages_crawled") or 0
                )
                screening.screening_meta_json = screening_meta
                run_screenings_for_quality_audit.append(screening)
                run_claims_by_screening_id[int(screening.id)] = [dict(row) for row in claim_records]
                if screening.ranking_eligible:
                    ranking_eligible_count += 1
                decision_class_counts[decision.classification] = decision_class_counts.get(decision.classification, 0) + 1
                evidence_sufficiency_counts[decision.evidence_sufficiency] = (
                    evidence_sufficiency_counts.get(decision.evidence_sufficiency, 0) + 1
                )
                if entity_index % scoring_write_batch_size == 0 or entity_index == scoring_total:
                    job.progress = 0.55 + (0.35 * (entity_index / scoring_total))
                    job.progress_message = f"Scored {entity_index}/{scoring_total} canonical candidates..."
                    db.commit()

            validation_queue_candidates: list[dict[str, Any]] = []
            for screening in run_screenings_for_quality_audit:
                if not screening.ranking_eligible or not screening.candidate_entity_id:
                    continue
                entity_row = entity_row_by_id.get(int(screening.candidate_entity_id))
                if entity_row is None:
                    continue
                validation = validation_metadata(entity_row)
                validation_lane_ids = validation.get("lane_ids") or []
                validation_lane_labels = validation.get("lane_labels") or []
                validation_source_families = validation.get("source_families") or []
                if not (validation_lane_ids or validation_lane_labels) and set(validation_source_families) <= {"directory"}:
                    continue
                source_summary = screening.source_summary_json if isinstance(screening.source_summary_json, dict) else {}
                validation_queue_candidates.append(
                    {
                        "candidate_entity_id": int(entity_row.id),
                        "company_name": screening.candidate_name,
                        "canonical_name": entity_row.canonical_name,
                        "official_website_url": screening.candidate_official_website or entity_row.canonical_website,
                        "discovery_url": screening.candidate_discovery_url or entity_row.discovery_primary_url,
                        "entity_type": entity_row.entity_type,
                        "decision_classification": screening.decision_classification,
                        "priority_score": float(validation.get("priority_score") or screening.total_score or 0.0),
                        "multi_origin_count": len(validation.get("origin_types") or []),
                        "validation_status": validation.get("status") or VALIDATION_STATUS_QUEUED,
                        "promoted_to_cards": bool(validation.get("promoted_to_cards")),
                        "validation_lane_ids": validation_lane_ids,
                        "validation_lane_labels": validation_lane_labels,
                        "validation_query_families": validation.get("query_families") or [],
                        "validation_source_families": validation_source_families,
                    }
                )

            validation_queue_ranked = build_diversified_validation_queue(
                validation_queue_candidates,
                limit=max(1, int(getattr(settings, "discovery_validation_queue_limit", 36))),
                lane_cap=max(1, int(getattr(settings, "discovery_validation_lane_cap", 6))),
                family_cap=max(1, int(getattr(settings, "discovery_validation_query_family_cap", 4))),
                source_family_cap=max(1, int(getattr(settings, "discovery_validation_source_family_cap", 18))),
            )
            queue_rank_by_entity_id = {
                int(item["candidate_entity_id"]): int(item.get("queue_rank") or 0)
                for item in validation_queue_ranked
            }
            for entity_id, entity_row in entity_row_by_id.items():
                validation = validation_metadata(entity_row)
                validation["queue_rank"] = int(queue_rank_by_entity_id.get(int(entity_id), 0))
                if validation["queue_rank"] > 0 and not str(validation.get("status") or "").strip():
                    validation["status"] = VALIDATION_STATUS_QUEUED
                validation["validation_state"] = str(validation.get("status") or VALIDATION_STATUS_QUEUED)
                set_validation_metadata(entity_row, validation)

            discovery_candidate_graph_sync = {"status": "not_run"}
            try:
                discovery_graph_candidates: list[dict[str, Any]] = []
                for entity in canonical_entities:
                    entity_id = int(entity.get("db_entity_id") or 0)
                    entity_row = entity_row_by_id.get(entity_id)
                    if entity_id <= 0 or entity_row is None:
                        continue
                    lane_ids, lane_labels = _candidate_validation_lane_metadata(
                        entity,
                        scope_buckets=_dedupe_strings(
                            [
                                str((origin.get("metadata") or {}).get("scope_bucket") or "").strip().lower()
                                for origin in (entity.get("origins") or [])
                                if isinstance(origin, dict) and isinstance(origin.get("metadata"), dict)
                            ]
                        ),
                    )
                    discovery_query_families, discovery_source_families = _candidate_origin_discovery_families(
                        [origin for origin in (entity.get("origins") or []) if isinstance(origin, dict)]
                    )
                    discovery_graph_candidates.append(
                        {
                            "candidate_entity_id": entity_id,
                            "company_name": entity_row.canonical_name,
                            "official_website_url": entity_row.canonical_website,
                            "discovery_url": entity_row.discovery_primary_url,
                            "entity_type": entity_row.entity_type,
                            "discovery_score": float((entity_row.metadata_json or {}).get("discovery_score") or _candidate_priority_score(entity)),
                            "lane_ids": lane_ids,
                            "lane_labels": lane_labels,
                            "query_families": discovery_query_families,
                            "source_families": discovery_source_families,
                            "validation_status": validation_metadata(entity_row).get("status") or VALIDATION_STATUS_QUEUED,
                            "promoted_to_cards": bool(validation_metadata(entity_row).get("promoted_to_cards")),
                            "directness": "adjacent" if lane_ids or lane_labels else "broad_market",
                        }
                    )
                graph_payload = build_discovery_candidate_graph_payload(
                    workspace_id=workspace.id,
                    candidates=discovery_graph_candidates,
                )
                discovery_candidate_graph_sync = Neo4jDiscoveryCandidateGraphStore().sync_graph(graph_payload)
            except Exception as discovery_graph_exc:
                discovery_candidate_graph_sync = {
                    "status": "failed",
                    "error": str(discovery_graph_exc),
                }
            _stage_finished(
                "stage_first_party_enrichment_parallel",
                enrichment_stage_started,
                settings.stage_enrichment_timeout_seconds,
            )
            _save_stage_checkpoint(
                job_id,
                "stage_first_party_enrichment_parallel",
                {
                    "scoring_entities_count": len(scoring_entities),
                    "first_party_crawl_attempted_count": first_party_crawl_attempted_count,
                    "first_party_crawl_success_count": first_party_crawl_success_count,
                    "first_party_crawl_pages_total": first_party_crawl_pages_total,
                    "first_party_hint_urls_used_count": first_party_hint_urls_used_count,
                    "first_party_hint_pages_crawled_total": first_party_hint_pages_crawled_total,
                },
            )
            stage_time_ms["stage_scoring_claims_persist"] = stage_time_ms.get(
                "stage_first_party_enrichment_parallel",
                0,
            )
            if stage_time_ms["stage_scoring_claims_persist"] > max(1, int(settings.stage_scoring_timeout_seconds)) * 1000:
                timeout_events.append(
                    {
                        "stage": "stage_scoring_claims_persist",
                        "duration_ms": stage_time_ms["stage_scoring_claims_persist"],
                        "timeout_seconds": int(settings.stage_scoring_timeout_seconds),
                    }
                )

            registry_queries_by_country: dict[str, int] = {}
            for source in [
                registry_identity_metrics.get("registry_queries_by_country", {}),
                registry_expansion_metrics.get("registry_queries_by_country", {}),
            ]:
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    registry_queries_by_country[str(key)] = registry_queries_by_country.get(str(key), 0) + int(value)
            registry_identity_queries_count = int(
                sum(
                    int(value)
                    for value in (registry_identity_metrics.get("registry_queries_by_country", {}) or {}).values()
                )
            )
            registry_neighbor_queries_count = int(
                sum(
                    int(value)
                    for value in (registry_expansion_metrics.get("registry_queries_by_country", {}) or {}).values()
                )
            )

            registry_raw_hits_by_country: dict[str, int] = {}
            for source in [
                registry_identity_metrics.get("registry_raw_hits_by_country", {}),
                registry_expansion_metrics.get("registry_raw_hits_by_country", {}),
            ]:
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    registry_raw_hits_by_country[str(key)] = registry_raw_hits_by_country.get(str(key), 0) + int(value)
            claims_graph_refresh = {"status": "not_run"}
            try:
                with db.begin_nested():
                    claims_graph_metrics = rebuild_workspace_claims_graph(db, workspace.id)
                claims_graph_refresh = {
                    "status": "rebuilt",
                    "metrics": claims_graph_metrics,
                }
            except Exception as graph_exc:
                claims_graph_refresh = {
                    "status": "failed",
                    "error": str(graph_exc),
                }

            quality_audit_v1 = build_quality_audit_v1(
                screenings=run_screenings_for_quality_audit,
                claims_by_screening=run_claims_by_screening_id,
                run_id=screening_run_id,
                thresholds=quality_audit_thresholds_from_settings(settings),
            )
            quality_audit_v1 = normalize_quality_audit_v1(quality_audit_v1) or quality_audit_v1
            quality_audit_passed = bool(
                isinstance(quality_audit_v1, dict) and quality_audit_v1.get("pass")
            )
            quality_validation_ready = bool(quality_audit_passed)
            quality_validation_blocked_reasons: list[str] = []
            if not quality_validation_ready:
                quality_validation_blocked_reasons.append("quality_audit_failed")
            discovery_ctx_snapshot = _load_discovery_context(job_id)
            pre_rerun_quality_audit_v1 = (
                discovery_ctx_snapshot.get("pre_rerun_quality_audit_v1")
                if isinstance(discovery_ctx_snapshot.get("pre_rerun_quality_audit_v1"), dict)
                else None
            )
            pre_rerun_quality_audit_run_id = str(
                discovery_ctx_snapshot.get("pre_rerun_quality_audit_run_id") or ""
            ).strip() or None
            pre_rerun_quality_validation_ready = bool(
                discovery_ctx_snapshot.get("pre_rerun_quality_validation_ready", False)
            )
            pre_rerun_quality_validation_blocked_reasons = [
                str(item).strip()
                for item in (
                    discovery_ctx_snapshot.get("pre_rerun_quality_validation_blocked_reasons")
                    or []
                )
                if str(item).strip()
            ]

            degraded_reasons: list[str] = []
            if llm_error:
                degraded_reasons.append("llm_discovery_error")
            if seed_llm_count <= 0:
                degraded_reasons.append("llm_discovery_failed")
            if claims_created < max(0, int(settings.quality_min_claims_created)):
                degraded_reasons.append("citation_threshold_not_met")
            if ranking_eligible_count < max(0, int(settings.quality_min_ranking_eligible_count)):
                degraded_reasons.append("ranking_threshold_not_met")
            critical_timeout_stages = {
                "stage_llm_discovery_fanout",
                "stage_registry_identity_expand",
                "stage_first_party_enrichment_parallel",
            }
            for event in timeout_events:
                stage_name = str(event.get("stage") or "")
                if stage_name in critical_timeout_stages:
                    degraded_reasons.append(f"{stage_name}_timeout")
            degraded_reasons = _dedupe_strings(degraded_reasons)
            pipeline_total_ms = int((time.perf_counter() - pipeline_started) * 1000)
            stage_time_ms["pipeline_total_ms"] = pipeline_total_ms
            if pipeline_total_ms > max(1, int(settings.discovery_global_timeout_seconds)) * 1000:
                degraded_reasons.append("pipeline_global_timeout")
                timeout_events.append(
                    {
                        "stage": "pipeline_total",
                        "duration_ms": pipeline_total_ms,
                        "timeout_seconds": int(settings.discovery_global_timeout_seconds),
                    }
                )
                degraded_reasons = _dedupe_strings(degraded_reasons)
            run_quality_tier = "high_quality" if not degraded_reasons else "degraded"
            quality_gate_passed = run_quality_tier == "high_quality"
            adaptive_hint_domain_hit_rates: dict[str, dict[str, Any]] = {}
            for hint_domain, stats in adaptive_hint_domain_stats.items():
                discovered_count = int(stats.get("discovered_count", 0))
                used_count = int(stats.get("used_count", 0))
                hint_pages = int(stats.get("hint_pages_crawled", 0))
                adaptive_hint_domain_hit_rates[hint_domain] = {
                    "discovered_count": discovered_count,
                    "used_count": used_count,
                    "hint_pages_crawled": hint_pages,
                    "hit_rate": round(hint_pages / max(1, used_count), 4),
                }
            adaptive_hint_domain_hit_rates = dict(
                sorted(
                    adaptive_hint_domain_hit_rates.items(),
                    key=lambda item: (
                        -float(item[1].get("hit_rate", 0.0)),
                        -int(item[1].get("used_count", 0)),
                    ),
                )[:60]
            )

            job.state = JobState.completed
            job.progress = 1.0
            job.progress_message = "Complete"
            job.result_json = {
                "candidates_found": len(canonical_entities),
                "llm_candidates_found": seed_llm_count,
                "external_search_candidates_found": seed_external_search_count,
                "seed_mentions_count": len(mention_records),
                "seed_directory_count": seed_directory_count,
                "seed_reference_count": seed_reference_count,
                "seed_benchmark_count": seed_benchmark_count,
                "seed_llm_count": seed_llm_count,
                "seed_external_search_count": seed_external_search_count,
                "buyer_employee_estimate": buyer_employee_estimate,
                "size_fit_policy": {
                    "window_ratio": float(
                        max(0.0, min(0.9, float(getattr(settings, "size_fit_window_ratio", SIZE_FIT_WINDOW_RATIO))))
                    ),
                    "boost_points": float(
                        max(0.0, float(getattr(settings, "size_fit_boost_points", SIZE_FIT_BOOST_POINTS)))
                    ),
                    "large_company_threshold": int(
                        max(1, int(getattr(settings, "size_large_company_threshold", SIZE_LARGE_COMPANY_THRESHOLD)))
                    ),
                    "large_company_penalty_points": float(
                        max(
                            0.0,
                            float(
                                getattr(
                                    settings,
                                    "size_large_company_penalty_points",
                                    SIZE_LARGE_COMPANY_PENALTY_POINTS,
                                )
                            ),
                        )
                    ),
                },
                "companies_created": created_companies,
                "vendors_updated": updated_existing,
                "kept_count": kept_count,
                "review_count": review_count,
                "rejected_count": rejected_count,
                "external_search_scored_count": external_search_scored_count,
                "external_search_kept_count": external_search_kept_count,
                "external_search_review_count": external_search_review_count,
                "external_search_kept_review_rate": round(
                    (external_search_kept_count + external_search_review_count) / max(1, external_search_scored_count),
                    4,
                ),
                "untrusted_sources_skipped": untrusted_sources_skipped,
                "claims_created": claims_created,
                "decision_class_counts": decision_class_counts,
                "evidence_sufficiency_counts": evidence_sufficiency_counts,
                "ranking_eligible_count": ranking_eligible_count,
                "solution_entity_screening_count": solution_entity_screening_count,
                "unresolved_directory_only_skipped_count": unresolved_directory_only_skipped_count,
                "screening_run_id": screening_run_id,
                "identity_resolved_count": int(identity_seed_stats.get("identity_resolved_count", 0))
                + int(identity_registry_stats.get("identity_resolved_count", 0)),
                "identity_fetch_count": int(identity_seed_stats.get("identity_fetch_count", 0))
                + int(identity_registry_stats.get("identity_fetch_count", 0)),
                "identity_budget_exhausted": int(identity_seed_stats.get("identity_budget_exhausted", 0))
                + int(identity_registry_stats.get("identity_budget_exhausted", 0)),
                "alias_clusters_count": alias_clusters_count,
                "duplicates_collapsed_count": int(canonical_metrics.get("duplicates_collapsed_count", 0)),
                "suspected_duplicate_count": int(canonical_metrics.get("suspected_duplicate_count", 0)),
                "registry_queries_count": int(sum(registry_queries_by_country.values())),
                "registry_identity_queries_count": registry_identity_queries_count,
                "registry_neighbor_queries_count": registry_neighbor_queries_count,
                "registry_neighbors_kept_count": int(registry_expansion_metrics.get("registry_neighbors_kept_count", 0)),
                "registry_neighbors_with_first_party_website_count": int(
                    registry_expansion_metrics.get("registry_neighbors_with_first_party_website_count", 0)
                ),
                "registry_neighbors_dropped_missing_official_website_count": int(
                    registry_expansion_metrics.get("registry_neighbors_dropped_missing_official_website_count", 0)
                ),
                "registry_origin_screening_counts": registry_expansion_metrics.get("registry_origin_screening_counts", {}),
                "registry_identity_candidates_count": int(registry_identity_metrics.get("registry_identity_candidates_count", 0)),
                "registry_identity_mapped_count": int(registry_identity_metrics.get("registry_identity_mapped_count", 0)),
                "registry_identity_country_breakdown": registry_identity_metrics.get("registry_identity_country_breakdown", {}),
                "registry_queries_by_country": registry_queries_by_country,
                "registry_raw_hits_by_country": registry_raw_hits_by_country,
                "registry_neighbors_kept_pre_dedupe": int(registry_expansion_metrics.get("registry_neighbors_kept_pre_dedupe", 0)),
                "registry_neighbors_unique_post_dedupe": int(registry_neighbors_unique_post_dedupe),
                "registry_reject_reason_breakdown": registry_expansion_metrics.get("registry_reject_reason_breakdown", {}),
                "final_universe_count": len(canonical_entities),
                "candidate_entity_cap": candidate_entity_cap,
                "trimmed_out_count": trimmed_out_count,
                "scoring_entities_count": len(scoring_entities),
                "scoring_entities_skipped_count": scoring_entities_skipped_count,
                "scoring_selection_meta": scoring_selection_meta,
                "validation_queue_count": len(validation_queue_ranked),
                "validation_queue_entity_ids": [int(item["candidate_entity_id"]) for item in validation_queue_ranked[:100]],
                "discovery_candidate_graph_sync": discovery_candidate_graph_sync,
                "first_party_fetch_budget_used": first_party_fetch_budget_default - first_party_fetch_budget,
                "first_party_crawl_budget_used": first_party_crawl_budget_default - first_party_crawl_budget,
                "first_party_hint_crawl_budget_used": first_party_hint_crawl_budget_default - first_party_hint_crawl_budget,
                "first_party_adaptive_hint_domain_budget_used": int(
                    max(0, adaptive_hint_domain_budget_default - adaptive_hint_domain_budget)
                ),
                "first_party_crawl_attempted_count": first_party_crawl_attempted_count,
                "first_party_crawl_success_count": first_party_crawl_success_count,
                "first_party_crawl_failed_count": first_party_crawl_failed_count,
                "first_party_crawl_deep_count": first_party_crawl_deep_count,
                "first_party_crawl_light_count": first_party_crawl_light_count,
                "first_party_crawl_fallback_count": first_party_crawl_fallback_count,
                "first_party_crawl_pages_total": first_party_crawl_pages_total,
                "first_party_hint_urls_used_count": first_party_hint_urls_used_count,
                "first_party_hint_pages_crawled_total": first_party_hint_pages_crawled_total,
                "first_party_hint_domain_stats": adaptive_hint_domain_hit_rates,
                "filter_reason_counts": filter_reason_counts,
                "penalty_reason_counts": penalties_count,
                "source_coverage": source_coverage,
                "origin_mix_distribution": origin_mix_distribution,
                "dedupe_quality_metrics": {
                    "pre_registry": pre_registry_metrics,
                    "final": canonical_metrics,
                },
                "registry_expansion_yield": {
                    **registry_expansion_metrics,
                    "registry_identity_queries_count": registry_identity_queries_count,
                    "registry_neighbor_queries_count": registry_neighbor_queries_count,
                    "registry_identity_candidates_count": int(registry_identity_metrics.get("registry_identity_candidates_count", 0)),
                    "registry_identity_mapped_count": int(registry_identity_metrics.get("registry_identity_mapped_count", 0)),
                    "registry_identity_country_breakdown": registry_identity_metrics.get("registry_identity_country_breakdown", {}),
                    "registry_neighbors_unique_post_dedupe": int(registry_neighbors_unique_post_dedupe),
                    "registry_neighbors_with_first_party_website_count": int(
                        registry_expansion_metrics.get("registry_neighbors_with_first_party_website_count", 0)
                    ),
                    "registry_neighbors_dropped_missing_official_website_count": int(
                        registry_expansion_metrics.get("registry_neighbors_dropped_missing_official_website_count", 0)
                    ),
                    "registry_origin_screening_counts": registry_expansion_metrics.get("registry_origin_screening_counts", {}),
                },
                "comparator_errors": comparator_errors,
                "identity_fetch_errors": {
                    **(identity_seed_stats.get("identity_fetch_errors") or {}),
                    **(identity_registry_stats.get("identity_fetch_errors") or {}),
                },
                "first_party_fetch_errors": first_party_error_cache,
                "first_party_crawl_errors": first_party_crawl_errors[:100],
                "llm_error": llm_error,
                "fallback_mode": bool(llm_error),
                "run_quality_tier": run_quality_tier,
                "quality_gate_passed": quality_gate_passed,
                "quality_audit_v1": quality_audit_v1 if isinstance(quality_audit_v1, dict) else None,
                "quality_audit_passed": quality_audit_passed,
                "quality_validation_ready": quality_validation_ready,
                "quality_validation_blocked_reasons": quality_validation_blocked_reasons,
                "pre_rerun_quality_audit_v1": (
                    pre_rerun_quality_audit_v1
                    if isinstance(pre_rerun_quality_audit_v1, dict)
                    else None
                ),
                "pre_rerun_quality_audit_run_id": pre_rerun_quality_audit_run_id,
                "pre_rerun_quality_validation_ready": pre_rerun_quality_validation_ready,
                "pre_rerun_quality_validation_blocked_reasons": pre_rerun_quality_validation_blocked_reasons,
                "degraded_reasons": degraded_reasons,
                "model_attempt_trace": model_attempt_trace[:200],
                "stage_time_ms": stage_time_ms,
                "timeout_events": timeout_events,
                "external_search_provider_mix": external_search_provider_mix,
                "external_search_query_plan_summary": query_plan_summary,
                "external_search_query_plan_error": llm_plan_error,
                "external_search_candidate_synthesis_error": llm_synthesis_error,
                "external_search_result_counts": external_search_result_counts,
                "external_search_brick_yield": external_search_brick_yield,
                "external_search_dedupe_stats": external_search_dedupe_stats,
                "external_search_errors": external_search_errors[:20],
                "known_entity_suppression_raw": known_entity_raw_stats if isinstance(known_entity_raw_stats, dict) else {},
                "known_entity_suppression_final": known_entity_final_stats if isinstance(known_entity_final_stats, dict) else {},
                "claims_graph_refresh": claims_graph_refresh,
            }
            job.finished_at = datetime.utcnow()
            db.commit()

            return {"success": True, "created": created_companies, "screening_run_id": screening_run_id}

        except Exception as e:
            job.state = JobState.failed
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(e)}

    finally:
        db.close()


def _record_stage_metric(job_id: int, stage_name: str, started_at: float, timeout_seconds: int) -> None:
    ctx = _load_discovery_context(job_id)
    stage_time_ms = ctx.get("stage_time_ms") if isinstance(ctx.get("stage_time_ms"), dict) else {}
    timeout_events = ctx.get("timeout_events") if isinstance(ctx.get("timeout_events"), list) else []
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    stage_time_ms[stage_name] = duration_ms
    if duration_ms > max(1, int(timeout_seconds)) * 1000:
        timeout_events.append(
            {
                "stage": stage_name,
                "duration_ms": duration_ms,
                "timeout_seconds": int(timeout_seconds),
            }
        )
    ctx["stage_time_ms"] = stage_time_ms
    ctx["timeout_events"] = timeout_events
    _save_discovery_context(job_id, ctx)


@celery_app.task(name="app.workers.workspace_tasks.run_discovery_universe")
def run_discovery_universe(job_id: int):
    """Queue staged discovery pipeline for a workspace job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        if job.state in {JobState.completed, JobState.failed}:
            return {"error": f"Job already terminal: {job.state.value}"}

        stale_runs_failed = _expire_stale_running_discovery_jobs(db, exclude_job_id=job_id)
        superseded_runs = (
            db.query(Job)
            .filter(
                Job.job_type == JobType.discovery_universe,
                Job.workspace_id == job.workspace_id,
                Job.state == JobState.running,
                Job.id != job_id,
            )
            .all()
        )
        for prior_job in superseded_runs:
            prior_job.state = JobState.failed
            prior_job.error_message = f"superseded_by_new_run:{job_id}"
            prior_job.finished_at = datetime.utcnow()
            prior_job.progress_message = "Superseded by newer discovery run"
        if superseded_runs:
            db.commit()

        job.state = JobState.running
        if not job.started_at:
            job.started_at = datetime.utcnow()
        job.progress = 0.05
        job.progress_message = "Queued staged discovery pipeline..."
        db.commit()

        ctx = _load_discovery_context(job_id)
        if not ctx.get("screening_run_id"):
            ctx["screening_run_id"] = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        ctx["job_id"] = job_id
        ctx["workspace_id"] = job.workspace_id
        ctx["pipeline_started_at"] = datetime.utcnow().isoformat()
        ctx["stage_time_ms"] = ctx.get("stage_time_ms") if isinstance(ctx.get("stage_time_ms"), dict) else {}
        ctx["timeout_events"] = ctx.get("timeout_events") if isinstance(ctx.get("timeout_events"), list) else []
        ctx["queue_wait_ms_by_stage"] = ctx.get("queue_wait_ms_by_stage") if isinstance(ctx.get("queue_wait_ms_by_stage"), dict) else {}
        ctx["stage_retry_counts"] = ctx.get("stage_retry_counts") if isinstance(ctx.get("stage_retry_counts"), dict) else {}
        ctx["stage_checkpoints"] = {}
        ctx["cache_stats_start"] = _read_retrieval_cache_stats_snapshot()
        discovery_mode = str(getattr(settings, "discovery_execution_mode", "live") or "live").strip().lower() or "live"
        if discovery_mode not in {"live", "fixture"}:
            discovery_mode = "live"
        ctx["stage_execution_mode"] = discovery_mode
        ctx["stale_runs_failed_count"] = int(stale_runs_failed)
        ctx["superseded_runs_count"] = int(len(superseded_runs))
        _save_discovery_context(job_id, ctx)
        _mark_stage_enqueued(job_id, "stage_seed_ingest")

        if discovery_mode == "fixture":
            pipeline = chain(
                stage_seed_ingest.s(job_id),
                stage_scoring_claims_persist.s(job_id),
                finalize_discovery_pipeline.s(job_id),
            )
        else:
            pipeline = chain(
                stage_seed_ingest.s(job_id),
                stage_llm_discovery_fanout.s(job_id),
                stage_registry_identity_expand.s(job_id),
                stage_first_party_enrichment_parallel.s(job_id),
                stage_scoring_claims_persist.s(job_id),
                finalize_discovery_pipeline.s(job_id),
            )
        pipeline.apply_async()
        discovery_watchdog.apply_async(
            args=[job_id],
            countdown=max(60, int(settings.discovery_global_timeout_seconds)),
        )
        return {"queued": True, "job_id": job_id}
    except Exception as exc:
        _fail_discovery_job(job_id, f"pipeline_queue_failed:{exc}")
        return {"error": str(exc)}
    finally:
        db.close()


@celery_app.task(
    name="app.workers.workspace_tasks.stage_seed_ingest",
    bind=True,
    max_retries=max(1, int(settings.stage_retry_max_attempts)),
)
def stage_seed_ingest(self, job_id: int):
    started = time.perf_counter()
    db = SessionLocal()
    try:
        _record_stage_queue_wait(job_id, "stage_seed_ingest")
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise RuntimeError("Job not found")
        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == job.workspace_id).first()
        company_context_pack = (
            db.query(CompanyContextPack)
            .filter(CompanyContextPack.workspace_id == job.workspace_id)
            .first()
        )
        from app.services.company_context import derive_discovery_scope_hints, normalize_expansion_brief

        scope_hints = (
            derive_discovery_scope_hints(company_context_pack, profile)
            if company_context_pack and profile
            else {}
        )
        scope_review_confirmed = bool(
            company_context_pack
            and normalize_expansion_brief(company_context_pack.expansion_brief_json or {}).get("confirmed_at")
        )
        if not workspace or not profile:
            raise RuntimeError("Missing workspace/profile data")
        discovery_readiness = build_workspace_discovery_readiness(
            expansion_confirmed=scope_review_confirmed,
            database_url_sync=settings.database_url_sync,
            # The job is already executing on a worker at this point. Rechecking
            # worker availability via inspect() is circular and can fail on a
            # healthy solo worker before it reports itself.
            check_worker=False,
        )
        if not discovery_readiness.get("runnable"):
            _save_stage_checkpoint(
                job_id,
                "stage_seed_ingest",
                {
                    "workspace_id": job.workspace_id,
                    "profile_ready": bool(profile),
                    "scope_review_ready": scope_review_confirmed,
                    "scope_hints_ready": bool(scope_hints),
                    "discovery_readiness": discovery_readiness,
                },
            )
            raise RuntimeError(
                "discovery_not_runnable:" + ",".join(discovery_readiness.get("reasons_blocked") or [])
            )
        job.progress = 0.2
        job.progress_message = "Stage 1/5: seed ingest checks complete"
        db.commit()

        previous_quality_audit, previous_run_id = _load_previous_completed_run_quality_audit(
            db=db,
            workspace_id=job.workspace_id,
        )
        pre_rerun_quality_validation_ready = bool(
            previous_quality_audit and bool(previous_quality_audit.get("pass"))
        )
        pre_rerun_quality_validation_blocked_reasons: list[str] = []
        if previous_run_id and not pre_rerun_quality_validation_ready:
            pre_rerun_quality_validation_blocked_reasons.append("pre_rerun_quality_audit_failed")
        elif not previous_run_id:
            pre_rerun_quality_validation_blocked_reasons.append("no_previous_completed_run")
        buyer_employee_estimate = _resolve_buyer_employee_estimate(workspace, profile)

        ctx = _load_discovery_context(job_id)
        ctx["pre_rerun_quality_audit_v1"] = previous_quality_audit if isinstance(previous_quality_audit, dict) else None
        ctx["pre_rerun_quality_audit_run_id"] = previous_run_id
        ctx["pre_rerun_quality_validation_ready"] = pre_rerun_quality_validation_ready
        ctx["pre_rerun_quality_validation_blocked_reasons"] = pre_rerun_quality_validation_blocked_reasons
        ctx["buyer_employee_estimate"] = buyer_employee_estimate
        _save_discovery_context(job_id, ctx)

        _save_stage_checkpoint(
            job_id,
            "stage_seed_ingest",
            {
                "workspace_id": job.workspace_id,
                "profile_ready": bool(profile),
                "scope_review_ready": scope_review_confirmed,
                "scope_hints_ready": bool(scope_hints),
                "discovery_readiness": discovery_readiness,
                "buyer_employee_estimate": buyer_employee_estimate,
                "pre_rerun_quality_audit_run_id": previous_run_id,
                "pre_rerun_quality_validation_ready": pre_rerun_quality_validation_ready,
                "pre_rerun_quality_validation_blocked_reasons": pre_rerun_quality_validation_blocked_reasons,
            },
        )
        _mark_stage_enqueued(job_id, "stage_llm_discovery_fanout")
        _record_stage_metric(job_id, "stage_seed_ingest", started, settings.stage_seed_ingest_timeout_seconds)
        return {"ok": True}
    except Exception as exc:
        if _is_retryable_stage_exception(exc) and self.request.retries < self.max_retries:
            _increment_stage_retry(job_id, "stage_seed_ingest")
            countdown = max(1, int(settings.stage_retry_backoff_seconds * (self.request.retries + 1)))
            raise self.retry(exc=exc, countdown=countdown)
        _fail_discovery_job(job_id, f"stage_seed_ingest_failed:{exc}")
        raise
    finally:
        db.close()


@celery_app.task(
    name="app.workers.workspace_tasks.stage_llm_discovery_fanout",
    bind=True,
    max_retries=max(1, int(settings.stage_retry_max_attempts)),
)
def stage_llm_discovery_fanout(self, _previous: Optional[dict[str, Any]], job_id: int):
    started = time.perf_counter()
    db = SessionLocal()
    try:
        _record_stage_queue_wait(job_id, "stage_llm_discovery_fanout")
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise RuntimeError("Job not found")
        job.progress = 0.35
        job.progress_message = "Stage 2/5: LLM provider fanout preflight"
        db.commit()

        attempts: list[dict[str, Any]] = []
        try:
            response = LLMOrchestrator().run_stage(
                LLMRequest(
                    stage=LLMStage.discovery_query_planning,
                    prompt='Return exactly [] as JSON.',
                    timeout_seconds=max(20, int(settings.stage_llm_discovery_timeout_seconds // 3)),
                    use_web_search=False,
                    expect_json=True,
                    metadata={"preflight": True, "job_id": job_id},
                )
            )
            attempts.extend([asdict(attempt) for attempt in response.attempts])
        except Exception as exc:
            if isinstance(exc, LLMOrchestrationError):
                attempts.extend([asdict(attempt) for attempt in exc.attempts])
            attempts.append(
                {
                    "stage": "discovery_query_planning",
                    "provider": "pipeline",
                    "model": "preflight",
                    "latency_ms": 0,
                    "status": "preflight_error",
                    "retry_count": 0,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc)[:500],
                    "started_at": datetime.utcnow().isoformat(),
                    "ended_at": datetime.utcnow().isoformat(),
                }
            )

        ctx = _load_discovery_context(job_id)
        trace = ctx.get("model_attempt_trace_preflight") if isinstance(ctx.get("model_attempt_trace_preflight"), list) else []
        trace.extend(attempts[:30])
        ctx["model_attempt_trace_preflight"] = trace[:100]
        _save_discovery_context(job_id, ctx)
        _save_stage_checkpoint(
            job_id,
            "stage_llm_discovery_fanout",
            {
                "attempt_count": len(attempts),
                "success_attempts": len([row for row in attempts if str(row.get("status")) == "success"]),
            },
        )
        _mark_stage_enqueued(job_id, "stage_registry_identity_expand")
        _record_stage_metric(job_id, "stage_llm_discovery_fanout", started, settings.stage_llm_discovery_timeout_seconds)
        return {"ok": True}
    except Exception as exc:
        if _is_retryable_stage_exception(exc) and self.request.retries < self.max_retries:
            _increment_stage_retry(job_id, "stage_llm_discovery_fanout")
            countdown = max(1, int(settings.stage_retry_backoff_seconds * (self.request.retries + 1)))
            raise self.retry(exc=exc, countdown=countdown)
        _fail_discovery_job(job_id, f"stage_llm_discovery_fanout_failed:{exc}")
        raise
    finally:
        db.close()


@celery_app.task(
    name="app.workers.workspace_tasks.stage_registry_identity_expand",
    bind=True,
    max_retries=max(1, int(settings.stage_retry_max_attempts)),
)
def stage_registry_identity_expand(self, _previous: Optional[dict[str, Any]], job_id: int):
    started = time.perf_counter()
    db = SessionLocal()
    try:
        _record_stage_queue_wait(job_id, "stage_registry_identity_expand")
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise RuntimeError("Job not found")
        job.progress = 0.45
        job.progress_message = "Stage 3/5: registry expansion preflight"
        db.commit()
        _save_stage_checkpoint(job_id, "stage_registry_identity_expand", {"status": "preflight_ok"})
        _mark_stage_enqueued(job_id, "stage_first_party_enrichment_parallel")
        _record_stage_metric(job_id, "stage_registry_identity_expand", started, settings.stage_registry_timeout_seconds)
        return {"ok": True}
    except Exception as exc:
        if _is_retryable_stage_exception(exc) and self.request.retries < self.max_retries:
            _increment_stage_retry(job_id, "stage_registry_identity_expand")
            countdown = max(1, int(settings.stage_retry_backoff_seconds * (self.request.retries + 1)))
            raise self.retry(exc=exc, countdown=countdown)
        _fail_discovery_job(job_id, f"stage_registry_identity_expand_failed:{exc}")
        raise
    finally:
        db.close()


@celery_app.task(
    name="app.workers.workspace_tasks.stage_first_party_enrichment_parallel",
    bind=True,
    max_retries=max(1, int(settings.stage_retry_max_attempts)),
)
def stage_first_party_enrichment_parallel(self, _previous: Optional[dict[str, Any]], job_id: int):
    started = time.perf_counter()
    db = SessionLocal()
    try:
        _record_stage_queue_wait(job_id, "stage_first_party_enrichment_parallel")
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise RuntimeError("Job not found")
        job.progress = 0.55
        job.progress_message = "Stage 4/5: first-party enrichment preflight"
        db.commit()
        _save_stage_checkpoint(job_id, "stage_first_party_enrichment_parallel", {"status": "preflight_ok"})
        _mark_stage_enqueued(job_id, "stage_scoring_claims_persist")
        _record_stage_metric(job_id, "stage_first_party_enrichment_parallel", started, settings.stage_enrichment_timeout_seconds)
        return {"ok": True}
    except Exception as exc:
        if _is_retryable_stage_exception(exc) and self.request.retries < self.max_retries:
            _increment_stage_retry(job_id, "stage_first_party_enrichment_parallel")
            countdown = max(1, int(settings.stage_retry_backoff_seconds * (self.request.retries + 1)))
            raise self.retry(exc=exc, countdown=countdown)
        _fail_discovery_job(job_id, f"stage_first_party_enrichment_parallel_failed:{exc}")
        raise
    finally:
        db.close()


@celery_app.task(
    name="app.workers.workspace_tasks.stage_scoring_claims_persist",
    bind=True,
    max_retries=2,
)
def stage_scoring_claims_persist(self, _previous: Optional[dict[str, Any]], job_id: int):
    started = time.perf_counter()
    try:
        _record_stage_queue_wait(job_id, "stage_scoring_claims_persist")
        if str(getattr(settings, "discovery_execution_mode", "live") or "live").strip().lower() == "fixture":
            result = _run_discovery_universe_fixture(job_id)
        else:
            result = _run_discovery_universe_monolith(job_id)
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result.get("error")))
        _save_stage_checkpoint(
            job_id,
            "stage_scoring_claims_persist",
            {
                "status": "completed",
                "created": int(result.get("created", 0)) if isinstance(result, dict) else 0,
                "screening_run_id": (result.get("screening_run_id") if isinstance(result, dict) else None),
            },
        )
        _mark_stage_enqueued(job_id, "finalize_discovery_pipeline")
        _record_stage_metric(job_id, "stage_scoring_claims_persist", started, settings.stage_scoring_timeout_seconds)
        return result
    except Exception as exc:
        if _is_deadlock_error(exc) and self.request.retries < self.max_retries:
            _increment_stage_retry(job_id, "stage_scoring_claims_persist")
            retry_delay = min(60, 5 * (2 ** self.request.retries))
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job and job.state != JobState.completed:
                    job.state = JobState.running
                    job.progress_message = (
                        f"Retrying scoring persistence after transient lock error "
                        f"({self.request.retries + 1}/{self.max_retries})"
                    )
                    db.commit()
            finally:
                db.close()
            raise self.retry(exc=exc, countdown=retry_delay)
        if _is_retryable_stage_exception(exc) and self.request.retries < self.max_retries:
            _increment_stage_retry(job_id, "stage_scoring_claims_persist")
            countdown = max(1, int(settings.stage_retry_backoff_seconds * (self.request.retries + 1)))
            raise self.retry(exc=exc, countdown=countdown)
        _fail_discovery_job(job_id, f"stage_scoring_claims_persist_failed:{exc}")
        raise


@celery_app.task(name="app.workers.workspace_tasks.finalize_discovery_pipeline")
def finalize_discovery_pipeline(_previous: Optional[dict[str, Any]], job_id: int):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        _record_stage_queue_wait(job_id, "finalize_discovery_pipeline")
        ctx = _load_discovery_context(job_id)
        if job.state == JobState.completed and isinstance(job.result_json, dict):
            stage_time_ms = ctx.get("stage_time_ms") if isinstance(ctx.get("stage_time_ms"), dict) else {}
            timeout_events = ctx.get("timeout_events") if isinstance(ctx.get("timeout_events"), list) else []
            queue_wait_ms_by_stage = (
                ctx.get("queue_wait_ms_by_stage")
                if isinstance(ctx.get("queue_wait_ms_by_stage"), dict)
                else {}
            )
            stage_retry_counts = (
                ctx.get("stage_retry_counts")
                if isinstance(ctx.get("stage_retry_counts"), dict)
                else {}
            )
            stage_checkpoints = (
                ctx.get("stage_checkpoints")
                if isinstance(ctx.get("stage_checkpoints"), dict)
                else {}
            )
            cache_stats_start = (
                ctx.get("cache_stats_start")
                if isinstance(ctx.get("cache_stats_start"), dict)
                else {}
            )
            cache_stats_end = _read_retrieval_cache_stats_snapshot()
            cache_hit_rates = _compute_cache_hit_rates(cache_stats_start, cache_stats_end)
            preflight_trace = (
                ctx.get("model_attempt_trace_preflight")
                if isinstance(ctx.get("model_attempt_trace_preflight"), list)
                else []
            )
            result = dict(job.result_json)
            merged_stage = result.get("stage_time_ms") if isinstance(result.get("stage_time_ms"), dict) else {}
            merged_stage.update({k: int(v) for k, v in stage_time_ms.items() if str(k).strip()})
            result["stage_time_ms"] = merged_stage
            merged_timeout = result.get("timeout_events") if isinstance(result.get("timeout_events"), list) else []
            if timeout_events:
                merged_timeout.extend(timeout_events)
            result["timeout_events"] = merged_timeout[:200]
            if preflight_trace:
                existing_trace = result.get("model_attempt_trace") if isinstance(result.get("model_attempt_trace"), list) else []
                result["model_attempt_trace"] = (preflight_trace + existing_trace)[:250]
            result["queue_wait_ms_by_stage"] = {
                str(k): int(v)
                for k, v in queue_wait_ms_by_stage.items()
                if str(k).strip()
            }
            result["stage_retry_counts"] = {
                str(k): int(v)
                for k, v in stage_retry_counts.items()
                if str(k).strip()
            }
            result["stage_checkpoints"] = stage_checkpoints
            screening_run_id = str(result.get("screening_run_id") or ctx.get("screening_run_id") or "").strip()
            if screening_run_id:
                result["stage_checkpoint_key"] = f"job:{int(job_id)}:run:{screening_run_id}"
            else:
                result["stage_checkpoint_key"] = f"job:{int(job_id)}"
            result["stage_execution_mode"] = str(
                result.get("stage_execution_mode")
                or ctx.get("stage_execution_mode")
                or "hybrid_preflight_monolith"
            )
            result["cache_hit_rates"] = cache_hit_rates
            result["cache_stats_snapshot"] = {
                "start": cache_stats_start,
                "end": cache_stats_end,
            }
            result["candidate_dropoff_funnel_v1"] = {
                "seed_total_count": int(
                    (result.get("seed_directory_count") or 0)
                    + (result.get("seed_reference_count") or 0)
                    + (result.get("seed_benchmark_count") or 0)
                    + (result.get("seed_llm_count") or 0)
                    + (result.get("seed_external_search_count") or 0)
                ),
                "registry_enriched_count": int(result.get("final_universe_count") or 0),
                "scoring_pool_count": int(result.get("scoring_entities_count") or 0),
                "ranking_eligible_count": int(result.get("ranking_eligible_count") or 0),
                "kept_count": int(result.get("kept_count") or 0),
                "review_count": int(result.get("review_count") or 0),
                "rejected_count": int(result.get("rejected_count") or 0),
            }
            if not isinstance(result.get("pre_rerun_quality_audit_v1"), dict):
                pre_audit = (
                    ctx.get("pre_rerun_quality_audit_v1")
                    if isinstance(ctx.get("pre_rerun_quality_audit_v1"), dict)
                    else None
                )
                if pre_audit:
                    result["pre_rerun_quality_audit_v1"] = pre_audit
            if "pre_rerun_quality_audit_run_id" not in result:
                result["pre_rerun_quality_audit_run_id"] = str(
                    ctx.get("pre_rerun_quality_audit_run_id") or ""
                ).strip() or None
            if "pre_rerun_quality_validation_ready" not in result:
                result["pre_rerun_quality_validation_ready"] = bool(
                    ctx.get("pre_rerun_quality_validation_ready", False)
                )
            if "pre_rerun_quality_validation_blocked_reasons" not in result:
                result["pre_rerun_quality_validation_blocked_reasons"] = [
                    str(item).strip()
                    for item in (ctx.get("pre_rerun_quality_validation_blocked_reasons") or [])
                    if str(item).strip()
                ]
            normalized_audit = normalize_quality_audit_v1(result.get("quality_audit_v1"))
            if normalized_audit:
                result["quality_audit_v1"] = normalized_audit
                result["quality_audit_passed"] = bool(normalized_audit.get("pass"))
            if "quality_validation_ready" not in result:
                result["quality_validation_ready"] = bool(result.get("quality_audit_passed", False))
            if "quality_validation_blocked_reasons" not in result:
                result["quality_validation_blocked_reasons"] = (
                    [] if bool(result.get("quality_validation_ready", False)) else ["quality_audit_failed"]
                )
            job.result_json = result
            db.commit()
        elif job.state == JobState.running:
            job.state = JobState.failed
            job.error_message = "Pipeline ended without terminal monolith outcome"
            job.finished_at = datetime.utcnow()
            db.commit()
        return {"job_id": job_id, "status": job.state.value}
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.fail_discovery_pipeline")
def fail_discovery_pipeline(job_id: int, reason: str):
    _fail_discovery_job(job_id, reason)
    return {"job_id": job_id, "status": "failed", "reason": reason}


@celery_app.task(name="app.workers.workspace_tasks.discovery_watchdog")
def discovery_watchdog(job_id: int):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        if job.state in {JobState.completed, JobState.failed}:
            return {"job_id": job_id, "status": job.state.value}
        fail_discovery_pipeline.delay(job_id, "pipeline_watchdog_timeout")
        return {"job_id": job_id, "status": "failed_by_watchdog"}
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.run_enrich_company")
def run_enrich_company(job_id: int):
    """Enrich a single company with dossier data."""
    from app.services.gemini_workspace import GeminiWorkspaceClient
    
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        
        job.state = JobState.running
        job.started_at = datetime.utcnow()
        job.progress = 0.1
        job.progress_message = "Starting enrichment..."
        db.commit()
        
        # Get company
        company = db.query(Company).filter(Company.id == job.company_id).first()
        if not company:
            job.state = JobState.failed
            job.error_message = "Company not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Company not found"}
        
        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        effective_policy = normalize_policy((workspace.decision_policy_json if workspace else None) or DEFAULT_EVIDENCE_POLICY)
        candidate_domain = normalize_domain(company.website)
        
        try:
            gemini = GeminiWorkspaceClient()
            
            job.progress = 0.3
            job.progress_message = "Researching..."
            db.commit()
            
            # Run appropriate enrichment based on job type
            if job.job_type == JobType.enrich_full:
                dossier_json = gemini.run_enrich_full(
                    company_url=company.website or "",
                    company_name=company.name,
                )
            elif job.job_type == JobType.enrich_modules:
                dossier_json = gemini.run_enrich_modules(company.website or "")
            elif job.job_type == JobType.enrich_customers:
                dossier_json = gemini.run_enrich_customers(company.website or "")
            elif job.job_type == JobType.enrich_hiring:
                dossier_json = gemini.run_enrich_hiring(company.website or "")
            else:
                dossier_json = {}
            
            job.progress = 0.8
            job.progress_message = "Saving dossier..."
            db.commit()
            
            # Get latest version
            existing_dossier = db.query(CompanyDossier).filter(
                CompanyDossier.company_id == company.id
            ).order_by(CompanyDossier.version.desc()).first()
            
            new_version = (existing_dossier.version + 1) if existing_dossier else 1
            
            # If partial enrichment, merge with existing
            if job.job_type != JobType.enrich_full and existing_dossier:
                merged = existing_dossier.dossier_json.copy() if existing_dossier.dossier_json else {}
                merged.update(dossier_json)
                dossier_json = merged
            
            # Create new dossier
            dossier = CompanyDossier(
                company_id=company.id,
                dossier_json=dossier_json,
                version=new_version
            )
            db.add(dossier)
            
            # Update company status
            company.status = CompanyStatus.enriched
            company.updated_at = datetime.utcnow()
            
            # Create evidence items from dossier
            canonical_sections = {
                "workflow": ("workflow", "workflow"),
                "customer": ("customer", "customer"),
                "business_model": ("business_model", "business_model"),
                "ownership": ("ownership", "ownership"),
                "transaction_feasibility": ("transaction_feasibility", "transaction_feasibility"),
            }
            for section_name, (dimension, claim_key) in canonical_sections.items():
                for entry in dossier_json.get(section_name, []):
                    if not isinstance(entry, dict):
                        continue
                    source_url = entry.get("evidence_url")
                    excerpt_text = str(entry.get("text") or "").strip()
                    if not source_url or not excerpt_text or not is_trusted_source_url(source_url):
                        continue
                    source_type = _source_type_for_url(source_url, candidate_domain)
                    source_tier = infer_source_tier(source_url, source_type, candidate_domain)
                    source_kind = infer_source_kind(source_url, source_type, candidate_domain)
                    claim_group = claim_group_for_dimension(dimension, claim_key)
                    ttl_days, valid_through = valid_through_from_claim_group(
                        claim_group,
                        policy=effective_policy,
                    )
                    evidence = SourceEvidence(
                        workspace_id=job.workspace_id,
                        company_id=company.id,
                        source_url=source_url,
                        excerpt_text=excerpt_text[:2000],
                        content_type="web",
                        source_tier=source_tier,
                        source_kind=source_kind,
                        freshness_ttl_days=ttl_days,
                        valid_through=valid_through,
                        asserted_by="run_enrich_company",
                    )
                    db.add(evidence)

            for metric_name, metric_payload in (dossier_json.get("kpis") or {}).items():
                if not isinstance(metric_payload, dict):
                    continue
                source_url = metric_payload.get("evidence_url")
                value = str(metric_payload.get("value") or "").strip()
                if not source_url or not value or not is_trusted_source_url(source_url):
                    continue
                source_type = _source_type_for_url(source_url, candidate_domain)
                source_tier = infer_source_tier(source_url, source_type, candidate_domain)
                source_kind = infer_source_kind(source_url, source_type, candidate_domain)
                claim_group = claim_group_for_dimension("financial", metric_name)
                ttl_days, valid_through = valid_through_from_claim_group(
                    claim_group,
                    policy=effective_policy,
                )
                evidence = SourceEvidence(
                    workspace_id=job.workspace_id,
                    company_id=company.id,
                    source_url=source_url,
                    excerpt_text=f"{metric_name}: {value}",
                    content_type="web",
                    source_tier=source_tier,
                    source_kind=source_kind,
                    freshness_ttl_days=ttl_days,
                    valid_through=valid_through,
                    asserted_by="run_enrich_company",
                )
                db.add(evidence)

            for module in dossier_json.get("modules", []):
                for url in module.get("evidence_urls", []):
                    if not is_trusted_source_url(url):
                        continue
                    source_type = _source_type_for_url(url, candidate_domain)
                    source_tier = infer_source_tier(url, source_type, candidate_domain)
                    source_kind = infer_source_kind(url, source_type, candidate_domain)
                    claim_group = claim_group_for_dimension("product", "module")
                    ttl_days, valid_through = valid_through_from_claim_group(
                        claim_group,
                        policy=effective_policy,
                    )
                    evidence = SourceEvidence(
                        workspace_id=job.workspace_id,
                        company_id=company.id,
                        source_url=url,
                        excerpt_text=module.get("description", ""),
                        content_type="web",
                        source_tier=source_tier,
                        source_kind=source_kind,
                        freshness_ttl_days=ttl_days,
                        valid_through=valid_through,
                        asserted_by="run_enrich_company",
                    )
                    db.add(evidence)
            
            for customer in dossier_json.get("customers", []):
                if customer.get("evidence_url") and is_trusted_source_url(customer.get("evidence_url")):
                    source_url = customer.get("evidence_url", "")
                    source_type = _source_type_for_url(source_url, candidate_domain)
                    source_tier = infer_source_tier(source_url, source_type, candidate_domain)
                    source_kind = infer_source_kind(source_url, source_type, candidate_domain)
                    claim_group = claim_group_for_dimension("customer", "customer")
                    ttl_days, valid_through = valid_through_from_claim_group(
                        claim_group,
                        policy=effective_policy,
                    )
                    evidence = SourceEvidence(
                        workspace_id=job.workspace_id,
                        company_id=company.id,
                        source_url=source_url,
                        excerpt_text=f"Customer: {customer.get('name', '')}",
                        content_type="case_study",
                        source_tier=source_tier,
                        source_kind=source_kind,
                        freshness_ttl_days=ttl_days,
                        valid_through=valid_through,
                        asserted_by="run_enrich_company",
                    )
                    db.add(evidence)
            
            job.state = JobState.completed
            job.progress = 1.0
            job.progress_message = "Complete"
            job.result_json = {
                "modules_found": len(dossier_json.get("modules", [])),
                "customers_found": len(dossier_json.get("customers", [])),
                "dossier_version": new_version
            }
            job.finished_at = datetime.utcnow()
            db.commit()
            
            return {"success": True}
            
        except Exception as e:
            job.state = JobState.failed
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(e)}
            
    finally:
        db.close()


def _extract_reference_tokens(profile: CompanyProfile | None) -> set[str]:
    tokens: set[str] = set()
    if not profile:
        return tokens

    urls = []
    if profile.buyer_company_url:
        urls.append(profile.buyer_company_url)
    if profile.comparator_seed_urls:
        urls.extend(profile.comparator_seed_urls)
    if profile.supporting_evidence_urls:
        urls.extend(profile.supporting_evidence_urls)

    for url in urls:
        domain = normalize_domain(url)
        if not domain:
            continue
        tokens.add(domain)
        first_label = domain.split(".")[0]
        if first_label:
            tokens.add(first_label)
    return tokens


def _context_pack_has_meaningful_content(context_pack: Any) -> bool:
    if context_pack is None:
        return False
    if str(getattr(context_pack, "summary", "") or "").strip():
        return True
    if any(getattr(context_pack, "signals", None) or []):
        return True
    if any(getattr(context_pack, "customer_evidence", None) or []):
        return True
    for page in getattr(context_pack, "pages", None) or []:
        if str(getattr(page, "raw_content", "") or "").strip():
            return True
        if any(getattr(page, "blocks", None) or []):
            return True
        if any(getattr(page, "signals", None) or []):
            return True
        if any(getattr(page, "customer_evidence", None) or []):
            return True
    return False


@celery_app.task(name="app.workers.workspace_tasks.generate_static_report")
def generate_static_report(job_id: int, filters: dict | None = None):
    """Generate immutable static report snapshot for a workspace."""
    db = SessionLocal()
    filters = filters or {}
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.state = JobState.running
        job.started_at = datetime.utcnow()
        job.progress = 0.05
        job.progress_message = "Preparing report snapshot..."
        db.commit()

        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        profile = db.query(CompanyProfile).filter(CompanyProfile.workspace_id == job.workspace_id).first()
        if not workspace:
            job.state = JobState.failed
            job.error_message = "Missing workspace for report generation"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Missing workspace"}
        effective_policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)

        geo_scope = profile.geo_scope if profile and profile.geo_scope else {}
        include_countries = {
            normalize_country(c)
            for c in geo_scope.get("include_countries", [])
            if normalize_country(c)
        }
        exclude_countries = {
            normalize_country(c)
            for c in geo_scope.get("exclude_countries", [])
            if normalize_country(c)
        }
        include_unknown_size = bool(filters.get("include_unknown_size", False))
        include_outside_sme = bool(filters.get("include_outside_sme", False))
        allowed_countries = include_countries if include_countries else set(DISCOVERY_COUNTRIES)

        promoted_entities = [
            entity
            for entity in (
                db.query(CandidateEntity)
                .filter(CandidateEntity.workspace_id == workspace.id)
                .order_by(CandidateEntity.created_at.asc())
                .all()
            )
            if bool(validation_metadata(entity).get("promoted_to_cards"))
        ]
        if not promoted_entities:
            job.state = JobState.failed
            job.error_message = "No promoted validation shortlist candidates available for Cards"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "No promoted validation shortlist candidates available for Cards"}

        promoted_entity_ids = [int(entity.id) for entity in promoted_entities]
        screenings = (
            db.query(CompanyScreening)
            .filter(
                CompanyScreening.workspace_id == workspace.id,
                CompanyScreening.candidate_entity_id.in_(promoted_entity_ids),
            )
            .order_by(CompanyScreening.created_at.desc())
            .all()
        )
        latest_screening_by_entity: dict[int, CompanyScreening] = {}
        for screening in screenings:
            if screening.candidate_entity_id and screening.candidate_entity_id not in latest_screening_by_entity:
                latest_screening_by_entity[screening.candidate_entity_id] = screening

        companies: list[Company] = []
        latest_screening_by_company: dict[int, CompanyScreening] = {}
        for entity in promoted_entities:
            screening = latest_screening_by_entity.get(int(entity.id))
            if screening is None:
                continue
            company_ref: Optional[Company] = None
            if screening.company_id:
                company_ref = (
                    db.query(Company)
                    .filter(Company.workspace_id == workspace.id, Company.id == screening.company_id)
                    .first()
                )
            if company_ref is None:
                metadata = screening.screening_meta_json if isinstance(screening.screening_meta_json, dict) else {}
                company_ref = Company(
                    workspace_id=workspace.id,
                    name=screening.candidate_name,
                    website=screening.candidate_official_website or entity.canonical_website,
                    hq_country=metadata.get("candidate_hq_country") or entity.country,
                    tags_custom=["validation:promoted_to_cards"],
                    status=CompanyStatus.kept,
                    why_relevant=[],
                    is_manual=False,
                )
                db.add(company_ref)
                db.flush()
                screening.company_id = company_ref.id
            elif not company_ref.is_manual:
                company_ref.status = CompanyStatus.kept
            companies.append(company_ref)
            latest_screening_by_company[int(company_ref.id)] = screening

        normalized_filters = {
            "size_bucket_default": "sme_in_range",
            "include_unknown_size": include_unknown_size,
            "include_outside_sme": include_outside_sme,
            "countries": DISCOVERY_COUNTRIES,
            "reliable_filings": sorted(RELIABLE_FILINGS_COUNTRIES),
        }
        if filters.get("name"):
            normalized_filters["name"] = str(filters["name"])

        snapshot_name = str(
            filters.get("name")
            or f"{workspace.name} static report {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # Build items first so snapshot persistence remains immutable and atomic.
        item_payloads: list[dict] = []
        bucket_counts = {"sme_in_range": 0, "unknown": 0, "outside_sme_range": 0}
        filing_fact_count = 0

        total = len(companies)
        for index, company in enumerate(companies):
            if not _company_passes_enterprise_screen(company.tags_custom or []):
                continue

            company_evidence = (
                db.query(SourceEvidence)
                .filter(SourceEvidence.company_id == company.id)
                .order_by(SourceEvidence.captured_at.desc())
                .all()
            )

            normalized_country = normalize_country(company.hq_country)

            # Promote filing-like discovery claims into structured evidence rows when source is reliable.
            filing_claims_created = False
            for claim in company.why_relevant or []:
                if not isinstance(claim, dict):
                    continue
                citation_url = claim.get("citation_url")
                excerpt_text = str(claim.get("text") or "").strip()
                if not citation_url or not excerpt_text:
                    continue
                if not is_reliable_filing_source_url(normalized_country, citation_url):
                    continue
                exists = (
                    db.query(SourceEvidence)
                    .filter(
                        SourceEvidence.company_id == company.id,
                        SourceEvidence.source_url == citation_url,
                        SourceEvidence.excerpt_text == excerpt_text,
                    )
                    .first()
                )
                if exists:
                    continue
                db.add(
                    SourceEvidence(
                        workspace_id=workspace.id,
                        company_id=company.id,
                        source_url=citation_url,
                        source_title="Discovery filing citation",
                        excerpt_text=excerpt_text[:2000],
                        content_type="registry",
                        source_tier="tier0_registry",
                        source_kind="registry",
                        freshness_ttl_days=365,
                        valid_through=datetime.utcnow() + timedelta(days=365),
                        asserted_by="generate_static_report",
                    )
                )
                filing_claims_created = True

            if filing_claims_created:
                db.flush()
                company_evidence = (
                    db.query(SourceEvidence)
                    .filter(SourceEvidence.company_id == company.id)
                    .order_by(SourceEvidence.captured_at.desc())
                    .all()
                )

            latest_dossier = (
                db.query(CompanyDossier)
                .filter(CompanyDossier.company_id == company.id)
                .order_by(CompanyDossier.version.desc())
                .first()
            )
            dossier_json = latest_dossier.dossier_json if latest_dossier else {}

            fallback_capabilities = [
                tag.split(":", 1)[1].strip()
                for tag in (company.tags_custom or [])
                if isinstance(tag, str) and tag.startswith("capability:")
            ]
            fallback_evidence_urls = [
                e.source_url
                for e in company_evidence
                if e.source_url and is_trusted_source_url(e.source_url)
            ]
            modules = modules_with_evidence(
                dossier_json,
                fallback_capabilities=fallback_capabilities,
                fallback_evidence_urls=fallback_evidence_urls,
            )
            customers, integrations = extract_customers_and_integrations(dossier_json)

            if allowed_countries:
                geo_match = normalized_country in allowed_countries
            else:
                geo_match = normalized_country in set(DISCOVERY_COUNTRIES) if normalized_country else False
            if normalized_country in exclude_countries:
                geo_match = False

            # Enforce V1 discovery geography (allow unknown-country records to remain reviewable).
            if normalized_country and not geo_match:
                continue

            existing_facts = (
                db.query(CompanyFact)
                .filter(CompanyFact.company_id == company.id)
                .all()
            )

            filing_facts = extract_filing_facts_from_evidence(normalized_country, company_evidence)
            facts_for_size = list(existing_facts)
            facts_for_size.extend(filing_facts)

            size_estimate = estimate_size_from_signals(
                dossier_json=dossier_json,
                facts=facts_for_size,
                evidence_items=company_evidence,
                tags_custom=company.tags_custom or [],
                why_relevant=company.why_relevant or [],
            )
            size_bucket = classify_size_bucket(size_estimate)
            if size_bucket == "outside_sme_range" and not include_outside_sme:
                continue
            if size_bucket == "unknown" and not include_unknown_size:
                continue
            bucket_counts[size_bucket] = bucket_counts.get(size_bucket, 0) + 1

            latest_screening = latest_screening_by_company.get(company.id)
            fit_score = 0.0
            if latest_screening and latest_screening.total_score is not None:
                fit_score = round(max(0.0, min(100.0, float(latest_screening.total_score))), 2)
            elif latest_screening and latest_screening.decision_classification == "good_target":
                fit_score = 70.0
            elif latest_screening and latest_screening.decision_classification == "borderline_watchlist":
                fit_score = 50.0
            elif latest_screening and latest_screening.decision_classification == "not_good_target":
                fit_score = 20.0

            fresh_evidence_count = len([row for row in company_evidence if is_fresh(row.valid_through)])
            evidence_score = round(
                min(
                    100.0,
                    (min(len(company_evidence), 12) * 5.0)
                    + (min(fresh_evidence_count, 12) * 3.0)
                    + (20.0 if normalized_country in RELIABLE_FILINGS_COUNTRIES else 0.0),
                ),
                2,
            )

            if filing_facts:
                filing_fact_count += len(filing_facts)
                for fact in filing_facts:
                    exists = (
                        db.query(CompanyFact)
                        .filter(
                            CompanyFact.company_id == company.id,
                            CompanyFact.fact_key == fact.fact_key,
                            CompanyFact.fact_value == fact.fact_value,
                            CompanyFact.period == fact.period,
                            CompanyFact.source_evidence_id == fact.source_evidence_id,
                        )
                        .first()
                    )
                    if exists:
                        continue
                    db.add(
                        CompanyFact(
                            company_id=company.id,
                            fact_key=fact.fact_key,
                            fact_value=fact.fact_value,
                            fact_unit=fact.fact_unit,
                            period=fact.period,
                            confidence=fact.confidence,
                            source_evidence_id=fact.source_evidence_id,
                            source_system="filing_parser",
                        )
                    )

            item_payloads.append(
                {
                    "company_id": company.id,
                    "compete_score": fit_score,
                    "complement_score": evidence_score,
                    "decision_classification": (
                        latest_screening_by_company[company.id].decision_classification
                        if company.id in latest_screening_by_company
                        else "insufficient_evidence"
                    ),
                    "reason_codes_json": (
                        {
                            "positive": latest_screening_by_company[company.id].positive_reason_codes_json or [],
                            "caution": latest_screening_by_company[company.id].caution_reason_codes_json or [],
                            "reject": latest_screening_by_company[company.id].reject_reason_codes_json or [],
                        }
                        if company.id in latest_screening_by_company
                        else {"positive": [], "caution": [], "reject": []}
                    ),
                    "evidence_summary_json": {
                        "evidence_count": len(company_evidence),
                        "freshness_ratio": round(
                            len([row for row in company_evidence if is_fresh(row.valid_through)]) / max(1, len(company_evidence)),
                            4,
                        ),
                        "tier_counts": {
                            tier: len([row for row in company_evidence if str(row.source_tier or "") == tier])
                            for tier in ["tier0_registry", "tier1_vendor", "tier2_partner_customer", "tier3_third_party", "tier4_discovery"]
                        },
                    },
                    "lens_breakdown_json": {
                        "fit_score": fit_score,
                        "evidence_score": evidence_score,
                        "size_bucket": size_bucket,
                        "size_estimate": size_estimate,
                        "country": normalized_country,
                        "modules_count": len(modules),
                        "customers_count": len(customers),
                        "integrations_count": len(integrations),
                        "filings_coverage": (
                            "reliable"
                            if normalized_country in RELIABLE_FILINGS_COUNTRIES
                            else "not_available_in_current_reliable_filings_coverage"
                        ),
                        "evidence_count": len(company_evidence),
                        "fresh_evidence_count": fresh_evidence_count,
                    },
                }
            )

            if total:
                job.progress = 0.10 + (0.75 * (index + 1) / total)
                job.progress_message = f"Scored {index + 1}/{total} companies"
                db.commit()

        job.progress = 0.9
        job.progress_message = "Persisting immutable report snapshot..."
        db.commit()

        snapshot = ReportSnapshot(
            workspace_id=workspace.id,
            name=snapshot_name,
            filters_json=normalized_filters,
            generated_at=datetime.utcnow(),
            status="completed",
            coverage_json={
                "discovery_countries": DISCOVERY_COUNTRIES,
                "reliable_filing_countries": sorted(RELIABLE_FILINGS_COUNTRIES),
                "bucket_counts": bucket_counts,
                "filing_facts_created": filing_fact_count,
            },
        )
        db.add(snapshot)
        db.flush()

        for payload in item_payloads:
                db.add(
                    ReportSnapshotItem(
                        report_id=snapshot.id,
                        company_id=payload["company_id"],
                        compete_score=payload["compete_score"],
                        complement_score=payload["complement_score"],
                        decision_classification=payload.get("decision_classification"),
                        reason_codes_json=payload.get("reason_codes_json") or {},
                        evidence_summary_json=payload.get("evidence_summary_json") or {},
                        lens_breakdown_json=payload["lens_breakdown_json"],
                    )
                )

        job.state = JobState.completed
        job.progress = 1.0
        job.progress_message = "Static report snapshot ready"
        job.result_json = {
            "report_id": snapshot.id,
            "total_items": len(item_payloads),
            "bucket_counts": bucket_counts,
            "filing_facts_created": filing_fact_count,
        }
        job.finished_at = datetime.utcnow()
        db.commit()

        return {"success": True, "report_id": snapshot.id, "items": len(item_payloads)}
    except Exception as exc:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.state = JobState.failed
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
        return {"error": str(exc)}
    finally:
        db.close()
