"""Celery tasks for workspace-based workflow."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from html import unescape
import hashlib
import json
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
from app.models.workspace import Workspace, CompanyProfile, BrickTaxonomy
from app.models.thesis import SearchLane
from app.models.vendor import Vendor, VendorDossier, VendorStatus
from app.models.job import Job, JobType, JobState
from app.models.workspace_evidence import WorkspaceEvidence
from app.models.report import ReportSnapshot, ReportSnapshotItem, VendorFact
from app.models.intelligence import (
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    ComparatorSourceRun,
    RegistryQueryLog,
    VendorMention,
    VendorScreening,
    VendorClaim,
)
from app.services.comparator_sources import (
    SOURCE_REGISTRY,
    ingest_source,
    resolve_external_website_from_profile,
)
from app.services.reporting import (
    DISCOVERY_COUNTRIES,
    RELIABLE_FILINGS_COUNTRIES,
    build_adjacency_map,
    classify_size_bucket,
    compute_lens_scores,
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
from app.services.llm.orchestrator import LLMOrchestrator
from app.services.llm.types import LLMRequest, LLMStage, LLMOrchestrationError
from app.services.retrieval.crawl_connectors import fetch_page_fast
from app.models.external_search import ExternalSearchRun, ExternalSearchResult
from app.services.retrieval.search_orchestrator import run_external_search_queries
from app.services.retrieval.url_normalization import normalize_url
from app.services.retrieval.cache import RetrievalCache
from app.services.crawler.connectors.chrome_devtools_mcp import render_page_via_chrome_devtools_mcp
from app.services.quality_audit import (
    build_quality_audit_v1,
    normalize_quality_audit_v1,
    quality_audit_thresholds_from_settings,
)

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
PRE_SCORE_UNIVERSE_CAP = 500
DIRECT_IDENTITY_RESOLUTION_TIMEOUT_SECONDS = 3

FIRST_PARTY_AUTO_HINT_PATHS = (
    "/client-stories/",
    "/customers/",
    "/case-studies/",
    "/resources/case-studies/",
    "/success-stories/",
)

FIRST_PARTY_HINT_URL_TOKENS = (
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


def _screening_run_id_from_screening(screening: VendorScreening) -> str:
    meta = screening.screening_meta_json if isinstance(screening.screening_meta_json, dict) else {}
    return str(meta.get("screening_run_id") or "").strip()


def _collect_run_screenings_and_claims(
    db,
    workspace_id: int,
    screening_run_id: str,
    *,
    max_screenings: int = 5000,
) -> tuple[list[VendorScreening], dict[int, list[VendorClaim]]]:
    target_run_id = str(screening_run_id or "").strip()
    if not target_run_id:
        return [], {}
    screening_rows = (
        db.query(VendorScreening)
        .filter(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
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
    claim_rows = db.query(VendorClaim).filter(VendorClaim.screening_id.in_(screening_ids)).all()
    claims_by_screening: dict[int, list[VendorClaim]] = {}
    for claim in claim_rows:
        if not claim.screening_id:
            continue
        claims_by_screening.setdefault(int(claim.screening_id), []).append(claim)
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
    entity["reference_input"] = bool(entity.get("reference_input")) or bool(candidate.get("reference_input"))
    entity["merged_candidates_count"] = int(entity.get("merged_candidates_count") or 1) + 1
    entity["merge_confidence"] = max(float(entity.get("merge_confidence") or 0.0), merge_confidence)
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
    url = "https://recherche-entreprises.api.gouv.fr/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MA-BuySide-Radar/1.0)",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            response = client.get(url, params={"q": query.strip(), "page": 1, "per_page": max_hits})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return [], f"fr_registry_search_failed:{exc}"

    results: list[dict[str, Any]] = []
    for row in payload.get("results", [])[:max_hits]:
        if not isinstance(row, dict):
            continue
        name = str(
            row.get("nom_complet")
            or row.get("nom_raison_sociale")
            or row.get("nom")
            or ""
        ).strip()
        siren = str(row.get("siren") or "").strip()
        website = row.get("site_internet") or row.get("site_web")
        if isinstance(website, list):
            website = website[0] if website else None
        website = str(website or "").strip() or None
        activity_code = _first_non_empty(
            row.get("activite_principale"),
            row.get("activite_principale_naf25"),
        )
        section_code = _first_non_empty(row.get("section_activite_principale"))
        activity_label = _first_non_empty(row.get("libelle_activite_principale"), activity_code)
        status = _first_non_empty(row.get("etat_administratif"), "A")
        context_text = " ".join(
            token
            for token in [
                str(activity_label or ""),
                str(activity_code or ""),
                str(section_code or ""),
                str(row.get("nature_juridique") or ""),
            ]
            if token
        ).strip()
        industry_codes = _dedupe_strings(
            [str(activity_code or "").upper(), str(section_code or "").upper()]
        )
        citation_url = f"https://annuaire-entreprises.data.gouv.fr/entreprise/{siren}" if siren else f"{url}?q={quote_plus(query)}"
        if not name:
            continue
        record = {
            "name": name,
            "website": website,
            "country": "FR",
            "registry_id": siren or None,
            "registry_source": "fr_recherche_entreprises",
            "registry_url": citation_url,
            "status": status,
            "is_active": _looks_active_status(status),
            "context_text": context_text,
            "industry_codes": industry_codes,
        }
        record["industry_keywords"] = _industry_keywords_from_record(record)
        results.append(
            record
        )
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
    origin_types = set(entity.get("origin_types") or [])
    score = 0.0
    entity_type = str(entity.get("entity_type") or "company").strip().lower()
    if entity_type == "solution":
        score -= 25.0
    if "reference_seed" in origin_types:
        score += 100.0
    if "benchmark_seed" in origin_types:
        score += 95.0
    if "directory_seed" in origin_types:
        score += 70.0
    if "registry_neighbor" in origin_types:
        score += 55.0
    if "llm_seed" in origin_types:
        score += 35.0
    if str(entity.get("identity_confidence") or "").lower() == "high":
        score += 10.0
    registry_identity = entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {}
    if registry_identity.get("id"):
        score += 12.0
    if float(registry_identity.get("match_confidence") or 0.0) >= 0.8:
        score += 8.0
    score += min(20.0, float(len(entity.get("why_relevant") or [])))
    score += min(10.0, float(len(entity.get("capability_signals") or [])))
    return score


def _trim_entities_for_universe(
    entities: list[dict[str, Any]],
    cap: int = PRE_SCORE_UNIVERSE_CAP,
) -> tuple[list[dict[str, Any]], int]:
    if len(entities) <= cap:
        return entities, 0
    ranked = sorted(entities, key=lambda entity: _candidate_priority_score(entity), reverse=True)
    trimmed_out = max(0, len(ranked) - cap)
    return ranked[:cap], trimmed_out


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
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

    text_candidates = [
        profile.buyer_context_summary,
        profile.context_pack_markdown,
    ]
    reference_summaries = profile.reference_summaries if isinstance(profile.reference_summaries, dict) else {}
    text_candidates.extend([str(value) for value in reference_summaries.values()])
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


def _vendor_passes_enterprise_screen(tags_custom: list[str] | None) -> bool:
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
            if title and title.text(strip=True):
                snippets.append(title.text(strip=True))
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


def _seed_candidates_from_mentions(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeded: list[dict[str, Any]] = []
    for mention in mentions:
        company_name = str(mention.get("company_name") or "").strip()
        if not company_name:
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
        seeded.append(
            {
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
                "_origins": [
                    {
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
                        },
                    }
                ],
                "why_relevant": [
                    {
                        "text": snippet_text[:700],
                        "citation_url": str(listing_url or profile_url or ""),
                        "dimension": "directory_context",
                    }
                ],
            }
        )
    return seeded


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
    taxonomy: BrickTaxonomy,
    mentions: list[dict[str, Any]],
) -> bool:
    text_chunks: list[str] = []
    for item in [profile.buyer_context_summary, profile.context_pack_markdown]:
        normalized = str(item or "").strip()
        if normalized:
            text_chunks.append(normalized[:4000])

    for value in (taxonomy.vertical_focus or []):
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
    "website": "https://example.com",
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
            entries.append({"query_text": query_text[:220], "query_intent": None, "brick_name": None, "lane_type": None})
        elif isinstance(item, dict):
            query_text = str(item.get("query") or item.get("query_text") or item.get("text") or "").strip()
            if not query_text:
                continue
            entries.append(
                {
                    "query_text": query_text[:220],
                    "query_intent": str(item.get("query_intent") or item.get("intent") or "").strip() or None,
                    "brick_name": str(item.get("brick_name") or "").strip() or None,
                    "lane_type": str(item.get("lane_type") or "").strip().lower() or None,
                }
            )
        if len(entries) >= max_items:
            break
    return entries


def _normalize_search_lane_payloads(search_lanes: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not isinstance(search_lanes, list):
        return payloads
    for lane in search_lanes:
        if isinstance(lane, SearchLane):
            payload = {
                "lane_type": lane.lane_type,
                "title": lane.title,
                "intent": lane.intent,
                "capabilities": lane.capabilities_json or [],
                "customer_tags": lane.customer_tags_json or [],
                "must_include_terms": lane.must_include_terms_json or [],
                "must_exclude_terms": lane.must_exclude_terms_json or [],
                "seed_urls": lane.seed_urls_json or [],
                "status": lane.status or "draft",
            }
        elif isinstance(lane, dict):
            payload = lane
        else:
            continue
        lane_type = str(payload.get("lane_type") or "").strip().lower()
        if lane_type not in {"core", "adjacent"}:
            continue
        payloads.append(
            {
                "lane_type": lane_type,
                "title": str(payload.get("title") or f"{lane_type.title()} sourcing lane").strip()[:255],
                "intent": str(payload.get("intent") or "").strip()[:500] or None,
                "capabilities": _normalize_string_list(payload.get("capabilities"), max_items=10, max_len=140),
                "customer_tags": _normalize_string_list(payload.get("customer_tags"), max_items=8, max_len=120),
                "must_include_terms": _normalize_string_list(payload.get("must_include_terms"), max_items=10, max_len=80),
                "must_exclude_terms": _normalize_string_list(payload.get("must_exclude_terms"), max_items=10, max_len=80),
                "seed_urls": _normalize_string_list(payload.get("seed_urls"), max_items=8, max_len=220),
                "status": str(payload.get("status") or "draft").strip().lower() or "draft",
            }
        )
    confirmed = [payload for payload in payloads if payload["status"] == "confirmed"]
    return confirmed or payloads


def _default_discovery_query_plan(
    taxonomy_bricks: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    vertical_focus: list[str],
    search_lanes: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    normalized_lanes = _normalize_search_lane_payloads(search_lanes)
    brick_names = [str(b.get("name") or "").strip() for b in (taxonomy_bricks or []) if str(b.get("name") or "").strip()]
    region = str((geo_scope or {}).get("region") or "EU+UK")
    if normalized_lanes:
        core_lane = next((lane for lane in normalized_lanes if lane["lane_type"] == "core"), normalized_lanes[0])
        adjacent_lane = next((lane for lane in normalized_lanes if lane["lane_type"] == "adjacent"), None)
        customer_hint = ", ".join((core_lane.get("customer_tags") or [])[:2]) or "B2B software"
        core_capabilities = core_lane.get("capabilities") or ["core workflow"]
        adjacent_capabilities = adjacent_lane.get("capabilities") or ["adjacent workflow"] if adjacent_lane else ["adjacent workflow"]
        precision_queries = [
            {
                "query_text": f"{customer_hint} {core_capabilities[0]} software vendor {region}",
                "query_intent": "capability_discovery",
                "brick_name": core_capabilities[0],
                "lane_type": "core",
            },
            {
                "query_text": f"{customer_hint} {core_capabilities[min(1, len(core_capabilities) - 1)]} platform {region}",
                "query_intent": "competitor_discovery",
                "brick_name": core_capabilities[min(1, len(core_capabilities) - 1)],
                "lane_type": "core",
            },
        ]
        recall_queries = [
            {
                "query_text": f"{customer_hint} {adjacent_capabilities[0]} software {region}",
                "query_intent": "adjacent",
                "brick_name": adjacent_capabilities[0],
                "lane_type": "adjacent",
            }
        ]
        return {
            "precision_queries": precision_queries,
            "recall_queries": recall_queries,
            "seed_urls": _dedupe_strings(
                [str(url) for lane in normalized_lanes for url in (lane.get("seed_urls") or []) if str(url).strip()]
            )[:8],
            "must_include_terms": _dedupe_strings(
                [
                    str(term)
                    for lane in normalized_lanes
                    for term in ((lane.get("must_include_terms") or []) + (lane.get("customer_tags") or []))
                    if str(term).strip()
                ]
            )[:10],
            "must_exclude_terms": _dedupe_strings(
                [str(term) for lane in normalized_lanes for term in (lane.get("must_exclude_terms") or []) if str(term).strip()]
            )[:10],
            "preferred_countries": [],
            "preferred_languages": [],
            "domain_allowlist": [],
            "domain_blocklist": [],
        }

    vertical_hint = ", ".join([str(v) for v in (vertical_focus or [])[:2] if str(v).strip()]) or "B2B software"
    brick_hint = ", ".join(brick_names[:2]) if brick_names else "core workflow"

    precision_queries = [
        {"query_text": f"{vertical_hint} {brick_hint} vendor {region}", "query_intent": "capability_discovery", "brick_name": brick_names[0] if brick_names else None, "lane_type": "core"},
        {"query_text": f"{vertical_hint} platform {region} {brick_hint}", "query_intent": "competitor_discovery", "brick_name": brick_names[1] if len(brick_names) > 1 else None, "lane_type": "core"},
    ]
    recall_queries = [
        {"query_text": f"{vertical_hint} SaaS {region}", "query_intent": "market_scan", "brick_name": None, "lane_type": "core"},
        {"query_text": f"{vertical_hint} adjacent workflow software {region}", "query_intent": "adjacent", "brick_name": None, "lane_type": "adjacent"},
    ]
    return {
        "precision_queries": precision_queries,
        "recall_queries": recall_queries,
        "seed_urls": [],
        "must_include_terms": [],
        "must_exclude_terms": [],
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
    search_lanes: Optional[list[dict[str, Any]]] = None,
) -> str:
    normalized_lanes = _normalize_search_lane_payloads(search_lanes)
    brick_names = [str(b.get("name") or "").strip() for b in (taxonomy_bricks or []) if str(b.get("name") or "").strip()]
    region = str((geo_scope or {}).get("region") or "EU+UK")
    include_countries = [str(c).strip() for c in ((geo_scope or {}).get("include_countries") or []) if str(c).strip()]
    verticals = [str(v).strip() for v in (vertical_focus or []) if str(v).strip()]

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
    {{"query": "string", "query_intent": "competitors|capability|pricing|integrations|market_scan", "brick_name": "optional capability name", "lane_type": "core|adjacent"}}
  ],
  "recall_queries": [
    {{"query": "string", "query_intent": "market_scan|alternatives|adjacent", "brick_name": "optional capability name", "lane_type": "core|adjacent"}}
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
- Include countries: {", ".join(include_countries) if include_countries else "auto"}
- Search lanes: {json.dumps(normalized_lanes[:2], ensure_ascii=False) if normalized_lanes else "none confirmed yet"}
- Legacy vertical hints: {", ".join(verticals) if verticals else "generic software"}
- Legacy capability hints: {", ".join(brick_names[:12]) if brick_names else "n/a"}

Comparator seeds:
{seeds_text}

Constraints:
- 3-5 precision queries, 1-3 recall queries.
- Keep queries short, vendor-oriented, and evidence-friendly.
- Seed URLs should be high-confidence competitor/vendor homepages only.
- Include must_exclude_terms for mega incumbents (Bloomberg, FIS, Fiserv, Broadridge, SS&C, Refinitiv).
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

    precision_queries = _normalize_query_entries(payload.get("precision_queries"), max_items=6)
    recall_queries = _normalize_query_entries(payload.get("recall_queries"), max_items=4)
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


def _build_external_search_queries_from_plan(
    plan: dict[str, Any],
    *,
    preferred_countries: Optional[list[str]] = None,
    preferred_languages: Optional[list[str]] = None,
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

    def _add_entries(entries: list[dict[str, Any]], query_type: str) -> None:
        for idx, entry in enumerate(entries, start=1):
            query_text = str(entry.get("query_text") or "").strip()
            if hint_suffix:
                query_text = f"{query_text} {hint_suffix}".strip()
            queries.append(
                {
                    "query_id": f"{query_type}_{idx}",
                    "query_text": query_text,
                    "query_type": query_type,
                    "query_intent": entry.get("query_intent"),
                    "brick_name": entry.get("brick_name"),
                    "lane_type": entry.get("lane_type"),
                    "must_include_terms": include_terms,
                    "must_exclude_terms": exclude_terms,
                    "domain_allowlist": allowlist,
                    "domain_blocklist": blocklist,
                }
            )

    _add_entries(precision_entries, "precision")
    _add_entries(recall_entries, "recall")

    summary = {
        "precision_queries": len(precision_entries),
        "recall_queries": len(recall_entries),
        "lane_types": _dedupe_strings(
            [str(entry.get("lane_type")) for entry in precision_entries + recall_entries if str(entry.get("lane_type") or "").strip()]
        ),
        "must_include_terms": include_terms,
        "must_exclude_terms": exclude_terms,
        "domain_allowlist": allowlist,
        "domain_blocklist": blocklist,
    }
    return queries, summary


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


def _candidates_from_retrieval_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in results:
        url = str(row.get("normalized_url") or row.get("url") or "").strip()
        if not url:
            continue
        candidates.append(
            {
                "name": str(row.get("title") or _domain_label_for_url(url)).strip()[:300],
                "website": url,
                "official_website_url": url,
                "discovery_url": url,
                "entity_type": "company",
                "first_party_domains": [],
                "hq_country": "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
                "why_relevant": [
                    {
                        "text": str(row.get("snippet") or "Externally discovered candidate signal.")[:400],
                        "citation_url": url,
                        "dimension": "external_search_seed",
                        "source_kind": "external_search_snippet",
                        "provider": row.get("provider"),
                        "query_id": row.get("query_id"),
                        "lane_type": row.get("lane_type"),
                        "rank": row.get("rank"),
                    }
                ],
                "_origins": [
                    {
                        "origin_type": "external_search_seed",
                        "origin_url": url,
                        "source_name": row.get("provider"),
                        "source_run_id": None,
                        "metadata": {
                            "query_id": row.get("query_id"),
                            "query_type": row.get("query_type"),
                            "lane_type": row.get("lane_type"),
                            "rank": row.get("rank"),
                        },
                    }
                ],
            }
        )
    return candidates


def _build_candidate_synthesis_prompt(
    retrieval_results: list[dict[str, Any]],
    context_pack: str,
    taxonomy_bricks: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    vertical_focus: list[str],
    search_lanes: Optional[list[dict[str, Any]]] = None,
) -> str:
    normalized_lanes = _normalize_search_lane_payloads(search_lanes)
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
                "rank": row.get("rank"),
            }
        )

    return f"""You are an M&A research analyst. Use ONLY the retrieval_context URLs below.
Return ONLY a JSON array of B2B software companies.

Buyer context:
{context_pack[:1800] if context_pack else "No buyer context provided."}

Target filters:
- Region: {region}
- Search lanes: {json.dumps(normalized_lanes[:2], ensure_ascii=False) if normalized_lanes else "none confirmed yet"}
- Legacy vertical hints: {", ".join(verticals) if verticals else "generic software"}
- Legacy capability hints: {", ".join(brick_names[:12]) if brick_names else "n/a"}

retrieval_context:
{json.dumps(retrieval_context, ensure_ascii=False)}

Output schema:
[
  {{
    "name": "Company Name",
    "website": "https://example.com",
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
- Every company must have website and citation_url from retrieval_context.
- Do not invent companies or URLs not present above.
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

    stats = {"dropped_missing_url": 0, "dropped_missing_evidence": 0}
    validated: list[dict[str, Any]] = []

    for candidate in candidates:
        website_raw = str(candidate.get("website") or "").strip()
        website = normalize_url(website_raw) if website_raw else ""
        if not website or website not in allowed_urls:
            stats["dropped_missing_url"] += 1
            continue

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
            provenance = retrieval_by_url.get(website)
            if provenance:
                reasons.append(
                    {
                        "text": str(provenance.get("snippet") or "Externally discovered candidate signal.")[:400],
                        "citation_url": website,
                        "dimension": "external_search_seed",
                        "source_kind": "external_search_snippet",
                        "provider": provenance.get("provider"),
                        "query_id": provenance.get("query_id"),
                        "rank": provenance.get("rank"),
                    }
                )

        if not reasons:
            stats["dropped_missing_evidence"] += 1
            continue

        candidate["website"] = website
        candidate["why_relevant"] = reasons
        candidate.setdefault("_origins", [])
        provenance = retrieval_by_url.get(website)
        if provenance:
            candidate["_origins"].append(
                {
                    "origin_type": "external_search_seed",
                    "origin_url": website,
                    "source_name": provenance.get("provider"),
                    "source_run_id": None,
                    "metadata": {
                        "query_id": provenance.get("query_id"),
                        "query_type": provenance.get("query_type"),
                        "rank": provenance.get("rank"),
                    },
                }
            )
        validated.append(candidate)

    return validated, stats


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
        if not name or not website:
            continue
        validated.append(
            {
                "name": name,
                "website": website,
                "hq_country": str(row.get("hq_country") or "Unknown").strip() or "Unknown",
                "likely_verticals": row.get("likely_verticals") if isinstance(row.get("likely_verticals"), list) else [],
                "employee_estimate": row.get("employee_estimate"),
                "capability_signals": row.get("capability_signals") if isinstance(row.get("capability_signals"), list) else [],
                "qualification": row.get("qualification") if isinstance(row.get("qualification"), dict) else {},
                "why_relevant": row.get("why_relevant") if isinstance(row.get("why_relevant"), list) else [],
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
        for url in (profile.reference_evidence_urls or []):
            register(url)
        for url in (profile.reference_vendor_urls or []):
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
    vendor_id: Optional[int],
    screening_id: int,
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
                "vendor_id": vendor_id,
                "screening_id": screening_id,
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
            Job.state.in_([JobState.queued, JobState.running, JobState.polling]),
        )
        .all()
    )

    for stale_job in superseded_jobs:
        stale_job.state = JobState.failed
        stale_job.error_message = "Superseded by newer company thesis run"
        stale_job.progress_message = "Superseded by newer company thesis run"
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
    from app.services.gemini_workspace import GeminiWorkspaceClient
    
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
            
            if profile.reference_vendor_urls:
                all_urls.extend(profile.reference_vendor_urls[:3])
            
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
            reference_summaries = {}
            all_context_packs = []  # Store full context packs for JSON export

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
                    
                    job.progress = site_start
                    job.progress_message = f"Starting crawl of {domain}..."
                    _append_job_live_event(job, f"[{domain}] Starting crawl")
                    db.commit()
                    
                    context_pack = loop.run_until_complete(crawler.crawl_for_context(url))
                    combined_markdown.append(context_pack.raw_markdown)
                    product_pages_total += context_pack.product_pages_count
                    all_context_packs.append((url, context_pack))  # Store for later
                    
                    # #region agent log
                    import json
                    log_path = "/app/debug.log"
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "K", "location": "workspace_tasks.py:123", "message": "context_pack result", "data": {"url": url, "product_pages_count": context_pack.product_pages_count, "total_pages": len(context_pack.pages), "product_pages_total": product_pages_total}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
                    
                    if url != profile.buyer_company_url:
                        reference_summaries[url] = context_pack.summary
                    
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
            
            # Summarize with Gemini
            job.progress = 0.8
            job.progress_message = "Generating summary..."
            db.commit()
            
            try:
                summary_prompt = (
                    "Summarize the buyer context in <= 250 words.\n"
                    "Focus on ICP, product capabilities, distribution, and constraints.\n"
                    "Return plain text only.\n\n"
                    f"Buyer URL: {profile.buyer_company_url or ''}\n\n"
                    f"Context:\n{raw_markdown[:14000]}"
                )
                summary_response = LLMOrchestrator().run_stage(
                    LLMRequest(
                        stage=LLMStage.context_summary,
                        prompt=summary_prompt,
                        timeout_seconds=60,
                        use_web_search=False,
                        expect_json=False,
                        metadata={"workspace_id": workspace.id, "job_id": job.id},
                    )
                )
                profile.buyer_context_summary = str(summary_response.text or "").strip()[:8000]
            except Exception as e:
                print(f"Error generating orchestrated summary: {e}")
                try:
                    gemini = GeminiWorkspaceClient()
                    summary = gemini.summarize_context_pack(raw_markdown, profile.buyer_company_url or "")
                    profile.buyer_context_summary = summary
                except Exception as legacy_exc:
                    print(f"Error generating summary: {legacy_exc}")
                    profile.buyer_context_summary = raw_markdown[:2000]
            
            # Update profile
            profile.context_pack_markdown = raw_markdown
            profile.context_pack_json = combined_context_pack_json
            profile.context_pack_generated_at = datetime.utcnow()
            profile.product_pages_found = product_pages_total
            profile.reference_summaries = reference_summaries
            
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
        max_vendors = int(monitor_cfg.get("max_vendors") or 80)
        stale_only = bool(monitor_cfg.get("stale_only", False))
        tracked_classes = set(
            monitor_cfg.get("classifications")
            or ["borderline_watchlist", "insufficient_evidence"]
        )

        screenings = (
            db.query(VendorScreening)
            .filter(VendorScreening.workspace_id == workspace.id)
            .order_by(VendorScreening.created_at.desc())
            .all()
        )
        latest_by_vendor: dict[int, VendorScreening] = {}
        for screening in screenings:
            if screening.vendor_id and screening.vendor_id not in latest_by_vendor:
                latest_by_vendor[screening.vendor_id] = screening

        candidate_rows = [
            screening
            for screening in latest_by_vendor.values()
            if str(screening.decision_classification or "insufficient_evidence") in tracked_classes
        ][:max_vendors]

        now = datetime.utcnow()
        triggered: list[dict[str, Any]] = []
        enrich_job_ids: list[int] = []

        total = max(1, len(candidate_rows))
        for index, screening in enumerate(candidate_rows):
            vendor_id = screening.vendor_id
            if not vendor_id:
                continue

            evidence_rows = (
                db.query(WorkspaceEvidence)
                .filter(WorkspaceEvidence.workspace_id == workspace.id, WorkspaceEvidence.vendor_id == vendor_id)
                .order_by(WorkspaceEvidence.captured_at.desc())
                .all()
            )
            claim_rows = (
                db.query(VendorClaim)
                .filter(VendorClaim.workspace_id == workspace.id, VendorClaim.vendor_id == vendor_id)
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
                job.progress_message = f"Checked {index + 1}/{total} watchlist vendors"
                db.commit()
                continue

            enrich_job = Job(
                workspace_id=workspace.id,
                vendor_id=vendor_id,
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
                    "vendor_id": vendor_id,
                    "classification": screening.decision_classification,
                    "trigger_reasons": trigger_reasons,
                    "stale_evidence_count": len(stale_evidence) + len(stale_claims) + len(missing_ttl_evidence),
                    "new_signal_count": new_evidence_since_screen + new_claims_since_screen,
                }
            )

            job.progress = 0.05 + (0.7 * (index + 1) / total)
            job.progress_message = f"Checked {index + 1}/{total} watchlist vendors"
            db.commit()

        for enrich_job_id in enrich_job_ids:
            run_enrich_vendor.delay(enrich_job_id)

        claims_graph_metrics = rebuild_workspace_claims_graph(db, workspace.id)
        job.state = JobState.completed
        job.progress = 1.0
        job.progress_message = "Monitoring delta complete"
        job.result_json = {
            "watchlist_candidates_checked": len(candidate_rows),
            "triggered_vendor_count": len(triggered),
            "triggered_vendor_ids": [entry["vendor_id"] for entry in triggered],
            "triggered": triggered[:200],
            "enrichment_job_ids": enrich_job_ids,
            "policy_version": effective_policy.get("version"),
            "claims_graph_refresh": claims_graph_metrics,
        }
        job.finished_at = datetime.utcnow()
        db.commit()
        return {"success": True, "triggered_vendor_count": len(triggered)}
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


def _run_discovery_universe_monolith(job_id: int):
    """Run discovery to find candidate universe."""

    db = SessionLocal()
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
        taxonomy = db.query(BrickTaxonomy).filter(BrickTaxonomy.workspace_id == job.workspace_id).first()

        if not workspace or not profile or not taxonomy:
            job.state = JobState.failed
            job.error_message = "Missing workspace data"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Missing workspace data"}
        effective_policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)
        discovery_ctx_snapshot = _load_discovery_context(job_id)
        buyer_employee_estimate = _sanitize_employee_estimate(
            discovery_ctx_snapshot.get("buyer_employee_estimate")
            if isinstance(discovery_ctx_snapshot, dict)
            else None
        )
        if buyer_employee_estimate is None:
            buyer_employee_estimate = _resolve_buyer_employee_estimate(workspace, profile)

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

            job.progress = 0.2
            job.progress_message = "Ingesting comparator directories..."
            db.commit()

            seed_stage_started = _stage_started()
            mention_records: list[dict[str, Any]] = []
            comparator_errors: list[str] = []
            source_names = [
                source_name
                for source_name in ["wealth_mosaic", "partner_graph_seed", "conference_exhibitors_seed"]
                if source_name in SOURCE_REGISTRY
            ]
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
                    record = VendorMention(
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
            _stage_finished("stage_seed_ingest", seed_stage_started, settings.stage_seed_ingest_timeout_seconds)
            _save_stage_checkpoint(
                job_id,
                "stage_seed_ingest",
                {
                    "mentions_count": len(mention_records),
                    "source_coverage_keys": list(source_coverage.keys())[:20],
                    "comparator_errors_count": len(comparator_errors),
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

            search_lanes = (
                db.query(SearchLane)
                .filter(SearchLane.workspace_id == workspace.id)
                .order_by(SearchLane.lane_type.asc(), SearchLane.id.asc())
                .all()
            )
            normalized_search_lanes = _normalize_search_lane_payloads(search_lanes)

            fallback_plan = _default_discovery_query_plan(
                taxonomy_bricks=taxonomy.bricks or [],
                geo_scope=profile.geo_scope or {},
                vertical_focus=taxonomy.vertical_focus or [],
                search_lanes=normalized_search_lanes,
            )
            query_plan = fallback_plan
            try:
                plan_prompt = _build_discovery_query_plan_prompt(
                    context_pack=profile.context_pack_markdown or "",
                    taxonomy_bricks=taxonomy.bricks or [],
                    geo_scope=profile.geo_scope or {},
                    vertical_focus=taxonomy.vertical_focus or [],
                    comparator_mentions=mention_records[:120],
                    search_lanes=normalized_search_lanes,
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

            preferred_countries = _normalize_string_list((profile.geo_scope or {}).get("include_countries"), max_items=8, max_len=8)
            preferred_languages = _normalize_string_list((profile.geo_scope or {}).get("languages"), max_items=6, max_len=8)
            search_queries, query_plan_summary = _build_external_search_queries_from_plan(
                query_plan,
                preferred_countries=preferred_countries,
                preferred_languages=preferred_languages,
            )
            query_brick_map = {
                str(entry.get("query_id")): str(entry.get("brick_name") or "").strip()
                for entry in search_queries
                if str(entry.get("brick_name") or "").strip()
            }
            query_lane_map = {
                str(entry.get("query_id")): str(entry.get("lane_type") or "").strip().lower()
                for entry in search_queries
                if str(entry.get("lane_type") or "").strip()
            }

            seed_urls = _normalize_string_list(query_plan.get("seed_urls"), max_items=8, max_len=220)
            for url in (profile.reference_vendor_urls or [])[:6]:
                seed_urls.append(str(url))
            seed_urls = _dedupe_strings([normalize_url(url) for url in seed_urls if normalize_url(url)])
            high_confidence_seed_urls: list[str] = []
            for seed_url in seed_urls:
                seed_domain = normalize_domain(seed_url)
                if not seed_domain or _is_non_first_party_profile_domain(seed_domain):
                    continue
                if not is_trusted_source_url(seed_url):
                    continue
                high_confidence_seed_urls.append(seed_url)
            similar_seed_cap = max(0, int(getattr(settings, "discovery_retrieval_similar_seed_cap", 4)))
            if similar_seed_cap > 0:
                for idx, seed_url in enumerate(high_confidence_seed_urls[:similar_seed_cap], start=1):
                    search_queries.append(
                        {
                            "query_id": f"seed_similar_{idx}",
                            "query_text": seed_url,
                            "query_type": "seed_similar",
                            "query_intent": "adjacent",
                            "seed_url": seed_url,
                            "lane_type": "adjacent",
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
            available_providers = [p for p in provider_order if provider_keys.get(p)]

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
                    external_search_errors = [
                        str(item) for item in (search_output.get("errors") or []) if str(item).strip()
                    ]
                    external_search_provider_mix = search_output.get("provider_mix") or {}
                    external_search_dedupe_stats = search_output.get("dedupe_stats") or {}
                    external_search_result_counts = {
                        "raw_count": int(search_output.get("dedupe_stats", {}).get("total_in", 0)),
                        "deduped_count": len(retrieval_results),
                        "per_query_counts": search_output.get("query_counts") or {},
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

            if retrieval_results:
                try:
                    synthesis_prompt = _build_candidate_synthesis_prompt(
                        retrieval_results,
                        context_pack=profile.context_pack_markdown or "",
                        taxonomy_bricks=taxonomy.bricks or [],
                        geo_scope=profile.geo_scope or {},
                        vertical_focus=taxonomy.vertical_focus or [],
                        search_lanes=normalized_search_lanes,
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

            if not llm_candidates and retrieval_results:
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

            seeded_candidates = _seed_candidates_from_mentions(mention_records)
            reference_seeded_candidates = _seed_candidates_from_reference_urls(
                profile.reference_vendor_urls or []
            )
            benchmark_seeded_candidates: list[dict[str, Any]] = []
            if _should_add_wealth_benchmark_seeds(profile, taxonomy, mention_records):
                benchmark_seeded_candidates = _seed_candidates_from_benchmark_list()
            first_party_hint_urls_by_domain = _build_first_party_hint_url_map(
                profile=profile,
                include_benchmark_hints=bool(benchmark_seeded_candidates),
            )
            source_coverage["benchmark_seeds"] = {
                "seeded_candidates": len(benchmark_seeded_candidates),
                "enabled": bool(benchmark_seeded_candidates),
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

            raw_candidates: list[dict[str, Any]] = []
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
            identity_seed_stats = _resolve_identities_for_candidates(
                raw_candidates,
                max_fetches=MAX_IDENTITY_FETCHES_PER_RUN,
                timeout_seconds=IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
                concurrency=IDENTITY_RESOLUTION_CONCURRENCY,
            )
            pre_registry_entities, pre_registry_metrics = _collapse_candidates_to_entities(raw_candidates)
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
            identity_registry_stats = _resolve_identities_for_candidates(
                registry_neighbor_candidates,
                max_fetches=MAX_IDENTITY_FETCHES_PER_RUN,
                timeout_seconds=IDENTITY_RESOLUTION_TIMEOUT_SECONDS,
                concurrency=IDENTITY_RESOLUTION_CONCURRENCY,
            )
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
                },
            )

            all_candidates = raw_candidates + registry_neighbor_candidates
            canonical_entities, canonical_metrics = _collapse_candidates_to_entities(all_candidates)
            pre_score_universe_cap = max(50, int(getattr(settings, "discovery_pre_score_universe_cap", PRE_SCORE_UNIVERSE_CAP)))
            canonical_entities, trimmed_out_count = _trim_entities_for_universe(
                canonical_entities,
                cap=pre_score_universe_cap,
            )
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
            previous_entities = db.query(CandidateEntity).filter(CandidateEntity.workspace_id == workspace.id).all()
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
            origin_mix_distribution: dict[str, int] = {}
            alias_clusters_count = 0
            for entity in canonical_entities:
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
                        "merge_rationale": entity.get("merge_reasons") or [],
                        "origin_types": entity.get("origin_types") or [],
                        "merged_candidates_count": int(entity.get("merged_candidates_count") or 1),
                        "merge_confidence": float(entity.get("merge_confidence") or 0.0),
                        "registry_identity": entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {},
                        "industry_signature": entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
                        "entity_type": str(entity.get("entity_type") or "company"),
                        "first_party_domains": entity.get("first_party_domains") if isinstance(entity.get("first_party_domains"), list) else [],
                        "solutions": entity.get("solutions") if isinstance(entity.get("solutions"), list) else [],
                        "discovery_primary_url": entity.get("discovery_primary_url"),
                        "suspected_duplicates": canonical_metrics.get("suspected_duplicates", []),
                    },
                )
                db.add(row)
                db.flush()
                entity_id_map[str(entity.get("temp_entity_id"))] = row.id
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

            # Existing vendors index
            existing_vendors = db.query(Vendor).filter(Vendor.workspace_id == workspace.id).all()
            existing_by_domain: dict[str, Vendor] = {}
            existing_by_name: dict[str, Vendor] = {}
            for vendor in existing_vendors:
                domain = normalize_domain(vendor.website)
                if domain:
                    existing_by_domain[domain] = vendor
                existing_by_name[vendor.name.strip().lower()] = vendor

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

            created_vendors = 0
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
            run_screenings_for_quality_audit: list[VendorScreening] = []
            run_claims_by_screening_id: dict[int, list[dict[str, Any]]] = {}
            solution_entity_screening_count = 0
            first_party_reason_cache: dict[str, list[dict[str, str]]] = {}
            first_party_capability_cache: dict[str, list[str]] = {}
            first_party_meta_cache: dict[str, dict[str, Any]] = {}
            first_party_error_cache: dict[str, str] = {}
            adaptive_hint_cache_by_domain: dict[str, list[str]] = {}
            adaptive_hint_domain_stats: dict[str, dict[str, int]] = {}
            adaptive_hint_domain_budget = max(
                0,
                int(getattr(settings, "first_party_adaptive_hint_domain_budget", 25)),
            )
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
            scoring_entities = ranked_scoring_entities[:max_scoring_entities]
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
                if is_solution_entity:
                    solution_entity_screening_count += 1
                candidate_domain = normalize_domain(candidate.get("website"))
                candidate_first_party_domains = _normalize_domain_list(candidate.get("first_party_domains") or [])
                origin_types = set(entity.get("origin_types") or [])
                priority_score = _candidate_priority_score(entity)
                if not is_solution_entity and candidate_domain and not _is_non_first_party_profile_domain(candidate_domain):
                    existing_vendor = existing_by_domain.get(candidate_domain)
                elif not is_solution_entity:
                    existing_vendor = existing_by_name.get(candidate_name.lower())
                else:
                    existing_vendor = None

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
                candidate_website = str(candidate.get("website") or "").strip()
                candidate_domain_for_fetch = normalize_domain(candidate_website)
                candidate_hint_domains = _normalize_domain_list(
                    ([candidate_domain_for_fetch] if candidate_domain_for_fetch else [])
                    + (candidate_first_party_domains or [])
                )
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
                    candidate_website
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

                    if cached_meta:
                        first_party_enrichment_meta = cached_meta
                    if cached_first_party:
                        normalized_reasons = _normalize_reasons(normalized_reasons + cached_first_party)
                    if cached_capabilities:
                        capability_signals = _dedupe_strings(capability_signals + cached_capabilities)

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
                candidate_lane_types = _dedupe_strings(
                    [
                        str((origin.get("metadata") or {}).get("lane_type") or "").strip().lower()
                        for origin in (entity.get("origins") or [])
                        if isinstance(origin, dict)
                        and isinstance(origin.get("metadata"), dict)
                        and str((origin.get("metadata") or {}).get("lane_type") or "").strip()
                    ]
                    + [
                        str(query_lane_map.get(str((origin.get("metadata") or {}).get("query_id") or "")) or "").strip().lower()
                        for origin in (entity.get("origins") or [])
                        if isinstance(origin, dict)
                        and isinstance(origin.get("metadata"), dict)
                        and str(query_lane_map.get(str((origin.get("metadata") or {}).get("query_id") or "")) or "").strip()
                    ]
                    + [
                        str(reason.get("lane_type") or "").strip().lower()
                        for reason in normalized_reasons
                        if isinstance(reason, dict)
                        and str(reason.get("lane_type") or "").strip()
                    ]
                )

                vendor_ref: Optional[Vendor] = existing_vendor

                if screening_status in {"kept", "review"} and not is_solution_entity:
                    target_vendor_status = VendorStatus.kept if screening_status == "kept" else VendorStatus.candidate
                    if vendor_ref is None:
                        vendor_ref = Vendor(
                            workspace_id=workspace.id,
                            name=candidate_name,
                            website=candidate.get("website"),
                            hq_country=candidate.get("hq_country", "Unknown"),
                            tags_vertical=likely_verticals,
                            tags_custom=tags_custom,
                            status=target_vendor_status,
                            why_relevant=trusted_reason_items[:10],
                            is_manual=False,
                        )
                        db.add(vendor_ref)
                        db.flush()
                        created_vendors += 1
                        if candidate_domain and not _is_non_first_party_profile_domain(candidate_domain):
                            existing_by_domain[candidate_domain] = vendor_ref
                        existing_by_name[candidate_name.lower()] = vendor_ref
                    else:
                        merged_reasons = _normalize_reasons((vendor_ref.why_relevant or []) + trusted_reason_items)
                        vendor_ref.why_relevant = merged_reasons[:12]
                        vendor_ref.tags_vertical = _dedupe_strings((vendor_ref.tags_vertical or []) + likely_verticals)
                        vendor_ref.tags_custom = _dedupe_strings((vendor_ref.tags_custom or []) + tags_custom)
                        if not vendor_ref.hq_country or vendor_ref.hq_country == "Unknown":
                            vendor_ref.hq_country = candidate.get("hq_country", vendor_ref.hq_country)
                        if not vendor_ref.is_manual:
                            vendor_ref.status = target_vendor_status
                        updated_existing += 1
                else:
                    if vendor_ref and not vendor_ref.is_manual and vendor_ref.status == VendorStatus.candidate:
                        vendor_ref.status = VendorStatus.removed
                        updated_existing += 1

                if vendor_ref and trusted_reason_items:
                    trusted_evidence_urls: list[str] = []
                    source_evidence_ids: dict[str, int] = {}
                    for item in trusted_reason_items:
                        citation_url = str(item.get("citation_url") or "").strip()
                        reason_text = str(item.get("text") or "").strip()
                        if not citation_url or not is_trusted_source_url(citation_url):
                            continue
                        exists = (
                            db.query(WorkspaceEvidence)
                            .filter(
                                WorkspaceEvidence.workspace_id == workspace.id,
                                WorkspaceEvidence.vendor_id == vendor_ref.id,
                                WorkspaceEvidence.source_url == citation_url,
                                WorkspaceEvidence.excerpt_text == reason_text,
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
                        evidence_row = WorkspaceEvidence(
                            workspace_id=workspace.id,
                            vendor_id=vendor_ref.id,
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

                    if not trusted_evidence_urls and vendor_ref.website and is_trusted_source_url(vendor_ref.website):
                        fallback_source_type = _source_type_for_url(
                            vendor_ref.website,
                            candidate_domain,
                            first_party_domains=candidate_first_party_domains,
                        )
                        if fallback_source_type == "first_party_website":
                            fallback_source_tier = infer_source_tier(vendor_ref.website, fallback_source_type, candidate_domain)
                            fallback_source_kind = infer_source_kind(vendor_ref.website, fallback_source_type, candidate_domain)
                            ttl_days, valid_through = valid_through_from_claim_group(
                                "identity_scope",
                                policy=effective_policy,
                            )
                            fallback_row = WorkspaceEvidence(
                                workspace_id=workspace.id,
                                vendor_id=vendor_ref.id,
                                source_url=vendor_ref.website,
                                source_title=f"{vendor_ref.name} website",
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
                            source_evidence_ids[vendor_ref.website.lower()] = fallback_row.id
                            trusted_evidence_urls.append(vendor_ref.website)

                    if screening_status in {"kept", "review"}:
                        existing_dossier = (
                            db.query(VendorDossier)
                            .filter(VendorDossier.vendor_id == vendor_ref.id)
                            .order_by(VendorDossier.version.desc())
                            .first()
                        )
                        if not existing_dossier:
                            bootstrap_dossier = {
                                "modules": _map_capabilities_to_modules(
                                    capabilities=capability_signals,
                                    taxonomy_bricks=taxonomy.bricks or [],
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
                                VendorDossier(
                                    vendor_id=vendor_ref.id,
                                    dossier_json=bootstrap_dossier,
                                    version=1,
                                )
                            )

                screening = VendorScreening(
                    workspace_id=workspace.id,
                    vendor_id=vendor_ref.id if vendor_ref else None,
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
                        "lane_types": candidate_lane_types,
                        "query_lane_types": candidate_lane_types,
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
                    vendor_id=vendor_ref.id if vendor_ref else None,
                    screening_id=screening.id,
                    candidate=candidate,
                    trusted_reasons=trusted_reason_items,
                    matched_mentions=matched_mentions,
                    policy=effective_policy,
                    source_evidence_ids=(source_evidence_ids if vendor_ref and trusted_reason_items else None),
                    first_party_domains=candidate_first_party_domains,
                )
                for record in claim_records:
                    db.add(VendorClaim(**record))
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
                "vendors_created": created_vendors,
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
                "pre_score_universe_cap": pre_score_universe_cap,
                "trimmed_out_count": trimmed_out_count,
                "scoring_entities_count": len(scoring_entities),
                "scoring_entities_skipped_count": scoring_entities_skipped_count,
                "first_party_fetch_budget_used": first_party_fetch_budget_default - first_party_fetch_budget,
                "first_party_crawl_budget_used": first_party_crawl_budget_default - first_party_crawl_budget,
                "first_party_hint_crawl_budget_used": first_party_hint_crawl_budget_default - first_party_hint_crawl_budget,
                "first_party_adaptive_hint_domain_budget_used": int(
                    max(0, int(getattr(settings, "first_party_adaptive_hint_domain_budget", 25)) - adaptive_hint_domain_budget)
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
                "claims_graph_refresh": claims_graph_refresh,
            }
            job.finished_at = datetime.utcnow()
            db.commit()

            return {"success": True, "created": created_vendors, "screening_run_id": screening_run_id}

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
        ctx["stage_execution_mode"] = "hybrid_preflight_monolith"
        ctx["stale_runs_failed_count"] = int(stale_runs_failed)
        ctx["superseded_runs_count"] = int(len(superseded_runs))
        _save_discovery_context(job_id, ctx)
        _mark_stage_enqueued(job_id, "stage_seed_ingest")

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
        taxonomy = db.query(BrickTaxonomy).filter(BrickTaxonomy.workspace_id == job.workspace_id).first()
        if not workspace or not profile or not taxonomy:
            raise RuntimeError("Missing workspace/profile/taxonomy")
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
                "taxonomy_ready": bool(taxonomy),
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


@celery_app.task(name="app.workers.workspace_tasks.run_enrich_vendor")
def run_enrich_vendor(job_id: int):
    """Enrich a single vendor with dossier data."""
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
        
        # Get vendor
        vendor = db.query(Vendor).filter(Vendor.id == job.vendor_id).first()
        if not vendor:
            job.state = JobState.failed
            job.error_message = "Vendor not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Vendor not found"}
        
        # Get taxonomy
        taxonomy = db.query(BrickTaxonomy).filter(BrickTaxonomy.workspace_id == job.workspace_id).first()
        taxonomy_bricks = taxonomy.bricks if taxonomy else []
        workspace = db.query(Workspace).filter(Workspace.id == job.workspace_id).first()
        effective_policy = normalize_policy((workspace.decision_policy_json if workspace else None) or DEFAULT_EVIDENCE_POLICY)
        candidate_domain = normalize_domain(vendor.website)
        
        try:
            gemini = GeminiWorkspaceClient()
            
            job.progress = 0.3
            job.progress_message = "Researching..."
            db.commit()
            
            # Run appropriate enrichment based on job type
            if job.job_type == JobType.enrich_full:
                dossier_json = gemini.run_enrich_full(
                    vendor_url=vendor.website or "",
                    vendor_name=vendor.name,
                    taxonomy_bricks=taxonomy_bricks
                )
            elif job.job_type == JobType.enrich_modules:
                dossier_json = gemini.run_enrich_modules(
                    vendor_url=vendor.website or "",
                    taxonomy_bricks=taxonomy_bricks
                )
            elif job.job_type == JobType.enrich_customers:
                dossier_json = gemini.run_enrich_customers(vendor.website or "")
            elif job.job_type == JobType.enrich_hiring:
                dossier_json = gemini.run_enrich_hiring(vendor.website or "")
            else:
                dossier_json = {}
            
            job.progress = 0.8
            job.progress_message = "Saving dossier..."
            db.commit()
            
            # Get latest version
            existing_dossier = db.query(VendorDossier).filter(
                VendorDossier.vendor_id == vendor.id
            ).order_by(VendorDossier.version.desc()).first()
            
            new_version = (existing_dossier.version + 1) if existing_dossier else 1
            
            # If partial enrichment, merge with existing
            if job.job_type != JobType.enrich_full and existing_dossier:
                merged = existing_dossier.dossier_json.copy() if existing_dossier.dossier_json else {}
                merged.update(dossier_json)
                dossier_json = merged
            
            # Create new dossier
            dossier = VendorDossier(
                vendor_id=vendor.id,
                dossier_json=dossier_json,
                version=new_version
            )
            db.add(dossier)
            
            # Update vendor status
            vendor.status = VendorStatus.enriched
            vendor.updated_at = datetime.utcnow()
            
            # Create evidence items from dossier
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
                    evidence = WorkspaceEvidence(
                        workspace_id=job.workspace_id,
                        vendor_id=vendor.id,
                        source_url=url,
                        excerpt_text=module.get("description", ""),
                        content_type="web",
                        brick_ids=module.get("brick_id", ""),
                        source_tier=source_tier,
                        source_kind=source_kind,
                        freshness_ttl_days=ttl_days,
                        valid_through=valid_through,
                        asserted_by="run_enrich_vendor",
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
                    evidence = WorkspaceEvidence(
                        workspace_id=job.workspace_id,
                        vendor_id=vendor.id,
                        source_url=source_url,
                        excerpt_text=f"Customer: {customer.get('name', '')}",
                        content_type="case_study",
                        source_tier=source_tier,
                        source_kind=source_kind,
                        freshness_ttl_days=ttl_days,
                        valid_through=valid_through,
                        asserted_by="run_enrich_vendor",
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
    if profile.reference_vendor_urls:
        urls.extend(profile.reference_vendor_urls)
    if profile.reference_evidence_urls:
        urls.extend(profile.reference_evidence_urls)

    for url in urls:
        domain = normalize_domain(url)
        if not domain:
            continue
        tokens.add(domain)
        first_label = domain.split(".")[0]
        if first_label:
            tokens.add(first_label)
    return tokens


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
        taxonomy = db.query(BrickTaxonomy).filter(BrickTaxonomy.workspace_id == job.workspace_id).first()
        if not workspace or not taxonomy:
            job.state = JobState.failed
            job.error_message = "Missing workspace or taxonomy for report generation"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Missing workspace or taxonomy"}
        effective_policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)

        priority_bricks = set(taxonomy.priority_brick_ids or [])
        if not priority_bricks:
            priority_bricks = {b.get("id") for b in (taxonomy.bricks or []) if b.get("id")}
        adjacency_map = build_adjacency_map(taxonomy.bricks or [])

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
        reference_tokens = _extract_reference_tokens(profile)
        vertical_focus = set(taxonomy.vertical_focus or [])
        include_unknown_size = bool(filters.get("include_unknown_size", False))
        include_outside_sme = bool(filters.get("include_outside_sme", False))
        allowed_countries = include_countries if include_countries else set(DISCOVERY_COUNTRIES)

        vendors = (
            db.query(Vendor)
            .filter(
                Vendor.workspace_id == workspace.id,
                Vendor.status.in_([VendorStatus.kept, VendorStatus.enriched]),
            )
            .order_by(Vendor.created_at.asc())
            .all()
        )
        screenings = (
            db.query(VendorScreening)
            .filter(VendorScreening.workspace_id == workspace.id)
            .order_by(VendorScreening.created_at.desc())
            .all()
        )
        latest_screening_by_vendor: dict[int, VendorScreening] = {}
        for screening in screenings:
            if screening.vendor_id and screening.vendor_id not in latest_screening_by_vendor:
                latest_screening_by_vendor[screening.vendor_id] = screening

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

        total = len(vendors)
        for index, vendor in enumerate(vendors):
            if not _vendor_passes_enterprise_screen(vendor.tags_custom or []):
                continue

            vendor_evidence = (
                db.query(WorkspaceEvidence)
                .filter(WorkspaceEvidence.vendor_id == vendor.id)
                .order_by(WorkspaceEvidence.captured_at.desc())
                .all()
            )

            normalized_country = normalize_country(vendor.hq_country)

            # Promote filing-like discovery claims into structured evidence rows when source is reliable.
            filing_claims_created = False
            for claim in vendor.why_relevant or []:
                if not isinstance(claim, dict):
                    continue
                citation_url = claim.get("citation_url")
                excerpt_text = str(claim.get("text") or "").strip()
                if not citation_url or not excerpt_text:
                    continue
                if not is_reliable_filing_source_url(normalized_country, citation_url):
                    continue
                exists = (
                    db.query(WorkspaceEvidence)
                    .filter(
                        WorkspaceEvidence.vendor_id == vendor.id,
                        WorkspaceEvidence.source_url == citation_url,
                        WorkspaceEvidence.excerpt_text == excerpt_text,
                    )
                    .first()
                )
                if exists:
                    continue
                db.add(
                    WorkspaceEvidence(
                        workspace_id=workspace.id,
                        vendor_id=vendor.id,
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
                vendor_evidence = (
                    db.query(WorkspaceEvidence)
                    .filter(WorkspaceEvidence.vendor_id == vendor.id)
                    .order_by(WorkspaceEvidence.captured_at.desc())
                    .all()
                )

            latest_dossier = (
                db.query(VendorDossier)
                .filter(VendorDossier.vendor_id == vendor.id)
                .order_by(VendorDossier.version.desc())
                .first()
            )
            dossier_json = latest_dossier.dossier_json if latest_dossier else {}

            fallback_capabilities = [
                tag.split(":", 1)[1].strip()
                for tag in (vendor.tags_custom or [])
                if isinstance(tag, str) and tag.startswith("capability:")
            ]
            fallback_evidence_urls = [
                e.source_url
                for e in vendor_evidence
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

            vertical_match = bool(set(vendor.tags_vertical or []).intersection(vertical_focus)) if vertical_focus else True
            geo_vertical_match = geo_match and vertical_match
            has_geo_vertical_evidence = bool(vendor_evidence)

            lens = compute_lens_scores(
                vendor_modules=modules,
                customers=customers,
                integrations=integrations,
                priority_bricks=priority_bricks,
                adjacency_map=adjacency_map,
                reference_tokens=reference_tokens,
                geo_vertical_match=geo_vertical_match,
                has_geo_vertical_evidence=has_geo_vertical_evidence,
            )

            existing_facts = (
                db.query(VendorFact)
                .filter(VendorFact.vendor_id == vendor.id)
                .all()
            )

            filing_facts = extract_filing_facts_from_evidence(normalized_country, vendor_evidence)
            facts_for_size = list(existing_facts)
            facts_for_size.extend(filing_facts)

            size_estimate = estimate_size_from_signals(
                dossier_json=dossier_json,
                facts=facts_for_size,
                evidence_items=vendor_evidence,
                tags_custom=vendor.tags_custom or [],
                why_relevant=vendor.why_relevant or [],
            )
            size_bucket = classify_size_bucket(size_estimate)
            if size_bucket == "outside_sme_range" and not include_outside_sme:
                continue
            if size_bucket == "unknown" and not include_unknown_size:
                continue
            bucket_counts[size_bucket] = bucket_counts.get(size_bucket, 0) + 1

            if filing_facts:
                filing_fact_count += len(filing_facts)
                for fact in filing_facts:
                    exists = (
                        db.query(VendorFact)
                        .filter(
                            VendorFact.vendor_id == vendor.id,
                            VendorFact.fact_key == fact.fact_key,
                            VendorFact.fact_value == fact.fact_value,
                            VendorFact.period == fact.period,
                            VendorFact.source_evidence_id == fact.source_evidence_id,
                        )
                        .first()
                    )
                    if exists:
                        continue
                    db.add(
                        VendorFact(
                            vendor_id=vendor.id,
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
                    "vendor_id": vendor.id,
                    "compete_score": lens["compete_score"],
                    "complement_score": lens["complement_score"],
                    "decision_classification": (
                        latest_screening_by_vendor[vendor.id].decision_classification
                        if vendor.id in latest_screening_by_vendor
                        else "insufficient_evidence"
                    ),
                    "reason_codes_json": (
                        {
                            "positive": latest_screening_by_vendor[vendor.id].positive_reason_codes_json or [],
                            "caution": latest_screening_by_vendor[vendor.id].caution_reason_codes_json or [],
                            "reject": latest_screening_by_vendor[vendor.id].reject_reason_codes_json or [],
                        }
                        if vendor.id in latest_screening_by_vendor
                        else {"positive": [], "caution": [], "reject": []}
                    ),
                    "evidence_summary_json": {
                        "evidence_count": len(vendor_evidence),
                        "freshness_ratio": round(
                            len([row for row in vendor_evidence if is_fresh(row.valid_through)]) / max(1, len(vendor_evidence)),
                            4,
                        ),
                        "tier_counts": {
                            tier: len([row for row in vendor_evidence if str(row.source_tier or "") == tier])
                            for tier in ["tier0_registry", "tier1_vendor", "tier2_partner_customer", "tier3_third_party", "tier4_discovery"]
                        },
                    },
                    "lens_breakdown_json": {
                        **lens,
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
                    },
                }
            )

            if total:
                job.progress = 0.10 + (0.75 * (index + 1) / total)
                job.progress_message = f"Scored {index + 1}/{total} vendors"
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
                        vendor_id=payload["vendor_id"],
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
