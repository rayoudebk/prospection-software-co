"""Celery tasks for workspace-based workflow."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re
from difflib import SequenceMatcher
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus
import unicodedata
import httpx
from selectolax.parser import HTMLParser
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.workers.celery_app import celery_app
from app.config import get_settings
from app.models.workspace import Workspace, CompanyProfile, BrickTaxonomy
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

MIN_PUBLIC_PRICE_USD = 250.0
MIN_SOFTWARE_HEAVINESS = 3
KEEP_SCORE_THRESHOLD = 45.0
REVIEW_SCORE_THRESHOLD = 30.0
MIN_TRUSTED_EVIDENCE_FOR_KEEP = 3
MIN_TRUSTED_EVIDENCE_FOR_REVIEW = 2
MAX_PENALTY_POINTS = 35.0
MAX_IDENTITY_FETCHES_PER_RUN = 600
IDENTITY_RESOLUTION_TIMEOUT_SECONDS = 4
IDENTITY_RESOLUTION_CONCURRENCY = 20
FIRST_PARTY_FETCH_BUDGET = 40
FIRST_PARTY_FETCH_TIMEOUT_SECONDS = 4
FIRST_PARTY_MAX_SIGNALS = 6
FIRST_PARTY_CRAWL_BUDGET = 35
FIRST_PARTY_CRAWL_DEEP_BUDGET = 15
FIRST_PARTY_CRAWL_LIGHT_MAX_PAGES = 5
FIRST_PARTY_CRAWL_DEEP_MAX_PAGES = 10
FIRST_PARTY_CRAWL_MAX_REASONS = 16
FIRST_PARTY_CRAWL_DEEP_PRIORITY_THRESHOLD = 95.0
PRE_SCORE_UNIVERSE_CAP = 500

REGISTRY_MAX_QUERIES = 300
REGISTRY_MAX_ACCEPTED_NEIGHBORS = 250
REGISTRY_MAX_RAW_HITS_PER_QUERY = 10
REGISTRY_MAX_NEIGHBORS_PER_ENTITY = 3
REGISTRY_IDENTITY_TOP_SEEDS = 120
REGISTRY_IDENTITY_MIN_SCORE = 0.68
REGISTRY_IDENTITY_BRAND_MATCH_MIN_SCORE = 0.62
REGISTRY_NEIGHBOR_MIN_SCORE = 28.0
REGISTRY_EXPANSION_COUNTRIES = {"FR", "UK", "DE"}
REGISTRY_MAX_DE_QUERIES = 8
REGISTRY_DE_ERROR_BREAKER = 3
REGISTRY_IDENTITY_MAX_SECONDS = 180
REGISTRY_NEIGHBOR_MAX_SECONDS = 240

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

# Sync engine for Celery workers
settings = get_settings()
sync_engine = create_engine(settings.database_url_sync, echo=settings.debug)
SessionLocal = sessionmaker(bind=sync_engine)


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
        if input_domain and _is_aggregator_domain(input_domain) and allow_network_resolution:
            cached = resolve_external_website_from_profile(website)
        elif input_domain and _is_aggregator_domain(input_domain):
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
    resolved_domain = normalize_domain(resolved) if resolved else input_domain

    candidate["original_website"] = website
    candidate["website"] = resolved or website
    candidate["identity"] = {
        "input_website": website,
        "official_website": resolved or website,
        "input_domain": input_domain,
        "canonical_domain": resolved_domain,
        "identity_confidence": cached.get("identity_confidence", "low"),
        "identity_sources": [cached.get("profile_url")] if cached.get("profile_url") else [],
        "captured_at": cached.get("captured_at"),
        "error": cached.get("error"),
    }
    if not candidate.get("hq_country"):
        inferred_country = _infer_country_from_domain(resolved_domain or input_domain)
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
    if _is_aggregator_domain(domain_a) or _is_aggregator_domain(domain_b):
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


def _apply_identity_payload(candidate: dict[str, Any], payload: dict[str, Any]) -> None:
    website = str(candidate.get("website") or "").strip()
    normalized_input = website if website.startswith(("http://", "https://")) else (f"https://{website}" if website else "")
    input_domain = normalize_domain(normalized_input)

    resolved = str(payload.get("official_website") or "").strip()
    if resolved and not resolved.startswith(("http://", "https://")):
        resolved = f"https://{resolved}"
    resolved_domain = normalize_domain(resolved) if resolved else input_domain

    candidate["original_website"] = normalized_input or website
    candidate["website"] = resolved or normalized_input or website
    candidate["identity"] = {
        "input_website": normalized_input or website or None,
        "official_website": resolved or normalized_input or website or None,
        "input_domain": input_domain,
        "canonical_domain": resolved_domain,
        "identity_confidence": payload.get("identity_confidence", "low"),
        "identity_sources": [payload.get("profile_url")] if payload.get("profile_url") else [],
        "captured_at": payload.get("captured_at"),
        "error": payload.get("error"),
    }
    if not candidate.get("hq_country"):
        inferred_country = _infer_country_from_domain(resolved_domain or input_domain)
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
    aggregator_seen: set[str] = set()
    for candidate in candidates:
        website = str(candidate.get("website") or "").strip()
        if not website:
            payload = {
                "profile_url": None,
                "official_website": None,
                "identity_confidence": "low",
                "captured_at": datetime.utcnow().isoformat(),
                "error": "missing_website",
            }
            _apply_identity_payload(candidate, payload)
            continue
        normalized = website if website.startswith(("http://", "https://")) else f"https://{website}"
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
        if not _is_aggregator_domain(domain):
            payload = {
                "profile_url": normalized,
                "official_website": normalized,
                "identity_confidence": "high",
                "captured_at": datetime.utcnow().isoformat(),
                "error": None,
            }
            _apply_identity_payload(candidate, payload)
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

    for candidate in candidates:
        website = str(candidate.get("website") or "").strip()
        if not website:
            continue
        normalized = website if website.startswith(("http://", "https://")) else f"https://{website}"
        domain = normalize_domain(normalized)
        if not domain or not _is_aggregator_domain(domain):
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
        "identity_fetch_count": len(fetch_urls),
        "identity_fetch_errors": fetch_errors,
        "identity_budget_exhausted": max(0, len(aggregator_urls) - len(fetch_urls)),
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
    if domain and not _is_aggregator_domain(domain):
        if not current_domain or _is_aggregator_domain(current_domain):
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
    suspected_duplicates: list[dict[str, Any]] = []

    for candidate in candidates:
        name = str(candidate.get("name") or "").strip()
        if not name:
            continue
        website = str(candidate.get("website") or "").strip() or None
        domain = normalize_domain(website)
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
            "registry_identity": candidate.get("registry_identity") if isinstance(candidate.get("registry_identity"), dict) else {},
            "industry_signature": candidate.get("industry_signature") if isinstance(candidate.get("industry_signature"), dict) else {},
        }
        entities.append(new_entity)
        entity_idx = len(entities) - 1
        if domain and not _is_aggregator_domain(domain) and not _is_registry_profile_domain(domain):
            domain_to_entity[domain] = entity_idx
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
        candidates.extend(["FR", "UK", "DE"])
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
    ranked_directory = [
        entity for entity in entities
        if "directory_seed" in set(entity.get("origin_types") or [])
    ]
    ranked_directory.sort(key=_candidate_priority_score, reverse=True)
    return ranked_directory[:top_n]


def _run_registry_search(country: str, query: str) -> tuple[list[dict[str, Any]], Optional[str], str]:
    if country == "FR":
        records, error = _search_fr_registry_neighbors(query)
        return records, error, "fr_recherche_entreprises"
    if country == "DE":
        records, error = _search_de_registry_neighbors(query)
        source_name = str(records[0].get("registry_source") or "de_registry") if records else "de_registry"
        return records, error, source_name
    records, error = _search_uk_registry_neighbors(query)
    return records, error, "uk_companies_house"


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
    deadline = time.monotonic() + max(30, REGISTRY_IDENTITY_MAX_SECONDS)
    timed_out = False

    identity_seed_entities = _select_registry_identity_seed_entities(entities, top_n=REGISTRY_IDENTITY_TOP_SEEDS)
    candidates_count = len(identity_seed_entities)
    query_budget = max(1, REGISTRY_MAX_QUERIES)

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
            if country == "DE" and de_query_count >= REGISTRY_MAX_DE_QUERIES:
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
                if country == "DE" and de_query_count >= REGISTRY_MAX_DE_QUERIES:
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
    deadline = time.monotonic() + max(30, REGISTRY_NEIGHBOR_MAX_SECONDS)
    timed_out = False

    ranked_entities = sorted(entities, key=_candidate_priority_score, reverse=True)

    for seed_entity in ranked_entities:
        if time.monotonic() > deadline:
            timed_out = True
            errors.append("registry_neighbor_timeout")
            break
        if len(accepted) >= max_neighbors or query_count >= max_queries:
            break

        seed_identity = seed_entity.get("registry_identity") if isinstance(seed_entity.get("registry_identity"), dict) else {}
        seed_country = normalize_country(seed_identity.get("country") or seed_entity.get("registry_country") or seed_entity.get("country"))
        if seed_country not in REGISTRY_EXPANSION_COUNTRIES:
            continue
        if seed_country == "DE" and de_disabled:
            continue

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
            if len(accepted) >= max_neighbors or query_count >= max_queries:
                break
            if seed_country == "DE" and de_query_count >= REGISTRY_MAX_DE_QUERIES:
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
                if len(accepted) >= max_neighbors:
                    break
                candidate_name = str(record.get("name") or "").strip()
                registry_id = str(record.get("registry_id") or "").strip()
                website = str(record.get("website") or "").strip()
                key = (
                    seed_country,
                    registry_id or _normalize_name_for_matching(candidate_name),
                    website.lower(),
                )
                if key in seen_neighbor_keys:
                    reject_reason_counts["duplicate_candidate"] = reject_reason_counts.get("duplicate_candidate", 0) + 1
                    query_rejects["duplicate_candidate"] = query_rejects.get("duplicate_candidate", 0) + 1
                    continue

                if not candidate_name or not (registry_id or website):
                    reject_reason_counts["missing_identity"] = reject_reason_counts.get("missing_identity", 0) + 1
                    query_rejects["missing_identity"] = query_rejects.get("missing_identity", 0) + 1
                    continue
                record_is_active_value = record.get("is_active")
                if isinstance(record_is_active_value, bool):
                    record_is_active = record_is_active_value
                else:
                    record_is_active = _looks_active_status(record.get("status"))
                if not record_is_active:
                    reject_reason_counts["inactive_status"] = reject_reason_counts.get("inactive_status", 0) + 1
                    query_rejects["inactive_status"] = query_rejects.get("inactive_status", 0) + 1
                    continue

                score, score_breakdown, signals = _score_registry_neighbor(seed_entity, record)
                if score < REGISTRY_NEIGHBOR_MIN_SCORE:
                    reject_reason_counts["score_below_threshold"] = reject_reason_counts.get("score_below_threshold", 0) + 1
                    query_rejects["score_below_threshold"] = query_rejects.get("score_below_threshold", 0) + 1
                    continue

                seen_neighbor_keys.add(key)
                kept_pre_dedupe += 1
                query_kept += 1
                citation_url = str(record.get("registry_url") or "").strip()
                industry_signature = _normalize_industry_signature(
                    record.get("industry_codes") or [],
                    record.get("industry_keywords") or [],
                )
                candidates_for_seed.append(
                    {
                        "name": candidate_name,
                        "website": record.get("website") or citation_url,
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
        "registry_reject_reason_breakdown": reject_reason_counts,
        "registry_neighbors_timed_out": timed_out,
        "registry_errors": errors[:100],
    }
    return accepted, diagnostics, query_logs


def _candidate_priority_score(entity: dict[str, Any]) -> float:
    origin_types = set(entity.get("origin_types") or [])
    score = 0.0
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


def _extract_employee_estimate(candidate: dict[str, Any]) -> int | None:
    direct = candidate.get("employee_estimate") or candidate.get("team_size_estimate")
    if isinstance(direct, int):
        return direct
    if isinstance(direct, float):
        return int(direct)

    possible_texts: list[str] = []
    if isinstance(direct, str):
        possible_texts.append(direct)

    for item in candidate.get("why_relevant", []) or []:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                possible_texts.append(text)

    for text in possible_texts:
        ranged = re.search(r"(\d{1,4})\s*(?:-|to|and|–)\s*(\d{1,4})", text)
        if ranged:
            low = int(ranged.group(1))
            high = int(ranged.group(2))
            if high >= low:
                return int(round((low + high) / 2.0))
        single = re.search(r"(?:employees|staff|team|headcount|effectif)[^\d]{0,12}(\d{1,4})", text, flags=re.IGNORECASE)
        if single:
            return int(single.group(1))
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
        reasons.append(
            {
                "text": text[:800],
                "citation_url": citation_url,
                "dimension": str(item.get("dimension") or "evidence")[:64],
            }
        )
    return reasons


def _extract_price_floor_usd(candidate: dict[str, Any], reasons: list[dict[str, str]]) -> float | None:
    qualification = candidate.get("qualification") if isinstance(candidate.get("qualification"), dict) else {}
    direct = _parse_float(qualification.get("public_price_floor_usd_month"))
    if direct is not None:
        return direct

    texts = [r.get("text", "") for r in reasons]
    for text in texts:
        lower = text.lower()
        if "$" in lower or "usd" in lower:
            m = re.search(r"(?:\$|usd\s*)(\d+(?:\.\d+)?)", lower)
            if m:
                return _parse_float(m.group(1))
        if "€" in lower or "eur" in lower:
            m = re.search(r"(?:€|eur\s*)(\d+(?:\.\d+)?)", lower)
            if m:
                # rough parity, sufficient for low-ticket filtering
                return _parse_float(m.group(1))
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
        deduped.append(
            {
                "text": text[:700],
                "citation_url": citation_url[:1000],
                "dimension": dimension[:64],
            }
        )
    return deduped


def _extract_first_party_signals_from_crawl(
    website: str,
    candidate_name: str,
    max_pages: int,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any], Optional[str]]:
    """Crawl first-party pages and extract richer buy-side signals."""
    normalized = website.strip()
    if not normalized:
        return [], [], {"method": "crawler", "pages_crawled": 0}, "missing_website"
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"

    domain = normalize_domain(normalized)
    if not domain or _is_aggregator_domain(domain) or _is_registry_profile_domain(domain):
        return [], [], {"method": "crawler", "pages_crawled": 0}, "invalid_domain"

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
        context_pack = loop.run_until_complete(crawler.crawl_for_context(normalized))
    except Exception as exc:
        return [], [], {"method": "crawler", "pages_crawled": 0}, f"first_party_crawl_failed:{exc}"
    finally:
        try:
            loop.close()
        except Exception:
            pass
        asyncio.set_event_loop(None)

    if context_pack is None:
        return [], [], {"method": "crawler", "pages_crawled": 0}, "first_party_crawl_empty"

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

    deduped_reasons = _dedupe_reason_items(reasons)[:FIRST_PARTY_CRAWL_MAX_REASONS]
    deduped_capabilities = _dedupe_strings(capability_signals)[:12]

    if not deduped_reasons:
        fallback_text = f"First-party website crawled for {candidate_name}: {normalized}"
        deduped_reasons = [
            {
                "text": fallback_text[:700],
                "citation_url": normalized,
                "dimension": "company_profile",
            }
        ]

    meta = {
        "method": "crawler",
        "pages_crawled": len(context_pack.pages or []),
        "signals_extracted": len(context_pack.signals or []),
        "customer_evidence_count": len(context_pack.customer_evidence or []),
        "page_types": page_types,
        "max_pages_requested": max_pages,
    }
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
    if not domain or _is_aggregator_domain(domain):
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


def _source_type_for_url(url: str, candidate_domain: Optional[str]) -> str:
    lowered = url.lower()
    host = normalize_domain(url) or ""
    if "thewealthmosaic.com" in lowered:
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
        )
    ):
        return "official_registry_filing"
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
    source_type_counts: dict[str, int] = {}
    for reason in reasons:
        source_url = str(reason.get("citation_url") or "")
        source_type = _source_type_for_url(source_url, candidate_domain) if source_url else "unknown"
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
        name_key = str(candidate.get("name", "")).strip().lower()
        if domain and any(domain == agg or domain.endswith(f".{agg}") for agg in aggregator_domains):
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
        if not existing.get("hq_country") and candidate.get("hq_country"):
            existing["hq_country"] = candidate.get("hq_country")
    return list(merged.values())


def _seed_candidates_from_mentions(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeded: list[dict[str, Any]] = []
    for mention in mentions:
        company_name = str(mention.get("company_name") or "").strip()
        if not company_name:
            continue
        company_url = mention.get("company_url")
        snippets = mention.get("listing_text_snippets") or []
        snippet_text = str(snippets[0]) if snippets else "Listed as comparator in domain directory."
        combined_listing_text = " ".join(
            str(item) for item in (mention.get("listing_text_snippets") or []) if str(item).strip()
        )
        inferred_country = (
            _infer_country_from_text(combined_listing_text)
            or _infer_country_from_domain(normalize_domain(company_url))
            or "Unknown"
        )
        seeded.append(
            {
                "name": company_name,
                "website": company_url,
                "hq_country": inferred_country,
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
                "_origins": [
                    {
                        "origin_type": "directory_seed",
                        "origin_url": str(mention.get("listing_url") or company_url or ""),
                        "source_name": str(mention.get("source_name") or "wealth_mosaic"),
                        "source_run_id": mention.get("source_run_id"),
                        "metadata": {
                            "category_tags": mention.get("category_tags") or [],
                        },
                    }
                ],
                "why_relevant": [
                    {
                        "text": snippet_text[:700],
                        "citation_url": str(mention.get("listing_url") or company_url or ""),
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
        name_token = domain.split(".")[0].replace("-", " ").replace("_", " ").strip().title()
        seeded.append(
            {
                "name": name_token or domain,
                "website": website if website.startswith(("http://", "https://")) else f"https://{website}",
                "hq_country": _infer_country_from_domain(domain) or "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "reference_input": True,
                "qualification": {},
                "_origins": [
                    {
                        "origin_type": "reference_seed",
                        "origin_url": website if website.startswith(("http://", "https://")) else f"https://{website}",
                        "source_name": "user_reference",
                        "source_run_id": None,
                        "metadata": {},
                    }
                ],
                "why_relevant": [
                    {
                        "text": "Reference comparator provided as explicit user input.",
                        "citation_url": website if website.startswith(("http://", "https://")) else f"https://{website}",
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


def _build_mention_indexes(mentions: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for mention in mentions:
        domain = normalize_domain(str(mention.get("company_url") or ""))
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
                mention_url = str(mention.get("company_url") or "").strip().lower()
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
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate_domain = normalize_domain(str(candidate.get("website") or ""))

    def add_claim(dimension: str, text: str, source_url: str, confidence: str = "medium", claim_key: Optional[str] = None):
        source_url = str(source_url or "").strip()
        claim_text = str(text or "").strip()
        if not claim_text or not source_url:
            return
        if not is_trusted_source_url(source_url):
            return
        parsed_key, numeric_value, numeric_unit, period = _numeric_from_claim_text(claim_text)
        records.append(
            {
                "workspace_id": workspace_id,
                "vendor_id": vendor_id,
                "screening_id": screening_id,
                "dimension": (dimension or "evidence")[:64],
                "claim_key": (claim_key or parsed_key),
                "claim_text": claim_text[:3000],
                "source_url": source_url[:1000],
                "source_type": _source_type_for_url(source_url, candidate_domain),
                "confidence": confidence,
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
    if website and is_trusted_source_url(website):
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

    return records


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
            
            def update_progress(message: str):
                """Update job progress message."""
                job.progress_message = message
                db.commit()
            
            for i, url in enumerate(all_urls):
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc or url
                    
                    # Create unified crawler with progress callback for this URL
                    # Use default argument to capture domain in closure
                    crawler = UnifiedCrawler(
                        max_pages=30,
                        progress_callback=lambda msg, d=domain: update_progress(f"[{d}] {msg}")
                    )
                    
                    job.progress = 0.2 + (0.5 * i / len(all_urls))
                    job.progress_message = f"Starting crawl of {domain}..."
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
                    
                    job.progress = 0.2 + (0.5 * (i + 1) / len(all_urls))
                    job.progress_message = f"Completed {domain} ({len(context_pack.pages)} pages)"
                    db.commit()
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
                    job.progress_message = f"Error crawling {url}: {str(e)[:100]}"
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
                gemini = GeminiWorkspaceClient()
                summary = gemini.summarize_context_pack(raw_markdown, profile.buyer_company_url or "")
                profile.buyer_context_summary = summary
            except Exception as e:
                print(f"Error generating summary: {e}")
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
            job.result_json = {
                "urls_crawled": len(all_urls),
                "product_pages_found": product_pages_total,
                "markdown_length": len(raw_markdown)
            }
            job.finished_at = datetime.utcnow()
            db.commit()
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "L", "location": "workspace_tasks.py:181", "message": "after db commit", "data": {"product_pages_found": profile.product_pages_found}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            return {"success": True, "product_pages": product_pages_total}
            
        except Exception as e:
            job.state = JobState.failed
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(e)}
            
    finally:
        db.close()


@celery_app.task(name="app.workers.workspace_tasks.run_discovery_universe")
def run_discovery_universe(job_id: int):
    """Run discovery to find candidate universe."""
    from app.services.gemini_workspace import GeminiWorkspaceClient

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

        try:
            screening_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            source_coverage: dict[str, Any] = {}

            job.progress = 0.2
            job.progress_message = "Ingesting comparator directories..."
            db.commit()

            mention_records: list[dict[str, Any]] = []
            comparator_errors: list[str] = []
            if "wealth_mosaic" in SOURCE_REGISTRY:
                source_result = ingest_source("wealth_mosaic")
                run_row = ComparatorSourceRun(
                    workspace_id=workspace.id,
                    source_name=source_result["source_name"],
                    source_url=source_result["source_url"],
                    status="completed" if not source_result.get("errors") else "completed_with_errors",
                    mentions_found=len(source_result.get("mentions", [])),
                    metadata_json={
                        "pages_crawled": source_result.get("pages_crawled", 0),
                        "errors": source_result.get("errors", []),
                    },
                )
                db.add(run_row)
                db.flush()

                for mention in source_result.get("mentions", []):
                    record = VendorMention(
                        workspace_id=workspace.id,
                        source_run_id=run_row.id,
                        source_name=source_result["source_name"],
                        listing_url=str(mention.get("listing_url") or source_result["source_url"]),
                        company_name=str(mention.get("company_name") or "")[:300],
                        company_url=str(mention.get("company_url") or "")[:1000] or None,
                        category_tags=mention.get("category_tags") or [],
                        listing_text_snippets=mention.get("listing_text_snippets") or [],
                        provenance_json=mention.get("provenance") or {},
                    )
                    db.add(record)
                    mention_records.append(
                        {
                            "company_name": record.company_name,
                            "company_url": record.company_url,
                            "listing_url": record.listing_url,
                            "category_tags": record.category_tags or [],
                            "listing_text_snippets": record.listing_text_snippets or [],
                            "source_name": record.source_name,
                            "source_run_id": run_row.id,
                        }
                    )
                comparator_errors.extend(source_result.get("errors", []))
                source_coverage["wealth_mosaic"] = {
                    "mentions": len(source_result.get("mentions", [])),
                    "pages_crawled": source_result.get("pages_crawled", 0),
                    "errors": source_result.get("errors", []),
                }
                db.commit()

            mentions_by_domain, mentions_by_name = _build_mention_indexes(mention_records)

            job.progress = 0.35
            job.progress_message = "Searching and expanding candidate universe..."
            db.commit()

            llm_candidates: list[dict[str, Any]] = []
            llm_error: Optional[str] = None
            try:
                gemini = GeminiWorkspaceClient()
                llm_candidates = gemini.run_discovery_universe(
                    context_pack=profile.context_pack_markdown or "",
                    taxonomy_bricks=taxonomy.bricks or [],
                    geo_scope=profile.geo_scope or {},
                    vertical_focus=taxonomy.vertical_focus or [],
                    comparator_mentions=mention_records[:120],
                )
            except Exception as exc:
                llm_error = str(exc)

            seeded_candidates = _seed_candidates_from_mentions(mention_records)
            reference_seeded_candidates = _seed_candidates_from_reference_urls(
                profile.reference_vendor_urls or []
            )
            benchmark_seeded_candidates: list[dict[str, Any]] = []
            if _should_add_wealth_benchmark_seeds(profile, taxonomy, mention_records):
                benchmark_seeded_candidates = _seed_candidates_from_benchmark_list()
            source_coverage["benchmark_seeds"] = {
                "seeded_candidates": len(benchmark_seeded_candidates),
                "enabled": bool(benchmark_seeded_candidates),
            }
            for candidate in llm_candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate.setdefault(
                    "_origins",
                    [
                        {
                            "origin_type": "llm_seed",
                            "origin_url": str(candidate.get("website") or ""),
                            "source_name": "gemini_discovery",
                            "source_run_id": None,
                            "metadata": {},
                        }
                    ],
                )

            raw_candidates: list[dict[str, Any]] = []
            raw_candidates.extend([c for c in llm_candidates if isinstance(c, dict)])
            raw_candidates.extend(seeded_candidates)
            raw_candidates.extend(reference_seeded_candidates)
            raw_candidates.extend(benchmark_seeded_candidates)
            seed_llm_count = len([c for c in llm_candidates if isinstance(c, dict)])
            seed_directory_count = len(seeded_candidates)
            seed_reference_count = len(reference_seeded_candidates)
            seed_benchmark_count = len(benchmark_seeded_candidates)

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

            all_candidates = raw_candidates + registry_neighbor_candidates
            canonical_entities, canonical_metrics = _collapse_candidates_to_entities(all_candidates)
            canonical_entities, trimmed_out_count = _trim_entities_for_universe(canonical_entities, cap=PRE_SCORE_UNIVERSE_CAP)
            registry_neighbors_unique_post_dedupe = len(
                [
                    entity for entity in canonical_entities
                    if "registry_neighbor" in set(entity.get("origin_types") or [])
                ]
            )

            job.progress = 0.55
            job.progress_message = f"Scoring {len(canonical_entities)} canonical candidates..."
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

            created_vendors = 0
            updated_existing = 0
            kept_count = 0
            review_count = 0
            rejected_count = 0
            untrusted_sources_skipped = 0
            filter_reason_counts: dict[str, int] = {}
            penalties_count: dict[str, int] = {}
            claims_created = 0
            first_party_reason_cache: dict[str, list[dict[str, str]]] = {}
            first_party_capability_cache: dict[str, list[str]] = {}
            first_party_meta_cache: dict[str, dict[str, Any]] = {}
            first_party_error_cache: dict[str, str] = {}
            first_party_fetch_budget = FIRST_PARTY_FETCH_BUDGET
            first_party_crawl_budget = FIRST_PARTY_CRAWL_BUDGET
            first_party_crawl_deep_budget = FIRST_PARTY_CRAWL_DEEP_BUDGET
            first_party_crawl_attempted_count = 0
            first_party_crawl_success_count = 0
            first_party_crawl_failed_count = 0
            first_party_crawl_deep_count = 0
            first_party_crawl_light_count = 0
            first_party_crawl_fallback_count = 0
            first_party_crawl_pages_total = 0
            first_party_crawl_errors: list[str] = []

            scoring_entities = sorted(canonical_entities, key=_candidate_priority_score, reverse=True)

            for entity in scoring_entities:
                candidate = {
                    "name": str(entity.get("canonical_name") or "").strip(),
                    "website": entity.get("canonical_website"),
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
                    "original_website": entity.get("canonical_website"),
                    "identity": {
                        "input_website": entity.get("canonical_website"),
                        "official_website": entity.get("canonical_website"),
                        "canonical_domain": normalize_domain(entity.get("canonical_website")),
                        "identity_confidence": entity.get("identity_confidence"),
                        "identity_sources": [origin.get("origin_url") for origin in (entity.get("origins") or []) if origin.get("origin_url")][:3],
                        "captured_at": datetime.utcnow().isoformat(),
                        "error": entity.get("identity_error"),
                    },
                }
                candidate_name = str(candidate.get("name") or "").strip()
                if not candidate_name:
                    continue
                candidate_domain = normalize_domain(candidate.get("website"))
                if candidate_domain and not _is_aggregator_domain(candidate_domain):
                    existing_vendor = existing_by_domain.get(candidate_domain)
                else:
                    existing_vendor = existing_by_name.get(candidate_name.lower())

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
                if (
                    candidate_website
                    and candidate_domain_for_fetch
                    and not _is_aggregator_domain(candidate_domain_for_fetch)
                    and not _is_registry_profile_domain(candidate_domain_for_fetch)
                    and is_trusted_source_url(candidate_website)
                ):
                    cached_first_party = first_party_reason_cache.get(candidate_domain_for_fetch)
                    cached_capabilities = first_party_capability_cache.get(candidate_domain_for_fetch, [])
                    cached_meta = first_party_meta_cache.get(candidate_domain_for_fetch, {})
                    if cached_first_party is None:
                        if first_party_crawl_budget > 0:
                            first_party_crawl_attempted_count += 1
                            priority_score = _candidate_priority_score(entity)
                            use_deep_crawl = (
                                first_party_crawl_deep_budget > 0
                                and (
                                    bool(candidate.get("reference_input"))
                                    or "reference_seed" in set(entity.get("origin_types") or [])
                                    or "benchmark_seed" in set(entity.get("origin_types") or [])
                                    or priority_score >= FIRST_PARTY_CRAWL_DEEP_PRIORITY_THRESHOLD
                                )
                            )
                            max_pages = FIRST_PARTY_CRAWL_DEEP_MAX_PAGES if use_deep_crawl else FIRST_PARTY_CRAWL_LIGHT_MAX_PAGES
                            if use_deep_crawl:
                                first_party_crawl_deep_budget -= 1
                                first_party_crawl_deep_count += 1
                            else:
                                first_party_crawl_light_count += 1

                            crawled_reasons, crawled_capabilities, crawl_meta, crawl_error = _extract_first_party_signals_from_crawl(
                                candidate_website,
                                candidate_name,
                                max_pages=max_pages,
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
                    has_website_reason = any(
                        str(reason.get("citation_url") or "").strip().lower() == website_url.strip().lower()
                        for reason in trusted_reason_items
                    )
                    if not has_website_reason:
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
                )
                for penalty in penalties:
                    reason_key = str(penalty.get("reason") or "unknown_penalty")
                    penalties_count[reason_key] = penalties_count.get(reason_key, 0) + 1

                screening_status = str(score_meta.get("screening_status") or "rejected")
                if not passed_gate and gate_meta.get("hard_fail"):
                    screening_status = "rejected"
                if screening_status == "rejected" and not reject_reasons:
                    reject_reasons.append("score_below_threshold")
                if screening_status == "kept":
                    kept_count += 1
                elif screening_status == "review":
                    review_count += 1
                else:
                    rejected_count += 1
                    for reason in reject_reasons:
                        filter_reason_counts[reason] = filter_reason_counts.get(reason, 0) + 1

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

                vendor_ref: Optional[Vendor] = existing_vendor

                if screening_status in {"kept", "review"}:
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
                        if candidate_domain and not _is_aggregator_domain(candidate_domain):
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
                            continue
                        db.add(
                            WorkspaceEvidence(
                                workspace_id=workspace.id,
                                vendor_id=vendor_ref.id,
                                source_url=citation_url,
                                source_title=source_label_for_url(citation_url),
                                excerpt_text=reason_text[:1200],
                                content_type="web",
                            )
                        )
                        trusted_evidence_urls.append(citation_url)

                    if not trusted_evidence_urls and vendor_ref.website and is_trusted_source_url(vendor_ref.website):
                        db.add(
                            WorkspaceEvidence(
                                workspace_id=workspace.id,
                                vendor_id=vendor_ref.id,
                                source_url=vendor_ref.website,
                                source_title=f"{vendor_ref.name} website",
                                excerpt_text="First-party website baseline source.",
                                content_type="web",
                            )
                        )
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
                    screening_status=screening_status,
                    total_score=total_score,
                    component_scores_json=component_scores,
                    penalties_json=penalties,
                    reject_reasons_json=_dedupe_strings(reject_reasons),
                    screening_meta_json={
                        "job_id": job.id,
                        "screening_run_id": screening_run_id,
                        "passed_enterprise_gate": passed_gate,
                        "hard_fail_gate": bool(gate_meta.get("hard_fail")),
                        "qualification": candidate.get("qualification") or {},
                        "matched_mentions": len(matched_mentions),
                        "candidate_hq_country": candidate.get("hq_country"),
                        "identity": candidate.get("identity") or {},
                        "input_website": candidate.get("original_website"),
                        "resolved_website": candidate.get("website"),
                        "candidate_entity_id": entity.get("db_entity_id"),
                        "aliases": entity.get("alias_names") or [],
                        "merge_rationale": entity.get("merge_reasons") or [],
                        "origin_types": entity.get("origin_types") or [],
                        "registry_identity": entity.get("registry_identity") if isinstance(entity.get("registry_identity"), dict) else {},
                        "industry_signature": entity.get("industry_signature") if isinstance(entity.get("industry_signature"), dict) else {},
                        "first_party_enrichment": {
                            "method": first_party_enrichment_meta.get("method"),
                            "tier": first_party_enrichment_meta.get("tier"),
                            "pages_crawled": int(first_party_enrichment_meta.get("pages_crawled") or 0),
                            "signals_extracted": int(first_party_enrichment_meta.get("signals_extracted") or 0),
                            "customer_evidence_count": int(first_party_enrichment_meta.get("customer_evidence_count") or 0),
                            "page_types": first_party_enrichment_meta.get("page_types") if isinstance(first_party_enrichment_meta.get("page_types"), dict) else {},
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
                        "first_party_enrichment": {
                            "method": first_party_enrichment_meta.get("method"),
                            "tier": first_party_enrichment_meta.get("tier"),
                            "pages_crawled": int(first_party_enrichment_meta.get("pages_crawled") or 0),
                        },
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
                )
                for record in claim_records:
                    db.add(VendorClaim(**record))
                claims_created += len(claim_records)

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

            job.state = JobState.completed
            job.progress = 1.0
            job.progress_message = "Complete"
            job.result_json = {
                "candidates_found": len(canonical_entities),
                "llm_candidates_found": seed_llm_count,
                "seed_mentions_count": len(mention_records),
                "seed_directory_count": seed_directory_count,
                "seed_reference_count": seed_reference_count,
                "seed_benchmark_count": seed_benchmark_count,
                "seed_llm_count": seed_llm_count,
                "vendors_created": created_vendors,
                "vendors_updated": updated_existing,
                "kept_count": kept_count,
                "review_count": review_count,
                "rejected_count": rejected_count,
                "untrusted_sources_skipped": untrusted_sources_skipped,
                "claims_created": claims_created,
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
                "registry_identity_candidates_count": int(registry_identity_metrics.get("registry_identity_candidates_count", 0)),
                "registry_identity_mapped_count": int(registry_identity_metrics.get("registry_identity_mapped_count", 0)),
                "registry_identity_country_breakdown": registry_identity_metrics.get("registry_identity_country_breakdown", {}),
                "registry_queries_by_country": registry_queries_by_country,
                "registry_raw_hits_by_country": registry_raw_hits_by_country,
                "registry_neighbors_kept_pre_dedupe": int(registry_expansion_metrics.get("registry_neighbors_kept_pre_dedupe", 0)),
                "registry_neighbors_unique_post_dedupe": int(registry_neighbors_unique_post_dedupe),
                "registry_reject_reason_breakdown": registry_expansion_metrics.get("registry_reject_reason_breakdown", {}),
                "final_universe_count": len(canonical_entities),
                "trimmed_out_count": trimmed_out_count,
                "first_party_fetch_budget_used": FIRST_PARTY_FETCH_BUDGET - first_party_fetch_budget,
                "first_party_crawl_budget_used": FIRST_PARTY_CRAWL_BUDGET - first_party_crawl_budget,
                "first_party_crawl_attempted_count": first_party_crawl_attempted_count,
                "first_party_crawl_success_count": first_party_crawl_success_count,
                "first_party_crawl_failed_count": first_party_crawl_failed_count,
                "first_party_crawl_deep_count": first_party_crawl_deep_count,
                "first_party_crawl_light_count": first_party_crawl_light_count,
                "first_party_crawl_fallback_count": first_party_crawl_fallback_count,
                "first_party_crawl_pages_total": first_party_crawl_pages_total,
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
                    evidence = WorkspaceEvidence(
                        workspace_id=job.workspace_id,
                        vendor_id=vendor.id,
                        source_url=url,
                        excerpt_text=module.get("description", ""),
                        content_type="web",
                        brick_ids=module.get("brick_id", "")
                    )
                    db.add(evidence)
            
            for customer in dossier_json.get("customers", []):
                if customer.get("evidence_url") and is_trusted_source_url(customer.get("evidence_url")):
                    evidence = WorkspaceEvidence(
                        workspace_id=job.workspace_id,
                        vendor_id=vendor.id,
                        source_url=customer.get("evidence_url", ""),
                        excerpt_text=f"Customer: {customer.get('name', '')}",
                        content_type="case_study"
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
