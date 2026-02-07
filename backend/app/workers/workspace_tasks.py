"""Celery tasks for workspace-based workflow."""
import asyncio
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
    ComparatorSourceRun,
    VendorMention,
    VendorScreening,
    VendorClaim,
)
from app.services.comparator_sources import SOURCE_REGISTRY, ingest_source
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
KEEP_SCORE_THRESHOLD = 60.0
MIN_TRUSTED_EVIDENCE_FOR_KEEP = 2

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
    }
    return len(reject_reasons) == 0, reject_reasons, meta


def _extract_period(text: str) -> Optional[str]:
    match = re.search(r"(20\d{2})", text)
    if not match:
        return None
    return f"FY{match.group(1)}"


def _source_type_for_url(url: str, candidate_domain: Optional[str]) -> str:
    lowered = url.lower()
    host = normalize_domain(url) or ""
    if "thewealthmosaic.com" in lowered:
        return "directory_comparator"
    if any(pattern in lowered for pattern in ("companieshouse.gov.uk", "pappers.fr", "infogreffe.fr", "inpi.fr")):
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
) -> tuple[float, dict[str, float], list[dict[str, Any]], bool]:
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
    final_score = max(0.0, weighted_score - penalty_total)
    keep = (
        final_score >= KEEP_SCORE_THRESHOLD
        and trusted_count >= MIN_TRUSTED_EVIDENCE_FOR_KEEP
        and not any(
            r in {"go_to_market_b2c", "retail_only_icp", "consumer_language_without_institutional_icp"}
            for r in reject_reasons
        )
    )
    return round(final_score, 2), components, penalties, keep


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
        seeded.append(
            {
                "name": company_name,
                "website": company_url,
                "hq_country": "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
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
                "hq_country": "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
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
            candidates = _merge_candidates_by_domain(
                llm_candidates + seeded_candidates + reference_seeded_candidates
            )

            job.progress = 0.55
            job.progress_message = f"Scoring {len(candidates)} candidates..."
            db.commit()

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
            rejected_count = 0
            untrusted_sources_skipped = 0
            filter_reason_counts: dict[str, int] = {}
            penalties_count: dict[str, int] = {}
            claims_created = 0

            for candidate in candidates:
                candidate_name = str(candidate.get("name") or "").strip()
                if not candidate_name:
                    continue
                candidate_domain = normalize_domain(candidate.get("website"))
                existing_vendor = (
                    existing_by_domain.get(candidate_domain)
                    if candidate_domain
                    else existing_by_name.get(candidate_name.lower())
                )

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

                normalized_reasons = _normalize_reasons(why_relevant)
                passed_gate, reject_reasons, gate_meta = _evaluate_enterprise_b2b_fit(candidate, normalized_reasons)

                trusted_reason_items = [
                    reason
                    for reason in normalized_reasons
                    if is_trusted_source_url(reason.get("citation_url"))
                ]
                untrusted_sources_skipped += max(0, len(normalized_reasons) - len(trusted_reason_items))

                if not trusted_reason_items and candidate.get("website") and is_trusted_source_url(candidate.get("website")):
                    trusted_reason_items = [
                        {
                            "text": "Company first-party website used as baseline evidence.",
                            "citation_url": str(candidate.get("website")),
                            "dimension": "company_profile",
                        }
                    ]
                if not trusted_reason_items:
                    reject_reasons.append("no_trusted_evidence")
                    filter_reason_counts["no_trusted_evidence"] = filter_reason_counts.get("no_trusted_evidence", 0) + 1

                total_score, component_scores, penalties, keep_recommendation = _score_buy_side_candidate(
                    candidate=candidate,
                    reasons=trusted_reason_items,
                    capability_signals=capability_signals,
                    gate_meta=gate_meta,
                    reject_reasons=reject_reasons,
                )
                for penalty in penalties:
                    reason_key = str(penalty.get("reason") or "unknown_penalty")
                    penalties_count[reason_key] = penalties_count.get(reason_key, 0) + 1

                # Keep gate hard-fails as rejected even if score is high.
                if not passed_gate:
                    keep_recommendation = False

                screening_status = "kept" if keep_recommendation else "rejected"
                if screening_status == "rejected" and not reject_reasons:
                    reject_reasons.append("score_below_threshold")
                if screening_status == "kept":
                    kept_count += 1
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

                if screening_status == "kept":
                    if vendor_ref is None:
                        vendor_ref = Vendor(
                            workspace_id=workspace.id,
                            name=candidate_name,
                            website=candidate.get("website"),
                            hq_country=candidate.get("hq_country", "Unknown"),
                            tags_vertical=likely_verticals,
                            tags_custom=tags_custom,
                            status=VendorStatus.kept,
                            why_relevant=trusted_reason_items[:10],
                            is_manual=False,
                        )
                        db.add(vendor_ref)
                        db.flush()
                        created_vendors += 1
                        if candidate_domain:
                            existing_by_domain[candidate_domain] = vendor_ref
                        existing_by_name[candidate_name.lower()] = vendor_ref
                    else:
                        merged_reasons = _normalize_reasons((vendor_ref.why_relevant or []) + trusted_reason_items)
                        vendor_ref.why_relevant = merged_reasons[:12]
                        vendor_ref.tags_vertical = _dedupe_strings((vendor_ref.tags_vertical or []) + likely_verticals)
                        vendor_ref.tags_custom = _dedupe_strings((vendor_ref.tags_custom or []) + tags_custom)
                        if not vendor_ref.hq_country or vendor_ref.hq_country == "Unknown":
                            vendor_ref.hq_country = candidate.get("hq_country", vendor_ref.hq_country)
                        if vendor_ref.status in {VendorStatus.candidate, VendorStatus.removed}:
                            vendor_ref.status = VendorStatus.kept
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

                    if screening_status == "kept":
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
                        "qualification": candidate.get("qualification") or {},
                        "matched_mentions": len(matched_mentions),
                        "candidate_hq_country": candidate.get("hq_country"),
                    },
                    source_summary_json={
                        "trusted_reason_count": len(trusted_reason_items),
                        "source_urls": _dedupe_strings(
                            [str(reason.get("citation_url") or "") for reason in trusted_reason_items if reason.get("citation_url")]
                        )[:12],
                        "mention_listing_urls": _dedupe_strings(
                            [str(m.get("listing_url") or "") for m in matched_mentions if m.get("listing_url")]
                        )[:8],
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

            job.state = JobState.completed
            job.progress = 1.0
            job.progress_message = "Complete"
            job.result_json = {
                "candidates_found": len(candidates),
                "llm_candidates_found": len(llm_candidates),
                "seed_mentions_count": len(mention_records),
                "vendors_created": created_vendors,
                "vendors_updated": updated_existing,
                "kept_count": kept_count,
                "rejected_count": rejected_count,
                "untrusted_sources_skipped": untrusted_sources_skipped,
                "claims_created": claims_created,
                "screening_run_id": screening_run_id,
                "filter_reason_counts": filter_reason_counts,
                "penalty_reason_counts": penalties_count,
                "source_coverage": source_coverage,
                "comparator_errors": comparator_errors,
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
