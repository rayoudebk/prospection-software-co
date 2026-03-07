"""Thesis-pack bootstrap, normalization, and search-lane derivation helpers."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.models.thesis import BuyerThesisPack, SearchLane
from app.models.workspace import BrickTaxonomy, CompanyProfile
from app.services.reporting import normalize_domain
from app.services.retrieval.url_normalization import normalize_url

THESIS_SECTIONS = {
    "core_capability",
    "adjacent_capability",
    "business_model",
    "customer_profile",
    "deployment_model",
    "size_signal",
    "geography",
    "include_constraint",
    "exclude_constraint",
}

CLAIM_RENDERINGS = {"fact", "hypothesis"}
CLAIM_USER_STATUSES = {"system", "confirmed", "edited", "removed"}
LANE_TYPES = {"core", "adjacent"}
LANE_STATUSES = {"draft", "confirmed"}

DEFAULT_OPEN_QUESTIONS = [
    "Which customer segment is most strategic for target discovery?",
    "What revenue model matters most for this sourcing pass?",
    "What size window should the system optimize for?",
]

CUSTOMER_KEYWORDS = (
    "asset manager",
    "wealth manager",
    "private equity",
    "fund administrator",
    "bank",
    "insurer",
    "advisor",
    "enterprise",
    "mid-market",
    "SMB",
    "fund manager",
    "portfolio manager",
    "operations team",
    "finance team",
    "compliance team",
)

BUSINESS_MODEL_PATTERNS = (
    ("saas", "SaaS / subscription software"),
    ("subscription", "Subscription-based software"),
    ("recurring", "Recurring revenue model"),
    ("license", "License-based software"),
    ("implementation", "Implementation and onboarding services"),
    ("managed service", "Managed services offering"),
    ("professional services", "Professional services revenue"),
    ("consulting", "Consulting-led services"),
    ("multi-year", "Multi-year enterprise contracts"),
    ("annual contract", "Annual contract model"),
)

DEPLOYMENT_PATTERNS = (
    ("cloud", "Cloud-delivered product"),
    ("api", "API / integration-led deployment"),
    ("integrat", "Integration-heavy deployment"),
    ("on-prem", "On-premise deployment option"),
    ("hosted", "Hosted deployment option"),
    ("implementation", "High-touch implementation required"),
)


def _slugify(value: str, *, max_len: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized[:max_len] or "item"


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "||".join([prefix, *[str(part or "").strip().lower() for part in parts]])
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _normalize_string_list(values: Any, *, max_items: int = 20, max_len: int = 180) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    iterable = values if isinstance(values, list) else [values]
    for item in iterable:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text[:max_len])
        if len(results) >= max_items:
            break
    return results


def _clamp_confidence(value: Any, *, default: float = 0.65) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = default
    return max(0.0, min(1.0, round(confidence, 2)))


def _pill_label_for_url(
    url: str,
    *,
    buyer_url: Optional[str],
    reference_vendor_urls: Iterable[str],
    reference_evidence_urls: Iterable[str],
) -> str:
    normalized = normalize_url(url)
    domain = normalize_domain(normalized) or "source"
    if normalized and normalize_domain(normalized) == normalize_domain(buyer_url):
        return "Buyer website"
    if normalized in {normalize_url(item) for item in reference_vendor_urls if normalize_url(item)}:
        return f"Comparator seed: {domain}"
    if normalized in {normalize_url(item) for item in reference_evidence_urls if normalize_url(item)}:
        return f"Evidence source: {domain}"
    return domain


def normalize_source_pills(
    source_pills: Any,
    *,
    buyer_url: Optional[str] = None,
    reference_vendor_urls: Optional[Iterable[str]] = None,
    reference_evidence_urls: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    pills: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    vendor_urls = list(reference_vendor_urls or [])
    evidence_urls = list(reference_evidence_urls or [])

    if not isinstance(source_pills, list):
        source_pills = []

    for item in source_pills:
        if isinstance(item, str):
            payload = {"url": item}
        elif isinstance(item, dict):
            payload = item
        else:
            continue
        url = normalize_url(payload.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        label = str(payload.get("label") or "").strip() or _pill_label_for_url(
            url,
            buyer_url=buyer_url,
            reference_vendor_urls=vendor_urls,
            reference_evidence_urls=evidence_urls,
        )
        pills.append(
            {
                "id": str(payload.get("id") or _stable_id("pill", url)),
                "label": label[:120],
                "url": url,
            }
        )
    return pills[:40]


def normalize_thesis_claims(
    claims: Any,
    *,
    available_pill_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    available_pills = available_pill_ids or set()
    normalized_claims: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    if not isinstance(claims, list):
        claims = []

    for item in claims:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "").strip()
        value = str(item.get("value") or "").strip()
        if not section or not value or section not in THESIS_SECTIONS:
            continue
        pair = (section, value.lower())
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rendering = str(item.get("rendering") or "fact").strip().lower()
        if rendering not in CLAIM_RENDERINGS:
            rendering = "fact"
        user_status = str(item.get("user_status") or "system").strip().lower()
        if user_status not in CLAIM_USER_STATUSES:
            user_status = "system"
        source_pill_ids = [
            str(pill_id)
            for pill_id in _normalize_string_list(item.get("source_pill_ids"), max_items=8, max_len=64)
            if not available_pills or str(pill_id) in available_pills
        ]
        normalized_claims.append(
            {
                "id": str(item.get("id") or _stable_id("claim", section, value)),
                "section": section,
                "value": value[:280],
                "rendering": rendering,
                "confidence": _clamp_confidence(
                    item.get("confidence"),
                    default=0.8 if rendering == "fact" else 0.5,
                ),
                "source_pill_ids": source_pill_ids,
                "user_status": user_status,
            }
        )
    return normalized_claims[:80]


def normalize_open_questions(open_questions: Any) -> list[str]:
    return _normalize_string_list(open_questions, max_items=12, max_len=240)


def _extract_context_text(profile: CompanyProfile) -> str:
    parts: list[str] = []
    reference_summaries = (
        list((profile.reference_summaries or {}).values())
        if isinstance(profile.reference_summaries, dict)
        else []
    )
    for value in [profile.buyer_context_summary, profile.context_pack_markdown, *reference_summaries]:
        text = str(value or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_employee_signal(text: str) -> Optional[str]:
    if not text:
        return None
    ranged = re.search(
        r"(?:between\s+)?(\d{1,4})\s*(?:-|to|and|–)\s*(\d{1,4})\s*(?:employees|employee|staff|people|headcount|team)",
        text,
        flags=re.IGNORECASE,
    )
    if ranged:
        return f"Employee estimate: {ranged.group(1)}-{ranged.group(2)} employees"
    explicit = re.search(
        r"(\d{1,4})\s*(?:employees|employee|staff|people|headcount|team)",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return f"Employee estimate: {explicit.group(1)} employees"
    return None


def _claim_from_value(
    claims: list[dict[str, Any]],
    *,
    section: str,
    value: str,
    rendering: str,
    confidence: float,
    source_pill_ids: Optional[list[str]] = None,
    user_status: str = "system",
) -> None:
    cleaned = str(value or "").strip()
    if not cleaned or section not in THESIS_SECTIONS:
        return
    claims.append(
        {
            "section": section,
            "value": cleaned,
            "rendering": rendering,
            "confidence": confidence,
            "source_pill_ids": source_pill_ids or [],
            "user_status": user_status,
        }
    )


def _site_source_pill_ids_by_url(profile: CompanyProfile, source_pills: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pill in source_pills:
        url = normalize_url(pill.get("url"))
        if url:
            mapping[url] = str(pill.get("id"))
    context_pack = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        site_url = normalize_url(site.get("url"))
        if site_url and site_url not in mapping:
            pill = {
                "id": _stable_id("pill", site_url),
                "label": str(site.get("company_name") or normalize_domain(site_url) or "source")[:120],
                "url": site_url,
            }
            source_pills.append(pill)
            mapping[site_url] = pill["id"]
        for page in site.get("pages") or []:
            if not isinstance(page, dict):
                continue
            page_url = normalize_url(page.get("url"))
            if page_url and page_url not in mapping:
                pill = {
                    "id": _stable_id("pill", page_url),
                    "label": str(page.get("title") or normalize_domain(page_url) or "source")[:120],
                    "url": page_url,
                }
                source_pills.append(pill)
                mapping[page_url] = pill["id"]
    normalized = normalize_source_pills(
        source_pills,
        buyer_url=profile.buyer_company_url,
        reference_vendor_urls=profile.reference_vendor_urls or [],
        reference_evidence_urls=profile.reference_evidence_urls or [],
    )
    source_pills[:] = normalized
    return {normalize_url(pill.get("url")): str(pill.get("id")) for pill in normalized if normalize_url(pill.get("url"))}


def _extract_customer_profile_claims(text: str) -> list[str]:
    lower_text = text.lower()
    claims: list[str] = []
    for keyword in CUSTOMER_KEYWORDS:
        if keyword.lower() in lower_text:
            claims.append(f"Targets {keyword} customers")
    return _normalize_string_list(claims, max_items=6, max_len=120)


def _extract_keyword_claims(text: str, patterns: Iterable[tuple[str, str]]) -> list[str]:
    lower_text = text.lower()
    claims: list[str] = []
    for token, label in patterns:
        if token in lower_text:
            claims.append(label)
    return _normalize_string_list(claims, max_items=6, max_len=160)


def _derive_source_pills_from_profile(profile: CompanyProfile) -> list[dict[str, Any]]:
    source_items: list[dict[str, Any]] = []
    for url in [profile.buyer_company_url, *(profile.reference_vendor_urls or []), *(profile.reference_evidence_urls or [])]:
        if normalize_url(url):
            source_items.append({"url": url})

    context_pack = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        if normalize_url(site.get("url")):
            source_items.append(
                {
                    "url": site.get("url"),
                    "label": str(site.get("company_name") or normalize_domain(site.get("url")) or "source"),
                }
            )
        for signal in site.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            if normalize_url(signal.get("source_url")):
                source_items.append({"url": signal.get("source_url")})
        for page in site.get("pages") or []:
            if not isinstance(page, dict):
                continue
            if normalize_url(page.get("url")):
                source_items.append({"url": page.get("url"), "label": page.get("title")})
            for signal in page.get("signals") or []:
                if not isinstance(signal, dict):
                    continue
                if normalize_url(signal.get("source_url")):
                    source_items.append({"url": signal.get("source_url")})
            for evidence in page.get("customer_evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                if normalize_url(evidence.get("source_url")):
                    source_items.append({"url": evidence.get("source_url")})

    return normalize_source_pills(
        source_items,
        buyer_url=profile.buyer_company_url,
        reference_vendor_urls=profile.reference_vendor_urls or [],
        reference_evidence_urls=profile.reference_evidence_urls or [],
    )


def bootstrap_thesis_payload(
    profile: CompanyProfile,
    taxonomy: Optional[BrickTaxonomy] = None,
) -> dict[str, Any]:
    source_pills = _derive_source_pills_from_profile(profile)
    pill_id_by_url = _site_source_pill_ids_by_url(profile, source_pills)
    claims: list[dict[str, Any]] = []
    context_text = _extract_context_text(profile)
    buyer_url = normalize_url(profile.buyer_company_url)
    buyer_domain = normalize_domain(buyer_url)
    context_pack = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    buyer_site: dict[str, Any] = {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        site_domain = normalize_domain(site.get("url"))
        if buyer_domain and site_domain == buyer_domain:
            buyer_site = site
            break
    if not buyer_site and isinstance(context_pack.get("sites"), list) and context_pack.get("sites"):
        first_site = context_pack.get("sites")[0]
        if isinstance(first_site, dict):
            buyer_site = first_site

    buyer_site_url = normalize_url(buyer_site.get("url")) if isinstance(buyer_site, dict) else buyer_url
    buyer_site_pill = pill_id_by_url.get(buyer_site_url or "") if buyer_site_url else None

    for signal in (buyer_site.get("signals") or [])[:60]:
        if not isinstance(signal, dict):
            continue
        signal_type = str(signal.get("type") or "").strip().lower()
        signal_value = str(signal.get("value") or "").strip()
        signal_url = normalize_url(signal.get("source_url")) or buyer_site_url or buyer_url or ""
        pill_ids = [pill_id_by_url.get(signal_url)] if pill_id_by_url.get(signal_url) else ([buyer_site_pill] if buyer_site_pill else [])
        if signal_type == "capability":
            _claim_from_value(
                claims,
                section="core_capability",
                value=signal_value,
                rendering="fact",
                confidence=0.84,
                source_pill_ids=pill_ids,
            )
        elif signal_type == "service":
            _claim_from_value(
                claims,
                section="adjacent_capability",
                value=signal_value,
                rendering="hypothesis",
                confidence=0.58,
                source_pill_ids=pill_ids,
            )
        elif signal_type == "integration":
            _claim_from_value(
                claims,
                section="deployment_model",
                value=f"Integration signal: {signal_value}",
                rendering="fact",
                confidence=0.72,
                source_pill_ids=pill_ids,
            )

    for customer in (buyer_site.get("customer_evidence") or [])[:8]:
        if not isinstance(customer, dict):
            continue
        customer_name = str(customer.get("name") or "").strip()
        source_url = normalize_url(customer.get("source_url")) or buyer_site_url or ""
        pill_ids = [pill_id_by_url.get(source_url)] if pill_id_by_url.get(source_url) else ([buyer_site_pill] if buyer_site_pill else [])
        if customer_name:
            _claim_from_value(
                claims,
                section="customer_profile",
                value=f"Named customer proof: {customer_name}",
                rendering="fact",
                confidence=0.73,
                source_pill_ids=pill_ids,
            )

    for claim_value in _extract_customer_profile_claims(context_text):
        _claim_from_value(
            claims,
            section="customer_profile",
            value=claim_value,
            rendering="hypothesis",
            confidence=0.55,
            source_pill_ids=[buyer_site_pill] if buyer_site_pill else [],
        )

    for claim_value in _extract_keyword_claims(context_text, BUSINESS_MODEL_PATTERNS):
        _claim_from_value(
            claims,
            section="business_model",
            value=claim_value,
            rendering="hypothesis",
            confidence=0.58,
            source_pill_ids=[buyer_site_pill] if buyer_site_pill else [],
        )

    for claim_value in _extract_keyword_claims(context_text, DEPLOYMENT_PATTERNS):
        _claim_from_value(
            claims,
            section="deployment_model",
            value=claim_value,
            rendering="hypothesis",
            confidence=0.56,
            source_pill_ids=[buyer_site_pill] if buyer_site_pill else [],
        )

    employee_signal = _extract_employee_signal(context_text)
    if employee_signal:
        _claim_from_value(
            claims,
            section="size_signal",
            value=employee_signal,
            rendering="fact",
            confidence=0.62,
            source_pill_ids=[buyer_site_pill] if buyer_site_pill else [],
        )

    geo_scope = profile.geo_scope or {}
    region = str(geo_scope.get("region") or "").strip()
    if region:
        _claim_from_value(
            claims,
            section="geography",
            value=f"Primary sourcing region: {region}",
            rendering="fact",
            confidence=0.9,
            source_pill_ids=[],
        )
    include_countries = _normalize_string_list(geo_scope.get("include_countries"), max_items=8, max_len=32)
    if include_countries:
        _claim_from_value(
            claims,
            section="include_constraint",
            value=f"Prefer companies active in {', '.join(include_countries)}",
            rendering="fact",
            confidence=0.9,
            source_pill_ids=[],
        )
    exclude_countries = _normalize_string_list(geo_scope.get("exclude_countries"), max_items=8, max_len=32)
    if exclude_countries:
        _claim_from_value(
            claims,
            section="exclude_constraint",
            value=f"Exclude companies limited to {', '.join(exclude_countries)}",
            rendering="fact",
            confidence=0.9,
            source_pill_ids=[],
        )

    if taxonomy:
        priority_ids = set(taxonomy.priority_brick_ids or [])
        prioritized = [
            brick for brick in (taxonomy.bricks or [])
            if isinstance(brick, dict) and str(brick.get("id") or "").strip() in priority_ids
        ]
        if not prioritized:
            prioritized = [brick for brick in (taxonomy.bricks or []) if isinstance(brick, dict)][:3]
        for brick in prioritized[:6]:
            name = str(brick.get("name") or "").strip()
            if not name:
                continue
            _claim_from_value(
                claims,
                section="core_capability",
                value=name,
                rendering="hypothesis",
                confidence=0.45,
                source_pill_ids=[],
            )
        for vertical in _normalize_string_list(taxonomy.vertical_focus or [], max_items=4, max_len=60):
            _claim_from_value(
                claims,
                section="include_constraint",
                value=f"Prioritize vendors serving {vertical}",
                rendering="hypothesis",
                confidence=0.48,
                source_pill_ids=[],
            )

    normalized_claims = normalize_thesis_claims(
        claims,
        available_pill_ids={pill["id"] for pill in source_pills},
    )

    active_claims = [claim for claim in normalized_claims if claim["user_status"] != "removed"]
    open_questions: list[str] = []
    if not any(claim["section"] == "business_model" for claim in active_claims):
        open_questions.append("Confirm the dominant revenue model (SaaS, license, services, or mixed).")
    if not any(claim["section"] == "adjacent_capability" for claim in active_claims):
        open_questions.append("Identify one or two adjacent capabilities worth sourcing against.")
    if not any(claim["section"] == "size_signal" for claim in active_claims):
        open_questions.append("Set an employee or company-size window for sourcing.")
    if not active_claims:
        open_questions.extend(DEFAULT_OPEN_QUESTIONS)

    summary = str(
        profile.buyer_context_summary
        or buyer_site.get("summary")
        or context_text[:1200]
        or "System-generated sourcing thesis pending confirmation."
    ).strip()[:8000]

    return {
        "summary": summary,
        "claims": normalized_claims,
        "source_pills": source_pills,
        "open_questions": normalize_open_questions(open_questions),
        "generated_at": datetime.utcnow(),
        "confirmed_at": None,
    }


def derive_search_lane_payloads(
    thesis_pack: BuyerThesisPack | dict[str, Any],
    profile: CompanyProfile,
    taxonomy: Optional[BrickTaxonomy] = None,
) -> list[dict[str, Any]]:
    if isinstance(thesis_pack, BuyerThesisPack):
        claims = thesis_pack.claims_json or []
        confirmed_at = thesis_pack.confirmed_at
    else:
        claims = thesis_pack.get("claims") or []
        confirmed_at = thesis_pack.get("confirmed_at")
    active_claims = [
        claim for claim in normalize_thesis_claims(claims)
        if str(claim.get("user_status") or "system") != "removed"
    ]

    section_values: dict[str, list[str]] = {}
    for claim in active_claims:
        section = str(claim.get("section"))
        section_values.setdefault(section, [])
        value = str(claim.get("value") or "").strip()
        if value and value not in section_values[section]:
            section_values[section].append(value)

    if taxonomy:
        priority_ids = set(taxonomy.priority_brick_ids or [])
        fallback_bricks = [
            str(brick.get("name") or "").strip()
            for brick in (taxonomy.bricks or [])
            if isinstance(brick, dict)
            and (
                (priority_ids and str(brick.get("id") or "").strip() in priority_ids)
                or not priority_ids
            )
            and str(brick.get("name") or "").strip()
        ]
        for value in fallback_bricks[:6]:
            section_values.setdefault("core_capability", [])
            if value not in section_values["core_capability"]:
                section_values["core_capability"].append(value)

    include_terms = section_values.get("include_constraint", [])
    exclude_terms = section_values.get("exclude_constraint", [])
    customer_tags = section_values.get("customer_profile", [])
    reference_seed_urls = _normalize_string_list(profile.reference_vendor_urls or [], max_items=8, max_len=220)

    core_capabilities = section_values.get("core_capability", [])[:8]
    adjacent_capabilities = section_values.get("adjacent_capability", [])[:8]
    business_model = section_values.get("business_model", [])[:2]

    core_lane = {
        "lane_type": "core",
        "title": "Core sourcing lane",
        "intent": (
            "Source direct-fit companies aligned with the confirmed buyer thesis."
            if core_capabilities
            else "Source direct-fit companies once core capabilities are confirmed."
        ),
        "capabilities": core_capabilities,
        "customer_tags": customer_tags[:6],
        "must_include_terms": include_terms[:6] + business_model[:2],
        "must_exclude_terms": exclude_terms[:6],
        "seed_urls": reference_seed_urls[:6],
        "status": "confirmed" if confirmed_at else "draft",
    }
    adjacent_lane = {
        "lane_type": "adjacent",
        "title": "Adjacent sourcing lane",
        "intent": (
            "Source adjacent capability extensions and neighboring product lines."
            if adjacent_capabilities
            else "Capture adjacent sourcing ideas beyond the direct product footprint."
        ),
        "capabilities": adjacent_capabilities,
        "customer_tags": customer_tags[:4],
        "must_include_terms": include_terms[:4],
        "must_exclude_terms": exclude_terms[:6],
        "seed_urls": reference_seed_urls[:3],
        "status": "confirmed" if confirmed_at else "draft",
    }
    return [normalize_search_lane_payload(core_lane), normalize_search_lane_payload(adjacent_lane)]


def normalize_search_lane_payload(payload: dict[str, Any]) -> dict[str, Any]:
    lane_type = str(payload.get("lane_type") or "").strip().lower()
    if lane_type not in LANE_TYPES:
        raise ValueError("Unsupported lane type")
    status = str(payload.get("status") or "draft").strip().lower()
    if status not in LANE_STATUSES:
        status = "draft"
    title = str(payload.get("title") or f"{lane_type.title()} sourcing lane").strip()[:255]
    intent = str(payload.get("intent") or "").strip()[:1000] or None
    return {
        "lane_type": lane_type,
        "title": title,
        "intent": intent,
        "capabilities": _normalize_string_list(payload.get("capabilities"), max_items=12, max_len=140),
        "customer_tags": _normalize_string_list(payload.get("customer_tags"), max_items=10, max_len=140),
        "must_include_terms": _normalize_string_list(payload.get("must_include_terms"), max_items=12, max_len=140),
        "must_exclude_terms": _normalize_string_list(payload.get("must_exclude_terms"), max_items=12, max_len=140),
        "seed_urls": [normalize_url(url) for url in _normalize_string_list(payload.get("seed_urls"), max_items=10, max_len=240) if normalize_url(url)],
        "status": status,
    }


def apply_thesis_adjustment_operations(
    thesis_pack: BuyerThesisPack | dict[str, Any],
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(thesis_pack, BuyerThesisPack):
        payload = {
            "summary": thesis_pack.summary,
            "claims": deepcopy(thesis_pack.claims_json or []),
            "source_pills": deepcopy(thesis_pack.source_pills_json or []),
            "open_questions": deepcopy(thesis_pack.open_questions_json or []),
            "generated_at": thesis_pack.generated_at,
            "confirmed_at": thesis_pack.confirmed_at,
        }
    else:
        payload = {
            "summary": thesis_pack.get("summary"),
            "claims": deepcopy(thesis_pack.get("claims") or []),
            "source_pills": deepcopy(thesis_pack.get("source_pills") or []),
            "open_questions": deepcopy(thesis_pack.get("open_questions") or []),
            "generated_at": thesis_pack.get("generated_at"),
            "confirmed_at": thesis_pack.get("confirmed_at"),
        }

    claims = normalize_thesis_claims(
        payload["claims"],
        available_pill_ids={str(pill.get("id")) for pill in (payload["source_pills"] or []) if isinstance(pill, dict)},
    )
    open_questions = normalize_open_questions(payload["open_questions"])
    applied_ops: list[dict[str, Any]] = []

    def _claim_index() -> dict[str, dict[str, Any]]:
        return {str(claim.get("id")): claim for claim in claims}

    for raw_op in operations or []:
        if not isinstance(raw_op, dict):
            continue
        op = str(raw_op.get("op") or "").strip().lower()
        if not op:
            continue
        claim_id = str(raw_op.get("claim_id") or "").strip()
        claim_lookup = _claim_index()
        if op in {"remove_claim", "confirm_claim", "set_status", "edit_claim", "update_claim"} and claim_id and claim_id not in claim_lookup:
            continue
        if op == "remove_claim":
            claim_lookup[claim_id]["user_status"] = "removed"
            applied_ops.append({"op": op, "claim_id": claim_id})
        elif op in {"confirm_claim", "set_status"}:
            new_status = str(raw_op.get("user_status") or "confirmed").strip().lower()
            if new_status not in CLAIM_USER_STATUSES:
                new_status = "confirmed"
            claim_lookup[claim_id]["user_status"] = new_status
            applied_ops.append({"op": op, "claim_id": claim_id, "user_status": new_status})
        elif op in {"edit_claim", "update_claim"}:
            claim = claim_lookup[claim_id]
            if raw_op.get("value") is not None:
                claim["value"] = str(raw_op.get("value") or "").strip()[:280]
            if raw_op.get("section") in THESIS_SECTIONS:
                claim["section"] = str(raw_op.get("section"))
            if raw_op.get("rendering") in CLAIM_RENDERINGS:
                claim["rendering"] = str(raw_op.get("rendering"))
            if raw_op.get("confidence") is not None:
                claim["confidence"] = _clamp_confidence(raw_op.get("confidence"), default=claim.get("confidence") or 0.65)
            if raw_op.get("source_pill_ids") is not None:
                claim["source_pill_ids"] = _normalize_string_list(raw_op.get("source_pill_ids"), max_items=8, max_len=64)
            claim["user_status"] = "edited"
            applied_ops.append({"op": op, "claim_id": claim_id})
        elif op == "add_claim":
            section = str(raw_op.get("section") or "").strip()
            value = str(raw_op.get("value") or "").strip()
            if section not in THESIS_SECTIONS or not value:
                continue
            claims.append(
                {
                    "id": str(raw_op.get("id") or _stable_id("claim", section, value)),
                    "section": section,
                    "value": value[:280],
                    "rendering": str(raw_op.get("rendering") or "hypothesis"),
                    "confidence": _clamp_confidence(raw_op.get("confidence"), default=0.55),
                    "source_pill_ids": _normalize_string_list(raw_op.get("source_pill_ids"), max_items=8, max_len=64),
                    "user_status": "edited",
                }
            )
            applied_ops.append({"op": op, "section": section, "value": value[:280]})
        elif op == "replace_summary":
            summary = str(raw_op.get("summary") or "").strip()
            if summary:
                payload["summary"] = summary[:8000]
                applied_ops.append({"op": op})
        elif op == "add_open_question":
            question = str(raw_op.get("value") or "").strip()
            if question:
                open_questions.append(question[:240])
                applied_ops.append({"op": op, "value": question[:240]})
        elif op == "remove_open_question":
            question = str(raw_op.get("value") or "").strip().lower()
            if question:
                open_questions = [item for item in open_questions if item.strip().lower() != question]
                applied_ops.append({"op": op, "value": question})

    normalized_claims = normalize_thesis_claims(
        claims,
        available_pill_ids={str(pill.get("id")) for pill in (payload["source_pills"] or []) if isinstance(pill, dict)},
    )
    payload["claims"] = normalized_claims
    payload["open_questions"] = normalize_open_questions(open_questions)
    payload["confirmed_at"] = datetime.utcnow()
    payload["applied_operations"] = applied_ops
    return payload


def infer_adjustment_operations_from_message(
    message: str,
    thesis_pack: BuyerThesisPack | dict[str, Any],
) -> list[dict[str, Any]]:
    text = str(message or "").strip()
    if not text:
        return []
    claims = thesis_pack.claims_json if isinstance(thesis_pack, BuyerThesisPack) else thesis_pack.get("claims") or []
    normalized_claims = normalize_thesis_claims(claims)
    lowered_message = text.lower()
    operations: list[dict[str, Any]] = []

    for claim in normalized_claims:
        claim_value = str(claim.get("value") or "").strip()
        if not claim_value:
            continue
        claim_value_lower = claim_value.lower()
        if claim_value_lower in lowered_message or _slugify(claim_value_lower) in _slugify(lowered_message):
            if any(token in lowered_message for token in ["remove", "drop", "delete", "wrong"]):
                operations.append({"op": "remove_claim", "claim_id": claim["id"]})
            elif any(token in lowered_message for token in ["confirm", "keep", "correct", "right"]):
                operations.append({"op": "confirm_claim", "claim_id": claim["id"], "user_status": "confirmed"})

    line_ops = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    for line in line_ops:
        lower_line = line.lower()
        if lower_line.startswith("add adjacent:"):
            operations.append({"op": "add_claim", "section": "adjacent_capability", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("add core:"):
            operations.append({"op": "add_claim", "section": "core_capability", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("business model:"):
            operations.append({"op": "add_claim", "section": "business_model", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("customer profile:"):
            operations.append({"op": "add_claim", "section": "customer_profile", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("deployment:"):
            operations.append({"op": "add_claim", "section": "deployment_model", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("include:"):
            operations.append({"op": "add_claim", "section": "include_constraint", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("exclude:"):
            operations.append({"op": "add_claim", "section": "exclude_constraint", "value": line.split(":", 1)[1].strip()})
        elif lower_line.startswith("question:"):
            operations.append({"op": "add_open_question", "value": line.split(":", 1)[1].strip()})

    return operations[:24]


def thesis_pack_to_payload(thesis_pack: BuyerThesisPack) -> dict[str, Any]:
    source_pills = normalize_source_pills(thesis_pack.source_pills_json or [])
    return {
        "id": thesis_pack.id,
        "workspace_id": thesis_pack.workspace_id,
        "summary": thesis_pack.summary,
        "claims": normalize_thesis_claims(
            thesis_pack.claims_json or [],
            available_pill_ids={str(pill.get("id")) for pill in source_pills},
        ),
        "source_pills": source_pills,
        "open_questions": normalize_open_questions(thesis_pack.open_questions_json or []),
        "generated_at": thesis_pack.generated_at,
        "confirmed_at": thesis_pack.confirmed_at,
    }


def search_lane_to_payload(search_lane: SearchLane) -> dict[str, Any]:
    return {
        "id": search_lane.id,
        "workspace_id": search_lane.workspace_id,
        "lane_type": search_lane.lane_type,
        "title": search_lane.title,
        "intent": search_lane.intent,
        "capabilities": _normalize_string_list(search_lane.capabilities_json, max_items=12, max_len=140),
        "customer_tags": _normalize_string_list(search_lane.customer_tags_json, max_items=10, max_len=140),
        "must_include_terms": _normalize_string_list(search_lane.must_include_terms_json, max_items=12, max_len=140),
        "must_exclude_terms": _normalize_string_list(search_lane.must_exclude_terms_json, max_items=12, max_len=140),
        "seed_urls": [normalize_url(url) for url in _normalize_string_list(search_lane.seed_urls_json, max_items=10, max_len=240) if normalize_url(url)],
        "status": search_lane.status if search_lane.status in LANE_STATUSES else "draft",
        "confirmed_at": search_lane.confirmed_at,
    }
