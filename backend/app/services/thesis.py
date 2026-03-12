"""Thesis-pack bootstrap, normalization, and search-lane derivation helpers."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.models.thesis import BuyerThesisPack, SearchLane
from app.models.workspace import CompanyProfile
from app.services.company_profile_context import (
    get_generated_context_summary,
    get_manual_brief_text,
)
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

BUYER_EVIDENCE_MIN_SCORE = 3

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
    "healthcare",
    "healthcare provider",
    "hospital",
    "doctor",
    "physician",
    "clinic",
    "pharmacy",
    "medical practice",
)

BUSINESS_MODEL_PATTERNS = (
    ("saas", "SaaS / subscription software"),
    ("subscription", "Subscription-based software"),
    ("recurring", "Recurring revenue model"),
    ("license", "License-based software"),
    ("licence", "License-based software"),
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

NEGATIVE_CONSTRAINT_PATTERNS = (
    ("no saas", "Exclude SaaS-first software companies"),
    ("non-saas", "Exclude SaaS-first software companies"),
    ("not saas", "Exclude SaaS-first software companies"),
    ("no services", "Exclude services-led businesses"),
    ("no consulting", "Exclude consulting-led businesses"),
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
    reference_company_urls: Iterable[str],
    reference_evidence_urls: Iterable[str],
) -> str:
    normalized = normalize_url(url)
    domain = normalize_domain(normalized) or "source"
    if normalized and normalize_domain(normalized) == normalize_domain(buyer_url):
        return "Buyer website"
    if normalized in {normalize_url(item) for item in reference_company_urls if normalize_url(item)}:
        return f"Comparator seed: {domain}"
    if normalized in {normalize_url(item) for item in reference_evidence_urls if normalize_url(item)}:
        return f"Evidence source: {domain}"
    return domain


def normalize_source_pills(
    source_pills: Any,
    *,
    buyer_url: Optional[str] = None,
    reference_company_urls: Optional[Iterable[str]] = None,
    reference_evidence_urls: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    pills: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    company_urls = list(reference_company_urls or [])
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
            reference_company_urls=company_urls,
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
    for value in [
        get_manual_brief_text(profile),
        get_generated_context_summary(profile),
        profile.context_pack_markdown,
        *reference_summaries,
    ]:
        text = str(value or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _site_has_meaningful_content(site: Any) -> bool:
    if not isinstance(site, dict):
        return False
    if str(site.get("summary") or "").strip():
        return True
    if any(isinstance(signal, dict) and str(signal.get("value") or "").strip() for signal in site.get("signals") or []):
        return True
    if any(
        isinstance(customer, dict)
        and (str(customer.get("name") or "").strip() or str(customer.get("context") or "").strip())
        for customer in site.get("customer_evidence") or []
    ):
        return True
    for page in site.get("pages") or []:
        if not isinstance(page, dict):
            continue
        if str(page.get("raw_content") or "").strip():
            return True
        if any(isinstance(block, dict) and str(block.get("content") or "").strip() for block in page.get("blocks") or []):
            return True
        if any(isinstance(signal, dict) and str(signal.get("value") or "").strip() for signal in page.get("signals") or []):
            return True
        if any(
            isinstance(customer, dict)
            and (str(customer.get("name") or "").strip() or str(customer.get("context") or "").strip())
            for customer in page.get("customer_evidence") or []
        ):
            return True
    return False


def _extract_site_context_text(site: Any, *, max_chars: int = 12000) -> str:
    if not isinstance(site, dict):
        return ""

    parts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            parts.append(text)

    add(site.get("company_name"))
    add(site.get("summary"))

    for signal in site.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        add(signal.get("value"))
        add(signal.get("snippet"))

    for customer in site.get("customer_evidence") or []:
        if not isinstance(customer, dict):
            continue
        name = str(customer.get("name") or "").strip()
        context = str(customer.get("context") or "").strip()
        if name and context:
            add(f"{name}: {context}")
        else:
            add(name or context)

    for page in site.get("pages") or []:
        if not isinstance(page, dict):
            continue
        add(page.get("title"))
        add(page.get("raw_content"))
        for block in page.get("blocks") or []:
            if isinstance(block, dict):
                add(block.get("content"))
        for signal in page.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            add(signal.get("value"))
            add(signal.get("snippet"))
        for customer in page.get("customer_evidence") or []:
            if not isinstance(customer, dict):
                continue
            name = str(customer.get("name") or "").strip()
            context = str(customer.get("context") or "").strip()
            if name and context:
                add(f"{name}: {context}")
            else:
                add(name or context)

    return "\n".join(parts)[:max_chars]


def _identity_tokens_from_url(url: Any) -> set[str]:
    normalized = normalize_domain(normalize_url(url))
    if not normalized:
        return set()
    tokens = {normalized.lower()}
    first_label = normalized.split(".")[0].strip().lower()
    if first_label:
        tokens.add(first_label)
    return {token for token in tokens if token}


def _identity_tokens_from_name(value: Any) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not cleaned:
        return set()
    tokens = {part for part in cleaned.split() if len(part) >= 3}
    if cleaned.replace(" ", ""):
        tokens.add(cleaned.replace(" ", ""))
    return tokens


def _generated_summary_for_buyer(profile: CompanyProfile, buyer_site: dict[str, Any]) -> str:
    summary = str(get_generated_context_summary(profile) or "").strip()
    if not summary:
        return ""

    lowered = summary.lower()
    buyer_tokens = _identity_tokens_from_url(profile.buyer_company_url)
    buyer_tokens.update(_identity_tokens_from_name(buyer_site.get("company_name")))
    reference_tokens: set[str] = set()
    for url in profile.reference_company_urls or []:
        reference_tokens.update(_identity_tokens_from_url(url))

    mentions_buyer = any(token in lowered for token in buyer_tokens if token)
    mentions_reference = any(token in lowered for token in reference_tokens if token)
    if mentions_reference and not mentions_buyer:
        return ""
    return summary


def _resolve_buyer_site(profile: CompanyProfile) -> dict[str, Any]:
    buyer_url = normalize_url(profile.buyer_company_url)
    buyer_domain = normalize_domain(buyer_url)
    context_pack = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        site_domain = normalize_domain(site.get("url"))
        if buyer_domain and site_domain == buyer_domain:
            return site
    if isinstance(context_pack.get("sites"), list) and context_pack.get("sites"):
        first_site = context_pack.get("sites")[0]
        if isinstance(first_site, dict):
            return first_site
    return {}


def assess_buyer_evidence(profile: CompanyProfile) -> dict[str, Any]:
    buyer_url = normalize_url(profile.buyer_company_url)
    if not buyer_url:
        return {
            "mode": "thesis_only",
            "status": "not_applicable",
            "score": 0,
            "used_for_inference": True,
            "warning": None,
            "metrics": {
                "pages_crawled": 0,
                "content_pages": 0,
                "signal_count": 0,
                "customer_evidence_count": 0,
                "summary_chars": 0,
            },
        }

    buyer_site = _resolve_buyer_site(profile)
    pages = buyer_site.get("pages") or []
    pages_crawled = len([page for page in pages if isinstance(page, dict)])
    content_pages = 0
    page_signal_count = 0
    page_customer_count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        has_page_content = bool(
            str(page.get("raw_content") or "").strip()
            or any(
                isinstance(block, dict) and str(block.get("content") or "").strip()
                for block in page.get("blocks") or []
            )
            or any(
                isinstance(signal, dict) and str(signal.get("value") or "").strip()
                for signal in page.get("signals") or []
            )
            or any(
                isinstance(customer, dict)
                and (str(customer.get("name") or "").strip() or str(customer.get("context") or "").strip())
                for customer in page.get("customer_evidence") or []
            )
        )
        if has_page_content:
            content_pages += 1
        page_signal_count += len([signal for signal in page.get("signals") or [] if isinstance(signal, dict)])
        page_customer_count += len([customer for customer in page.get("customer_evidence") or [] if isinstance(customer, dict)])

    top_level_signal_count = len([signal for signal in buyer_site.get("signals") or [] if isinstance(signal, dict)])
    top_level_customer_count = len([customer for customer in buyer_site.get("customer_evidence") or [] if isinstance(customer, dict)])
    signal_count = top_level_signal_count + page_signal_count
    customer_evidence_count = top_level_customer_count + page_customer_count
    summary_chars = len(str(buyer_site.get("summary") or "").strip())

    score = 0
    if summary_chars >= 120:
        score += 2
    elif summary_chars >= 40:
        score += 1
    score += min(content_pages, 2)
    score += min(signal_count, 2)
    score += min(customer_evidence_count, 2)

    sufficient = score >= BUYER_EVIDENCE_MIN_SCORE
    return {
        "mode": "buyer_website",
        "status": "sufficient" if sufficient else "insufficient",
        "score": score,
        "used_for_inference": sufficient,
        "warning": (
            None
            if sufficient
            else (
                "Buyer evidence is too weak for reliable inference. Add first-party product pages, PDFs, "
                "case studies, or supporting evidence before trusting customer, capability, and deployment claims."
            )
        ),
        "metrics": {
            "pages_crawled": pages_crawled,
            "content_pages": content_pages,
            "signal_count": signal_count,
            "customer_evidence_count": customer_evidence_count,
            "summary_chars": summary_chars,
        },
    }


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


def _extract_revenue_signal(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(
        r"(?:less than|under|below|<)\s*\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|b|bn|billion|k|thousand)?\s*(?:in\s+)?revenue",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    amount = match.group(1)
    suffix = (match.group(2) or "").strip().lower()
    if suffix in {"m", "mm", "million"}:
        normalized = f"${amount}M"
    elif suffix in {"b", "bn", "billion"}:
        normalized = f"${amount}B"
    elif suffix in {"k", "thousand"}:
        normalized = f"${amount}K"
    else:
        normalized = f"${amount}"
    return f"Prefer companies under {normalized} revenue"


def _sanitize_focus_phrase(value: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", str(value or "")).strip(" ,.;:-")
    cleaned = cleaned.split("(", 1)[0].strip(" ,.;:-")
    cleaned = re.split(
        r"\b(?:as|with|that|which|who|where|when|and\s+they|and\s+that|but)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    words = cleaned.split()
    return " ".join(words[:8]).strip()


def _extract_brief_capability_claims(text: str) -> list[str]:
    if not text:
        return []
    claims: list[str] = []
    patterns = (
        r"(?:provide|provides|providing|sell|sells|selling|build|builds|building|offer|offers|offering)\s+software\s+(?:to|for)\s+([^.,;\n]+)",
        r"software\s+(?:to|for)\s+([^.,;\n]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = _sanitize_focus_phrase(match.group(1))
            if phrase:
                claims.append(f"Software for {phrase}")
    return _normalize_string_list(claims, max_items=3, max_len=140)


def _humanize_customer_profile(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned.lower().startswith("targets "):
        cleaned = cleaned[8:]
    cleaned = re.sub(r"\s+customers?$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^named customer proof:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _fallback_lane_focus(
    customer_tags: list[str],
    include_terms: list[str],
    business_model: list[str],
) -> Optional[str]:
    for tag in customer_tags:
        customer_focus = _humanize_customer_profile(tag)
        if not customer_focus:
            continue
        if "health" in customer_focus.lower() or "hospital" in customer_focus.lower() or "doctor" in customer_focus.lower():
            return "Healthcare provider software"
        return f"Software for {customer_focus}"
    for term in include_terms:
        if term.lower().startswith("prefer companies under "):
            continue
        if term.lower().startswith("prefer companies active in "):
            continue
        return term
    if business_model:
        return business_model[0]
    return None


def _lane_title(lane_type: str, capabilities: list[str], customer_tags: list[str], business_model: list[str]) -> str:
    prefix = "Core" if lane_type == "core" else "Adjacent"
    focus = (capabilities or [])[0] if capabilities else _fallback_lane_focus(customer_tags, [], business_model)
    if focus:
        return f"{prefix}: {focus}"[:255]
    return f"{prefix} sourcing lane"


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
        reference_company_urls=profile.reference_company_urls or [],
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


def _extract_business_model_claims(text: str) -> list[str]:
    lower_text = text.lower()
    claims: list[str] = []
    negative_saas = any(token in lower_text for token in ["no saas", "non-saas", "not saas"])
    negative_subscription = "no subscription" in lower_text or "non-subscription" in lower_text
    for token, label in BUSINESS_MODEL_PATTERNS:
        if token not in lower_text:
            continue
        if token == "saas" and negative_saas:
            continue
        if token == "subscription" and negative_subscription:
            continue
        claims.append(label)
    return _normalize_string_list(claims, max_items=6, max_len=160)


def _derive_source_pills_from_profile(profile: CompanyProfile) -> list[dict[str, Any]]:
    source_items: list[dict[str, Any]] = []
    for url in [profile.buyer_company_url, *(profile.reference_company_urls or []), *(profile.reference_evidence_urls or [])]:
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
        reference_company_urls=profile.reference_company_urls or [],
        reference_evidence_urls=profile.reference_evidence_urls or [],
    )


def bootstrap_thesis_payload(
    profile: CompanyProfile,
) -> dict[str, Any]:
    source_pills = _derive_source_pills_from_profile(profile)
    pill_id_by_url = _site_source_pill_ids_by_url(profile, source_pills)
    claims: list[dict[str, Any]] = []
    buyer_url = normalize_url(profile.buyer_company_url)
    buyer_site = _resolve_buyer_site(profile)
    buyer_evidence = assess_buyer_evidence(profile)

    buyer_site_has_content = _site_has_meaningful_content(buyer_site)
    buyer_site_context_text = _extract_site_context_text(buyer_site) if buyer_site_has_content else ""
    buyer_generated_summary = (
        _generated_summary_for_buyer(profile, buyer_site)
        if buyer_url and isinstance(buyer_site, dict) and buyer_site_has_content
        else ""
    )
    if buyer_url:
        buyer_only_context_text = "\n".join(
            part for part in [buyer_generated_summary, buyer_site_context_text] if str(part or "").strip()
        )
        context_text = buyer_only_context_text if bool(buyer_evidence.get("used_for_inference")) else ""
    else:
        context_text = _extract_context_text(profile)
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

    for claim_value in _extract_brief_capability_claims(context_text):
        _claim_from_value(
            claims,
            section="core_capability",
            value=claim_value,
            rendering="hypothesis",
            confidence=0.52,
            source_pill_ids=[buyer_site_pill] if buyer_site_pill else [],
        )

    for claim_value in _extract_business_model_claims(context_text):
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

    for claim_value in _extract_keyword_claims(context_text, NEGATIVE_CONSTRAINT_PATTERNS):
        _claim_from_value(
            claims,
            section="exclude_constraint",
            value=claim_value,
            rendering="hypothesis",
            confidence=0.72,
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

    revenue_signal = _extract_revenue_signal(context_text)
    if revenue_signal:
        _claim_from_value(
            claims,
            section="include_constraint",
            value=revenue_signal,
            rendering="hypothesis",
            confidence=0.72,
            source_pill_ids=[buyer_site_pill] if buyer_site_pill else [],
        )

    if not profile.geo_scope and re.search(r"\beurope|european\b", context_text, flags=re.IGNORECASE):
        _claim_from_value(
            claims,
            section="geography",
            value="Primary sourcing region: Europe",
            rendering="hypothesis",
            confidence=0.72,
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

    normalized_claims = normalize_thesis_claims(
        claims,
        available_pill_ids={pill["id"] for pill in source_pills},
    )

    active_claims = [claim for claim in normalized_claims if claim["user_status"] != "removed"]
    open_questions: list[str] = []
    if buyer_evidence.get("status") == "insufficient" and buyer_evidence.get("warning"):
        open_questions.append(str(buyer_evidence.get("warning")))
    if not any(claim["section"] == "business_model" for claim in active_claims):
        open_questions.append("Confirm the dominant revenue model (SaaS, license, services, or mixed).")
    if not any(claim["section"] == "core_capability" for claim in active_claims):
        open_questions.append("Define the main workflow, product category, or capability to anchor the core lane.")
    if not any(claim["section"] == "adjacent_capability" for claim in active_claims):
        open_questions.append("Identify one or two adjacent capabilities worth sourcing against.")
    if not any(claim["section"] == "size_signal" for claim in active_claims):
        open_questions.append("Set an employee or company-size window for sourcing.")
    if not active_claims:
        open_questions.extend(DEFAULT_OPEN_QUESTIONS)

    buyer_site_summary = str(buyer_site.get("summary") or "").strip() if isinstance(buyer_site, dict) else ""
    summary = str(
        get_manual_brief_text(profile)
        or buyer_generated_summary
        or (buyer_site_summary if buyer_url else "")
        or (buyer_site_context_text[:1200] if buyer_url and buyer_site_context_text else "")
        or (get_generated_context_summary(profile) if not buyer_url else "")
        or context_text[:1200]
        or (
            "Buyer website crawled, but no first-party product or customer evidence was extracted yet. "
            "Add supporting evidence or regenerate after improving the crawl target."
            if buyer_url
            else "System-generated sourcing thesis pending confirmation."
        )
    ).strip()[:8000]

    return {
        "summary": summary,
        "claims": normalized_claims,
        "source_pills": source_pills,
        "open_questions": normalize_open_questions(open_questions),
        "buyer_evidence": buyer_evidence,
        "generated_at": datetime.utcnow(),
        "confirmed_at": None,
    }


def derive_search_lane_payloads(
    thesis_pack: BuyerThesisPack | dict[str, Any],
    profile: CompanyProfile,
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

    include_terms = section_values.get("include_constraint", [])
    exclude_terms = section_values.get("exclude_constraint", [])
    customer_tags = section_values.get("customer_profile", [])
    reference_seed_urls = _normalize_string_list(profile.reference_company_urls or [], max_items=8, max_len=220)

    core_capabilities = section_values.get("core_capability", [])[:8]
    adjacent_capabilities = section_values.get("adjacent_capability", [])[:8]
    business_model = section_values.get("business_model", [])[:2]
    if not core_capabilities:
        fallback_core = _fallback_lane_focus(customer_tags, include_terms, business_model)
        if fallback_core:
            core_capabilities = [fallback_core]
    if not adjacent_capabilities and core_capabilities:
        adjacent_capabilities = [f"Adjacent workflows around {core_capabilities[0]}"][:1]

    core_lane = {
        "lane_type": "core",
        "title": _lane_title("core", core_capabilities, customer_tags, business_model),
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
        "title": _lane_title("adjacent", adjacent_capabilities, customer_tags, business_model),
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


def thesis_pack_to_payload(
    thesis_pack: BuyerThesisPack,
    profile: CompanyProfile | None = None,
) -> dict[str, Any]:
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
        "buyer_evidence": assess_buyer_evidence(profile) if profile else None,
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
