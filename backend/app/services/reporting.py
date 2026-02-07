"""Static report generation helpers for SME M&A radar."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse


DISCOVERY_COUNTRIES = [
    "UK",
    "IE",
    "FR",
    "BE",
    "NL",
    "LU",
    "DE",
    "ES",
    "PT",
]

RELIABLE_FILINGS_COUNTRIES = {"FR", "UK"}

COUNTRY_NORMALIZATION = {
    "united kingdom": "UK",
    "uk": "UK",
    "great britain": "UK",
    "england": "UK",
    "france": "FR",
    "french": "FR",
    "ireland": "IE",
    "belgium": "BE",
    "netherlands": "NL",
    "luxembourg": "LU",
    "germany": "DE",
    "spain": "ES",
    "portugal": "PT",
}

FILING_SOURCE_PATTERNS = (
    "companieshouse.gov.uk",
    "find-and-update.company-information.service.gov.uk",
    "pappers.fr",
    "infogreffe.fr",
    "data.inpi.fr",
)

UNTRUSTED_SOURCE_HOSTS = {
    "vertexaisearch.cloud.google.com",
}

UK_FILING_SOURCE_PATTERNS = (
    "companieshouse.gov.uk",
    "find-and-update.company-information.service.gov.uk",
)

FR_FILING_SOURCE_PATTERNS = (
    "pappers.fr",
    "infogreffe.fr",
    "data.inpi.fr",
)


@dataclass
class MetricFact:
    fact_key: str
    fact_value: str
    fact_unit: Optional[str]
    period: Optional[str]
    confidence: str
    source_system: str
    source_url: str
    source_title: Optional[str]
    source_evidence_id: Optional[int]
    captured_at: Optional[datetime]


def normalize_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    normalized = url.strip()
    if not normalized:
        return None
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    try:
        parsed = urlparse(normalized)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or None
    except Exception:
        return None


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    c = country.strip().lower()
    if not c:
        return None
    return COUNTRY_NORMALIZATION.get(c, country.strip().upper())


def _url_host(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"
        host = urlparse(normalized).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def is_trusted_source_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            return False
        host = _url_host(normalized)
        if not host:
            return False
        if host in UNTRUSTED_SOURCE_HOSTS:
            return False
        if any(blocked in host for blocked in UNTRUSTED_SOURCE_HOSTS):
            return False
        return True
    except Exception:
        return False


def is_reliable_filing_source_url(country: Optional[str], url: Optional[str]) -> bool:
    normalized_country = normalize_country(country)
    if normalized_country not in RELIABLE_FILINGS_COUNTRIES:
        return False
    if not is_trusted_source_url(url):
        return False
    lower_url = (url or "").lower()
    if normalized_country == "UK":
        return any(pattern in lower_url for pattern in UK_FILING_SOURCE_PATTERNS)
    if normalized_country == "FR":
        return any(pattern in lower_url for pattern in FR_FILING_SOURCE_PATTERNS)
    return any(pattern in lower_url for pattern in FILING_SOURCE_PATTERNS)


def parse_size_estimate(dossier_json: Optional[Dict[str, Any]]) -> Optional[int]:
    if not dossier_json:
        return None
    team_size = (
        dossier_json.get("hiring", {})
        .get("mix_summary", {})
        .get("team_size_estimate")
    )
    if not team_size:
        return None
    if isinstance(team_size, int):
        return team_size
    if isinstance(team_size, float):
        return int(team_size)

    text = str(team_size).lower()
    numbers = [int(n) for n in re.findall(r"\d{1,4}", text)]
    if not numbers:
        return None
    if len(numbers) >= 2:
        return int(round((numbers[0] + numbers[1]) / 2.0))
    return numbers[0]


def _parse_size_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    source = str(text).lower()

    # e.g. "15-40 employees", "between 25 and 60 staff"
    ranged = re.search(
        r"(?:between\s+)?(\d{1,4})\s*(?:-|to|and|–)\s*(\d{1,4})\s*(?:employees|employee|staff|people|headcount|team|effectif)",
        source,
        flags=re.IGNORECASE,
    )
    if ranged:
        low = int(ranged.group(1))
        high = int(ranged.group(2))
        if high >= low:
            return int(round((low + high) / 2.0))

    # e.g. "85 employees"
    explicit = re.search(
        r"(\d{1,4})\s*(?:employees|employee|staff|people|headcount|team|effectif)",
        source,
        flags=re.IGNORECASE,
    )
    if explicit:
        return int(explicit.group(1))

    # e.g. "team size: 60"
    prefixed = re.search(
        r"(?:team size|headcount|employees|effectif)[^\d]{0,12}(\d{1,4})",
        source,
        flags=re.IGNORECASE,
    )
    if prefixed:
        return int(prefixed.group(1))

    return None


def classify_size_bucket(size_estimate: Optional[int]) -> str:
    if size_estimate is None:
        return "unknown"
    if 15 <= size_estimate <= 100:
        return "sme_in_range"
    return "outside_sme_range"


def modules_with_evidence(
    dossier_json: Optional[Dict[str, Any]],
    fallback_capabilities: Optional[Iterable[str]] = None,
    fallback_evidence_urls: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if not dossier_json:
        dossier_json = {}
    modules: List[Dict[str, Any]] = []
    for raw in dossier_json.get("modules", []):
        evidence_urls = [
            u
            for u in raw.get("evidence_urls", [])
            if isinstance(u, str) and u and is_trusted_source_url(u)
        ]
        modules.append(
            {
                "name": raw.get("name") or raw.get("brick_name") or raw.get("brick_id") or "Unknown capability",
                "brick_id": raw.get("brick_id"),
                "evidence_urls": evidence_urls,
                "has_evidence": bool(evidence_urls),
            }
        )

    if not modules and fallback_capabilities:
        trusted_urls = [
            u for u in (fallback_evidence_urls or []) if isinstance(u, str) and is_trusted_source_url(u)
        ]
        for capability in fallback_capabilities:
            if not isinstance(capability, str):
                continue
            name = capability.strip()
            if not name:
                continue
            modules.append(
                {
                    "name": name,
                    "brick_id": None,
                    "evidence_urls": trusted_urls[:1],
                    "has_evidence": bool(trusted_urls),
                }
            )
    return modules


def estimate_size_from_signals(
    dossier_json: Optional[Dict[str, Any]],
    facts: Optional[Iterable[Any]] = None,
    evidence_items: Optional[Iterable[Any]] = None,
    tags_custom: Optional[Iterable[Any]] = None,
    why_relevant: Optional[Iterable[Any]] = None,
) -> Optional[int]:
    size = parse_size_estimate(dossier_json)
    if size is not None:
        return size

    for fact in facts or []:
        if getattr(fact, "fact_key", None) != "employees":
            continue
        try:
            return int(str(getattr(fact, "fact_value", "")).strip())
        except Exception:
            continue

    for tag in tags_custom or []:
        if not isinstance(tag, str):
            continue
        if not tag.startswith("employee_estimate:"):
            continue
        raw = tag.split(":", 1)[1]
        parsed = _parse_size_from_text(raw)
        if parsed is not None:
            return parsed

    for evidence in evidence_items or []:
        parsed = _parse_size_from_text(getattr(evidence, "excerpt_text", None))
        if parsed is not None:
            return parsed

    for item in why_relevant or []:
        if isinstance(item, dict):
            parsed = _parse_size_from_text(item.get("text"))
        else:
            parsed = _parse_size_from_text(str(item))
        if parsed is not None:
            return parsed

    return None


def extract_customers_and_integrations(dossier_json: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not dossier_json:
        return [], []

    customers: List[Dict[str, Any]] = []
    for customer in dossier_json.get("customers", []):
        url = customer.get("evidence_url")
        trusted = is_trusted_source_url(url)
        customers.append(
            {
                "name": customer.get("name") or "Unknown customer",
                "context": customer.get("context") or "unknown",
                "source_url": url if trusted else None,
                "has_evidence": trusted,
            }
        )

    integrations: List[Dict[str, Any]] = []
    for integration in dossier_json.get("integrations", []):
        url = integration.get("evidence_url")
        trusted = is_trusted_source_url(url)
        integrations.append(
            {
                "name": integration.get("name") or "Unknown integration",
                "type": integration.get("type") or "integration",
                "source_url": url if trusted else None,
                "has_evidence": trusted,
            }
        )

    return customers, integrations


def build_adjacency_map(bricks: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = {}
    ids = [b.get("id") for b in bricks if b.get("id")]
    for idx, brick_id in enumerate(ids):
        near: Set[str] = set()
        if idx > 0:
            near.add(ids[idx - 1])
        if idx < len(ids) - 1:
            near.add(ids[idx + 1])
        adjacency[brick_id] = near
    return adjacency


def compute_entity_overlap_score(entities: Iterable[str], reference_tokens: Set[str]) -> Tuple[float, bool]:
    normalized = {
        token.strip().lower()
        for token in entities
        if isinstance(token, str) and token.strip()
    }
    if not normalized or not reference_tokens:
        return 0.0, False

    overlap_count = 0
    for token in normalized:
        if any(ref in token or token in ref for ref in reference_tokens):
            overlap_count += 1

    if overlap_count == 0:
        return 0.0, False
    return min(1.0, overlap_count / max(1, len(normalized))), True


def compute_lens_scores(
    vendor_modules: List[Dict[str, Any]],
    customers: List[Dict[str, Any]],
    integrations: List[Dict[str, Any]],
    priority_bricks: Set[str],
    adjacency_map: Dict[str, Set[str]],
    reference_tokens: Set[str],
    geo_vertical_match: bool,
    has_geo_vertical_evidence: bool,
) -> Dict[str, Any]:
    vendor_bricks = {m["brick_id"] for m in vendor_modules if m.get("brick_id")}
    overlap = vendor_bricks & priority_bricks if priority_bricks else set()
    overlap_evidence = [m for m in vendor_modules if m.get("brick_id") in overlap and m.get("has_evidence")]

    if priority_bricks:
        overlap_ratio = len(overlap) / max(1, len(priority_bricks))
    else:
        overlap_ratio = 0.0

    brick_component = overlap_ratio if overlap_evidence else 0.0

    customer_entities = [c["name"] for c in customers if c.get("has_evidence")]
    integration_entities = [i["name"] for i in integrations if i.get("has_evidence")]
    entity_overlap_score, has_entity_evidence = compute_entity_overlap_score(
        customer_entities + integration_entities,
        reference_tokens,
    )
    customer_partner_component = entity_overlap_score if has_entity_evidence else 0.0

    geo_vertical_component = 1.0 if geo_vertical_match and has_geo_vertical_evidence else 0.0

    compete_score = round(
        100.0
        * (
            0.50 * brick_component
            + 0.25 * customer_partner_component
            + 0.25 * geo_vertical_component
        ),
        2,
    )

    adjacent_candidates: Set[str] = set()
    for b in overlap:
        adjacent_candidates.update(adjacency_map.get(b, set()))
    adjacent_non_overlap = adjacent_candidates - overlap
    vendor_adjacent = vendor_bricks & adjacent_non_overlap
    adjacent_evidence = [m for m in vendor_modules if m.get("brick_id") in vendor_adjacent and m.get("has_evidence")]

    adjacent_component = (
        min(1.0, len(vendor_adjacent) / max(1, len(adjacent_non_overlap)))
        if adjacent_non_overlap and adjacent_evidence
        else 0.0
    )

    has_module_evidence = any(m.get("has_evidence") and m.get("brick_id") for m in vendor_modules)
    low_overlap_component = (1.0 - min(1.0, overlap_ratio)) if has_module_evidence else 0.0

    integration_component = (
        min(1.0, len([i for i in integrations if i.get("has_evidence")]) / 3.0)
        if any(i.get("has_evidence") for i in integrations)
        else 0.0
    )

    complement_score = round(
        100.0
        * (
            0.45 * adjacent_component
            + 0.25 * low_overlap_component
            + 0.30 * integration_component
        ),
        2,
    )

    return {
        "compete_score": compete_score,
        "complement_score": complement_score,
        "compete": {
            "brick_overlap": {
                "value": round(brick_component, 4),
                "weight": 0.50,
                "overlap_bricks": sorted(overlap),
                "evidence_count": len(overlap_evidence),
            },
            "customer_partner_overlap": {
                "value": round(customer_partner_component, 4),
                "weight": 0.25,
                "evidence_count": len(customer_entities) + len(integration_entities),
            },
            "geo_vertical_overlap": {
                "value": geo_vertical_component,
                "weight": 0.25,
                "evidence_count": 1 if has_geo_vertical_evidence else 0,
            },
        },
        "complement": {
            "adjacent_brick_potential": {
                "value": round(adjacent_component, 4),
                "weight": 0.45,
                "adjacent_bricks": sorted(vendor_adjacent),
                "evidence_count": len(adjacent_evidence),
            },
            "low_overlap_penalty": {
                "value": round(low_overlap_component, 4),
                "weight": 0.25,
                "evidence_count": len(overlap_evidence),
            },
            "integration_adjacency": {
                "value": round(integration_component, 4),
                "weight": 0.30,
                "evidence_count": len([i for i in integrations if i.get("has_evidence")]),
            },
        },
    }


def _parse_metric_number(raw: str) -> str:
    return raw.replace(" ", "").replace("\u202f", "").replace(",", ".")


def extract_filing_facts_from_evidence(country: Optional[str], evidence_items: Iterable[Any]) -> List[MetricFact]:
    normalized_country = normalize_country(country)
    if normalized_country not in RELIABLE_FILINGS_COUNTRIES:
        return []

    facts: List[MetricFact] = []

    for evidence in evidence_items:
        source_url = getattr(evidence, "source_url", None)
        if not source_url:
            continue
        if not is_reliable_filing_source_url(normalized_country, source_url):
            continue

        excerpt = (getattr(evidence, "excerpt_text", "") or "")[:4000]
        if not excerpt:
            continue

        period_match = re.search(r"(20\d{2})", excerpt)
        period = f"FY{period_match.group(1)}" if period_match else None

        revenue_match = re.search(
            r"(?:revenue|turnover|chiffre\s*d['’]?affaires)[^\d]{0,24}(\d[\d\s.,]{1,20})\s*(m|million|k|thousand)?\s*(eur|€|gbp|£)?",
            excerpt,
            flags=re.IGNORECASE,
        )
        if revenue_match:
            raw = _parse_metric_number(revenue_match.group(1))
            mag = (revenue_match.group(2) or "").lower()
            currency = (revenue_match.group(3) or "").upper().replace("€", "EUR").replace("£", "GBP")
            unit = currency or None
            if mag in {"m", "million"}:
                unit = f"{unit or ''}m".strip()
            elif mag in {"k", "thousand"}:
                unit = f"{unit or ''}k".strip()

            facts.append(
                MetricFact(
                    fact_key="revenue",
                    fact_value=raw,
                    fact_unit=unit,
                    period=period,
                    confidence="high",
                    source_system="filing",
                    source_url=source_url,
                    source_title=getattr(evidence, "source_title", None),
                    source_evidence_id=getattr(evidence, "id", None),
                    captured_at=getattr(evidence, "captured_at", None),
                )
            )

        employee_match = re.search(
            r"(?:employees|effectif|staff)[^\d]{0,18}(\d{1,5})",
            excerpt,
            flags=re.IGNORECASE,
        )
        if employee_match:
            employee_value = employee_match.group(1)
        else:
            parsed_size = _parse_size_from_text(excerpt)
            employee_value = str(parsed_size) if parsed_size is not None else None

        if employee_value:
            facts.append(
                MetricFact(
                    fact_key="employees",
                    fact_value=employee_value,
                    fact_unit="people",
                    period=period,
                    confidence="high",
                    source_system="filing",
                    source_url=source_url,
                    source_title=getattr(evidence, "source_title", None),
                    source_evidence_id=getattr(evidence, "id", None),
                    captured_at=getattr(evidence, "captured_at", None),
                )
            )

    deduped: Dict[Tuple[str, str, Optional[str]], MetricFact] = {}
    for fact in facts:
        key = (fact.fact_key, fact.fact_value, fact.period)
        if key not in deduped:
            deduped[key] = fact
    return list(deduped.values())


def source_label_for_url(url: str) -> str:
    lower = url.lower()
    if "companieshouse.gov.uk" in lower or "company-information.service.gov.uk" in lower:
        return "UK Companies House filing"
    if "pappers.fr" in lower or "infogreffe.fr" in lower or "inpi.fr" in lower:
        return "French public filing"
    return "Public source"


def make_source_payload(url: str, document_id: Optional[str], captured_at: Optional[datetime]) -> Dict[str, Any]:
    return {
        "label": source_label_for_url(url),
        "url": url,
        "document_id": document_id,
        "captured_at": captured_at.isoformat() if captured_at else None,
    }
