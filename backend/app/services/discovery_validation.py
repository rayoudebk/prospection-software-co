from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


VALIDATION_STATUS_QUEUED = "queued_for_validation"
VALIDATION_STATUS_KEEP = "validated_keep"
VALIDATION_STATUS_WATCHLIST = "validated_watchlist"
VALIDATION_STATUS_REJECT = "validated_reject"
VALIDATION_STATUS_REMOVED = "removed_from_validation"
VALIDATION_STATUS_PROMOTED = "promoted_to_cards"

VALIDATION_STATUSES = {
    VALIDATION_STATUS_QUEUED,
    VALIDATION_STATUS_KEEP,
    VALIDATION_STATUS_WATCHLIST,
    VALIDATION_STATUS_REJECT,
    VALIDATION_STATUS_REMOVED,
    VALIDATION_STATUS_PROMOTED,
}

VALIDATION_RECOMMENDATION_BY_CLASSIFICATION = {
    "good_target": VALIDATION_STATUS_KEEP,
    "borderline_watchlist": VALIDATION_STATUS_WATCHLIST,
    "not_good_target": VALIDATION_STATUS_REJECT,
    "insufficient_evidence": VALIDATION_STATUS_QUEUED,
}

CLASSIFICATION_RANK = {
    "good_target": 0,
    "borderline_watchlist": 1,
    "insufficient_evidence": 2,
    "not_good_target": 3,
}

DISCOVERY_QUERY_FAMILIES = {
    "category_vendor",
    "competitor_direct",
    "alternatives",
    "buyer_substitute",
    "adjacent_substitute",
    "peer_expansion",
    "local_market",
    "comparative_source",
    "traffic_affinity",
    "registry_lookup",
    "unknown",
}

DISCOVERY_SOURCE_FAMILIES = {
    "first_party_vendor",
    "directory_category_map",
    "comparative_list",
    "traffic_affinity",
    "review_marketplace",
    "registry",
    "editorial_analyst",
    "community_social",
    "unknown",
}

SOCIAL_HOST_TOKENS = ("reddit.com", "linkedin.com", "x.com", "twitter.com", "facebook.com", "youtube.com")
REVIEW_HOST_TOKENS = ("g2.com", "capterra.com", "gartner.com", "g2crowd.com")
DIRECTORY_HOST_TOKENS = ("thewealthmosaic.com", "crunchbase.com", "vendor-showcase", "vendors/")
TRAFFIC_HOST_TOKENS = ("similarweb.com",)


def _first_non_empty(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
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


def _clean_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _title_case_token(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return " ".join(token.capitalize() for token in raw.replace("_", " ").replace("-", " ").split())


def _normalize_query_family(value: Any, *, query_text: Any = None, provider: Any = None) -> str:
    explicit = str(value or "").strip().lower()
    if explicit in DISCOVERY_QUERY_FAMILIES:
        return explicit
    combined = " ".join(
        part
        for part in [
            explicit,
            str(query_text or "").strip().lower(),
            str(provider or "").strip().lower(),
        ]
        if part
    )
    if not combined:
        return "unknown"
    if any(token in combined for token in ("competitor", "compete", "vs ", "compare")):
        return "competitor_direct"
    if "alternative" in combined:
        return "alternatives"
    if any(token in combined for token in ("local_market", "france", "germany", "belgium", "uk", "europe", "local")):
        return "local_market"
    if any(token in combined for token in ("peer_expand", "peer_expansion", "companies like", "similar to")):
        return "peer_expansion"
    if any(token in combined for token in ("traffic", "similarweb", "audience overlap")):
        return "traffic_affinity"
    if any(token in combined for token in ("review", "ratings", "marketplace")):
        return "review_marketplace"
    if any(token in combined for token in ("directory", "category map", "vendor list", "landscape", "top ", "best ")):
        return "comparative_source"
    if any(token in combined for token in ("buyer", "hospital", "used by", "for teams", "for hospitals")):
        return "buyer_substitute"
    if any(token in combined for token in ("adjacent", "workflow", "replacement", "scheduling", "staffing", "planning")):
        return "adjacent_substitute"
    if any(token in combined for token in ("vendor", "software", "platform", "provider", "solution")):
        return "category_vendor"
    return "unknown"


def _source_family_from_origin_and_metadata(origin_type: Any, metadata: dict[str, Any], origin_url: Any) -> str:
    normalized = _clean_token(origin_type)
    lowered_url = str(origin_url or "").strip().lower()
    source_name = str(metadata.get("source_name") or "").strip().lower()
    query_family = _normalize_query_family(
        metadata.get("query_type"),
        query_text=metadata.get("query_text"),
        provider=metadata.get("provider"),
    )
    if any(token in lowered_url for token in SOCIAL_HOST_TOKENS):
        return "community_social"
    if any(token in lowered_url for token in TRAFFIC_HOST_TOKENS) or source_name == "similarweb":
        return "traffic_affinity"
    if normalized.startswith("registry"):
        return "registry"
    if normalized in {"directory_seed", "directory_profile", "reference_seed"}:
        if any(token in lowered_url for token in REVIEW_HOST_TOKENS):
            return "review_marketplace"
        if any(token in lowered_url for token in TRAFFIC_HOST_TOKENS):
            return "traffic_affinity"
        return "directory_category_map"
    if normalized in {"benchmark_seed"}:
        return "comparative_list"
    if normalized in {"external_search_seed", "llm_seed"}:
        if query_family == "traffic_affinity":
            return "traffic_affinity"
        if query_family in {"competitor_direct", "alternatives", "comparative_source", "peer_expansion"}:
            return "comparative_list"
    return "editorial_analyst"
    if normalized == "expansion_company_seed":
        return "first_party_vendor"
    return "unknown"


def _source_role_for_family(source_family: str) -> str:
    if source_family in {"first_party_vendor"}:
        return "validation"
    if source_family in {"registry"}:
        return "dedupe"
    if source_family in {"community_social"}:
        return "rejection"
    return "discovery"


def normalize_discovery_query_family(value: Any, *, query_text: Any = None, provider: Any = None) -> str:
    return _normalize_query_family(value, query_text=query_text, provider=provider)


def normalize_discovery_source_family(origin_type: Any, metadata: dict[str, Any], origin_url: Any) -> str:
    return _source_family_from_origin_and_metadata(origin_type, metadata, origin_url)


def discovery_source_role(source_family: str) -> str:
    return _source_role_for_family(source_family)


def compute_discovery_score(
    *,
    source_families: list[str],
    query_families: list[str],
    lane_ids: list[str],
    geo_signals: list[str],
    origin_types: list[str],
    identity_confidence: Optional[str],
    directness: str,
) -> float:
    return _discovery_score(
        source_families=source_families,
        query_families=query_families,
        lane_ids=lane_ids,
        geo_signals=geo_signals,
        origin_types=origin_types,
        identity_confidence=identity_confidence,
        directness=directness,
    )


def _short_description_from_reasons(reasons: Iterable[Any]) -> Optional[str]:
    for item in reasons:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        return text[:240]
    return None


def _candidate_directness(
    *,
    lane_ids: list[str],
    lane_labels: list[str],
    scope_buckets: list[str],
    query_families: list[str],
) -> str:
    normalized_buckets = {_clean_token(item) for item in scope_buckets}
    normalized_queries = {_clean_token(item) for item in query_families}
    if normalized_buckets & {"core", "direct", "source", "core_adjacent"}:
        return "direct"
    if normalized_queries & {"competitor_direct", "alternatives"}:
        return "direct"
    if lane_ids or lane_labels:
        return "adjacent"
    return "broad_market"


def _discovery_score(
    *,
    source_families: list[str],
    query_families: list[str],
    lane_ids: list[str],
    geo_signals: list[str],
    origin_types: list[str],
    identity_confidence: Optional[str],
    directness: str,
) -> float:
    score = 0.0
    score += min(24.0, float(len(source_families)) * 6.0)
    score += min(18.0, float(len(query_families)) * 4.5)
    score += min(16.0, float(len(lane_ids)) * 4.0)
    score += min(8.0, float(len(geo_signals)) * 2.0)
    score += min(12.0, max(0.0, float(len(origin_types) - 1) * 2.0))
    if identity_confidence == "high":
        score += 8.0
    elif identity_confidence == "medium":
        score += 4.0
    if directness == "direct":
        score += 10.0
    elif directness == "adjacent":
        score += 6.0
    return round(score, 3)


def validation_metadata(entity: Any) -> dict[str, Any]:
    metadata = _as_dict(getattr(entity, "metadata_json", None))
    validation = _as_dict(metadata.get("validation"))
    status = str(validation.get("status") or validation.get("validation_state") or "").strip()
    if status not in VALIDATION_STATUSES:
        validation["status"] = VALIDATION_STATUS_QUEUED
    else:
        validation["status"] = status
    validation["validation_state"] = validation["status"]
    validation["promoted_to_cards"] = bool(validation.get("promoted_to_cards", False))
    validation["queue_rank"] = int(validation.get("queue_rank") or 0)
    validation["priority_score"] = float(validation.get("priority_score") or 0.0)
    validation["lane_ids"] = _dedupe_strings(validation.get("lane_ids") or [])
    validation["lane_labels"] = _dedupe_strings(validation.get("lane_labels") or [])
    validation["query_families"] = _dedupe_strings(validation.get("query_families") or [])
    validation["source_families"] = _dedupe_strings(validation.get("source_families") or [])
    validation["origin_types"] = _dedupe_strings(validation.get("origin_types") or [])
    validation["recommendation"] = str(validation.get("recommendation") or validation["status"]).strip() or VALIDATION_STATUS_QUEUED
    validation["identity_confidence"] = str(validation.get("identity_confidence") or "").strip() or None
    validation["vendor_classification"] = str(validation.get("vendor_classification") or "").strip() or None
    validation["official_website_confidence"] = str(validation.get("official_website_confidence") or "").strip() or None
    validation["page_classification"] = str(validation.get("page_classification") or "").strip() or None
    validation["short_description"] = str(validation.get("short_description") or "").strip() or None
    validation["validated_description"] = str(validation.get("validated_description") or "").strip() or None
    validation["validation_score"] = float(validation.get("validation_score") or 0.0)
    diagnostics = _as_dict(validation.get("identity_diagnostics"))
    validation["identity_diagnostics"] = {
        "identity_error": str(diagnostics.get("identity_error") or "").strip() or None,
        "resolved_via_redirect": bool(diagnostics.get("resolved_via_redirect", False)),
        "first_party_method": str(diagnostics.get("first_party_method") or "").strip() or None,
        "first_party_tier": str(diagnostics.get("first_party_tier") or "").strip() or None,
        "pages_crawled": int(diagnostics.get("pages_crawled") or 0),
        "signals_extracted": int(diagnostics.get("signals_extracted") or 0),
        "has_first_party_evidence": bool(diagnostics.get("has_first_party_evidence", False)),
    }
    return validation


def set_validation_metadata(entity: Any, payload: dict[str, Any]) -> None:
    metadata = _as_dict(getattr(entity, "metadata_json", None))
    validation = validation_metadata(entity)
    validation.update(_as_dict(payload))
    status = str(validation.get("status") or validation.get("validation_state") or "").strip()
    if status not in VALIDATION_STATUSES:
        validation["status"] = VALIDATION_STATUS_QUEUED
    else:
        validation["status"] = status
    validation["validation_state"] = validation["status"]
    validation["promoted_to_cards"] = bool(validation.get("promoted_to_cards", False))
    validation["lane_ids"] = _dedupe_strings(validation.get("lane_ids") or [])
    validation["lane_labels"] = _dedupe_strings(validation.get("lane_labels") or [])
    validation["query_families"] = _dedupe_strings(validation.get("query_families") or [])
    validation["source_families"] = _dedupe_strings(validation.get("source_families") or [])
    validation["origin_types"] = _dedupe_strings(validation.get("origin_types") or [])
    validation["page_classification"] = str(validation.get("page_classification") or "").strip() or None
    validation["short_description"] = str(validation.get("short_description") or "").strip() or None
    validation["validated_description"] = str(validation.get("validated_description") or "").strip() or None
    validation["validation_score"] = float(validation.get("validation_score") or 0.0)
    metadata["validation"] = validation
    entity.metadata_json = metadata


def validation_status_from_classification(classification: Any) -> str:
    normalized = str(classification or "insufficient_evidence").strip().lower()
    return VALIDATION_RECOMMENDATION_BY_CLASSIFICATION.get(normalized, VALIDATION_STATUS_QUEUED)


def vendor_classification_from_screening(
    *,
    official_website_url: Optional[str],
    entity_type: Optional[str],
    origin_types: Iterable[Any],
    screening_classification: Optional[str],
) -> str:
    normalized_entity_type = _clean_token(entity_type)
    normalized_origins = {_clean_token(item) for item in origin_types}
    normalized_classification = _clean_token(screening_classification)
    if normalized_entity_type == "solution":
        return "solution_profile"
    if official_website_url:
        return "vendor_candidate"
    if "external_search_seed" in normalized_origins:
        return "vendor_candidate"
    if normalized_classification in {"good_target", "borderline_watchlist"}:
        return "vendor_candidate"
    if "directory_seed" in normalized_origins:
        return "directory_only_candidate"
    return "unclassified_candidate"


def official_website_confidence(official_website_url: Optional[str], discovery_sources: Iterable[Any]) -> str:
    if official_website_url:
        return "high"
    if any(str(item or "").strip() for item in discovery_sources):
        return "medium"
    return "low"


def identity_diagnostics_from_context(
    *,
    entity: Any,
    screening_meta: dict[str, Any],
    entity_validation: dict[str, Any],
) -> dict[str, Any]:
    existing = _as_dict(entity_validation.get("identity_diagnostics"))
    first_party = _as_dict(screening_meta.get("first_party_enrichment"))
    return {
        "identity_error": (
            str(existing.get("identity_error") or "").strip()
            or str(getattr(entity, "identity_error", "") or "").strip()
            or None
        ),
        "resolved_via_redirect": bool(existing.get("resolved_via_redirect", False)),
        "first_party_method": (
            str(existing.get("first_party_method") or "").strip()
            or str(first_party.get("method") or "").strip()
            or None
        ),
        "first_party_tier": (
            str(existing.get("first_party_tier") or "").strip()
            or str(first_party.get("tier") or "").strip()
            or None
        ),
        "pages_crawled": int(existing.get("pages_crawled") or first_party.get("pages_crawled") or 0),
        "signals_extracted": int(existing.get("signals_extracted") or first_party.get("signals_extracted") or 0),
        "has_first_party_evidence": bool(
            existing.get("has_first_party_evidence", False)
            or bool(int(first_party.get("signals_extracted") or 0) > 0)
        ),
    }


def _lane_context_from_origin(origin_metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    lane_ids = _dedupe_strings(origin_metadata.get("fit_to_adjacency_box_ids") or [])
    lane_labels = _dedupe_strings(origin_metadata.get("fit_to_adjacency_box_labels") or [])
    if lane_ids or lane_labels:
        return lane_ids, lane_labels
    scope_bucket = str(origin_metadata.get("scope_bucket") or "").strip().lower()
    if scope_bucket:
        return [scope_bucket], [_title_case_token(scope_bucket)]
    source_caps = _dedupe_strings(origin_metadata.get("source_capability_matches") or [])
    if source_caps:
        return source_caps[:4], source_caps[:4]
    brick_name = str(origin_metadata.get("brick_name") or "").strip()
    if brick_name:
        return [_clean_token(brick_name)], [brick_name]
    return [], []


def _query_family_from_origin(origin_type: Any, origin_metadata: dict[str, Any]) -> str:
    provider = str(origin_metadata.get("provider") or "").strip().lower()
    query_type = str(
        origin_metadata.get("query_family")
        or origin_metadata.get("query_intent")
        or origin_metadata.get("query_type")
        or ""
    ).strip().lower()
    brick_name = str(origin_metadata.get("brick_name") or "").strip().lower()
    source_name = str(origin_metadata.get("source_name") or "").strip().lower()
    if query_type:
        return _normalize_query_family(query_type, query_text=origin_metadata.get("query_text"), provider=provider)
    if brick_name:
        return _normalize_query_family(brick_name, query_text=brick_name, provider=provider)
    return _clean_token(origin_type) or source_name or "unknown"


def _source_family_from_origin(origin_type: Any) -> str:
    return _source_family_from_origin_and_metadata(origin_type, {}, None)


def build_candidate_discovery_context(
    *,
    entity: Any,
    origins: list[Any],
) -> dict[str, Any]:
    metadata = _as_dict(getattr(entity, "metadata_json", None))
    lane_ids: list[str] = []
    lane_labels: list[str] = []
    query_families: list[str] = []
    source_families: list[str] = []
    source_roles: list[str] = []
    origin_types: list[str] = []
    discovery_sources: list[str] = []
    scope_buckets: list[str] = []

    for origin in origins:
        origin_type = getattr(origin, "origin_type", None)
        origin_url = getattr(origin, "origin_url", None)
        metadata_json = _as_dict(getattr(origin, "metadata_json", None))
        origin_types.append(str(origin_type or "").strip())
        if origin_url:
            discovery_sources.append(str(origin_url))
        ids, labels = _lane_context_from_origin(metadata_json)
        lane_ids.extend(ids)
        lane_labels.extend(labels)
        scope_bucket = str(metadata_json.get("scope_bucket") or "").strip().lower()
        if scope_bucket:
            scope_buckets.append(scope_bucket)
        query_family = _normalize_query_family(
            metadata_json.get("query_family") or metadata_json.get("query_intent") or metadata_json.get("query_type"),
            query_text=metadata_json.get("query_text"),
            provider=metadata_json.get("provider"),
        )
        query_families.append(query_family)
        source_family = _source_family_from_origin_and_metadata(origin_type, metadata_json, origin_url)
        source_families.append(source_family)
        source_roles.append(_source_role_for_family(source_family))

    lane_ids = _dedupe_strings(lane_ids)
    lane_labels = _dedupe_strings(lane_labels)
    query_families = _dedupe_strings(query_families)
    source_families = _dedupe_strings(source_families)
    source_roles = _dedupe_strings(source_roles)
    origin_types = _dedupe_strings(origin_types or metadata.get("origin_types") or [])
    discovery_sources = _dedupe_strings(discovery_sources)

    geo_signals = _dedupe_strings(
        [
            getattr(entity, "country", None),
            metadata.get("registry_country"),
            metadata.get("country"),
        ]
    )
    identity_confidence = str(
        metadata.get("identity_confidence") or getattr(entity, "identity_confidence", "") or ""
    ).strip() or None
    why_relevant = metadata.get("why_relevant") if isinstance(metadata.get("why_relevant"), list) else []
    short_description = (
        _first_non_empty(
            [
                metadata.get("short_description"),
                _short_description_from_reasons(why_relevant),
                _first_non_empty(metadata.get("solutions") or []),
            ]
        )
        or None
    )
    directness = str(metadata.get("directness") or "").strip() or _candidate_directness(
        lane_ids=lane_ids,
        lane_labels=lane_labels,
        scope_buckets=scope_buckets,
        query_families=query_families,
    )
    stored_node_fit = _as_dict(metadata.get("node_fit_summary"))
    matched_node_ids = _dedupe_strings(
        list(stored_node_fit.get("matched_node_ids") or [])
        + lane_ids
    )
    matched_node_labels = _dedupe_strings(
        list(stored_node_fit.get("matched_node_labels") or [])
        + lane_labels
    )
    core_match_count = int(stored_node_fit.get("core_match_count") or (1 if directness == "direct" and matched_node_labels else 0))
    adjacent_match_count = int(
        stored_node_fit.get("adjacent_match_count")
        or ((len(matched_node_labels) if directness == "adjacent" else 0))
    )
    node_fit_score = float(
        stored_node_fit.get("node_fit_score")
        or (core_match_count * 10.0 + adjacent_match_count * 5.0)
    )
    discovery_score = float(metadata.get("discovery_score") or 0.0)
    if discovery_score <= 0:
        discovery_score = _discovery_score(
            source_families=source_families,
            query_families=query_families,
            lane_ids=lane_ids,
            geo_signals=geo_signals,
            origin_types=origin_types,
            identity_confidence=identity_confidence,
            directness=directness,
        )
    provenance_sources = [
        str(source).strip()
        for source in discovery_sources[:4]
        if str(source).strip()
    ]
    provenance_summary = {
        "origin_count": len(origin_types),
        "source_family_count": len(source_families),
        "query_family_count": len(query_families),
        "discovery_links_count": len(discovery_sources),
        "top_source_urls": provenance_sources,
    }
    official_website_url = (
        str(getattr(entity, "canonical_website", "") or "").strip()
        or str(metadata.get("official_website_url") or "").strip()
        or None
    )
    return {
        "candidate_entity_id": int(getattr(entity, "id", 0) or 0),
        "company_name": str(getattr(entity, "canonical_name", "") or "").strip(),
        "official_website_url": official_website_url,
        "discovery_url": str(getattr(entity, "discovery_primary_url", "") or "").strip() or None,
        "entity_type": str(getattr(entity, "entity_type", None) or metadata.get("entity_type") or "company"),
        "identity_confidence": identity_confidence,
        "short_description": short_description,
        "discovery_score": float(discovery_score),
        "source_families": source_families,
        "query_families": query_families,
        "lane_ids": lane_ids,
        "lane_labels": lane_labels,
        "geo_signals": geo_signals,
        "directness": directness,
        "node_fit_summary": {
            "matched_node_ids": matched_node_ids,
            "matched_node_labels": matched_node_labels,
            "core_match_count": core_match_count,
            "adjacent_match_count": adjacent_match_count,
            "node_fit_score": round(node_fit_score, 3),
        },
        "origin_types": origin_types,
        "discovery_sources": discovery_sources,
        "source_roles": source_roles,
        "provenance_summary": provenance_summary,
    }


def build_candidate_validation_context(
    *,
    entity: Any,
    origins: list[Any],
    screening: Any = None,
) -> dict[str, Any]:
    screening_meta = _as_dict(getattr(screening, "screening_meta_json", None))
    source_summary = _as_dict(getattr(screening, "source_summary_json", None))
    entity_validation = validation_metadata(entity) if entity is not None else {}
    discovery = build_candidate_discovery_context(entity=entity, origins=origins)
    lane_ids = discovery["lane_ids"] or _dedupe_strings(screening_meta.get("fit_to_adjacency_box_ids") or screening_meta.get("scope_buckets") or [])
    lane_labels = discovery["lane_labels"] or _dedupe_strings(screening_meta.get("fit_to_adjacency_box_labels") or screening_meta.get("capability_signals") or [])
    query_families = discovery["query_families"]
    source_families = discovery["source_families"]
    origin_types = discovery["origin_types"] or _dedupe_strings(screening_meta.get("origin_types") or [])
    discovery_sources = discovery["discovery_sources"] or _dedupe_strings(source_summary.get("source_urls") or [])

    official_site = (
        str(getattr(screening, "candidate_official_website", "") or "").strip()
        or str(getattr(entity, "canonical_website", "") or "").strip()
        or None
    )
    recommendation = str(entity_validation.get("recommendation") or "").strip()
    if not recommendation:
        recommendation = validation_status_from_classification(getattr(screening, "decision_classification", None))
    status = str(entity_validation.get("status") or "").strip() or recommendation
    if status == VALIDATION_STATUS_PROMOTED:
        promoted_to_cards = True
    else:
        promoted_to_cards = bool(entity_validation.get("promoted_to_cards", False))
    if status not in VALIDATION_STATUSES:
        status = VALIDATION_STATUS_QUEUED

    source_count = len(discovery_sources)
    multi_origin_count = len(origin_types)
    validation_score = float(entity_validation.get("validation_score") or 0.0)
    if validation_score <= 0:
        validation_score = float(discovery.get("discovery_score") or 0.0)
        validation_score += min(6.0, float(multi_origin_count))
        validation_score += min(4.0, float(source_count))
        if official_site:
            validation_score += 4.0
    identity_confidence = str(
        entity_validation.get("identity_confidence") or getattr(entity, "identity_confidence", "") or ""
    ).strip() or None
    if identity_confidence == "high":
        validation_score += 3.0

    diagnostics = identity_diagnostics_from_context(
        entity=entity,
        screening_meta=screening_meta,
        entity_validation=entity_validation,
    )
    if diagnostics["has_first_party_evidence"]:
        validation_score += 2.0

    short_description = str(
        entity_validation.get("validated_description")
        or entity_validation.get("short_description")
        or discovery.get("short_description")
        or ""
    ).strip() or None

    return {
        "status": status,
        "recommendation": str(entity_validation.get("recommendation") or recommendation),
        "promoted_to_cards": promoted_to_cards,
        "promoted_to_cards_at": entity_validation.get("promoted_to_cards_at"),
        "lane_ids": lane_ids,
        "lane_labels": lane_labels,
        "query_families": query_families,
        "source_families": source_families,
        "origin_types": origin_types,
        "priority_score": round(validation_score, 3),
        "validation_score": round(validation_score, 3),
        "queue_rank": int(entity_validation.get("queue_rank") or 0),
        "identity_confidence": identity_confidence,
        "vendor_classification": str(
            entity_validation.get("vendor_classification")
            or vendor_classification_from_screening(
                official_website_url=official_site,
                entity_type=getattr(entity, "entity_type", None) or screening_meta.get("entity_type"),
                origin_types=origin_types,
                screening_classification=getattr(screening, "decision_classification", None),
            )
        ).strip(),
        "official_website_confidence": str(
            entity_validation.get("official_website_confidence")
            or official_website_confidence(official_site, discovery_sources)
        ).strip(),
        "identity_diagnostics": diagnostics,
        "discovery_sources": discovery_sources,
        "entity_type": str(getattr(entity, "entity_type", None) or screening_meta.get("entity_type") or "company"),
        "short_description": short_description,
        "validated_description": str(entity_validation.get("validated_description") or "").strip() or None,
        "page_classification": str(entity_validation.get("page_classification") or "").strip() or None,
        "provenance_summary": discovery.get("provenance_summary") or {},
        "directness": discovery.get("directness") or "broad_market",
        "geo_signals": discovery.get("geo_signals") or [],
    }


def build_diversified_validation_queue(
    items: list[dict[str, Any]],
    *,
    limit: int,
    lane_cap: int,
    family_cap: int,
    source_family_cap: int,
) -> list[dict[str, Any]]:
    if not items:
        return []

    status_rank = {
        VALIDATION_STATUS_PROMOTED: 0,
        VALIDATION_STATUS_KEEP: 1,
        VALIDATION_STATUS_WATCHLIST: 2,
        VALIDATION_STATUS_QUEUED: 3,
        VALIDATION_STATUS_REJECT: 4,
        VALIDATION_STATUS_REMOVED: 5,
    }

    sorted_items = sorted(
        items,
        key=lambda item: (
            status_rank.get(str(item.get("validation_status") or item.get("status") or VALIDATION_STATUS_QUEUED), 99),
            -float(item.get("validation_score") or item.get("priority_score") or 0.0),
            -int(item.get("multi_origin_count") or 0),
            int(item.get("candidate_entity_id") or 0),
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    lane_counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)

    def _candidate_ok(item: dict[str, Any], *, relax_lane: bool) -> bool:
        lane_keys = item.get("lane_ids") or item.get("lane_labels") or ["unscoped"]
        family_keys = item.get("query_families") or ["unknown"]
        source_keys = item.get("source_families") or ["unknown"]
        if not relax_lane and all(lane_counts[str(key)] >= lane_cap for key in lane_keys):
            return False
        if all(family_counts[str(key)] >= family_cap for key in family_keys):
            return False
        if all(source_counts[str(key)] >= source_family_cap for key in source_keys):
            return False
        return True

    def _mark(item: dict[str, Any]) -> None:
        lane_keys = item.get("lane_ids") or item.get("lane_labels") or ["unscoped"]
        family_keys = item.get("query_families") or ["unknown"]
        source_keys = item.get("source_families") or ["unknown"]
        for key in lane_keys[:1]:
            lane_counts[str(key)] += 1
        for key in family_keys[:1]:
            family_counts[str(key)] += 1
        for key in source_keys[:1]:
            source_counts[str(key)] += 1

    for relax_lane in (False, True):
        for item in sorted_items:
            entity_id = int(item.get("candidate_entity_id") or 0)
            if not entity_id or entity_id in selected_ids:
                continue
            if not _candidate_ok(item, relax_lane=relax_lane):
                continue
            selected.append(item)
            selected_ids.add(entity_id)
            _mark(item)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in sorted_items:
            entity_id = int(item.get("candidate_entity_id") or 0)
            if not entity_id or entity_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(entity_id)
            if len(selected) >= limit:
                break

    for index, item in enumerate(selected, start=1):
        item["queue_rank"] = index
    return selected


def build_diversified_universe_candidates(
    items: list[dict[str, Any]],
    *,
    limit: int,
    lane_cap: int,
    family_cap: int,
    source_family_cap: int,
) -> list[dict[str, Any]]:
    if not items:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: (
            -float(item.get("discovery_score") or 0.0),
            -int(item.get("multi_origin_count") or 0),
            int(item.get("candidate_entity_id") or 0),
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    lane_counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)

    def _candidate_ok(item: dict[str, Any], *, relax_lane: bool) -> bool:
        lane_keys = item.get("lane_ids") or item.get("lane_labels") or ["unscoped"]
        family_keys = item.get("query_families") or ["unknown"]
        source_keys = item.get("source_families") or ["unknown"]
        if not relax_lane and all(lane_counts[str(key)] >= lane_cap for key in lane_keys):
            return False
        if all(family_counts[str(key)] >= family_cap for key in family_keys):
            return False
        if all(source_counts[str(key)] >= source_family_cap for key in source_keys):
            return False
        return True

    def _mark(item: dict[str, Any]) -> None:
        for key in (item.get("lane_ids") or item.get("lane_labels") or ["unscoped"])[:1]:
            lane_counts[str(key)] += 1
        for key in (item.get("query_families") or ["unknown"])[:1]:
            family_counts[str(key)] += 1
        for key in (item.get("source_families") or ["unknown"])[:1]:
            source_counts[str(key)] += 1

    for relax_lane in (False, True):
        for item in sorted_items:
            entity_id = int(item.get("candidate_entity_id") or 0)
            if not entity_id or entity_id in selected_ids:
                continue
            if not _candidate_ok(item, relax_lane=relax_lane):
                continue
            selected.append(item)
            selected_ids.add(entity_id)
            _mark(item)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in sorted_items:
            entity_id = int(item.get("candidate_entity_id") or 0)
            if not entity_id or entity_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(entity_id)
            if len(selected) >= limit:
                break

    return selected
