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


def validation_metadata(entity: Any) -> dict[str, Any]:
    metadata = _as_dict(getattr(entity, "metadata_json", None))
    validation = _as_dict(metadata.get("validation"))
    status = str(validation.get("status") or "").strip()
    if status not in VALIDATION_STATUSES:
        validation["status"] = VALIDATION_STATUS_QUEUED
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
    if str(validation.get("status") or "").strip() not in VALIDATION_STATUSES:
        validation["status"] = VALIDATION_STATUS_QUEUED
    validation["promoted_to_cards"] = bool(validation.get("promoted_to_cards", False))
    validation["lane_ids"] = _dedupe_strings(validation.get("lane_ids") or [])
    validation["lane_labels"] = _dedupe_strings(validation.get("lane_labels") or [])
    validation["query_families"] = _dedupe_strings(validation.get("query_families") or [])
    validation["source_families"] = _dedupe_strings(validation.get("source_families") or [])
    validation["origin_types"] = _dedupe_strings(validation.get("origin_types") or [])
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
    query_type = str(origin_metadata.get("query_type") or "").strip().lower()
    brick_name = str(origin_metadata.get("brick_name") or "").strip().lower()
    source_name = str(origin_metadata.get("source_name") or "").strip().lower()
    if query_type or brick_name or provider:
        parts = [part for part in [provider, query_type, brick_name] if part]
        return "::".join(parts[:3])
    return _clean_token(origin_type) or source_name or "unknown"


def _source_family_from_origin(origin_type: Any) -> str:
    normalized = _clean_token(origin_type)
    if not normalized:
        return "unknown"
    if normalized in {"directory_seed", "directory_profile", "reference_seed"}:
        return "directory"
    if normalized in {"external_search_seed", "llm_seed", "benchmark_seed"}:
        return "search"
    if normalized.startswith("registry"):
        return "registry"
    return normalized


def build_candidate_validation_context(
    *,
    screening: Any,
    entity: Any,
    origins: list[Any],
) -> dict[str, Any]:
    screening_meta = _as_dict(getattr(screening, "screening_meta_json", None))
    source_summary = _as_dict(getattr(screening, "source_summary_json", None))
    entity_validation = validation_metadata(entity) if entity is not None else {}
    lane_ids: list[str] = []
    lane_labels: list[str] = []
    query_families: list[str] = []
    source_families: list[str] = []
    origin_types: list[str] = []
    discovery_sources: list[str] = []
    for origin in origins:
        origin_type = getattr(origin, "origin_type", None)
        origin_types.append(str(origin_type or "").strip())
        origin_url = getattr(origin, "origin_url", None)
        if origin_url:
            discovery_sources.append(str(origin_url))
        metadata = _as_dict(getattr(origin, "metadata_json", None))
        ids, labels = _lane_context_from_origin(metadata)
        lane_ids.extend(ids)
        lane_labels.extend(labels)
        query_families.append(_query_family_from_origin(origin_type, metadata))
        source_families.append(_source_family_from_origin(origin_type))

    lane_ids = _dedupe_strings(lane_ids or screening_meta.get("fit_to_adjacency_box_ids") or screening_meta.get("scope_buckets") or [])
    lane_labels = _dedupe_strings(lane_labels or screening_meta.get("fit_to_adjacency_box_labels") or screening_meta.get("capability_signals") or [])
    query_families = _dedupe_strings(query_families)
    source_families = _dedupe_strings(source_families)
    origin_types = _dedupe_strings(origin_types or screening_meta.get("origin_types") or [])
    discovery_sources = _dedupe_strings(discovery_sources or source_summary.get("source_urls") or [])

    official_site = (
        str(getattr(screening, "candidate_official_website", "") or "").strip()
        or str(getattr(entity, "canonical_website", "") or "").strip()
        or None
    )
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
    priority_score = float(getattr(screening, "total_score", 0.0) or 0.0)
    priority_score += min(6.0, float(multi_origin_count))
    priority_score += min(4.0, float(source_count))
    if official_site:
        priority_score += 4.0
    identity_confidence = str(
        entity_validation.get("identity_confidence") or getattr(entity, "identity_confidence", "") or ""
    ).strip() or None
    if identity_confidence == "high":
        priority_score += 3.0

    diagnostics = identity_diagnostics_from_context(
        entity=entity,
        screening_meta=screening_meta,
        entity_validation=entity_validation,
    )
    if diagnostics["has_first_party_evidence"]:
        priority_score += 2.0

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
        "priority_score": round(priority_score, 3),
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

    sorted_items = sorted(
        items,
        key=lambda item: (
            CLASSIFICATION_RANK.get(str(item.get("decision_classification") or "insufficient_evidence"), 99),
            -float(item.get("priority_score") or 0.0),
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
