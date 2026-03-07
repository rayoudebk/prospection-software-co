"""Quality audit helpers for discovery false-positive/false-negative diagnostics."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE = "fp_low_ticket_without_pricing_evidence"
PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT = "fn_missing_vertical_with_institutional_workflow_text"
PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT = "fp_registry_or_directory_overweight"
PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING = "fn_customer_proof_present_but_thin_grouping"

QUALITY_AUDIT_PATTERN_ORDER = [
    PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE,
    PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT,
    PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT,
    PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING,
]

DEFAULT_QUALITY_AUDIT_THRESHOLDS = {
    PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE: 0,
    PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT: 8,
    PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT: 5,
    PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING: 8,
}

PRICING_CONTEXT_TOKENS = {
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
}

NON_PRICING_TOKENS = {
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
    "investor",
    "investors",
    "valuation",
}

INSTITUTIONAL_TOKENS = {
    "asset manager",
    "asset managers",
    "wealth manager",
    "wealth managers",
    "private bank",
    "private banks",
    "institutional",
    "fund admin",
    "fund administration",
    "custodian",
    "broker",
    "insurance",
    "advisory firm",
    "banks",
    "bank",
}

WORKFLOW_MARKET_TOKENS = {
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
    "securities",
    "trading",
    "fund",
    "funds",
}

CUSTOMER_TOKENS = {
    "customer",
    "customers",
    "client",
    "clients",
    "case study",
    "case studies",
    "client story",
    "client stories",
    "trusted by",
    "adopted by",
    "partners with",
}

CURRENCY_MARKERS = ("$", "usd", "€", "eur", "£", "gbp")


def quality_audit_thresholds_from_settings(settings: Any) -> Dict[str, int]:
    return {
        PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE: int(
            getattr(settings, "audit_max_fp_low_ticket_without_pricing_evidence", 0)
        ),
        PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT: int(
            getattr(settings, "audit_max_fn_missing_vertical_with_institutional_text", 8)
        ),
        PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT: int(
            getattr(settings, "audit_max_fp_registry_or_directory_overweight", 5)
        ),
        PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING: int(
            getattr(settings, "audit_max_fn_customer_proof_but_thin_grouping", 8)
        ),
    }


def _string(value: Any) -> str:
    return str(value or "").strip()


def _screening_value(screening: Any, key: str, default: Any = None) -> Any:
    if isinstance(screening, dict):
        return screening.get(key, default)
    return getattr(screening, key, default)


def _claim_value(claim: Any, key: str, default: Any = None) -> Any:
    if isinstance(claim, dict):
        return claim.get(key, default)
    return getattr(claim, key, default)


def _token_match(text: str, tokens: set[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _has_pricing_claim_evidence(claims: Sequence[Any]) -> bool:
    for claim in claims:
        dimension = _string(_claim_value(claim, "dimension")).lower()
        text = _string(_claim_value(claim, "claim_text"))
        lowered = text.lower()
        if not any(marker in lowered for marker in CURRENCY_MARKERS):
            continue
        has_pricing_context = dimension == "pricing_gtm" or _token_match(lowered, PRICING_CONTEXT_TOKENS)
        if not has_pricing_context:
            continue
        if dimension != "pricing_gtm" and _token_match(lowered, NON_PRICING_TOKENS):
            continue
        return True
    return False


def _has_institutional_workflow_claim(claims: Sequence[Any]) -> bool:
    for claim in claims:
        text = _string(_claim_value(claim, "claim_text"))
        lowered = text.lower()
        if _token_match(lowered, INSTITUTIONAL_TOKENS) and _token_match(lowered, WORKFLOW_MARKET_TOKENS):
            return True
    return False


def _source_type_counts(claims: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for claim in claims:
        source_type = _string(_claim_value(claim, "source_type")).lower() or "unknown"
        counts[source_type] += 1
    return dict(counts)


def _has_customer_proof(claims: Sequence[Any]) -> bool:
    for claim in claims:
        dimension = _string(_claim_value(claim, "dimension")).lower()
        text = _string(_claim_value(claim, "claim_text")).lower()
        if dimension in {"customer", "customers", "case_study"}:
            return True
        if _token_match(text, CUSTOMER_TOKENS):
            return True
    return False


def _is_registry_directory_overweight(screening: Any, claims: Sequence[Any]) -> bool:
    counts = _source_type_counts(claims)
    total = int(sum(counts.values()))
    if total < 3:
        return False
    first_party_count = int(counts.get("first_party_website", 0))
    registry_or_directory = int(counts.get("official_registry_filing", 0)) + int(counts.get("directory_comparator", 0))
    decision = _string(_screening_value(screening, "decision_classification")).lower()
    screening_status = _string(_screening_value(screening, "screening_status")).lower()
    ranking_eligible = bool(_screening_value(screening, "ranking_eligible", False))
    is_positive_path = decision in {"good_target", "borderline_watchlist"} or screening_status in {"kept", "review"} or ranking_eligible
    if not is_positive_path:
        return False
    if first_party_count > 0:
        return False
    return registry_or_directory >= max(2, int(total * 0.6))


def _normalize_thresholds(thresholds: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    merged = dict(DEFAULT_QUALITY_AUDIT_THRESHOLDS)
    for key in QUALITY_AUDIT_PATTERN_ORDER:
        if thresholds is None:
            continue
        try:
            merged[key] = int(thresholds.get(key, merged[key]))  # type: ignore[arg-type]
        except Exception:
            continue
    return merged


def _append_pattern_hit(
    impacted: Dict[int, Dict[str, Any]],
    pattern_hits: Dict[str, list[Dict[str, Any]]],
    pattern_key: str,
    screening_id: int,
    candidate_name: str,
) -> None:
    pattern_hits[pattern_key].append(
        {
            "screening_id": int(screening_id),
            "candidate_name": candidate_name,
        }
    )
    entry = impacted.setdefault(
        int(screening_id),
        {
            "screening_id": int(screening_id),
            "candidate_name": candidate_name,
            "reasons": [],
        },
    )
    reasons = entry["reasons"]
    if pattern_key not in reasons:
        reasons.append(pattern_key)


def normalize_quality_audit_v1(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    run_id = _string(payload.get("run_id"))
    if not run_id:
        return None
    thresholds_input = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    thresholds = _normalize_thresholds(thresholds_input)

    patterns_in = payload.get("patterns") if isinstance(payload.get("patterns"), list) else []
    by_pattern: Dict[str, Dict[str, Any]] = {}
    for row in patterns_in:
        if not isinstance(row, dict):
            continue
        key = _string(row.get("pattern_key"))
        if key not in QUALITY_AUDIT_PATTERN_ORDER:
            continue
        by_pattern[key] = {
            "pattern_key": key,
            "count": max(0, int(row.get("count") or 0)),
            "sample_screening_ids": [
                int(item)
                for item in (row.get("sample_screening_ids") or [])
                if str(item).strip()
            ][:20],
            "sample_candidate_names": [
                _string(item)
                for item in (row.get("sample_candidate_names") or [])
                if _string(item)
            ][:20],
        }

    patterns: list[Dict[str, Any]] = []
    for key in QUALITY_AUDIT_PATTERN_ORDER:
        patterns.append(
            by_pattern.get(
                key,
                {
                    "pattern_key": key,
                    "count": 0,
                    "sample_screening_ids": [],
                    "sample_candidate_names": [],
                },
            )
        )

    impacted_in = payload.get("top_impacted_candidates") if isinstance(payload.get("top_impacted_candidates"), list) else []
    top_impacted_candidates: list[Dict[str, Any]] = []
    for row in impacted_in:
        if not isinstance(row, dict):
            continue
        screening_id = int(row.get("screening_id") or 0)
        candidate_name = _string(row.get("candidate_name"))
        if screening_id <= 0 or not candidate_name:
            continue
        reasons = [
            _string(reason)
            for reason in (row.get("reasons") or [])
            if _string(reason) in QUALITY_AUDIT_PATTERN_ORDER
        ]
        top_impacted_candidates.append(
            {
                "screening_id": screening_id,
                "candidate_name": candidate_name,
                "reasons": reasons,
            }
        )
    top_impacted_candidates = top_impacted_candidates[:30]

    pattern_count_map = {
        row["pattern_key"]: int(row["count"])
        for row in patterns
    }
    computed_pass = all(
        int(pattern_count_map.get(key, 0)) <= int(thresholds.get(key, 0))
        for key in QUALITY_AUDIT_PATTERN_ORDER
    )
    pass_value = bool(payload.get("pass")) if "pass" in payload else computed_pass
    if pass_value != computed_pass:
        pass_value = computed_pass

    return {
        "run_id": run_id,
        "pass": pass_value,
        "patterns": patterns,
        "thresholds": thresholds,
        "top_impacted_candidates": top_impacted_candidates,
        "generated_at": _string(payload.get("generated_at")) or datetime.utcnow().isoformat(),
    }


def build_quality_audit_v1(
    screenings: Sequence[Any],
    claims_by_screening: Mapping[int, Sequence[Any]],
    run_id: str,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_thresholds = _normalize_thresholds(thresholds)
    pattern_hits: Dict[str, list[Dict[str, Any]]] = {
        key: []
        for key in QUALITY_AUDIT_PATTERN_ORDER
    }
    impacted: Dict[int, Dict[str, Any]] = {}

    for screening in screenings:
        screening_id = int(_screening_value(screening, "id", 0) or 0)
        if screening_id <= 0:
            continue
        candidate_name = _string(_screening_value(screening, "candidate_name")) or f"screening_{screening_id}"
        claims = claims_by_screening.get(screening_id, []) or []
        reject_reasons = {
            _string(reason)
            for reason in (_screening_value(screening, "reject_reasons_json", []) or [])
            if _string(reason)
        }
        missing_groups = {
            _string(group)
            for group in (_screening_value(screening, "missing_claim_groups_json", []) or [])
            if _string(group)
        }
        evidence_sufficiency = _string(_screening_value(screening, "evidence_sufficiency")).lower()

        if "low_ticket_public_pricing" in reject_reasons and not _has_pricing_claim_evidence(claims):
            _append_pattern_hit(
                impacted,
                pattern_hits,
                PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE,
                screening_id,
                candidate_name,
            )

        if "vertical_workflow" in missing_groups and _has_institutional_workflow_claim(claims):
            _append_pattern_hit(
                impacted,
                pattern_hits,
                PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT,
                screening_id,
                candidate_name,
            )

        if _is_registry_directory_overweight(screening, claims):
            _append_pattern_hit(
                impacted,
                pattern_hits,
                PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT,
                screening_id,
                candidate_name,
            )

        if evidence_sufficiency == "insufficient" and missing_groups and _has_customer_proof(claims):
            _append_pattern_hit(
                impacted,
                pattern_hits,
                PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING,
                screening_id,
                candidate_name,
            )

    patterns: list[Dict[str, Any]] = []
    pass_all = True
    for key in QUALITY_AUDIT_PATTERN_ORDER:
        hits = pattern_hits.get(key, [])
        count = len(hits)
        threshold = int(normalized_thresholds.get(key, 0))
        if count > threshold:
            pass_all = False
        patterns.append(
            {
                "pattern_key": key,
                "count": count,
                "sample_screening_ids": [int(row["screening_id"]) for row in hits[:20]],
                "sample_candidate_names": [_string(row["candidate_name"]) for row in hits[:20]],
            }
        )

    impacted_rows = list(impacted.values())
    impacted_rows.sort(
        key=lambda row: (
            -len(row.get("reasons") or []),
            int(row.get("screening_id") or 0),
        )
    )
    top_impacted = [
        {
            "screening_id": int(row.get("screening_id") or 0),
            "candidate_name": _string(row.get("candidate_name")),
            "reasons": [
                _string(reason)
                for reason in (row.get("reasons") or [])
                if _string(reason) in QUALITY_AUDIT_PATTERN_ORDER
            ],
        }
        for row in impacted_rows[:30]
        if int(row.get("screening_id") or 0) > 0 and _string(row.get("candidate_name"))
    ]

    return {
        "run_id": _string(run_id),
        "pass": bool(pass_all),
        "patterns": patterns,
        "thresholds": normalized_thresholds,
        "top_impacted_candidates": top_impacted,
        "generated_at": datetime.utcnow().isoformat(),
    }

