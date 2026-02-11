"""Evidence policy defaults and helpers (tiers, TTLs, freshness, grouping)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from app.services.reporting import normalize_domain


DEFAULT_EVIDENCE_POLICY: Dict[str, Any] = {
    "version": "evidence_policy_v1",
    "source_tier_hierarchy": [
        "tier0_registry",
        "tier1_vendor",
        "tier2_partner_customer",
        "tier3_third_party",
        "tier4_discovery",
    ],
    "claim_group_ttl_days": {
        "identity_scope": 365,
        "vertical_workflow": 365,
        "product_depth": 365,
        "traction": 730,
        "ecosystem_defensibility": 365,
    },
    "gate_requirements": {
        "context_pack": {"required_claim_groups": ["identity_scope", "product_depth", "vertical_workflow"], "min_required_groups_met": 2},
        "brick_model": {"min_priority_bricks": 3, "require_evidence_mapped_bricks": 2},
        "universe": {"min_decision_qualified_vendors": 5, "allowed_classes": ["good_target", "borderline_watchlist"], "max_insufficient_ratio": 0.5},
        "enrichment": {"min_enriched_vendors": 5, "required_groups_per_vendor": ["product_depth", "traction"]},
    },
    "contradiction_resolution": "higher_tier_preferred_but_visible",
    "tier4_cannot_justify_good_target": True,
}


REGISTRY_HOST_TOKENS = (
    "companieshouse.gov.uk",
    "company-information.service.gov.uk",
    "pappers.fr",
    "infogreffe.fr",
    "inpi.fr",
    "annuaire-entreprises.data.gouv.fr",
    "handelsregister.de",
    "unternehmensregister.de",
    "gleif.org",
)

DIRECTORY_HOST_TOKENS = (
    "thewealthmosaic.com",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
)


def infer_source_tier(source_url: Optional[str], source_type: Optional[str], candidate_domain: Optional[str]) -> str:
    url = str(source_url or "").strip().lower()
    stype = str(source_type or "").strip().lower()
    if any(token in url for token in REGISTRY_HOST_TOKENS) or stype == "official_registry_filing":
        return "tier0_registry"
    if any(token in url for token in DIRECTORY_HOST_TOKENS) or stype == "directory_comparator":
        return "tier4_discovery"
    host = normalize_domain(source_url) or ""
    if candidate_domain and not any(token in host for token in DIRECTORY_HOST_TOKENS) and (host == candidate_domain or host.endswith(f".{candidate_domain}")):
        return "tier1_vendor"
    if stype in {"first_party_partner", "customer_case_study"}:
        return "tier2_partner_customer"
    return "tier3_third_party"


def infer_source_kind(source_url: Optional[str], source_type: Optional[str], candidate_domain: Optional[str]) -> str:
    tier = infer_source_tier(source_url, source_type, candidate_domain)
    if tier == "tier0_registry":
        return "registry"
    if tier == "tier1_vendor":
        return "first_party"
    if tier == "tier2_partner_customer":
        return "customer_partner"
    if tier == "tier4_discovery":
        return "directory"
    return "third_party"


def claim_group_for_dimension(dimension: Optional[str], claim_key: Optional[str] = None) -> str:
    dim = str(dimension or "").strip().lower()
    key = str(claim_key or "").strip().lower()
    if dim in {"company_profile", "registry_identity", "registry_neighbor", "registry_lookup", "identity"}:
        return "identity_scope"
    if dim in {"icp", "target_customer", "vertical", "industry"}:
        return "vertical_workflow"
    if dim in {"product", "capability", "services", "directory_context"}:
        return "product_depth"
    if dim in {"customer", "customers", "case_study"}:
        return "traction"
    if dim in {"integration", "partnership", "moat", "defensibility"}:
        return "ecosystem_defensibility"
    if key in {"employees", "revenue"}:
        return "identity_scope"
    return "product_depth"


def ttl_days_for_claim_group(claim_group: Optional[str], policy: Optional[Dict[str, Any]] = None) -> int:
    cfg = (policy or DEFAULT_EVIDENCE_POLICY).get("claim_group_ttl_days", {})
    try:
        return int(cfg.get(str(claim_group or ""), 365))
    except Exception:
        return 365


def valid_through_from_claim_group(
    claim_group: Optional[str],
    captured_at: Optional[datetime] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Tuple[int, datetime]:
    ttl = ttl_days_for_claim_group(claim_group, policy=policy)
    base = captured_at or datetime.utcnow()
    return ttl, base + timedelta(days=max(1, ttl))


def is_fresh(valid_through: Optional[datetime], as_of: Optional[datetime] = None) -> bool:
    if not valid_through:
        return False
    now = as_of or datetime.utcnow()
    return valid_through >= now


def normalize_policy(raw_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_EVIDENCE_POLICY)
    candidate = raw_policy if isinstance(raw_policy, dict) else {}
    for key, value in candidate.items():
        if key in {"claim_group_ttl_days", "gate_requirements"} and isinstance(value, dict):
            merged[key] = {**merged.get(key, {}), **value}
        else:
            merged[key] = value
    return merged
