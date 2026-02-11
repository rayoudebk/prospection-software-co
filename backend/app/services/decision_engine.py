"""Deterministic decision engine for buy-side sourcing classifications."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from app.services.decision_catalog import reason_text
from app.services.evidence_policy import DEFAULT_EVIDENCE_POLICY, normalize_policy


@dataclass
class DecisionResult:
    classification: str
    evidence_sufficiency: str
    positive_reason_codes: List[str]
    caution_reason_codes: List[str]
    reject_reason_codes: List[str]
    missing_claim_groups: List[str]
    unresolved_contradictions_count: int
    rationale_summary: str
    rationale_markdown: str
    decision_engine_version: str
    gating_passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "evidence_sufficiency": self.evidence_sufficiency,
            "positive_reason_codes": self.positive_reason_codes,
            "caution_reason_codes": self.caution_reason_codes,
            "reject_reason_codes": self.reject_reason_codes,
            "missing_claim_groups": self.missing_claim_groups,
            "unresolved_contradictions_count": self.unresolved_contradictions_count,
            "rationale_summary": self.rationale_summary,
            "rationale_markdown": self.rationale_markdown,
            "decision_engine_version": self.decision_engine_version,
            "gating_passed": self.gating_passed,
        }


DECISION_ENGINE_VERSION = "decision_engine_v1"

REJECT_REASON_CODE_MAP = {
    "go_to_market_b2c": "REJ-01",
    "retail_only_icp": "REJ-01",
    "consumer_language_without_institutional_icp": "REJ-01",
    "low_software_heaviness": "REJ-02",
    "public_self_serve_pricing": "REJ-05",
    "low_ticket_public_pricing": "REJ-05",
    "no_trusted_evidence": "CAU-05",
}


def _claim_group_coverage(claims: Iterable[Any]) -> Dict[str, int]:
    counts = {
        "identity_scope": 0,
        "vertical_workflow": 0,
        "product_depth": 0,
        "traction": 0,
        "ecosystem_defensibility": 0,
    }
    for claim in claims:
        group = str(getattr(claim, "claim_group", None) or (claim.get("claim_group") if isinstance(claim, dict) else "")).strip()
        status = str(getattr(claim, "claim_status", None) or (claim.get("claim_status") if isinstance(claim, dict) else "fact")).strip().lower()
        if group in counts and status in {"fact", "assumption"}:
            counts[group] += 1
    return counts


def _contradictions_count(claims: Iterable[Any]) -> int:
    groups = set()
    count = 0
    for claim in claims:
        conflicting = bool(getattr(claim, "is_conflicting", False) if not isinstance(claim, dict) else claim.get("is_conflicting"))
        if not conflicting:
            continue
        count += 1
        group_id = getattr(claim, "contradiction_group_id", None) if not isinstance(claim, dict) else claim.get("contradiction_group_id")
        if group_id:
            groups.add(str(group_id))
    if groups:
        return len(groups)
    return count


def _reason_codes_from_reject_reasons(reject_reasons: Iterable[str]) -> tuple[List[str], List[str]]:
    caution: List[str] = []
    reject: List[str] = []
    for reason in reject_reasons:
        code = REJECT_REASON_CODE_MAP.get(str(reason), "CAU-05")
        if code.startswith("REJ-"):
            reject.append(code)
        else:
            caution.append(code)
    return sorted(set(caution)), sorted(set(reject))


def _positive_reason_codes(
    coverage: Dict[str, int],
    source_type_counts: Dict[str, int],
    component_scores: Dict[str, float],
) -> List[str]:
    positive: List[str] = []
    if coverage.get("vertical_workflow", 0) > 0:
        positive.append("POS-01")
    if coverage.get("product_depth", 0) >= 2:
        positive.append("POS-02")
    if component_scores.get("enterprise_gtm", 0.0) >= 0.5:
        positive.append("POS-03")
    if coverage.get("traction", 0) >= 2:
        positive.append("POS-04")
    if coverage.get("ecosystem_defensibility", 0) > 0:
        positive.append("POS-05")
    if source_type_counts.get("official_registry_filing", 0) > 0:
        positive.append("POS-06")
    if component_scores.get("defensibility_moat", 0.0) >= 0.5:
        positive.append("POS-07")
    if component_scores.get("services_implementation_complexity", 0.0) >= 0.4:
        positive.append("POS-08")
    return sorted(set(positive))


def _missing_claim_groups(coverage: Dict[str, int], required: Iterable[str]) -> List[str]:
    return [group for group in required if coverage.get(group, 0) <= 0]


def _build_summary(
    classification: str,
    positive_codes: List[str],
    caution_codes: List[str],
    reject_codes: List[str],
    missing: List[str],
    contradictions: int,
) -> tuple[str, str]:
    header_map = {
        "good_target": "Good target",
        "borderline_watchlist": "Borderline/watchlist",
        "not_good_target": "Not a good target",
        "insufficient_evidence": "Insufficient evidence",
    }
    header = header_map.get(classification, classification)
    highlights = [reason_text(code) for code in (positive_codes[:3] + caution_codes[:2] + reject_codes[:2])]
    if not highlights:
        highlights = ["Evidence needs additional validation."]
    summary = f"{header}: {highlights[0]}"
    markdown_lines = [f"### {header}", "", "Why:", ""]
    for item in highlights[:5]:
        markdown_lines.append(f"- {item}")
    if missing:
        markdown_lines.extend(["", "Known unknowns:", ""])
        for group in missing:
            markdown_lines.append(f"- Missing claim group: {group}")
    if contradictions > 0:
        markdown_lines.extend(["", f"- Unresolved contradictions: {contradictions}"])
    return summary[:480], "\n".join(markdown_lines)[:4000]


def evaluate_decision(
    screening_status: str,
    reject_reasons: List[str],
    claims: Iterable[Any],
    component_scores: Optional[Dict[str, float]] = None,
    source_type_counts: Optional[Dict[str, int]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> DecisionResult:
    effective_policy = normalize_policy(policy or DEFAULT_EVIDENCE_POLICY)
    component_scores = component_scores or {}
    source_type_counts = source_type_counts or {}
    coverage = _claim_group_coverage(claims)
    contradictions = _contradictions_count(claims)

    required_groups = ["identity_scope", "vertical_workflow", "product_depth"]
    missing = _missing_claim_groups(coverage, required_groups)

    if contradictions > 0:
        evidence_sufficiency = "contradictory"
    elif len(missing) == 0 and sum(coverage.values()) >= 5:
        evidence_sufficiency = "sufficient"
    else:
        evidence_sufficiency = "insufficient"

    caution_codes, reject_codes = _reason_codes_from_reject_reasons(reject_reasons)
    positive_codes = _positive_reason_codes(coverage, source_type_counts, component_scores)

    if evidence_sufficiency == "contradictory":
        caution_codes = sorted(set(caution_codes + ["CAU-01", "CAU-07"]))
    if evidence_sufficiency == "insufficient":
        caution_codes = sorted(set(caution_codes + ["CAU-05"]))
    if source_type_counts.get("directory_comparator", 0) > 0 and source_type_counts.get("first_party_website", 0) == 0:
        caution_codes = sorted(set(caution_codes + ["CAU-03"]))
    if "REJ-01" in reject_codes or "REJ-02" in reject_codes:
        positive_codes = []

    if reject_codes:
        classification = "not_good_target"
    elif screening_status == "kept" and evidence_sufficiency == "sufficient":
        classification = "good_target"
    elif screening_status == "review":
        classification = "borderline_watchlist"
    elif evidence_sufficiency == "insufficient":
        classification = "insufficient_evidence"
    else:
        classification = "borderline_watchlist"

    gate_requirements = effective_policy.get("gate_requirements", {}).get("universe", {})
    allowed = set(gate_requirements.get("allowed_classes", ["good_target", "borderline_watchlist"]))
    gating_passed = classification in allowed and evidence_sufficiency in {"sufficient", "contradictory"}
    if classification == "good_target" and evidence_sufficiency != "sufficient":
        gating_passed = False

    summary, markdown = _build_summary(
        classification,
        positive_codes,
        caution_codes,
        reject_codes,
        missing,
        contradictions,
    )

    return DecisionResult(
        classification=classification,
        evidence_sufficiency=evidence_sufficiency,
        positive_reason_codes=positive_codes,
        caution_reason_codes=caution_codes,
        reject_reason_codes=reject_codes,
        missing_claim_groups=missing,
        unresolved_contradictions_count=contradictions,
        rationale_summary=summary,
        rationale_markdown=markdown,
        decision_engine_version=DECISION_ENGINE_VERSION,
        gating_passed=gating_passed,
    )
