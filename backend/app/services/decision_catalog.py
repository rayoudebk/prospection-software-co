"""Decision reason code catalog for buy-side sourcing narratives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ReasonCode:
    code: str
    reason_type: str  # positive|caution|reject
    definition: str
    min_evidence: str
    user_text: str
    source_examples: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "reason_type": self.reason_type,
            "definition": self.definition,
            "min_evidence": self.min_evidence,
            "user_text": self.user_text,
            "source_examples": self.source_examples,
        }


CATALOG: List[ReasonCode] = [
    ReasonCode("POS-01", "positive", "Core institutional vertical fit.", "1 tier-1 + 1 corroborator", "Built for institutional workflows.", ["product page", "customer PR"]),
    ReasonCode("POS-02", "positive", "Depth in priority bricks.", "1 module source + 1 depth artifact", "Real module depth on priority capabilities.", ["docs", "release notes"]),
    ReasonCode("POS-03", "positive", "Enterprise ICP and GTM.", "1 ICP source + 1 enterprise proxy", "Positioned for enterprise financial institutions.", ["ICP page", "case study"]),
    ReasonCode("POS-04", "positive", "Credible named customers.", "2 customer proofs with >=1 non-logo", "Verified adoption by institutional customers.", ["customer PR", "case study"]),
    ReasonCode("POS-05", "positive", "Embedded ecosystem integrations.", "1 integration source + 1 corroborator", "Integrates into incumbent stack.", ["integration docs", "partner marketplace"]),
    ReasonCode("POS-06", "positive", "Trust/compliance posture.", "1 trust/compliance source in TTL", "Bank-grade trust/compliance posture.", ["security page", "certification"]),
    ReasonCode("POS-07", "positive", "Defensible differentiation.", "1 claim + 1 supporting artifact", "Differentiation appears hard to replicate.", ["architecture", "patent"]),
    ReasonCode("POS-08", "positive", "Scalable delivery model.", "1 delivery statement + 1 evidence", "Delivery model looks repeatable.", ["implementation guide", "partner program"]),
    ReasonCode("CAU-01", "caution", "Mixed or ambiguous ICP.", "conflicting ICP signals", "Target segment is mixed; confirm institutional fit.", ["case studies", "pricing page"]),
    ReasonCode("CAU-02", "caution", "Services-heavy execution risk.", "services emphasis + corroborator", "Delivery may be services-led.", ["services page", "job posts"]),
    ReasonCode("CAU-03", "caution", "Platform claim without depth proof.", "platform claim with weak depth evidence", "Platform depth remains weakly evidenced.", ["marketing page", "missing docs"]),
    ReasonCode("CAU-04", "caution", "Customer evidence is logo-only.", "customer claims without non-logo corroboration", "Customer traction is not independently verified.", ["logo wall", "testimonial page"]),
    ReasonCode("CAU-05", "caution", "Evidence sparsity.", "required groups missing", "Insufficient evidence for a confident decision.", ["sparse site", "few third-party references"]),
    ReasonCode("CAU-06", "caution", "Geo/localization mismatch risk.", "primary focus outside EU/UK", "EU/UK delivery coverage is unclear.", ["offices page", "customer geography"]),
    ReasonCode("CAU-07", "caution", "Entity/group ambiguity.", "conflicting legal identity evidence", "Corporate structure needs clarification.", ["registry", "about/legal pages"]),
    ReasonCode("REJ-01", "reject", "B2C/retail focus mismatch.", "consumer-focused evidence + no institutional proof", "Primarily consumer-focused; outside scope.", ["consumer pricing", "app listing"]),
    ReasonCode("REJ-02", "reject", "Consultancy/dev shop profile.", "services-led evidence + no product depth", "Services-led provider, not a product target.", ["services page", "portfolio page"]),
    ReasonCode("REJ-03", "reject", "Out-of-vertical.", "primary domain outside target scope", "Primary domain outside target thesis.", ["product pages", "directory taxonomy"]),
    ReasonCode("REJ-04", "reject", "Not a relevant software product.", "offering type mismatch evidence", "Offering is not a relevant software platform.", ["datasheet", "website"]),
    ReasonCode("REJ-05", "reject", "Non-enterprise micro-focus mismatch.", "SMB/retail emphasis without enterprise evidence", "Enterprise fit unlikely based on customer focus.", ["pricing", "customer pages"]),
    ReasonCode("REJ-06", "reject", "Identity cannot be verified.", "failed legal identity verification", "Legal entity identity cannot be verified.", ["registry checks", "legal pages"]),
    ReasonCode("REJ-07", "reject", "Duplicate/non-distinct target.", "canonical merge indicates duplicate", "Not a distinct standalone target.", ["entity mapping", "group structure"]),
    ReasonCode("REJ-08", "reject", "Material public risk flags.", "credible enforcement/sanction evidence", "Material public risk flags; deprioritized.", ["regulator notices", "major press"]),
]

CATALOG_BY_CODE = {row.code: row for row in CATALOG}


def get_catalog_payload() -> Dict[str, Any]:
    return {
        "version": "decision_catalog_v1",
        "codes": [row.to_dict() for row in CATALOG],
    }


def reason_text(code: str) -> str:
    row = CATALOG_BY_CODE.get(code)
    return row.user_text if row else code

