from app.services.quality_audit import (
    PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING,
    PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT,
    PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE,
    PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT,
    build_quality_audit_v1,
    normalize_quality_audit_v1,
)


def _pattern_count(audit: dict, key: str) -> int:
    for row in audit.get("patterns", []):
        if row.get("pattern_key") == key:
            return int(row.get("count") or 0)
    return 0


def test_quality_audit_flags_low_ticket_without_pricing_context():
    screenings = [
        {
            "id": 101,
            "candidate_name": "Upvest",
            "reject_reasons_json": ["low_ticket_public_pricing"],
            "missing_claim_groups_json": ["vertical_workflow"],
            "decision_classification": "not_good_target",
            "screening_status": "review",
            "evidence_sufficiency": "insufficient",
            "ranking_eligible": False,
        }
    ]
    claims_by_screening = {
        101: [
            {
                "dimension": "product",
                "claim_text": "Upvest closes a €30m fundraising round with participation from institutional investors.",
                "source_type": "first_party_website",
            }
        ]
    }
    audit = build_quality_audit_v1(
        screenings=screenings,
        claims_by_screening=claims_by_screening,
        run_id="20260212T151814",
    )
    assert audit["pass"] is False
    assert _pattern_count(audit, PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE) == 1


def test_quality_audit_uses_pricing_context_to_avoid_false_positive():
    screenings = [
        {
            "id": 102,
            "candidate_name": "Priced Co",
            "reject_reasons_json": ["low_ticket_public_pricing"],
            "missing_claim_groups_json": [],
            "decision_classification": "not_good_target",
            "screening_status": "rejected",
            "evidence_sufficiency": "insufficient",
            "ranking_eligible": False,
        }
    ]
    claims_by_screening = {
        102: [
            {
                "dimension": "pricing_gtm",
                "claim_text": "Plans start at €49/month with online self-serve onboarding.",
                "source_type": "first_party_website",
            }
        ]
    }
    audit = build_quality_audit_v1(
        screenings=screenings,
        claims_by_screening=claims_by_screening,
        run_id="20260212T151814",
    )
    assert _pattern_count(audit, PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE) == 0


def test_quality_audit_flags_vertical_and_registry_customer_patterns():
    screenings = [
        {
            "id": 201,
            "candidate_name": "Allvue Systems",
            "reject_reasons_json": [],
            "missing_claim_groups_json": ["vertical_workflow"],
            "decision_classification": "good_target",
            "screening_status": "kept",
            "evidence_sufficiency": "insufficient",
            "ranking_eligible": True,
        }
    ]
    claims_by_screening = {
        201: [
            {
                "dimension": "product",
                "claim_text": "Platform adopted by asset managers and banks for portfolio workflows.",
                "source_type": "directory_comparator",
            },
            {
                "dimension": "customer",
                "claim_text": "Client story: major private bank migrated to the platform.",
                "source_type": "official_registry_filing",
            },
            {
                "dimension": "company_profile",
                "claim_text": "Registry confirms legal entity status.",
                "source_type": "official_registry_filing",
            },
        ]
    }
    audit = build_quality_audit_v1(
        screenings=screenings,
        claims_by_screening=claims_by_screening,
        run_id="20260212T151814",
    )
    assert _pattern_count(audit, PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT) == 1
    assert _pattern_count(audit, PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT) == 1
    assert _pattern_count(audit, PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING) == 1


def test_normalize_quality_audit_v1_repairs_and_validates_shape():
    normalized = normalize_quality_audit_v1(
        {
            "run_id": "run-1",
            "pass": True,
            "patterns": [
                {
                    "pattern_key": PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE,
                    "count": 0,
                    "sample_screening_ids": [1, "2"],
                    "sample_candidate_names": ["Alpha"],
                }
            ],
            "thresholds": {
                PATTERN_FP_LOW_TICKET_WITHOUT_PRICING_EVIDENCE: 0,
                PATTERN_FN_MISSING_VERTICAL_WITH_INSTITUTIONAL_TEXT: 8,
                PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT: 5,
                PATTERN_FN_CUSTOMER_PROOF_BUT_THIN_GROUPING: 8,
            },
            "top_impacted_candidates": [
                {"screening_id": 10, "candidate_name": "Alpha", "reasons": [PATTERN_FP_REGISTRY_OR_DIRECTORY_OVERWEIGHT]}
            ],
        }
    )
    assert normalized is not None
    assert normalized["run_id"] == "run-1"
    assert isinstance(normalized["patterns"], list)
    assert isinstance(normalized["thresholds"], dict)
    assert normalized["top_impacted_candidates"][0]["candidate_name"] == "Alpha"
