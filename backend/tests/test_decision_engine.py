from app.services.decision_engine import evaluate_decision


def test_good_target_classification_when_required_evidence_is_sufficient():
    claims = [
        {"claim_group": "identity_scope", "claim_status": "fact"},
        {"claim_group": "vertical_workflow", "claim_status": "fact"},
        {"claim_group": "product_depth", "claim_status": "fact"},
        {"claim_group": "product_depth", "claim_status": "fact"},
        {"claim_group": "traction", "claim_status": "fact"},
        {"claim_group": "ecosystem_defensibility", "claim_status": "fact"},
    ]
    decision = evaluate_decision(
        screening_status="kept",
        reject_reasons=[],
        claims=claims,
        component_scores={"enterprise_gtm": 0.8, "defensibility_moat": 0.7},
        source_type_counts={"official_registry_filing": 1, "first_party_website": 3},
    )
    assert decision.classification == "good_target"
    assert decision.evidence_sufficiency == "sufficient"
    assert decision.gating_passed is True
    assert "POS-01" in decision.positive_reason_codes
    assert "POS-02" in decision.positive_reason_codes


def test_reject_reason_forces_not_good_target():
    claims = [
        {"claim_group": "identity_scope", "claim_status": "fact"},
        {"claim_group": "vertical_workflow", "claim_status": "fact"},
        {"claim_group": "product_depth", "claim_status": "fact"},
    ]
    decision = evaluate_decision(
        screening_status="review",
        reject_reasons=["go_to_market_b2c"],
        claims=claims,
        component_scores={},
        source_type_counts={},
    )
    assert decision.classification == "not_good_target"
    assert "REJ-01" in decision.reject_reason_codes


def test_contradictory_claims_mark_evidence_as_contradictory():
    claims = [
        {"claim_group": "identity_scope", "claim_status": "fact", "is_conflicting": False},
        {"claim_group": "vertical_workflow", "claim_status": "fact", "is_conflicting": True, "contradiction_group_id": "icp:1"},
        {"claim_group": "product_depth", "claim_status": "fact", "is_conflicting": False},
        {"claim_group": "traction", "claim_status": "fact", "is_conflicting": False},
        {"claim_group": "ecosystem_defensibility", "claim_status": "fact", "is_conflicting": False},
    ]
    decision = evaluate_decision(
        screening_status="kept",
        reject_reasons=[],
        claims=claims,
        component_scores={},
        source_type_counts={},
    )
    assert decision.evidence_sufficiency == "contradictory"
    assert "CAU-01" in decision.caution_reason_codes
    assert "CAU-07" in decision.caution_reason_codes
    assert decision.unresolved_contradictions_count == 1


def test_score_below_threshold_is_not_exposed_as_user_reason_code():
    decision = evaluate_decision(
        screening_status="rejected",
        reject_reasons=["score_below_threshold"],
        claims=[],
        component_scores={},
        source_type_counts={},
    )
    # Unknown/internal reject reasons collapse to caution-based evidence insufficiency.
    assert "CAU-05" in decision.caution_reason_codes
    assert all(not code.startswith("score_") for code in (decision.caution_reason_codes + decision.reject_reason_codes))
