from app.services.decision_engine import evaluate_decision


def test_tier4_only_evidence_cannot_be_good_target():
    claims = [
        {"claim_group": "identity_scope", "claim_status": "fact"},
        {"claim_group": "vertical_workflow", "claim_status": "fact"},
        {"claim_group": "product_depth", "claim_status": "fact"},
        {"claim_group": "traction", "claim_status": "fact"},
        {"claim_group": "ecosystem_defensibility", "claim_status": "fact"},
    ]
    result = evaluate_decision(
        screening_status="kept",
        reject_reasons=[],
        claims=claims,
        component_scores={},
        source_type_counts={"external_search_snippet": 3},
        policy={"tier4_cannot_justify_good_target": True},
    )
    assert result.classification == "insufficient_evidence"
