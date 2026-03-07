from datetime import datetime, timedelta

from app.services.evidence_policy import (
    claim_group_for_dimension,
    infer_source_kind,
    infer_source_tier,
    is_fresh,
    normalize_policy,
    valid_through_from_claim_group,
)


def test_infer_source_tier_for_registry_vendor_and_directory():
    assert (
        infer_source_tier(
            "https://find-and-update.company-information.service.gov.uk/company/12345678",
            "official_registry_filing",
            "example.com",
        )
        == "tier0_registry"
    )
    assert (
        infer_source_tier(
            "https://example.com/security",
            "first_party_website",
            "example.com",
        )
        == "tier1_vendor"
    )
    assert (
        infer_source_tier(
            "https://www.thewealthmosaic.com/vendors/acme",
            "directory_comparator",
            "acme.io",
        )
        == "tier4_discovery"
    )


def test_infer_source_kind_from_tier_logic():
    assert infer_source_kind("https://gleif.org/en", "official_registry_filing", "foo.com") == "registry"
    assert infer_source_kind("https://foo.com/docs", "first_party_website", "foo.com") == "first_party"
    assert infer_source_kind("https://bar.com/listing", "directory_comparator", "foo.com") == "directory"


def test_claim_group_and_freshness_helpers():
    assert claim_group_for_dimension("icp") == "vertical_workflow"
    assert claim_group_for_dimension("integration") == "ecosystem_defensibility"
    ttl_days, valid_through = valid_through_from_claim_group("traction")
    assert ttl_days > 0
    assert is_fresh(valid_through) is True
    assert is_fresh(datetime.utcnow() - timedelta(days=1)) is False


def test_normalize_policy_preserves_defaults_and_overrides():
    policy = normalize_policy(
        {
            "claim_group_ttl_days": {"traction": 999},
            "gate_requirements": {"universe": {"min_decision_qualified_vendors": 9}},
            "tier4_cannot_justify_good_target": True,
        }
    )
    assert policy["claim_group_ttl_days"]["traction"] == 999
    assert policy["gate_requirements"]["universe"]["min_decision_qualified_vendors"] == 9
    # Defaults remain for untouched sections.
    assert "context_pack" in policy["gate_requirements"]
