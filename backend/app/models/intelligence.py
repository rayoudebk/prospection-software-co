"""Comparator intelligence models for source ingestion, canonical entities, claims, and screening."""
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship

from app.models.base import Base


class ComparatorSourceRun(Base):
    """Metadata for a single comparator source ingestion run."""

    __tablename__ = "comparator_source_runs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)

    source_name = Column(String(120), nullable=False, index=True)
    source_url = Column(String(1000), nullable=False)
    status = Column(String(40), nullable=False, default="completed")
    mentions_found = Column(Integer, nullable=False, default=0)
    metadata_json = Column(JSON, default=dict)

    captured_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="comparator_source_runs")


class VendorMention(Base):
    """Source mention from directory/comparator listings."""

    __tablename__ = "vendor_mentions"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    source_run_id = Column(Integer, ForeignKey("comparator_source_runs.id"), nullable=True, index=True)

    source_name = Column(String(120), nullable=False, index=True)
    listing_url = Column(String(1000), nullable=False)
    company_name = Column(String(300), nullable=False, index=True)
    company_url = Column(String(1000), nullable=True, index=True)
    profile_url = Column(String(1000), nullable=True, index=True)
    official_website_url = Column(String(1000), nullable=True, index=True)
    company_slug = Column(String(180), nullable=True, index=True)
    solution_slug = Column(String(220), nullable=True, index=True)
    entity_type = Column(String(32), nullable=False, default="company", index=True)  # company|solution|service_line

    category_tags = Column(JSON, default=list)
    listing_text_snippets = Column(JSON, default=list)
    provenance_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="vendor_mentions")
    source_run = relationship("ComparatorSourceRun")


class CandidateEntity(Base):
    """Canonical candidate entity assembled from multi-source discovery signals."""

    __tablename__ = "candidate_entities"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)

    canonical_name = Column(String(300), nullable=False, index=True)
    canonical_website = Column(String(1000), nullable=True)
    canonical_domain = Column(String(255), nullable=True, index=True)
    discovery_primary_url = Column(String(1000), nullable=True)
    entity_type = Column(String(32), nullable=False, default="company", index=True)  # company|solution|service_line
    first_party_domains_json = Column(JSON, default=list)
    solutions_json = Column(JSON, default=list)
    country = Column(String(32), nullable=True, index=True)

    identity_confidence = Column(String(20), nullable=False, default="low")
    identity_error = Column(String(255), nullable=True)

    registry_country = Column(String(16), nullable=True)
    registry_id = Column(String(128), nullable=True, index=True)
    registry_source = Column(String(120), nullable=True)

    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="candidate_entities")
    aliases = relationship(
        "CandidateEntityAlias",
        back_populates="entity",
        cascade="all, delete-orphan",
    )
    origins = relationship(
        "CandidateOriginEdge",
        back_populates="entity",
        cascade="all, delete-orphan",
    )
    screenings = relationship("VendorScreening", back_populates="candidate_entity")


class CandidateEntityAlias(Base):
    """Alias attached to a canonical candidate entity."""

    __tablename__ = "candidate_entity_aliases"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("candidate_entities.id"), nullable=False, index=True)

    alias_name = Column(String(300), nullable=True, index=True)
    alias_website = Column(String(1000), nullable=True, index=True)
    source_name = Column(String(120), nullable=True)
    merge_confidence = Column(Float, nullable=False, default=0.0)
    merge_reason = Column(String(255), nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    entity = relationship("CandidateEntity", back_populates="aliases")


class CandidateOriginEdge(Base):
    """Origin/provenance edge for a canonical candidate entity."""

    __tablename__ = "candidate_origin_edges"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("candidate_entities.id"), nullable=False, index=True)

    origin_type = Column(String(40), nullable=False, index=True)
    origin_url = Column(String(1000), nullable=True)
    source_run_id = Column(Integer, ForeignKey("comparator_source_runs.id"), nullable=True, index=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    entity = relationship("CandidateEntity", back_populates="origins")
    source_run = relationship("ComparatorSourceRun")


class RegistryQueryLog(Base):
    """Per-query registry diagnostics for identity mapping and neighbor expansion."""

    __tablename__ = "registry_query_logs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)

    run_id = Column(String(40), nullable=False, index=True)
    seed_entity_name = Column(String(300), nullable=True, index=True)
    query_type = Column(String(40), nullable=False, index=True)  # identity_map | neighbor_expand
    country = Column(String(16), nullable=False, index=True)  # FR | UK
    source_name = Column(String(120), nullable=False)
    query = Column(String(300), nullable=False)

    raw_hits = Column(Integer, nullable=False, default=0)
    kept_hits = Column(Integer, nullable=False, default=0)
    reject_reasons_json = Column(JSON, default=dict)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="registry_query_logs")


class VendorScreening(Base):
    """Evidence-weighted screening decision for kept vs rejected candidates."""

    __tablename__ = "vendor_screenings"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True, index=True)
    candidate_entity_id = Column(Integer, ForeignKey("candidate_entities.id"), nullable=True, index=True)

    candidate_name = Column(String(300), nullable=False)
    candidate_website = Column(String(1000), nullable=True)
    candidate_discovery_url = Column(String(1000), nullable=True)
    candidate_official_website = Column(String(1000), nullable=True)
    screening_status = Column(String(20), nullable=False, index=True)  # kept | review | rejected
    total_score = Column(Float, nullable=False, default=0.0)

    component_scores_json = Column(JSON, default=dict)
    penalties_json = Column(JSON, default=list)
    reject_reasons_json = Column(JSON, default=list)
    positive_reason_codes_json = Column(JSON, default=list)
    caution_reason_codes_json = Column(JSON, default=list)
    reject_reason_codes_json = Column(JSON, default=list)
    missing_claim_groups_json = Column(JSON, default=list)
    unresolved_contradictions_count = Column(Integer, nullable=False, default=0)
    decision_classification = Column(
        String(40),
        nullable=False,
        default="insufficient_evidence",
        index=True,
    )  # good_target|borderline_watchlist|not_good_target|insufficient_evidence
    evidence_sufficiency = Column(
        String(24),
        nullable=False,
        default="insufficient",
        index=True,
    )  # sufficient|insufficient|contradictory
    rationale_summary = Column(Text, nullable=True)
    rationale_markdown = Column(Text, nullable=True)
    top_claim_json = Column(JSON, default=dict)
    decision_engine_version = Column(String(64), nullable=True)
    gating_passed = Column(Boolean, nullable=False, default=False)
    ranking_eligible = Column(Boolean, nullable=False, default=False)
    screening_meta_json = Column(JSON, default=dict)
    source_summary_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="vendor_screenings")
    vendor = relationship("Vendor", overlaps="screenings")
    candidate_entity = relationship("CandidateEntity", back_populates="screenings")
    claims = relationship(
        "VendorClaim",
        back_populates="screening",
        cascade="all, delete-orphan",
    )


class VendorClaim(Base):
    """Atomic evidence claim tagged by dimension with optional normalized numeric payload."""

    __tablename__ = "vendor_claims"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True, index=True)
    screening_id = Column(Integer, ForeignKey("vendor_screenings.id"), nullable=True, index=True)

    dimension = Column(String(64), nullable=False, index=True)
    claim_key = Column(String(120), nullable=True, index=True)
    claim_text = Column(Text, nullable=False)

    source_url = Column(String(1000), nullable=False)
    source_type = Column(String(64), nullable=False, default="trusted_third_party")
    source_tier = Column(String(32), nullable=False, default="tier3_third_party", index=True)
    source_evidence_id = Column(Integer, ForeignKey("workspace_evidence.id"), nullable=True, index=True)
    confidence = Column(String(20), nullable=False, default="medium")
    claim_group = Column(String(64), nullable=True, index=True)
    claim_status = Column(String(24), nullable=False, default="fact", index=True)  # fact|assumption|unknown|contradicted
    contradiction_group_id = Column(String(120), nullable=True, index=True)
    freshness_ttl_days = Column(Integer, nullable=True)
    valid_through = Column(DateTime, nullable=True)

    numeric_value = Column(Float, nullable=True)
    numeric_unit = Column(String(32), nullable=True)
    period = Column(String(32), nullable=True)
    is_conflicting = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="vendor_claims")
    vendor = relationship("Vendor", overlaps="claims")
    screening = relationship("VendorScreening", back_populates="claims")
    source_evidence = relationship("WorkspaceEvidence")
