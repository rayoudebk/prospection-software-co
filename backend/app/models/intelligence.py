"""Comparator intelligence models for source ingestion, claims, and screening."""
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

    category_tags = Column(JSON, default=list)
    listing_text_snippets = Column(JSON, default=list)
    provenance_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="vendor_mentions")
    source_run = relationship("ComparatorSourceRun")


class VendorScreening(Base):
    """Evidence-weighted screening decision for kept vs rejected candidates."""

    __tablename__ = "vendor_screenings"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True, index=True)

    candidate_name = Column(String(300), nullable=False)
    candidate_website = Column(String(1000), nullable=True)
    screening_status = Column(String(20), nullable=False, index=True)  # kept | rejected
    total_score = Column(Float, nullable=False, default=0.0)

    component_scores_json = Column(JSON, default=dict)
    penalties_json = Column(JSON, default=list)
    reject_reasons_json = Column(JSON, default=list)
    screening_meta_json = Column(JSON, default=dict)
    source_summary_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="vendor_screenings")
    vendor = relationship("Vendor", overlaps="screenings")
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
    confidence = Column(String(20), nullable=False, default="medium")

    numeric_value = Column(Float, nullable=True)
    numeric_unit = Column(String(32), nullable=True)
    period = Column(String(32), nullable=True)
    is_conflicting = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", overlaps="vendor_claims")
    vendor = relationship("Vendor", overlaps="claims")
    screening = relationship("VendorScreening", back_populates="claims")
