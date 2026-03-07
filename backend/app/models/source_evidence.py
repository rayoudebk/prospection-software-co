"""Source evidence model."""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base


class SourceEvidence(Base):
    """Evidence items for a workspace with optional company linkage."""
    __tablename__ = "source_evidence"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)  # Nullable for workspace-level evidence
    
    # Source information
    source_url = Column(String(1000), nullable=False)
    source_title = Column(String(500), nullable=True)
    
    # Content
    content_type = Column(String(50), default="web")  # web, pdf, registry, careers, case_study
    excerpt_text = Column(Text, nullable=True)
    
    # Metadata
    captured_at = Column(DateTime, default=datetime.utcnow)
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    freshness_ttl_days = Column(Integer, nullable=True)
    valid_through = Column(DateTime, nullable=True)
    source_tier = Column(
        String(32),
        nullable=False,
        default="tier3_third_party",
        index=True,
    )  # tier0_registry|tier1_vendor|tier2_partner_customer|tier3_third_party|tier4_discovery
    source_kind = Column(
        String(32),
        nullable=False,
        default="third_party",
        index=True,
    )  # registry|first_party|customer_partner|third_party|directory
    asserted_by = Column(String(120), nullable=True)

    # Relationships
    workspace = relationship("Workspace", back_populates="evidence_items")
    company = relationship("Company", back_populates="evidence_items")
