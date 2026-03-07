"""Workspace models - Core entity for M&A sourcing work."""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class RegionScope(enum.Enum):
    eu_uk = "EU+UK"
    us = "US"
    apac = "APAC"
    global_ = "Global"


class Workspace(Base):
    """Core workspace entity - replaces Strategy."""
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    region_scope = Column(String(50), default="EU+UK")
    decision_policy_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    company_profile = relationship("CompanyProfile", back_populates="workspace", uselist=False, cascade="all, delete-orphan")
    thesis_pack = relationship("BuyerThesisPack", back_populates="workspace", uselist=False, cascade="all, delete-orphan")
    search_lanes = relationship("SearchLane", back_populates="workspace", cascade="all, delete-orphan")
    companies = relationship("Company", back_populates="workspace", cascade="all, delete-orphan")
    evidence_items = relationship("SourceEvidence", back_populates="workspace", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="workspace", cascade="all, delete-orphan")
    report_snapshots = relationship(
        "ReportSnapshot",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    comparator_source_runs = relationship(
        "ComparatorSourceRun",
        cascade="all, delete-orphan",
    )
    company_mentions = relationship(
        "CompanyMention",
        cascade="all, delete-orphan",
    )
    company_screenings = relationship(
        "CompanyScreening",
        cascade="all, delete-orphan",
    )
    company_claims = relationship(
        "CompanyClaim",
        cascade="all, delete-orphan",
    )
    registry_query_logs = relationship(
        "RegistryQueryLog",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    candidate_entities = relationship(
        "CandidateEntity",
        cascade="all, delete-orphan",
    )
    feedback_events = relationship(
        "WorkspaceFeedbackEvent",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )


class CompanyProfile(Base):
    """1:1 with Workspace - buyer context and geo scope."""
    __tablename__ = "company_profiles"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, unique=True)
    
    # Buyer company
    buyer_company_url = Column(String(500), nullable=True)
    buyer_context_summary = Column(Text, nullable=True)  # AI-generated summary of buyer
    
    # Reference companies (2-3 examples)
    reference_company_urls = Column(JSON, default=list)  # ["https://company1.com", ...]
    reference_evidence_urls = Column(JSON, default=list)  # ["https://vendor.com/blog/case-study", ...]
    reference_summaries = Column(JSON, default=dict)  # {"url": "summary", ...}
    
    # Geographic scope
    geo_scope = Column(JSON, default=dict)  # {region: "EU+UK", include_countries: [], exclude_countries: []}
    
    # Context pack content
    context_pack_markdown = Column(Text, nullable=True)
    context_pack_json = Column(JSON, nullable=True)  # Full structured data from crawler
    context_pack_generated_at = Column(DateTime, nullable=True)
    product_pages_found = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    workspace = relationship("Workspace", back_populates="company_profile")
