"""Workspace models - Core entity for M&A market maps."""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Enum, Boolean
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
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    company_profile = relationship("CompanyProfile", back_populates="workspace", uselist=False, cascade="all, delete-orphan")
    brick_taxonomy = relationship("BrickTaxonomy", back_populates="workspace", uselist=False, cascade="all, delete-orphan")
    vendors = relationship("Vendor", back_populates="workspace", cascade="all, delete-orphan")
    evidence_items = relationship("WorkspaceEvidence", back_populates="workspace", cascade="all, delete-orphan")
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
    vendor_mentions = relationship(
        "VendorMention",
        cascade="all, delete-orphan",
    )
    vendor_screenings = relationship(
        "VendorScreening",
        cascade="all, delete-orphan",
    )
    vendor_claims = relationship(
        "VendorClaim",
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
    
    # Reference vendors (2-3 examples)
    reference_vendor_urls = Column(JSON, default=list)  # ["https://vendor1.com", ...]
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


# Default bricks for fintech/wealthtech
DEFAULT_BRICKS = [
    "PMS",
    "OMS",
    "Pre-trade compliance",
    "Post-trade compliance",
    "Risk & limits",
    "Performance & attribution",
    "Client reporting",
    "Data hub/IBOR/positions",
    "Market data",
    "Connectivity (FIX/SWIFT/custodians)",
    "Corporate actions",
    "Fund admin/TA",
    "Accounting/GL/reg reporting",
    "Digital wealth channels",
    "Workflow/approvals/audit",
]


class BrickTaxonomy(Base):
    """1:1 with Workspace - the capability building blocks."""
    __tablename__ = "brick_taxonomies"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, unique=True)
    
    # Bricks as JSON array of objects: [{"id": "uuid", "name": "PMS", "description": "..."}, ...]
    bricks = Column(JSON, default=list)
    
    # Priority brick IDs (subset of bricks)
    priority_brick_ids = Column(JSON, default=list)
    
    # Vertical focus (used for discovery filtering)
    vertical_focus = Column(JSON, default=list)  # ["asset_manager", "wealth_manager", ...]
    
    # Version for tracking changes
    version = Column(Integer, default=1)
    confirmed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    workspace = relationship("Workspace", back_populates="brick_taxonomy")


class BrickMapping(Base):
    """Maps vendors (including buyer/references) to bricks with evidence."""
    __tablename__ = "brick_mappings"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    
    # Can be vendor_id OR reference company URL
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True)
    reference_url = Column(String(500), nullable=True)  # For buyer/reference companies
    
    brick_id = Column(String(100), nullable=False)  # References brick in taxonomy
    
    # Evidence for this mapping
    evidence_ids = Column(JSON, default=list)  # IDs of WorkspaceEvidence items
    confidence = Column(String(20), default="inferred")  # verified, inferred, weak
    
    created_at = Column(DateTime, default=datetime.utcnow)
