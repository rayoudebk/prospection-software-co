"""Vendor models - Replaces Target for workspace-based approach."""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class VendorStatus(enum.Enum):
    candidate = "candidate"
    kept = "kept"
    removed = "removed"
    enriched = "enriched"


class Vendor(Base):
    """A potential acquisition target within a workspace."""
    __tablename__ = "vendors"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    
    # Basic info
    name = Column(String(255), nullable=False)
    website = Column(String(500), nullable=True)
    hq_country = Column(String(100), nullable=True)
    operating_countries = Column(JSON, default=list)  # ["UK", "DE", "FR", ...]
    
    # Tags/classification
    tags_vertical = Column(JSON, default=list)  # ["asset_manager", "wealth_manager", ...]
    tags_custom = Column(JSON, default=list)  # User-defined tags
    
    # Status
    status = Column(Enum(VendorStatus), default=VendorStatus.candidate)
    
    # Why relevant (from discovery)
    why_relevant = Column(JSON, default=list)  # [{"text": "...", "citation_url": "..."}, ...]
    
    # Manually added?
    is_manual = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    workspace = relationship("Workspace", back_populates="vendors")
    dossiers = relationship("VendorDossier", back_populates="vendor", cascade="all, delete-orphan")
    evidence_items = relationship("WorkspaceEvidence", back_populates="vendor", cascade="all, delete-orphan")
    facts = relationship("VendorFact", back_populates="vendor", cascade="all, delete-orphan")
    screenings = relationship("VendorScreening", cascade="all, delete-orphan")
    claims = relationship("VendorClaim", cascade="all, delete-orphan")


class VendorDossier(Base):
    """Versioned enrichment data for a vendor."""
    __tablename__ = "vendor_dossiers"

    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    
    # The full dossier as JSON
    # Schema:
    # {
    #   "modules": [{"name": "...", "brick_id": "...", "evidence_urls": [...]}],
    #   "customers": [{"name": "...", "context": "case_study", "evidence_url": "..."}],
    #   "hiring": {"postings": [...], "mix_summary": {...}},
    #   "integrations": [{"name": "...", "type": "...", "evidence_url": "..."}]
    # }
    dossier_json = Column(JSON, nullable=False)
    
    # Version tracking
    version = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    vendor = relationship("Vendor", back_populates="dossiers")

    @property
    def modules(self):
        return self.dossier_json.get("modules", []) if self.dossier_json else []
    
    @property
    def customers(self):
        return self.dossier_json.get("customers", []) if self.dossier_json else []
    
    @property
    def hiring(self):
        return self.dossier_json.get("hiring", {}) if self.dossier_json else {}
    
    @property
    def integrations(self):
        return self.dossier_json.get("integrations", []) if self.dossier_json else []
