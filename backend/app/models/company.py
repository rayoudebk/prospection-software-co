"""Company models for target sourcing workflows."""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class CompanyStatus(enum.Enum):
    candidate = "candidate"
    kept = "kept"
    removed = "removed"
    enriched = "enriched"


class Company(Base):
    """A potential acquisition target within a workspace."""
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    
    # Basic info
    name = Column(String(255), nullable=False)
    website = Column(String(500), nullable=True)
    hq_country = Column(String(100), nullable=True)
    operating_countries = Column(JSON, default=list)  # ["UK", "DE", "FR", ...]
    
    # Tags/classification
    tags_custom = Column(JSON, default=list)  # User-defined tags
    
    # Status
    status = Column(Enum(CompanyStatus, name="companystatus"), default=CompanyStatus.candidate)
    
    # Why relevant (from discovery)
    why_relevant = Column(JSON, default=list)  # [{"text": "...", "citation_url": "..."}, ...]
    
    # Manually added?
    is_manual = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    workspace = relationship("Workspace", back_populates="companies")
    dossiers = relationship("CompanyDossier", back_populates="company", cascade="all, delete-orphan")
    evidence_items = relationship("SourceEvidence", back_populates="company", cascade="all, delete-orphan")
    facts = relationship("CompanyFact", back_populates="company", cascade="all, delete-orphan")
    screenings = relationship("CompanyScreening", back_populates="company", cascade="all, delete-orphan")
    claims = relationship("CompanyClaim", back_populates="company", cascade="all, delete-orphan")


class CompanyDossier(Base):
    """Versioned enrichment data for a company."""
    __tablename__ = "company_dossiers"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    
    # The full dossier as JSON. The canonical structure is organized into
    # evidence-backed company-card buckets:
    # {
    #   "workflow": [{"text": "...", "evidence_url": "..."}],
    #   "customer": [{"text": "...", "evidence_url": "..."}],
    #   "business_model": [{"text": "...", "evidence_url": "..."}],
    #   "ownership": [{"text": "...", "evidence_url": "..."}],
    #   "transaction_feasibility": [{"text": "...", "evidence_url": "..."}],
    #   "kpis": {
    #     "revenue": {"value": "...", "period": "...", "confidence": "...", "evidence_url": "..."},
    #     "net_income": {...},
    #     "employee_count": {...},
    #     "debt": {...},
    #     "retained_earnings": {...},
    #     "book_value": {...}
    #   },
    #   "modules": [...],        # legacy compatibility
    #   "customers": [...],      # legacy compatibility
    #   "hiring": {...},         # legacy compatibility
    #   "integrations": [...]    # legacy compatibility
    # }
    dossier_json = Column(JSON, nullable=False)
    
    # Version tracking
    version = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    company = relationship("Company", back_populates="dossiers")

    @property
    def modules(self):
        return self.dossier_json.get("modules", []) if self.dossier_json else []

    @property
    def workflow(self):
        return self.dossier_json.get("workflow", []) if self.dossier_json else []
    
    @property
    def customers(self):
        return self.dossier_json.get("customers", []) if self.dossier_json else []

    @property
    def customer(self):
        return self.dossier_json.get("customer", []) if self.dossier_json else []
    
    @property
    def hiring(self):
        return self.dossier_json.get("hiring", {}) if self.dossier_json else {}

    @property
    def business_model(self):
        return self.dossier_json.get("business_model", []) if self.dossier_json else []
    
    @property
    def integrations(self):
        return self.dossier_json.get("integrations", []) if self.dossier_json else []

    @property
    def ownership(self):
        return self.dossier_json.get("ownership", []) if self.dossier_json else []

    @property
    def transaction_feasibility(self):
        return self.dossier_json.get("transaction_feasibility", []) if self.dossier_json else []

    @property
    def kpis(self):
        return self.dossier_json.get("kpis", {}) if self.dossier_json else {}
