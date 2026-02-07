"""Job model - Enhanced job types for workspace-based workflow."""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class JobType(enum.Enum):
    # Context & taxonomy
    context_pack = "context_pack"
    build_taxonomy = "build_taxonomy"
    
    # Discovery
    discovery_universe = "discovery_universe"
    
    # Enrichment
    enrich_modules = "enrich_modules"
    enrich_customers = "enrich_customers"
    enrich_hiring = "enrich_hiring"
    enrich_full = "enrich_full"  # All enrichment in one

    # Static report generation
    generate_report_snapshot = "generate_report_snapshot"
    
    # Legacy (keep for backwards compat)
    landscape = "landscape"
    deep_profile = "deep_profile"


class JobState(enum.Enum):
    queued = "queued"
    running = "running"
    polling = "polling"  # For async providers
    completed = "completed"
    failed = "failed"


class JobProvider(enum.Enum):
    gemini_flash = "gemini_flash"
    gemini_deep_research = "gemini_deep_research"
    crawler = "crawler"


class Job(Base):
    """Research/enrichment job for workspace workflow."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True)  # Nullable for workspace-level jobs
    
    job_type = Column(Enum(JobType), nullable=False)
    state = Column(Enum(JobState), default=JobState.queued)
    provider = Column(Enum(JobProvider), default=JobProvider.gemini_flash)
    
    # For async providers (Gemini Interactions API, etc.)
    interaction_id = Column(String(255), nullable=True)
    
    # Results
    result_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Progress tracking (0.0 - 1.0)
    progress = Column(Float, default=0.0)
    progress_message = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    # Relationships
    workspace = relationship("Workspace", back_populates="jobs")
    vendor = relationship("Vendor")
