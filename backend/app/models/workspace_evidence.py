"""Workspace-level evidence model."""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base


class WorkspaceEvidence(Base):
    """Evidence items for workspace - can be linked to vendor or workspace-level."""
    __tablename__ = "workspace_evidence"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True)  # Nullable for workspace-level evidence
    
    # Source information
    source_url = Column(String(1000), nullable=False)
    source_title = Column(String(500), nullable=True)
    
    # Content
    content_type = Column(String(50), default="web")  # web, pdf, registry, careers, case_study
    excerpt_text = Column(Text, nullable=True)
    
    # Metadata
    captured_at = Column(DateTime, default=datetime.utcnow)
    
    # Which brick(s) this evidence supports
    brick_ids = Column(String(500), nullable=True)  # Comma-separated brick IDs

    # Relationships
    workspace = relationship("Workspace", back_populates="evidence_items")
    vendor = relationship("Vendor", back_populates="evidence_items")
