"""Workspace-era feedback events for decision overrides and rationale tuning."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base


class WorkspaceFeedbackEvent(Base):
    __tablename__ = "workspace_feedback_events"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True, index=True)
    screening_id = Column(Integer, ForeignKey("vendor_screenings.id"), nullable=True, index=True)

    feedback_type = Column(String(64), nullable=False, default="classification_override")
    previous_classification = Column(String(40), nullable=True)
    new_classification = Column(String(40), nullable=True)
    reason_codes_json = Column(JSON, default=list)
    comment = Column(Text, nullable=True)
    metadata_json = Column(JSON, default=dict)

    created_by = Column(String(120), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="feedback_events")
    vendor = relationship("Vendor")
    screening = relationship("VendorScreening")

