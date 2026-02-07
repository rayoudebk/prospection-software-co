"""Report snapshot and fact models for static M&A radar output."""
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.models.base import Base


class ReportSnapshot(Base):
    """Immutable report snapshot generated for a workspace."""

    __tablename__ = "report_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    filters_json = Column(JSON, default=dict)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(32), default="completed", nullable=False)
    coverage_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="report_snapshots")
    items = relationship(
        "ReportSnapshotItem",
        back_populates="report_snapshot",
        cascade="all, delete-orphan",
    )


class ReportSnapshotItem(Base):
    """A scored vendor item inside a report snapshot."""

    __tablename__ = "report_snapshot_items"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("report_snapshots.id"), nullable=False, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False, index=True)

    compete_score = Column(Float, default=0.0)
    complement_score = Column(Float, default=0.0)
    lens_breakdown_json = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    report_snapshot = relationship("ReportSnapshot", back_populates="items")
    vendor = relationship("Vendor")


class VendorFact(Base):
    """Source-backed normalized facts associated with a vendor."""

    __tablename__ = "vendor_facts"

    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False, index=True)

    fact_key = Column(String(100), nullable=False, index=True)
    fact_value = Column(String(255), nullable=False)
    fact_unit = Column(String(50), nullable=True)
    period = Column(String(50), nullable=True)
    confidence = Column(String(20), default="medium", nullable=False)

    source_evidence_id = Column(Integer, ForeignKey("workspace_evidence.id"), nullable=True)
    source_system = Column(String(100), nullable=False, default="unknown")

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    vendor = relationship("Vendor", back_populates="facts")
    source_evidence = relationship("WorkspaceEvidence")
