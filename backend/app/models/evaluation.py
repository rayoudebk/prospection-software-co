"""Evaluation harness artifacts for precision/recall and stability replays."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text

from app.models.base import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    run_type = Column(String(64), nullable=False, default="gold_set_replay")
    status = Column(String(24), nullable=False, default="completed")
    model_version = Column(String(64), nullable=True)
    metrics_json = Column(JSON, default=dict)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EvaluationSampleResult(Base):
    __tablename__ = "evaluation_sample_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True, index=True)
    expected_classification = Column(String(40), nullable=True)
    predicted_classification = Column(String(40), nullable=True)
    matched = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=True)
    details_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

