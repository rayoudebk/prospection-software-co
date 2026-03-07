from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.models.base import Base


class ExternalSearchRun(Base):
    __tablename__ = "external_search_runs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, index=True)
    run_id = Column(String(80), nullable=False, index=True, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    provider_order = Column(String(200), nullable=False)
    caps_json = Column(JSON, default=dict)
    query_plan_json = Column(JSON, default=dict)
    query_plan_hash = Column(String(64), nullable=True)

    results = relationship(
        "ExternalSearchResult",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class ExternalSearchResult(Base):
    __tablename__ = "external_search_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(80), ForeignKey("external_search_runs.run_id"), nullable=False, index=True)
    provider = Column(String(40), nullable=False)
    query_id = Column(String(80), nullable=False)
    query_type = Column(String(40), nullable=False)
    query_text = Column(String(500), nullable=False)
    rank = Column(Integer, nullable=False)
    url = Column(String(1000), nullable=False)
    url_fingerprint = Column(String(40), nullable=False, index=True)
    domain_fingerprint = Column(String(40), nullable=True, index=True)
    title = Column(String(500), nullable=True)
    snippet = Column(String(2000), nullable=True)
    retrieved_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("ExternalSearchRun", back_populates="results")
