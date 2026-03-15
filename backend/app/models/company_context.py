"""Workspace company-context persistence models."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base


class CompanyContextPack(Base):
    __tablename__ = "company_context_packs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, unique=True, index=True)

    market_map_brief_json = Column(JSON, default=dict)
    expansion_brief_json = Column(JSON, default=dict)
    taxonomy_nodes_json = Column(JSON, default=list)
    taxonomy_edges_json = Column(JSON, default=list)
    lens_seeds_json = Column(JSON, default=list)
    company_context_graph_ref = Column(String(255), nullable=True, index=True)
    company_context_graph_cache_json = Column(JSON, default=dict)
    graph_sync_status = Column(String(32), nullable=False, default="not_synced")
    graph_sync_error = Column(Text, nullable=True)
    graph_stats_json = Column(JSON, default=dict)
    graph_synced_at = Column(DateTime, nullable=True)

    generated_at = Column(DateTime, nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="company_context_pack")
