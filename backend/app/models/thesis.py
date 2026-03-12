"""Workspace thesis and search-lane persistence models."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.models.base import Base


class BuyerThesisPack(Base):
    __tablename__ = "buyer_thesis_packs"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, unique=True, index=True)

    summary = Column(Text, nullable=True)
    claims_json = Column(JSON, default=list)
    source_pills_json = Column(JSON, default=list)
    open_questions_json = Column(JSON, default=list)
    market_map_brief_json = Column(JSON, default=dict)
    taxonomy_nodes_json = Column(JSON, default=list)
    taxonomy_edges_json = Column(JSON, default=list)
    lens_seeds_json = Column(JSON, default=list)

    generated_at = Column(DateTime, nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="thesis_pack")


class SearchLane(Base):
    __tablename__ = "search_lanes"
    __table_args__ = (
        UniqueConstraint("workspace_id", "lane_type", name="uq_search_lanes_workspace_lane_type"),
    )

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)

    lane_type = Column(String(32), nullable=False)
    title = Column(String(255), nullable=False)
    intent = Column(Text, nullable=True)
    capabilities_json = Column(JSON, default=list)
    customer_tags_json = Column(JSON, default=list)
    must_include_terms_json = Column(JSON, default=list)
    must_exclude_terms_json = Column(JSON, default=list)
    seed_urls_json = Column(JSON, default=list)
    status = Column(String(32), nullable=False, default="draft")

    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="search_lanes")
