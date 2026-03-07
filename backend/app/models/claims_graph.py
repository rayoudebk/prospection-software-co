"""Normalized claims graph entities and relations for cross-vendor reasoning."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base


class ClaimGraphNode(Base):
    __tablename__ = "claim_graph_nodes"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    node_type = Column(String(64), nullable=False, index=True)  # company|customer|integration|module|workflow
    canonical_name = Column(String(300), nullable=False, index=True)
    external_id = Column(String(120), nullable=True, index=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ClaimGraphEdge(Base):
    __tablename__ = "claim_graph_edges"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False, index=True)
    from_node_id = Column(Integer, ForeignKey("claim_graph_nodes.id"), nullable=False, index=True)
    to_node_id = Column(Integer, ForeignKey("claim_graph_nodes.id"), nullable=False, index=True)

    relation_type = Column(String(80), nullable=False, index=True)  # serves|integrates_with|offers_module|adjacent_to
    confidence = Column(Float, nullable=False, default=0.5)
    evidence_count = Column(Integer, nullable=False, default=0)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    from_node = relationship("ClaimGraphNode", foreign_keys=[from_node_id])
    to_node = relationship("ClaimGraphNode", foreign_keys=[to_node_id])


class ClaimGraphEdgeEvidence(Base):
    __tablename__ = "claim_graph_edge_evidence"

    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(Integer, ForeignKey("claim_graph_edges.id"), nullable=False, index=True)
    claim_id = Column(Integer, ForeignKey("vendor_claims.id"), nullable=True, index=True)
    source_evidence_id = Column(Integer, ForeignKey("workspace_evidence.id"), nullable=True, index=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    edge = relationship("ClaimGraphEdge")
    claim = relationship("VendorClaim")
    source_evidence = relationship("WorkspaceEvidence")

