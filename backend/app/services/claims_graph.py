"""Claims graph builder from vendor claims and evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from sqlalchemy.orm import Session

from app.models.claims_graph import ClaimGraphEdge, ClaimGraphEdgeEvidence, ClaimGraphNode
from app.models.intelligence import VendorClaim


def _node_key(node_type: str, name: str) -> str:
    return f"{node_type}:{name.strip().lower()}"


def rebuild_workspace_claims_graph(db: Session, workspace_id: int) -> Dict[str, int]:
    claims = (
        db.query(VendorClaim)
        .filter(VendorClaim.workspace_id == workspace_id)
        .order_by(VendorClaim.created_at.desc())
        .all()
    )

    db.query(ClaimGraphEdgeEvidence).filter(
        ClaimGraphEdgeEvidence.edge_id.in_(
            db.query(ClaimGraphEdge.id).filter(ClaimGraphEdge.workspace_id == workspace_id)
        )
    ).delete(synchronize_session=False)
    db.query(ClaimGraphEdge).filter(ClaimGraphEdge.workspace_id == workspace_id).delete(synchronize_session=False)
    db.query(ClaimGraphNode).filter(ClaimGraphNode.workspace_id == workspace_id).delete(synchronize_session=False)
    db.flush()

    nodes: Dict[str, ClaimGraphNode] = {}
    edges: Dict[Tuple[str, str, str], List[VendorClaim]] = defaultdict(list)

    for claim in claims:
        vendor_name = f"vendor-{claim.vendor_id}" if claim.vendor_id else "unknown-vendor"
        vendor_key = _node_key("company", vendor_name)
        if vendor_key not in nodes:
            nodes[vendor_key] = ClaimGraphNode(
                workspace_id=workspace_id,
                node_type="company",
                canonical_name=vendor_name,
                metadata_json={"vendor_id": claim.vendor_id},
            )

        if claim.dimension in {"customer", "customers", "case_study"}:
            rel_type = "serves"
            target_name = (claim.claim_text or "").split(":")[0][:180] or "unknown-customer"
            target_type = "customer"
        elif claim.dimension in {"integration", "partnership"}:
            rel_type = "integrates_with"
            target_name = (claim.claim_text or "").split(":")[0][:180] or "unknown-integration"
            target_type = "integration"
        elif claim.dimension in {"product", "capability", "services"}:
            rel_type = "offers_module"
            target_name = (claim.claim_key or claim.dimension or "module")[:180]
            target_type = "module"
        else:
            continue

        target_key = _node_key(target_type, target_name)
        if target_key not in nodes:
            nodes[target_key] = ClaimGraphNode(
                workspace_id=workspace_id,
                node_type=target_type,
                canonical_name=target_name,
                metadata_json={},
            )

        edges[(vendor_key, target_key, rel_type)].append(claim)

    for node in nodes.values():
        db.add(node)
    db.flush()

    node_id_by_key = {key: node.id for key, node in nodes.items()}

    edge_count = 0
    evidence_count = 0
    for (from_key, to_key, rel_type), rel_claims in edges.items():
        edge = ClaimGraphEdge(
            workspace_id=workspace_id,
            from_node_id=node_id_by_key[from_key],
            to_node_id=node_id_by_key[to_key],
            relation_type=rel_type,
            confidence=min(1.0, 0.4 + (0.1 * len(rel_claims))),
            evidence_count=len(rel_claims),
            metadata_json={},
        )
        db.add(edge)
        db.flush()
        edge_count += 1
        for claim in rel_claims[:20]:
            db.add(
                ClaimGraphEdgeEvidence(
                    edge_id=edge.id,
                    claim_id=claim.id,
                    source_evidence_id=claim.source_evidence_id,
                    explanation=(claim.claim_text or "")[:500],
                )
            )
            evidence_count += 1

    return {
        "claims_count": len(claims),
        "nodes_count": len(nodes),
        "edges_count": edge_count,
        "edge_evidence_count": evidence_count,
    }

