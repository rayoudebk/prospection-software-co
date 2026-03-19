from __future__ import annotations

from typing import Any, Dict, Iterable, List

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None

from app.config import get_settings
from app.services.company_context_graph import _stable_id, company_context_graph_namespace


def _candidate_graph_ref(workspace_id: int) -> str:
    namespace = company_context_graph_namespace()
    return f"{namespace}-workspace-{int(workspace_id)}-discovery-candidates"


def build_discovery_candidate_graph_payload(
    *,
    workspace_id: int,
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    graph_namespace = company_context_graph_namespace()
    graph_ref = _candidate_graph_ref(workspace_id)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(label: str, payload: dict[str, Any]) -> None:
        node_id = str(payload["id"])
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        payload["workspace_id"] = workspace_id
        payload["graph_namespace"] = graph_namespace
        payload["graph_ref"] = graph_ref
        payload["_label"] = label
        nodes.append(payload)

    def add_edge(edge_type: str, from_id: str, to_id: str, metadata: dict[str, Any] | None = None) -> None:
        edge_id = _stable_id("discovery_edge", edge_type, from_id, to_id)
        if edge_id in seen_edges:
            return
        seen_edges.add(edge_id)
        payload = {
            "id": edge_id,
            "from_id": from_id,
            "to_id": to_id,
            "type": edge_type,
            "workspace_id": workspace_id,
            "graph_namespace": graph_namespace,
            "graph_ref": graph_ref,
        }
        if metadata:
            payload.update(metadata)
        edges.append(payload)

    for candidate in candidates:
        entity_id = int(candidate.get("candidate_entity_id") or 0)
        if entity_id <= 0:
            continue
        candidate_node_id = _stable_id("discovered_candidate", workspace_id, entity_id)
        add_node(
            "DiscoveredCandidate",
            {
                "id": candidate_node_id,
                "candidate_entity_id": entity_id,
                "name": str(candidate.get("canonical_name") or candidate.get("company_name") or ""),
                "website": str(candidate.get("official_website_url") or "") or None,
                "discovery_url": str(candidate.get("discovery_url") or "") or None,
                "entity_type": str(candidate.get("entity_type") or "company"),
                "status": str(candidate.get("validation_status") or "queued_for_validation"),
                "promoted_to_cards": bool(candidate.get("promoted_to_cards")),
                "priority_score": float(candidate.get("priority_score") or 0.0),
            },
        )
        for lane_id in candidate.get("validation_lane_ids") or []:
            adjacency_id = _stable_id("adjacency_box_ref", workspace_id, lane_id)
            add_node(
                "AdjacencyBoxRef",
                {
                    "id": adjacency_id,
                    "adjacency_box_id": str(lane_id),
                    "label": str(lane_id),
                },
            )
            add_edge("FITS_ADJACENCY_BOX", candidate_node_id, adjacency_id)
        for lane_label in candidate.get("validation_lane_labels") or []:
            adjacency_id = _stable_id("adjacency_box_ref", workspace_id, lane_label)
            add_node(
                "AdjacencyBoxRef",
                {
                    "id": adjacency_id,
                    "adjacency_box_id": str(lane_label),
                    "label": str(lane_label),
                },
            )
            add_edge("FITS_ADJACENCY_BOX", candidate_node_id, adjacency_id)
        for family in candidate.get("validation_query_families") or []:
            query_id = _stable_id("discovery_query", workspace_id, family)
            add_node(
                "DiscoveryQuery",
                {
                    "id": query_id,
                    "family": str(family),
                    "label": str(family),
                },
            )
            add_edge("DISCOVERED_VIA_QUERY", candidate_node_id, query_id)
        for source_family in candidate.get("validation_source_families") or []:
            if str(source_family) != "directory":
                continue
            directory_id = _stable_id("discovery_directory", workspace_id, source_family)
            add_node(
                "DiscoveryDirectory",
                {
                    "id": directory_id,
                    "family": str(source_family),
                    "label": str(source_family),
                },
            )
            add_edge("DISCOVERED_VIA_DIRECTORY", candidate_node_id, directory_id)

    return {
        "workspace_id": workspace_id,
        "graph_namespace": graph_namespace,
        "graph_ref": graph_ref,
        "nodes": nodes,
        "edges": edges,
    }


class Neo4jDiscoveryCandidateGraphStore:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _driver(self):
        if GraphDatabase is None:
            return None
        if not (
            str(self.settings.neo4j_uri or "").strip()
            and str(self.settings.neo4j_username or "").strip()
            and str(self.settings.neo4j_password or "").strip()
        ):
            return None
        return GraphDatabase.driver(
            str(self.settings.neo4j_uri),
            auth=(str(self.settings.neo4j_username), str(self.settings.neo4j_password)),
        )

    def sync_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        driver = self._driver()
        if driver is None:
            return {"ok": False, "skipped": True, "reason": "neo4j_not_configured"}
        workspace_id = int(graph.get("workspace_id") or 0)
        graph_namespace = str(graph.get("graph_namespace") or "")
        graph_ref = str(graph.get("graph_ref") or "")
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        with driver.session(database=str(self.settings.neo4j_database or "neo4j")) as session:
            session.run(
                """
                MATCH (n)
                WHERE n.workspace_id = $workspace_id
                  AND n.graph_namespace = $graph_namespace
                  AND n.graph_ref = $graph_ref
                DETACH DELETE n
                """,
                workspace_id=workspace_id,
                graph_namespace=graph_namespace,
                graph_ref=graph_ref,
            )
            for label in {"DiscoveredCandidate", "DiscoveryQuery", "DiscoveryDirectory", "AdjacencyBoxRef"}:
                rows = [{k: v for k, v in row.items() if row.get("_label") == label and k != "_label"} for row in nodes if row.get("_label") == label]
                if not rows:
                    continue
                session.run(
                    f"UNWIND $rows AS row CREATE (n:{label}) SET n = row",
                    rows=rows,
                )
            for edge_type in {"DISCOVERED_VIA_QUERY", "DISCOVERED_VIA_DIRECTORY", "FITS_ADJACENCY_BOX"}:
                rows = [row for row in edges if row.get("type") == edge_type]
                if not rows:
                    continue
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (from {{id: row.from_id, workspace_id: row.workspace_id, graph_namespace: row.graph_namespace, graph_ref: row.graph_ref}})
                    MATCH (to {{id: row.to_id, workspace_id: row.workspace_id, graph_namespace: row.graph_namespace, graph_ref: row.graph_ref}})
                    CREATE (from)-[r:{edge_type}]->(to)
                    SET r = row
                    """,
                    rows=rows,
                )
        driver.close()
        return {"ok": True, "nodes": len(nodes), "edges": len(edges)}
