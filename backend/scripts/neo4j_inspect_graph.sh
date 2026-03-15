#!/usr/bin/env bash
set -euo pipefail

GRAPH_REF="${1:-}"

if ! command -v cypher-shell >/dev/null 2>&1; then
  echo "cypher-shell is required. Install it first, e.g. 'brew install cypher-shell'." >&2
  exit 1
fi

if [[ -z "${NEO4J_URI:-}" || -z "${NEO4J_USERNAME:-}" || -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD before inspecting graphs." >&2
  exit 1
fi

if [[ -z "${GRAPH_REF}" ]]; then
  echo "Usage: backend/scripts/neo4j_inspect_graph.sh <graph_ref>" >&2
  exit 1
fi

DATABASE="${NEO4J_DATABASE:-neo4j}"

cypher-shell \
  --non-interactive \
  --format verbose \
  --address "${NEO4J_URI}" \
  --username "${NEO4J_USERNAME}" \
  --password "${NEO4J_PASSWORD}" \
  --database "${DATABASE}" \
  "MATCH (n:CompanyContext {graph_ref: '${GRAPH_REF}'})
   OPTIONAL MATCH (n)-[r]->(m:CompanyContext {graph_ref: '${GRAPH_REF}'})
   RETURN n.label AS node_label,
          count(DISTINCT n) AS node_count,
          collect(DISTINCT type(r))[0..10] AS relationship_types
   ORDER BY node_label;"
