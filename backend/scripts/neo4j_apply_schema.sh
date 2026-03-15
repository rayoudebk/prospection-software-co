#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_FILE="${SCRIPT_DIR}/../neo4j/company_context_schema.cypher"

if ! command -v cypher-shell >/dev/null 2>&1; then
  echo "cypher-shell is required. Install it first, e.g. 'brew install cypher-shell'." >&2
  exit 1
fi

if [[ -z "${NEO4J_URI:-}" || -z "${NEO4J_USERNAME:-}" || -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD before applying schema." >&2
  exit 1
fi

DATABASE="${NEO4J_DATABASE:-neo4j}"

cypher-shell \
  --non-interactive \
  --format plain \
  --address "${NEO4J_URI}" \
  --username "${NEO4J_USERNAME}" \
  --password "${NEO4J_PASSWORD}" \
  --database "${DATABASE}" \
  --file "${SCHEMA_FILE}"

echo "Applied company-context Neo4j schema to ${NEO4J_URI}/${DATABASE}"
