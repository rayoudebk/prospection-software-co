CREATE CONSTRAINT company_context_node_identity IF NOT EXISTS
FOR (n:CompanyContext)
REQUIRE (n.graph_ref, n.id) IS UNIQUE;

CREATE RANGE INDEX company_context_graph_ref IF NOT EXISTS
FOR (n:CompanyContext)
ON (n.graph_ref);

CREATE RANGE INDEX company_context_label IF NOT EXISTS
FOR (n:CompanyContext)
ON (n.label);

CREATE RANGE INDEX company_context_source_document_url IF NOT EXISTS
FOR (n:CompanyContext)
ON (n.url);
