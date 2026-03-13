# Phase 1 Market Map Brief

## Purpose

Phase 1 should turn a source company crawl into an evidence-backed market map brief.

This artifact is not a deal-readiness memo. It is a discovery-first brief that answers:

- what the source company appears to sell
- to whom it sells
- in which workflows
- how it is delivered or integrated
- which market-map lenses are worth activating next
- where evidence is still thin

## Inputs

The brief must be generated only from normalized phase-1 artifacts:

- `ContextPackV2`
- `taxonomy_nodes`
- `taxonomy_edges`
- `lens_seeds`

No fresh web reasoning should happen in this step.

## Output Shape

```json
{
  "source_summary": "string",
  "customer_nodes": [
    {
      "id": "taxonomy_xxx",
      "layer": "customer_archetype",
      "phrase": "private bank",
      "aliases": ["banques privées"],
      "confidence": 0.84,
      "evidence_ids": ["evidence_xxx"],
      "scope_status": "in_scope"
    }
  ],
  "workflow_nodes": [],
  "capability_nodes": [],
  "delivery_or_integration_nodes": [],
  "named_customer_proof": [
    {
      "name": "Milleis Banque Privée",
      "source_url": "https://source.example/customers",
      "context": "Trusted by ...",
      "evidence_type": "customer_logo",
      "evidence_id": "evidence_xxx"
    }
  ],
  "integration_partner_proof": [
    {
      "name": "BP2S",
      "source_url": "https://source.example/integrations",
      "evidence_id": "evidence_xxx"
    }
  ],
  "active_lenses": [
    {
      "id": "lens_xxx",
      "lens_type": "same_customer_different_product",
      "label": "Same Customer, Different Product",
      "query_phrase": "Front office titres",
      "rationale": "Named customer and customer-archetype evidence suggests adjacent products sold into the same buying accounts.",
      "supporting_node_ids": ["taxonomy_xxx"],
      "evidence_ids": ["evidence_xxx"],
      "confidence": 0.82
    }
  ],
  "adjacency_hypotheses": [
    "short evidence-backed hypothesis"
  ],
  "open_questions": [
    "short evidence gap or ambiguity"
  ],
  "confirmed_at": null
}
```

## Node Layer Definitions

### `customer_archetype`

Use for buyer or operator types.

Examples:

- `asset manager`
- `private bank`
- `online brokerage`
- `hospital operations team`

Do not use:

- named customer entities
- internal teams unless they are the actual buyer
- broad industry labels like `finance` or `healthcare`

### `workflow`

Use for the operating process or job the customer is trying to run.

Examples:

- `front office titres`
- `back office titres`
- `payments & cash operations`
- `shift replacement`

Do not use:

- product brand names
- delivery terms like `REST API`
- generic placeholders like `operations`

### `capability`

Use for the product function, module, or concrete feature cluster.

Examples:

- `Décisions de gestion et arbitrages`
- `Connectivité et routage multi-marchés`
- `Portfolio analytics`
- `Workforce pool management`

Do not use:

- named customers
- generic headings like `Capacités clés`
- long sentence fragments
- infrastructure labels

### `delivery_or_integration`

Use for architecture, integration surface, or delivery traits.

Examples:

- `REST API`
- `API documentation`
- `Cloud delivery`
- `Partner ecosystem`

Do not use:

- customer workflows
- product feature labels

## Evidence Rules

- Every surfaced node must keep `evidence_ids`.
- `named_customer_proof` and `integration_partner_proof` must stay separate.
- Fact-style claims must be source-backed.
- Hypotheses may combine multiple weak signals, but must still be tied to evidence in the reasoning layer.
- The brief should prefer source-company evidence over comparator evidence.

## Lens Activation Rules

### `same_customer_different_product`

Activate when:

- named customer proof exists, or
- customer archetype evidence is strong enough to anchor the buying account

Use to ask:

- what else these buyers likely purchase adjacent to the source capability

### `same_product_different_customer`

Activate when:

- capability evidence is strong enough to define the source solution clearly

Use to ask:

- who sells the same solution into other customer segments

### `different_product_different_customer_within_market_box`

Activate only when:

- the market box is bounded by shared customer archetype, workflow cluster, or capability neighborhood

Do not activate when the result would drift into generic industry search.

## Ranking Principles

When surfacing top nodes in the brief:

- prefer short, concrete, evidence-backed phrases
- prefer source-company feature phrases over generic page headings
- demote sentence-like fragments
- demote delivery/infrastructure terms from capability/workflow layers
- preserve the full taxonomy separately even if the brief only surfaces the top slice

## Non-Goals

Phase 1 should not require:

- ownership inference
- transaction feasibility scoring
- outreach strategy
- valuation or deal process readiness

Those belong later in the pipeline.
