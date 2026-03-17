# Expansion Brief V2

## Purpose

`Expansion Brief v2` is the canonical research-heavy artifact that complements the `Sourcing Brief`.

It is not a final strategic verdict. It is the product's replacement for the M&A senior associate's adjacency-mapping step:

- map one-hop adjacent workflows around the source company
- estimate which adjacencies look operationally critical versus peripheral
- identify likely buyer-neighbor segments
- surface seed companies and search-ready lanes for the universe step
- preserve hypotheses and evidence separately from source-company truth

This artifact should be the default expansion process, not an escalation-only path.

## Canonical Build Chain

1. `Primary + secondary crawl / extraction`
2. `Company context graph`
3. `Sourcing Brief`
4. `Expansion Brief v2`
5. `Scope Review`
6. `Universe`
7. `Company crawl / enrichment`
8. `Graph reinforcement or contradiction`

The `Expansion Brief v2` must be generated from a bounded packet:

- source-company brief
- graph-derived taxonomy nodes
- secondary context about the source company
- adjacent market inputs
- workspace geo scope

The model may use web search, but it must remain bounded to one-hop adjacency around the source evidence.

## Design Principles

### 1. Source truth stays separate from expansion research

`Sourcing Brief` remains the canonical answer to:

- what the source company clearly does
- who it clearly serves
- which workflows it clearly supports

`Expansion Brief v2` answers a different question:

- what adjacent capabilities, customers, workflows, and technology shifts deserve inclusion in the market map

The two artifacts must not collapse into one.

### 2. Expansion output must be retrieval-ready

The artifact should not be a prose memo only.

Each adjacency box must produce structure that can drive:

- graph writeback
- scope review
- search query planning
- company-seed retrieval

### 3. Criticality is a hypothesis, not a fact

Fields such as workflow criticality and switching-cost intensity are evidence-backed estimates.

They must remain clearly labeled as inferred research output.

### 4. Graph writeback is mandatory

If the deep-research result is not written back into the graph, the product loses compounding value.

Expansion output should become a reviewable inferred layer in the graph that later company discovery and crawls can strengthen, contradict, or prune.

## Canonical Schema

Machine-readable schema: [docs/schemas/expansion-brief-v2.schema.json](/Users/rayaneachich/Desktop/prospection-software-co/docs/schemas/expansion-brief-v2.schema.json)

Top-level shape:

- `version`
- `reasoning_status`
- `reasoning_warning`
- `reasoning_provider`
- `reasoning_model`
- `source_company`
- `graph_ref`
- `confirmed_at`
- `adjacency_boxes`
- `company_seeds`
- `confidence_gaps`
- `open_questions`

### Adjacency Box

`adjacency_boxes[]` is the canonical unit for scope review and graph extraction.

Each box represents one bounded adjacency lane.

Required fields:

- `id`
- `label`
- `adjacency_kind`
- `status`
- `confidence`
- `why_it_matters`
- `source_fit`
- `criticality`
- `supporting_node_ids`
- `related_source_node_ids`
- `likely_customer_segments`
- `likely_workflows`
- `evidence`
- `emerging_signals`
- `company_seed_ids`
- `retrieval_query_seeds`

`adjacency_kind`:

- `adjacent_capability`
- `adjacent_customer_segment`
- `adjacent_workflow`
- `technology_shift`

`status`:

- `hypothesis`
- `corroborated_expansion`
- `source_grounded`
- `user_kept`
- `user_removed`
- `user_deprioritized`

`source_fit` captures why the adjacency belongs near the source company:

- `shared_buyers`
- `shared_workflows`
- `shared_data_objects`
- `shared_integrations`
- `rationale`

`criticality` captures the operator-grade judgment the universe step needs:

- `market_importance`
- `operational_centrality`
- `workflow_criticality`
- `daily_operator_usage`
- `switching_cost_intensity`
- `strategic_value_hypothesis`
- `replicability`
- `market_density`

`emerging_signals[]` captures adjacent technology or market shifts only when they help explain why this lane matters for company discovery.

These signals are contextual, not first-class objects.

### Company Seeds

`company_seeds[]` contains named companies surfaced during expansion research that appear useful for later retrieval or validation.

Required fields:

- `id`
- `name`
- `website`
- `seed_type`
- `status`
- `confidence`
- `why_relevant`
- `fit_to_adjacency_box_ids`
- `evidence`

`seed_type`:

- `vendor`
- `platform`
- `specialist`
- `emerging`

`status`:

- `hypothesis`
- `corroborated_expansion`
- `retrieved`
- `crawled`
- `rejected`

## Graph Writeback

### Goal

Write expansion output into Neo4j as an inferred, reviewable layer that can later be reinforced or contradicted by universe discoveries and company crawls.

### Rule

Do not write expansion output back as first-party source truth.

Expansion output must remain:

- inferred
- reviewable
- status-bearing
- provenance-preserving

### Initial Graph Mapping

Use the current graph schema with minimal disruption first.

`adjacency_boxes[]`:

- upsert `Capability`, `Workflow`, or `CustomerArchetype` nodes when the label maps cleanly to an existing node type
- create a `Claim` node for the adjacency-box judgment itself
- connect the source company to that `Claim` via `MENTIONS`
- connect the `Claim` to related nodes via `MENTIONS`
- connect the `Claim` to source documents or evidence via `SUPPORTED_BY`

`company_seeds[]`:

- upsert `Company` nodes with `evidence_tier=inferred`
- connect the source company to the seed company via a `Claim` node or direct `MENTIONS`
- connect the seed company to relevant capability or customer nodes via `MENTIONS`
- preserve `status` so uncrawled seeds stay distinct from discovered companies

`adjacency_boxes[].emerging_signals[]`:

- write as properties on the adjacency-box `Claim` node initially
- only materialize them as separate graph objects later if they prove useful for retrieval or monitoring

### Node Properties To Preserve

Every expansion-derived node or claim should preserve:

- `status`
- `confidence`
- `graph_ref`
- `source_type=inferred`
- `evidence_tier=inferred`
- `evidence_type`
- `market_importance`
- `operational_centrality`
- `workflow_criticality`
- `daily_operator_usage`
- `switching_cost_intensity`
- `strategic_value_hypothesis`
- `replicability`
- `market_density`
- `adjacency_kind`

### Future Graph Extension

Once the inferred layer grows, add explicit support for:

- `AdjacencyBox` nodes
- `ADJACENT_TO`
- `SEEDS_DISCOVERY_IN`

Do not block `Expansion Brief v2` on those schema extensions.

The first pass should fit inside the existing `Claim`-centric graph model.

## Universe Consumption

The universe step should consume approved expansion nodes, not raw prose.

Approved adjacency boxes should drive:

- query seeds
- comparator and adjacent-vendor search lanes
- named company seed expansion
- geo filters
- candidate clustering and ranking context

The retrieval planner should prefer:

- `user_kept`
- `source_grounded`
- `corroborated_expansion`

It should exclude:

- `user_removed`

It should only include `hypothesis` items when:

- they are explicitly kept by the user
- or they exceed a confidence threshold defined in workspace policy

## Reinforcement Loop

Later company discovery and crawling should not create a parallel truth system.

Instead they should:

1. attach new company evidence to existing adjacency-box claims where possible
2. upgrade or downgrade expansion statuses
3. add new seed companies or contradictory evidence
4. preserve the audit trail between:
   - source truth
   - expansion hypotheses
   - discovered company evidence

This is what makes the graph compound over time instead of forcing each workspace run to restart the market map from scratch.

## Hublo Example

This is a pressure-test example for a healthcare staffing / workforce-operations source company.

It is intentionally shaped around discovery lanes and company seeds, not a generic trend memo.

```json
{
  "version": "expansion_brief_v2",
  "reasoning_status": "success",
  "reasoning_warning": null,
  "reasoning_provider": "openai",
  "reasoning_model": "o3-deep-research",
  "graph_ref": "workspace-12-hublo",
  "confirmed_at": null,
  "source_company": {
    "name": "Hublo",
    "website": "https://www.hublo.com/en"
  },
  "adjacency_boxes": [
    {
      "id": "adj_box_internal_float_pool",
      "label": "Internal float pool and workforce deployment",
      "adjacency_kind": "adjacent_workflow",
      "status": "corroborated_expansion",
      "confidence": 0.82,
      "why_it_matters": "Hospitals buying shift replacement tools often also need internal pool deployment, cross-site mobility, and staffing coordination workflows.",
      "source_fit": {
        "shared_buyers": ["hospital", "care facility"],
        "shared_workflows": ["shift replacement", "internal mobility", "staffing operations"],
        "shared_data_objects": ["staff roster", "open shifts", "department coverage"],
        "shared_integrations": ["HRIS", "scheduling", "timekeeping"],
        "rationale": "The same operations teams often own both replacement and internal workforce deployment decisions."
      },
      "criticality": {
        "market_importance": "high",
        "operational_centrality": "core",
        "workflow_criticality": "high",
        "daily_operator_usage": "high",
        "switching_cost_intensity": "high",
        "strategic_value_hypothesis": "A company strong in this lane could deepen Hublo's position in daily staffing operations and increase operational stickiness.",
        "replicability": "moderate",
        "market_density": "mixed"
      },
      "supporting_node_ids": ["workflow_shift_replacement", "workflow_internal_mobility"],
      "related_source_node_ids": ["capability_staffing_platform", "workflow_staffing_operations"],
      "likely_customer_segments": ["hospital", "care facility", "healthcare provider"],
      "likely_workflows": ["internal mobility", "pool management", "staff deployment"],
      "evidence": [
        {
          "url": "https://www.hublo.com/en/solutions",
          "title": "Healthcare staffing platform",
          "publisher": "Hublo",
          "publisher_type": "source_company",
          "evidence_tier": "primary",
          "source_entity_name": "Hublo",
          "claim_text": "Hospitals use Hublo to manage staffing operations, replacement planning, and workforce pools across departments."
        }
      ],
      "emerging_signals": [
        {
          "label": "staffing orchestration automation",
          "theme_type": "workflow_modernization",
          "confidence": 0.68,
          "why_it_matters": "Vendors in this lane increasingly position around cross-site coordination and automated workforce allocation."
        }
      ],
      "company_seed_ids": ["seed_medgo", "seed_permuteo"],
      "retrieval_query_seeds": [
        "hospital internal mobility software",
        "hospital float pool management platform",
        "healthcare workforce deployment software"
      ]
    },
    {
      "id": "adj_box_workforce_planning",
      "label": "Workforce planning and demand forecasting",
      "adjacency_kind": "adjacent_capability",
      "status": "hypothesis",
      "confidence": 0.67,
      "why_it_matters": "Buyer teams managing replacement operations often also own upstream planning and coverage forecasting.",
      "source_fit": {
        "shared_buyers": ["hospital", "care facility"],
        "shared_workflows": ["staffing operations", "shift planning"],
        "shared_data_objects": ["roster", "absence data", "coverage demand"],
        "shared_integrations": ["HRIS", "planning", "analytics"],
        "rationale": "This lane sits upstream from urgent replacement and may open a broader workforce-operations search box."
      },
      "criticality": {
        "market_importance": "medium",
        "operational_centrality": "meaningful",
        "workflow_criticality": "high",
        "daily_operator_usage": "medium",
        "switching_cost_intensity": "medium",
        "strategic_value_hypothesis": "A planning adjacency could increase platform breadth, but may be less urgent than operational execution workflows.",
        "replicability": "moderate",
        "market_density": "crowded"
      },
      "supporting_node_ids": ["workflow_staffing_operations"],
      "related_source_node_ids": ["workflow_shift_replacement"],
      "likely_customer_segments": ["hospital", "care facility"],
      "likely_workflows": ["shift planning", "coverage forecasting"],
      "evidence": [
        {
          "url": "https://www.hublo.com/en/solutions",
          "title": "Healthcare staffing platform",
          "publisher": "Hublo",
          "publisher_type": "source_company",
          "evidence_tier": "primary",
          "source_entity_name": "Hublo",
          "claim_text": "Hublo helps hospitals and care providers manage staffing and replacement planning."
        }
      ],
      "emerging_signals": [
        {
          "label": "forecast-assisted staffing",
          "theme_type": "ai_assistive",
          "confidence": 0.61,
          "why_it_matters": "Planning vendors are increasingly using predictive demand and recommendation layers."
        }
      ],
      "company_seed_ids": ["seed_skello_health", "seed_combo"],
      "retrieval_query_seeds": [
        "hospital workforce planning software",
        "healthcare staffing forecasting platform"
      ]
    }
  ],
  "company_seeds": [
    {
      "id": "seed_medgo",
      "name": "MedGo",
      "website": "https://www.medgo.io",
      "seed_type": "specialist",
      "status": "hypothesis",
      "confidence": 0.63,
      "why_relevant": "Appears relevant to healthcare staffing workflow adjacency and could help test the internal deployment lane.",
      "fit_to_adjacency_box_ids": ["adj_box_internal_float_pool"],
      "evidence": [
        {
          "url": "https://play.google.com/store/apps/details?id=fr.medgo.medgo1",
          "title": "MedGo",
          "publisher": "Google Play",
          "publisher_type": "directory",
          "evidence_tier": "secondary",
          "source_entity_name": "MedGo",
          "claim_text": "Healthcare staffing-related deployment context."
        }
      ]
    },
    {
      "id": "seed_permuteo",
      "name": "Permuteo",
      "website": null,
      "seed_type": "specialist",
      "status": "hypothesis",
      "confidence": 0.58,
      "why_relevant": "Named as a likely nearby company to validate the internal mobility and workforce deployment lane.",
      "fit_to_adjacency_box_ids": ["adj_box_internal_float_pool"],
      "evidence": []
    }
  ],
  "confidence_gaps": [
    "Public evidence tying buyer budget ownership across urgent replacement and longer-horizon planning remains thin.",
    "Named customer evidence for adjacent workforce-planning vendors is still limited."
  ],
  "open_questions": [
    "Should the first discovery pass prioritize operational staffing execution or upstream planning software?",
    "Which healthcare buyer segment is most commercially relevant beyond large hospitals?"
  ]
}
```
