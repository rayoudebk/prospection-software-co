# Expansion Brief V3

## Purpose

`Expansion Brief v3` is the proposed next contract for the research-heavy expansion stage.

It keeps the core value of `v2`:

- bounded one-hop adjacency research
- company seeds for discovery
- reviewable inferred output, separate from source truth

But it strengthens three weak areas surfaced by the Hublo run:

- evidence discipline
- workflow semantics
- multilingual concept normalization

This is the right next step if the product should answer:

- what adjacent workflows the same buyers likely also buy
- how critical those workflows are in day-to-day operations
- which companies matter in those lanes
- which shifts in the market are raising demand for those lanes

without pretending the model is making a final strategic verdict.

## Why V2 Is Not Enough

`Expansion Brief v2` already adds real value over the `Sourcing Brief`, but the artifact is still too loose in several places.

Observed gaps:

- evidence is often URL-level, not claim-level
- categories like `VMS` or `credentialing` are not decomposed enough into concrete operator actions
- source truth, peer evidence, and analyst inference can blur together
- multilingual normalization relies too heavily on model judgment instead of explicit concept mapping
- confidence is too coarse for the kinds of judgments being made

## Design Principles

### 1. Truth-first, inference-explicit

Every important field should be attributable to one of:

- `source_grounded`
- `peer_observed`
- `analyst_inferred`

The artifact should let the user see not just the conclusion, but what kind of evidence produced it.

### 2. Workflow semantics over category labels

An adjacency box is only useful if it explains the workflow, not just the category name.

For example, `healthcare VMS` is not useful by itself. The artifact should explain:

- who uses it
- when they use it
- what actions they take
- what systems it touches
- what breaks if it fails

### 3. Cross-lingual concept normalization

The graph should not fork because French source evidence and English peer evidence describe the same concept differently.

`v3` should store:

- a canonical concept label
- original-language aliases
- evidence language
- optional language-specific terms used in search

### 4. Discovery-ready, not strategy-theater

The expansion stage should produce:

- adjacency lanes worth searching
- seed companies worth crawling
- trend claims worth monitoring

It should not pretend to output an acquisition recommendation.

## Canonical Build Chain

1. `Primary + secondary crawl / extraction`
2. `Company context graph`
3. `Sourcing Brief`
4. `Expansion Brief v3`
5. `Scope Review`
6. `Universe`
7. `Company crawl / enrichment`
8. `Graph reinforcement or contradiction`

## Top-Level Shape

Machine-readable schema: [docs/schemas/expansion-brief-v3.schema.json](/Users/rayaneachich/Desktop/prospection-software-co/docs/schemas/expansion-brief-v3.schema.json)

Top-level fields:

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
- `technology_shift_claims`
- `confidence_gaps`
- `open_questions`

## Adjacency Box

`adjacency_boxes[]` remains the canonical discovery lane.

Required fields:

- `id`
- `label`
- `canonical_concept_key`
- `adjacency_kind`
- `status`
- `confidence`
- `why_it_matters`
- `source_fit`
- `criticality`
- `workflow_anatomy`
- `evidence`
- `company_seed_ids`
- `retrieval_query_seeds`
- `priority_tier`

Important additions versus `v2`:

### `canonical_concept_key`

Used to collapse multilingual or synonym-heavy variants into one graph concept.

Examples:

- `workforce_scheduling_healthcare`
- `credentialing_regulatory_compliance_healthcare`
- `vendor_management_external_staffing_healthcare`
- `time_attendance_payroll_healthcare`

### `workflow_anatomy`

This is the main upgrade.

Each adjacency box should explain:

- `primary_operators`
- `primary_triggers`
- `core_actions`
- `systems_touched`
- `frequency`
- `failure_cost`
- `management_value`

This turns a category into a usable workflow map.

### `evidence[]`

Evidence should be claim-level, not just source-level.

Each item should preserve:

- `url`
- `title`
- `publisher`
- `publisher_type`
- `language`
- `evidence_tier`
- `source_entity_name`
- `claim_text`
- `supports`

`supports` should explain what the evidence item substantiates:

- `buyer_overlap`
- `workflow_criticality`
- `switching_cost`
- `market_shift`
- `company_seed_fit`
- `named_account`

### `confidence`

Keep a top-level confidence, but decompose the major judgments inside `criticality`.

`criticality` should now include:

- `market_importance`
- `operational_centrality`
- `workflow_criticality`
- `daily_operator_usage`
- `switching_cost_intensity`
- `strategic_value_hypothesis`
- `replicability`
- `market_density`
- `adjacency_confidence`
- `switching_cost_confidence`
- `trend_confidence`

## Company Seeds

`company_seeds[]` remain first-class because they directly bridge expansion research into discovery.

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

Recommended additional field:

- `seed_role`

`seed_role` values:

- `adjacent_specialist`
- `cross_category_incumbent`
- `geography_specific_analog`
- `trend_leader`
- `direct_neighbor`

This makes the seed list more useful during universe review.

## Technology Shift Claims

`technology_shift_claims[]` become explicit top-level objects only when they matter for discovery.

Use them sparingly.

A valid technology shift claim should answer:

- what shift is happening
- in which adjacency lane
- why that changes demand for a capability, integration, or service
- which evidence supports the shift
- which company seeds appear relevant because of it

This is a better fit for questions like:

- `AI scheduling optimization is increasing demand for healthcare rostering`
- `compliance automation is increasing demand for credentialing infrastructure`

## Multilingual Normalization

`v3` should make language handling explicit.

Every adjacency box and company seed should be able to preserve:

- canonical English label for graph consistency
- original-language aliases
- evidence language
- language-specific query seeds when useful

Minimum rule:

- graph concepts should be canonicalized into one language
- evidence should preserve its source language
- aliases should preserve local phrasing

This avoids losing richness from French source material while still allowing English-heavy peer research to reinforce the same concept.

## Graph Writeback

Graph writeback remains mandatory.

`adjacency_boxes[]`:

- write as inferred claim-like nodes linked to the source company
- preserve `workflow_anatomy`, `criticality`, `priority_tier`, `canonical_concept_key`
- attach claim-level evidence with support tags

`company_seeds[]`:

- write as inferred company nodes or inferred discovery candidates
- preserve `seed_role` and supporting evidence

`technology_shift_claims[]`:

- write as inferred claim nodes only when they directly steer discovery or monitoring

## Hublo Example

The Hublo run suggests that `v3` would have added real clarity in at least two places:

### `Workforce Scheduling & Rostering`

Useful today:

- clearly surfaced as `core_adjacent`
- strong buyer/workflow overlap
- strong switching-cost and workflow-criticality hypothesis

Still missing:

- exact operators: scheduler, cadre de sante, HR manager
- exact actions: build rota, assign shifts, approve changes, publish schedule
- failure cost: understaffing, unsafe ratios, overtime spikes

### `VMS / Agency Management`

Useful today:

- recognized as a meaningful adjacency
- seeded relevant companies

Still missing:

- exact semantics of `VMS` in the healthcare staffing context
- distinction between overflow staffing, procurement control, and invoice consolidation
- whether daily use is primarily operational or managerial

That is the kind of precision `workflow_anatomy` is meant to enforce.

## Migration Guidance

`v3` should replace `v2` only when:

- stage 1 prompts explicitly request claim-level evidence and workflow anatomy
- stage 2 normalization preserves the new fields
- graph writeback supports `canonical_concept_key`, `workflow_anatomy`, and evidence support tags

Until then, keep `v2` as the live contract and treat `v3` as the next target.
