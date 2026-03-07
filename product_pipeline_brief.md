# Prospection Software Co - Product and Pipeline Brief

## 1) Product Summary (Current State)

This product is an **evidence-first M&A target discovery platform** for software markets, organized as a 4-step workspace workflow:

1. Context Pack
2. Brick Model
3. Universe
4. Static Report

Primary use case: build a **curated longlist of acquisition targets** with source-backed rationale and snapshot reporting.

Core positioning characteristics in code/docs:
- Optimized for **static snapshots** (not continuous monitoring by default in core flow)
- Emphasizes **precision and source traceability** over broad recall
- Discovery geography defaults to **EU+UK focus**, with specific country controls
- Reliable filing-metric coverage is explicitly strongest for **FR + UK**

---

## 2) System Architecture

### Backend
- FastAPI (`backend/app/main.py`)
- SQLAlchemy async + PostgreSQL
- Celery + Redis for async jobs
- Workspace-centric API router (`/workspaces/*`)

### Frontend
- Next.js 14 App Router + React Query + Tailwind
- Workflow UI by workspace steps:
  - `/workspaces/[id]/context`
  - `/workspaces/[id]/bricks`
  - `/workspaces/[id]/universe`
  - `/workspaces/[id]/report`

### Key service layers
- Crawling/extraction: `app/services/crawler/*`
- Discovery orchestration + scoring: `app/workers/workspace_tasks.py`
- LLM routing: `app/services/llm/*`
- Decisioning: `app/services/decision_engine.py`
- Evidence policy + freshness tiers: `app/services/evidence_policy.py`
- Report generation/export: API + worker snapshot pipeline

---

## 3) End-to-End Pipeline (Operational)

## Stage A: Context Pack Generation

### Entry point
- `POST /workspaces/{id}/context-pack:refresh`
- Worker task: `generate_context_pack_v2`

### Inputs
- `buyer_company_url`
- `reference_vendor_urls` (UI limits to 3 reference competitors)
- `reference_evidence_urls`
- optional geo scope fields

### Processing
- Uses `UnifiedCrawler` (5 phases):
  1. URL discovery (robots/sitemaps/nav/BFS)
  2. page preview + scoring
  3. LLM triage for page selection
  4. structured extraction of content/signals/customer evidence
  5. context-pack synthesis
- Aggregates crawled content into:
  - markdown context pack
  - structured `context_pack_json`
- Generates buyer summary via LLM orchestrator (`context_summary` stage), fallback to Gemini legacy client

### Outputs
- Stored on `CompanyProfile`:
  - `context_pack_markdown`
  - `context_pack_json`
  - `buyer_context_summary`
  - `reference_summaries`
  - `product_pages_found`
- Job output:
  - `urls_crawled`
  - `product_pages_found`
  - `markdown_length`

---

## Stage B: Brick Model (Taxonomy)

### Entry points
- `GET/PATCH /workspaces/{id}/bricks`
- `POST /workspaces/{id}/bricks:confirm`

### Inputs
- Capability brick list (`id`, `name`, optional `description`)
- Priority brick IDs
- Vertical focus tags

### Processing
- User-curated taxonomy, versioned in DB
- Confirmation gate controls whether Universe step can proceed

### Outputs
- Confirmed taxonomy object:
  - `bricks`
  - `priority_brick_ids`
  - `vertical_focus`
  - `confirmed`

---

## Stage C: Universe Discovery + Screening

### Entry point
- `POST /workspaces/{id}/discovery:run`
- Worker entry: `run_discovery_universe`

### Runtime structure
The queue chain is staged as:
1. `stage_seed_ingest` (preflight)
2. `stage_llm_discovery_fanout` (preflight)
3. `stage_registry_identity_expand` (preflight)
4. `stage_first_party_enrichment_parallel` (preflight)
5. `stage_scoring_claims_persist` (executes monolith)
6. `finalize_discovery_pipeline`

Important: most heavy execution happens inside `_run_discovery_universe_monolith()` called by `stage_scoring_claims_persist`.

### Inputs
- Context pack markdown
- Confirmed bricks + vertical focus
- Geo scope
- Comparator source ingestion config
- LLM routing config and provider API keys
- External connector API keys (optional)

### Internal sub-steps (monolith)
1. Comparator seed ingest
- Pulls mentions from registered sources (e.g., Wealth Mosaic + seed sources)
- Writes source runs + mentions

2. LLM discovery fanout
- Builds orchestrated discovery prompt
- Calls LLM stage `discovery_retrieval` with web search enabled when supported
- Parses/repaints malformed JSON using `structured_normalization`
- Fallback path: legacy Gemini workspace client

3. External search candidate retrieval
- `discover_candidates_from_external_search(query, cap)`
- Providers currently integrated:
  - Tavily API (if key exists)
  - SerpAPI (if key exists)

4. Identity resolution + canonicalization
- Resolves official websites
- Collapses aliases/duplicates into canonical entities
- Tracks origin edges and provenance

5. Registry identity and neighbor expansion
- Country-focused deterministic registry query flow
- Enrichment metrics for kept/dropped neighbor candidates

6. First-party enrichment
- Adaptive hint URL discovery + crawl
- Light/deep crawl budgets and fallback fetch path
- Optional Chrome DevTools MCP rendered-browser fallback (feature-flagged)

7. Enterprise B2B gate + scoring
- Hard-fail and penalty logic
- Score components include institutional fit, product depth, GTM, defensibility, etc.
- Produces screening status: `kept` / `review` / `rejected`

8. Claim extraction + decision engine classification
- Builds `VendorClaim` records with source metadata
- Deterministic decision engine maps to:
  - `good_target`
  - `borderline_watchlist`
  - `not_good_target`
  - `insufficient_evidence`
- Computes evidence sufficiency and unresolved contradictions

9. Persist + diagnostics packaging
- Writes vendors/screenings/claims/evidence
- Builds rich run diagnostics payload in `Job.result_json`
- Adds funnel/dropoff, quality, stage timing, retries, cache stats

### Outputs
Primary persisted entities:
- `Vendor`
- `VendorScreening`
- `VendorClaim`
- `WorkspaceEvidence`
- `CandidateEntity`, aliases, origin edges
- `ComparatorSourceRun`, `VendorMention`, registry logs

Primary run outputs in job result:
- Seed counts by source
- Canonical/final universe counts
- Registry expansion and identity metrics
- First-party enrichment usage and errors
- Decision class distributions
- Ranking-eligible counts
- Quality tier + degraded reasons
- Stage timing/timeout/retry/checkpoint telemetry
- Candidate dropoff funnel object

### Universe API outputs used by frontend
- `/vendors` list with classification/rationale/citation summaries
- `/universe/top-candidates` ranking-eligible list (with degraded-run guard unless overridden)
- `/discovery:diagnostics` full funnel, quality, source and execution metrics

---

## Stage D: Vendor Enrichment (Post-universe)

### Entry point
- `POST /workspaces/{id}/vendors:enrich`
- Worker: `run_enrich_vendor`

### Inputs
- Vendor IDs
- Job type(s): `enrich_full`, `enrich_modules`, `enrich_customers`, `enrich_hiring`

### Processing
- Uses Gemini workspace methods to generate dossier sections
- Stores versioned `VendorDossier`
- Creates additional evidence rows for module/customer citations

### Outputs
- `VendorDossier` version increment
- Vendor status moved to `enriched`
- Job output: module/customer counts + version

---

## Stage E: Static Report Snapshot

### Entry point
- `POST /workspaces/{id}/reports:generate`
- Worker: `generate_static_report`

### Inputs
- Workspace vendors in kept/enriched states
- Latest screening data
- Taxonomy priority bricks + adjacency map
- Optional filters:
  - `include_unknown_size`
  - `include_outside_sme`
  - custom snapshot `name`

### Processing
- Applies enterprise-screen filters
- Computes compete/complement lens scores
- Derives size estimates and buckets
- Promotes reliable filing evidence into structured facts
- Writes immutable report snapshot + report items

### Outputs
- `ReportSnapshot`
- `ReportSnapshotItem[]`
- Report cards and lens views via API:
  - `/reports/{id}/cards`
  - `/reports/{id}/lenses?mode=compete|complement`
- Export endpoint:
  - `/reports/{id}/export?format=default|rich_json`

---

## 4) Input vs Output Matrix (Condensed)

| Pipeline step | Main inputs | Main outputs |
|---|---|---|
| Context Pack | buyer URL, references, evidence URLs | context markdown/json, buyer summary, crawl signals |
| Brick Model | capability bricks, priorities, vertical focus | confirmed taxonomy + gate unlock |
| Discovery Universe | context + taxonomy + geo + seed sources + LLM + retrieval connectors | canonical entities, vendors, screenings, claims, evidence, diagnostics |
| Enrichment | vendor IDs + enrichment type | versioned dossier modules/customers/hiring + new evidence |
| Static Report | kept/enriched vendors + screenings + taxonomy + filters | immutable report snapshot, cards, lenses, export payload |

---

## 5) Models and AI Providers Used

### Active orchestrated multi-provider layer
Configured in `backend/app/config.py` via provider:model routes.

Default routes currently:
- `discovery_retrieval`:
  - `gemini:gemini-2.0-flash`
  - `openai:gpt-4.1-mini`
  - `anthropic:claude-3-5-haiku-latest`
- `evidence_adjudication`:
  - `anthropic:claude-3-7-sonnet-latest`
  - `openai:gpt-4.1`
  - `gemini:gemini-2.0-flash`
- `structured_normalization`:
  - `openai:gpt-4.1-mini`
  - `anthropic:claude-3-5-haiku-latest`
  - `gemini:gemini-2.0-flash`
- `context_summary`:
  - `anthropic:claude-3-7-sonnet-latest`
  - `gemini:gemini-2.0-flash`
  - `openai:gpt-4.1-mini`
- `crawler_triage`:
  - `gemini:gemini-2.0-flash`
  - `openai:gpt-4.1-mini`
  - `anthropic:claude-3-5-haiku-latest`

### Provider behavior
- Gemini provider supports optional web search tool in orchestrator flow.
- OpenAI and Anthropic providers are plain chat/message calls in current implementation.
- Legacy `GeminiWorkspaceClient` still exists as fallback and enrichment path.

---

## 6) Non-LLM External Data/Vendor Integrations

Current integrated connectors (all optional by API key/flags):

Search/discovery:
- Tavily Search API
- SerpAPI (Google results)

Page content retrieval:
- Jina Reader API
- Firecrawl scrape API

Registry/data sources (direct site/query logic):
- Companies House (UK)
- French registry endpoints (INPI / Annuaire / related)
- German registry endpoints (Handelsregister patterns)

Browser-render fallback (feature-flagged):
- Chrome DevTools MCP endpoint

---

## 7) Evidence, Decision, and Quality Controls

### Evidence policy primitives
- Source tiers: registry -> first-party -> partner/customer -> third-party -> discovery directory
- Claim group TTL by claim type
- Freshness checks via `valid_through`
- Gate requirements per workflow stage

### Decision engine outputs
For each screened candidate:
- `classification`
- `evidence_sufficiency`
- reason code sets (positive/caution/reject)
- missing claim groups
- contradiction counts
- rationale summary + markdown
- gating pass boolean

### Run quality framework
Discovery run emits:
- `run_quality_tier`: `high_quality` or `degraded`
- degraded reasons
- quality audit v1 pass/fail
- stage-level timeouts/retries/checkpoints
- candidate dropoff funnel and variance hotspots

---

## 8) Persisted Artifacts Useful for Research-Agent Analysis

For pipeline capability and gap analysis, the most useful payloads are:

1. `GET /workspaces/{id}/discovery:diagnostics`
- Contains funnel, source mix, registry yield, quality signals, execution telemetry.

2. `GET /workspaces/{id}/universe/top-candidates`
- Shows ranking-eligible shortlist with decision/rationale metadata.

3. `GET /workspaces/{id}/reports/{report_id}/export?format=rich_json`
- Rich company-by-company output with screening, fit, evidence, and source pills.

4. Context export:
- `POST /workspaces/{id}/context-pack:export`
- Captures upstream buyer/context assumptions that shape downstream universe quality.

---

## 9) Pipeline Substitution Seams (Where External Vendors Could Replace/Improve)

This section is purely structural (where swaps are easiest), not a final recommendation.

1. External search seed generation
- Current: Tavily + SerpAPI + LLM web search behavior
- Swap seam: `discover_candidates_from_external_search()` and LLM discovery prompt fanout
- Candidate replacement type: Exa-style semantic web/company retrieval API

2. First-party content extraction
- Current: URL discovery + preview + triage + extraction + optional Jina/Firecrawl and optional browser-render fallback
- Swap seam: `UnifiedCrawler` extraction pipeline, `fetch_page_fast()`, crawl connectors
- Candidate replacement type: managed crawl + extract vendor with structured outputs

3. Identity resolution and entity canonicalization
- Current: internal URL/domain normalization + registry identity mapping + dedupe
- Swap seam: `_resolve_candidate_identity`, canonical collapse, alias/origin graph assembly
- Candidate replacement type: entity resolution vendor / company graph provider

4. Registry enrichment
- Current: deterministic query expansion with country-specific logic and heuristics
- Swap seam: registry query + neighbor expansion functions
- Candidate replacement type: commercial company intelligence datasets/APIs

5. Evidence adjudication and claim normalization
- Current: deterministic + LLM-assisted claim generation and classification
- Swap seam: claim build/normalize + decision engine inputs
- Candidate replacement type: specialized extraction/adjudication models

6. Ranking and scoring layer
- Current: handcrafted score components + penalties + thresholds
- Swap seam: `_score_buy_side_candidate`, gate policies, ranking eligibility logic
- Candidate replacement type: external scoring/ranking engine or learned model layer

---

## 10) Known Constraints and Design Tradeoffs

- Strong bias toward evidence-backed precision can reduce recall.
- Significant pipeline complexity lives in one monolithic discovery execution path.
- Stage chain includes preflight stages; heavy execution concentrated in final scoring/persist stage.
- Reliable filing metric coverage is geography-constrained.
- Some behaviors are key-dependent/optional (connector availability changes output quality).
- Decisioning is deterministic and transparent, but may underfit nuanced market signals without richer upstream evidence.

---

## 11) Suggested Artifacts to Feed a Research Agent

For competitor gap analysis and vendor-replacement evaluation, provide:
- This document
- One recent `discovery:diagnostics` JSON from a real workspace
- One `top-candidates` response (with and without degraded override when applicable)
- One `rich_json` report export
- Current environment config values for model routes and connector keys (redacted keys)

That combination gives enough context to assess:
- where recall/precision tradeoffs happen
- which steps are homegrown vs commodity
- where external APIs (e.g., Exa) could reduce latency/complexity or improve yield
