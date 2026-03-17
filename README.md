# Evidence-Backed M&A Sourcing Intelligence

Evidence-first acquisition target discovery for software markets. The product is optimized for **source-grounded market mapping and static snapshot analysis** rather than live monitoring, and prioritizes **precision + source traceability** over broad recall.

## V1 Product Flow

`Sourcing Brief -> Expansion Brief -> Universe -> Validation -> Cards`

1. `Sourcing Brief`: Start from a source company website, optional comparator sites, and optional supporting proof links. Crawl the relevant sources and generate the canonical source-scoped brief.
2. `Expansion Brief`: Run bounded deep research from the normalized graph plus the sourcing brief. The expansion step also hosts the scope-review board and confirmation gate.
3. `Universe`: Build and curate the candidate longlist from reviewed `adjacency_boxes` and `company_seeds`, with auditable source-backed rationale.
4. `Validation`: Manually choose which companies deserve simple enrichment before promoting only the shortlist into deep cards.
5. `Cards`: Generate immutable snapshot cards with:
- compete/complement lens scores
- source-backed claims with inline source pills
- filing metrics only when reliable evidence exists

## Current UX Behavior

- `buyer` in the data model means the **source company / anchor company** for the market map, not a target.
- The current workspace navigation is:
  - `context` -> `Sourcing Brief`
  - `bricks` -> `Expansion Brief`
  - `universe` -> `Universe`
  - `validation` -> `Validation`
  - `report` -> `Cards`
- The context step is brief-first in the UI:
  - `Source company website`
  - optional comparable companies
  - optional proof links / supporting evidence URLs
  - `Generate Sourcing Brief`, `Recrawl And Update Brief`, and `Regenerate Brief Only`
- The expansion step is the canonical `Expansion Brief v3` surface:
  - structured `adjacency_boxes`
  - `company_seeds`
  - `technology_shift_claims`
  - embedded scope-review board with keep/remove/deprioritize decisions
- Long-running jobs expose:
  - step-based progress
  - a stop control
  - rolling source activity from the crawl/search worker
  - a compact completed-run summary after the phase finishes
- Context-pack routes still exist as the crawl/input layer, but the user-facing workflow is sourcing-brief-first.

## Phase 1 Artifacts

Phase 1 now produces five canonical artifacts that stay reusable across different source companies:

1. `Context Pack`
   - selected first-party pages
   - evidence items
   - named customers
   - integrations / partners
   - extracted raw phrases
   - crawl coverage stats
2. `CompanyContextGraph`
   - canonical source-company graph in Neo4j + cached workspace payload in Postgres
   - evidence-tiered nodes and claims
3. `Sourcing Brief`
   - source summary
   - top customer/workflow/capability nodes
   - named customer proof
   - partner / integration proof
   - active lens recommendations
   - adjacency hypotheses
   - confidence gaps / open questions
4. `Expansion Brief v3`
   - `adjacency_boxes`
   - `company_seeds`
   - `technology_shift_claims`
   - `confidence_gaps`
   - `open_questions`
5. `Scope Review`
   - reviewed statuses on source baseline items and expansion lanes
   - confirmed scope used to drive universe discovery

## Company Context Graph

Phase 1 is now graph-first.

- Canonical phase-1 model: `CompanyContextGraph`
- Canonical API/UI surface: `company-context`
- Canonical evidence tiers:
  - `first_party`
  - `external_public`
  - `inferred`

Storage model:
- Neo4j AuraDB is the canonical graph store for phase-1 company context
- PostgreSQL keeps workspace metadata, job state, and a lightweight graph cache / sync status bridge

Graph structure:
- node labels:
  - `Company`
  - `CustomerEntity`
  - `CustomerArchetype`
  - `Workflow`
  - `Capability`
  - `DeliveryIntegration`
  - `PartnerEntity`
  - `Category`
  - `SourceDocument`
  - `Claim`
- relationship types:
  - `OFFERS_CAPABILITY`
  - `SUPPORTS_WORKFLOW`
  - `SERVES_CUSTOMER_ENTITY`
  - `SERVES_CUSTOMER_ARCHETYPE`
  - `INTEGRATES_WITH`
  - `LISTED_IN_CATEGORY`
  - `ANNOUNCED_BY_CUSTOMER`
  - `SUPPORTED_BY`
  - `CONTRADICTED_BY`
  - `MENTIONS`

## Crawl And Extraction Strategy

- Buyer/source sites are crawled more deeply than comparator sites.
- High-signal first-party routes are prioritized:
  - product / platform / solution pages
  - customer / reference / case-study pages
  - integration / docs / API pages
  - selected careers pages when they add product or workflow context
- Thin SPA sites are supported through:
  - JS bundle extraction for route labels, logos, and embedded signals
  - rendered page fallback for interactive product pages
  - accordion/disclosure expansion on selected product and solution routes
- Context packs are source-preserving:
  - page type
  - URL
  - headings
  - extracted evidence blocks/snippets
  - captured timestamps
  - confidence metadata

## Scope (V1)

- Discovery geography: UK, IE, FR, BE, NL, LU, DE, ES, PT
- Reliable filings metric coverage: **FR + UK only**
- SME policy: default shortlist targets 15-100 employees plus explicit unknown-size bucket
- Static snapshots: manual generation, no continuous watchlist feed

## Architecture

- Frontend: Next.js 14 (App Router) + Tailwind + React Query
- Backend: FastAPI + SQLAlchemy (async)
- Workers: Celery + Redis
- Databases: PostgreSQL + Neo4j AuraDB
- Research/enrichment: OpenAI/Gemini orchestration + structured first-party crawling + Playwright-backed rendering for interactive pages

## Quick Start

### Prerequisites

- Docker + Docker Compose
- At least one LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`)

### Environment

Create `.env` at repo root:

```bash
OPENAI_API_KEY=your-api-key
GEMINI_API_KEY=your-api-key
EXA_API_KEY=your-api-key
TAVILY_API_KEY=your-api-key
SERPAPI_API_KEY=your-api-key
FIRECRAWL_API_KEY=your-api-key
JINA_API_KEY=your-api-key
NEO4J_URI=neo4j+s://your-aura-host.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

Minimum practical setup for local sourcing:
- one LLM key: `OPENAI_API_KEY` or `GEMINI_API_KEY`
- web retrieval keys: `EXA_API_KEY` plus at least one fallback/provider
- Postgres + Redis via `docker-compose`

### Run

```bash
docker-compose up --build
```

Services:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

## API (Workspace Model)

### Workspace and Core Flow

- `POST /workspaces`
- `GET /workspaces`
- `GET /workspaces/{workspace_id}`
- `PATCH /workspaces/{workspace_id}`
- `DELETE /workspaces/{workspace_id}`

### Crawl Inputs

- `GET /workspaces/{workspace_id}/context-pack`
- `PATCH /workspaces/{workspace_id}/context-pack`
- `POST /workspaces/{workspace_id}/context-pack:refresh`
- `POST /workspaces/{workspace_id}/context-pack:export`

### Sourcing Brief + Expansion

- `GET /workspaces/{workspace_id}/company-context`
- `PATCH /workspaces/{workspace_id}/company-context`
- `POST /workspaces/{workspace_id}/company-context:refresh`
- `GET /workspaces/{workspace_id}/expansion-brief`
- `POST /workspaces/{workspace_id}/expansion-brief:generate`

### Scope Review

- `GET /workspaces/{workspace_id}/scope-review`
- `PATCH /workspaces/{workspace_id}/scope-review`
- `POST /workspaces/{workspace_id}/scope-review:confirm`

### Universe

- `POST /workspaces/{workspace_id}/discovery:run`
- `GET /workspaces/{workspace_id}/discovery:diagnostics`
- `GET /workspaces/{workspace_id}/universe/top-candidates`
- `GET /workspaces/{workspace_id}/companies`
- `POST /workspaces/{workspace_id}/companies`
- `PATCH /workspaces/{workspace_id}/companies/{company_id}`

### Validation + Enrichment

- `POST /workspaces/{workspace_id}/companies:enrich`
- `GET /workspaces/{workspace_id}/companies/{company_id}/dossier`

### Static Report Snapshots

- `POST /workspaces/{workspace_id}/reports:generate`
- `GET /workspaces/{workspace_id}/reports`
- `GET /workspaces/{workspace_id}/reports/{report_id}`
- `GET /workspaces/{workspace_id}/reports/{report_id}/cards`
- `GET /workspaces/{workspace_id}/reports/{report_id}/export`
- `GET /workspaces/{workspace_id}/reports/{report_id}/export?format=rich_json`

### Jobs and Gates

- `GET /workspaces/{workspace_id}/jobs`
- `GET /workspaces/{workspace_id}/jobs/{job_id}`
- `POST /workspaces/{workspace_id}/jobs/{job_id}:cancel`
- `GET /workspaces/{workspace_id}/gates`

## Notes

- Context-pack jobs now:
  - batch the LLM triage phase instead of classifying up to 100 preview pages in one call
  - emit rolling live events so the UI can show recent source activity
  - supersede older phase-1 jobs when a new brief run starts
  - prioritize buyer/source high-signal first-party routes over comparator depth
  - selectively include relevant careers pages
  - render interactive product/solution pages when static extraction is too thin
- Claims without source are rendered as `hypothesis`.
- Source-backed claims/metrics include source pill metadata (`label`, `url`, `captured_at`, optional `document_id`).
- Export APIs return payloads only; arbitrary server-side file path writes are not allowed.
- `rich_json` export includes both `kept` and `rejected` screening decisions with auditable reasons, ICP/product/customer evidence, numeric ranges, and source pills.

## Neo4j CLI

CLI-first graph operations are supported via `cypher-shell`.

Install locally:

```bash
brew install cypher-shell
```

Apply graph schema:

```bash
backend/scripts/neo4j_apply_schema.sh
```

Inspect one company graph:

```bash
backend/scripts/neo4j_inspect_graph.sh workspace-8-4tpm
```

The CLI scripts use:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Phase 1 Design Docs

- [Sourcing Brief -> Expansion Brief -> Scope Review -> Universe Flow](/Users/rayaneachich/Desktop/prospection-software-co/docs/source-brief-expansion-scope-universe-flow.md)
- [Expansion Brief V3](/Users/rayaneachich/Desktop/prospection-software-co/docs/expansion-brief-v3.md)
- [Expansion Brief Evaluation Rubric](/Users/rayaneachich/Desktop/prospection-software-co/docs/expansion-brief-evaluation-rubric.md)
- [Phase 1 Market Map Brief](/Users/rayaneachich/Desktop/prospection-software-co/docs/phase1-market-map-brief.md)
- [Phase 1 Market Map Reasoning](/Users/rayaneachich/Desktop/prospection-software-co/docs/phase1-market-map-reasoning.md)
