# Evidence-Backed M&A Sourcing Intelligence

Evidence-first acquisition target discovery for software markets. The product is optimized for **source-grounded market mapping and static snapshot analysis** rather than live monitoring, and prioritizes **precision + source traceability** over broad recall.

## V1 Product Flow

`Market Map Brief -> Search Lanes -> Universe -> Cards`

1. `Market Map Brief`: Start from a source company website, add comparable companies or supporting proof links, crawl the relevant sources, and generate a source-scoped market map brief.
2. `Search Lanes`: Review the derived core and adjacent lanes that will steer sourcing.
3. `Universe`: Build and curate a candidate longlist with evidence-backed fit rationale.
4. `Cards`: Generate immutable snapshot cards with:
- compete/complement lens scores
- source-backed claims with inline source pills
- filing metrics only when reliable evidence exists

## Current UX Behavior

- `buyer` in the data model means the **source company / anchor company** for the market map, not a target.
- Phase 1 is now brief-first in the UI:
  - `Source company website`
  - optional comparable companies
  - optional proof links / supporting evidence URLs
  - `Generate Market Map Brief`, `Recrawl And Update Brief`, and `Regenerate Brief Only`
- Long-running jobs expose:
  - step-based progress
  - a stop control
  - rolling source activity from the crawl/search worker
  - a compact completed-run summary after the phase finishes
- Context-pack routes still exist, but the product language and workflow are market-map-first.

## Phase 1 Artifacts

Phase 1 now produces four generic artifacts that stay reusable across different source companies:

1. `Source Company Context Pack`
   - selected first-party pages
   - evidence items
   - named customers
   - integrations / partners
   - extracted raw phrases
   - crawl coverage stats
2. `Taxonomy Map`
   - `customer_archetype`
   - `workflow`
   - `capability`
   - `delivery_or_integration`
3. `Market Map Brief`
   - source summary
   - top customer/workflow/capability nodes
   - named customer proof
   - integration proof
   - active lens recommendations
   - adjacency hypotheses
   - open questions / evidence gaps
4. `Lens Seeds`
   - `same_customer_different_product`
   - `same_product_different_customer`
   - `different_product_different_customer_within_market_box`

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
- Database: PostgreSQL
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

### Context

- `GET /workspaces/{workspace_id}/context-pack`
- `PATCH /workspaces/{workspace_id}/context-pack`
- `POST /workspaces/{workspace_id}/context-pack:refresh`
- `POST /workspaces/{workspace_id}/context-pack:export`

### Market Map Brief + Search Lanes

- `GET /workspaces/{workspace_id}/thesis-pack`
- `PATCH /workspaces/{workspace_id}/thesis-pack`
- `POST /workspaces/{workspace_id}/thesis-pack:refresh`
- `POST /workspaces/{workspace_id}/thesis-pack:apply-adjustment`
- `GET /workspaces/{workspace_id}/search-lanes`
- `PATCH /workspaces/{workspace_id}/search-lanes`
- `POST /workspaces/{workspace_id}/search-lanes:confirm`

### Universe

- `GET /workspaces/{workspace_id}/bricks`
- `PATCH /workspaces/{workspace_id}/bricks`
- `POST /workspaces/{workspace_id}/bricks:confirm`
- `POST /workspaces/{workspace_id}/discovery:run`
- `GET /workspaces/{workspace_id}/discovery:diagnostics`
- `GET /workspaces/{workspace_id}/vendors`
- `POST /workspaces/{workspace_id}/vendors`
- `PATCH /workspaces/{workspace_id}/vendors/{vendor_id}`

### Enrichment + Legacy Lenses

- `POST /workspaces/{workspace_id}/vendors:enrich`
- `GET /workspaces/{workspace_id}/vendors/{vendor_id}/dossier`
- `GET /workspaces/{workspace_id}/lenses/similarity`
- `GET /workspaces/{workspace_id}/lenses/complementarity`

### Static Report Snapshots

- `POST /workspaces/{workspace_id}/reports:generate`
- `GET /workspaces/{workspace_id}/reports`
- `GET /workspaces/{workspace_id}/reports/{report_id}`
- `GET /workspaces/{workspace_id}/reports/{report_id}/cards`
- `GET /workspaces/{workspace_id}/reports/{report_id}/lenses?mode=compete|complement`
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

## Phase 1 Design Docs

- [Phase 1 Market Map Brief](/Users/rayaneachich/Desktop/prospection-software-co/docs/phase1-market-map-brief.md)
- [Phase 1 Market Map Reasoning](/Users/rayaneachich/Desktop/prospection-software-co/docs/phase1-market-map-reasoning.md)
