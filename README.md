# Static SME M&A Target Radar

Evidence-first acquisition target discovery for software markets. The product is optimized for **static snapshot analysis** (not live monitoring) and prioritizes **precision + source traceability** over broad recall.

## V1 Product Flow

`Company Thesis -> Search Lanes -> Universe -> Cards`

1. `Company Thesis`: Start from your company website, add comparable companies or proof links, crawl the relevant sources, and generate a draft thesis.
2. `Search Lanes`: Review the derived core and adjacent lanes that will steer sourcing.
3. `Universe`: Build and curate a candidate longlist with evidence-backed fit rationale.
4. `Cards`: Generate immutable snapshot cards with:
- compete/complement lens scores
- source-backed claims with inline source pills
- filing metrics only when reliable evidence exists

## Current UX Behavior

- `buyer` in the data model means **your company**, not a target.
- The company-thesis step is now company-first in the UI:
  - `Your company website`
  - optional comparable companies
  - optional proof links
  - `Generate Draft Thesis`, `Recrawl And Update Draft`, and `Regenerate Draft Only`
- Long-running jobs expose:
  - step-based progress
  - a stop control
  - rolling source activity from the crawl/search worker
  - a compact completed-run summary after the phase finishes
- Context-pack routes still exist, but the product language and workflow are thesis-first.

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
- Research/enrichment: OpenAI/Gemini orchestration + lightweight web fetching (no browser agents)

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

### Thesis + Search Lanes

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
  - supersede older company-thesis jobs when a new thesis run starts
- Claims without source are rendered as `hypothesis`.
- Source-backed claims/metrics include source pill metadata (`label`, `url`, `captured_at`, optional `document_id`).
- Export APIs return payloads only; arbitrary server-side file path writes are not allowed.
- `rich_json` export includes both `kept` and `rejected` screening decisions with auditable reasons, ICP/product/customer evidence, numeric ranges, and source pills.
