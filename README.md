# Static SME M&A Target Radar

Evidence-first acquisition target discovery for software markets. The product is optimized for **static snapshot analysis** (not live monitoring) and prioritizes **precision + source traceability** over broad recall.

## V1 Product Flow

`Context Pack -> Bricks -> Universe -> Report`

1. `Context Pack`: Crawl buyer + reference sites and summarize context.
2. `Bricks`: Define capability taxonomy and priority bricks.
3. `Universe`: Build and curate a candidate longlist.
4. `Report`: Generate immutable snapshot cards with:
- compete/complement lens scores
- source-backed claims with inline source pills
- filing metrics only when reliable evidence exists

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
- Research/enrichment: Gemini + lightweight web fetching (no browser agents)

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Gemini API key

### Environment

Create `.env` at repo root:

```bash
GEMINI_API_KEY=your-api-key
```

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

### Bricks + Universe

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
- `GET /workspaces/{workspace_id}/gates`

## Notes

- Claims without source are rendered as `hypothesis`.
- Source-backed claims/metrics include source pill metadata (`label`, `url`, `captured_at`, optional `document_id`).
- Export APIs return payloads only; arbitrary server-side file path writes are not allowed.
- `rich_json` export includes both `kept` and `rejected` screening decisions with auditable reasons, ICP/product/customer evidence, numeric ranges, and source pills.
