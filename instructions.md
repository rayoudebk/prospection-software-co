# M&A Due Diligence Webapp (4TPM) — Product + Technical Plan

This document captures the context, product requirements, and an implementation plan for a collaborative M&A research and due diligence webapp for **4TPM**, focused on **Europe (EU + UK)** and explicitly designed to **avoid BPO-heavy / outsourcing-driven** companies in the securities/wealth processing ecosystem.

The research engine is **Gemini Deep Research** (via the **Gemini Interactions API**) for multi-step, cited web research. Long-running research runs in background mode and is polled asynchronously.

## 1) Goals and non-goals

### Goals
- Let multiple users (M&A + product + leadership) collaborate on:
  - building an **acquisition landscape** (candidate discovery + shortlisting)
  - producing **deep target profiles** with **cited evidence**
  - generating **diligence packs** (questions, red flags, request lists)
  - running **monitoring** for watched companies (news + registry changes + hiring signals)
- Encode 4TPM context through a **Strategy** object that includes:
  - **geo scope**
  - **intent** (capability / new vertical / geo expansion)
  - **seed URLs** (4TPM website + 2–3 competitor sites to anchor the landscape)
  - **negative constraints** (avoid BPO-heavy operators, etc.)
- Provide a feedback loop (thumbs up/down) to improve suggestions over time.

### Non-goals (initially)
- Full financial modeling / valuation automation (can be added later)
- Building a proprietary crawler for everything under the sun
- LinkedIn scraping as a core dependency (use safer proxies)

## 2) Users, permissions, collaboration

### Roles
- **Admin**: manages workspace, users, API keys, billing settings
- **Analyst**: creates strategies, runs research, edits targets, exports memos
- **Viewer**: read-only access, can leave comments/feedback

### Key collaboration features
- Comments on targets and evidence
- Change log for target fields (who changed what and why)
- Export (PDF / Markdown) for one-pagers, diligence packs, IC memos

## 3) Core workflow (UX)

### A. Create a “Target Strategy”
User inputs:
- Strategy name
- Region: EU + UK (with optional country filters)
- Intent: capability / new vertical / geo expansion
- Seed URLs:
  - 4TPM website
  - 2–3 competitor websites (known, trusted benchmarks)
- Exclusions / constraints:
  - “Avoid BPO-heavy companies” toggle (default ON)
  - keywords/phrases that indicate BPO/outsourcing
  - industries to exclude (e.g., pure outsourcing / call-center ops)

System actions:
- Crawl and snapshot seed URLs
- Produce a **Context Pack** (clean markdown) representing:
  - 4TPM capabilities, modules, market positioning
  - competitor positioning and common category language

### B. Generate the landscape (top 10)
- System returns a list of ~10 targets with:
  - short rationale (“why fit”)
  - sources/citations (evidence links)
  - early flags (BPO risk, overlap, integration complexity)
- User can **thumb up/down** and optionally tag the reason:
  - “BPO-heavy”
  - “not Europe/UK”
  - “not in scope / wrong vertical”
  - “good fit: OMS adjacency”
  - “good fit: regulatory/compliance module”
- Feedback is stored and used to improve future runs for that strategy.

### C. Deep research per target
For a selected target, generate a structured profile:
- What they do (product modules / value prop)
- ICP / clients (only if supported by sources / logo credentialas or case studies)
- Business model signals (license/SaaS vs services-led)
- Geography / regulatory footprint
- Tech and integration hints
- Risks + red flags
- Diligence questions and request list
- “Why 4TPM would buy this” (synergy hypotheses clearly labeled as hypotheses)

### D. Live monitoring (watchlist)
For watched targets, display a timeline of signals:
- News mentions and material events
- Registry changes (directors/officers, filings, ownership signals where available)
- Hiring proxy (job openings volume, engineering vs ops mix)
- “Key-person risk” events (director resignations, leadership changes)
Users can add/remove companies from the watchlist at any time.

## 4) Data sources (Europe-first)

### Registries (ground truth)
- **France**: Pappers API (SIREN/SIRET + documents)
- **UK**: Companies House API (company profile, filing history, officers, PSC)
- **Pan-Europe fallback**: OpenCorporates API (depending on licensing / pricing)

### Web research (positioning and context)
- Gemini Deep Research agent (web search + URL reading) generates cited reports.

### Monitoring (recommended approach)
- Prefer registries + news + job postings rather than LinkedIn scraping:
  - LinkedIn scraping often violates ToS and is brittle.
  - Instead, compute signals from publicly accessible sources and allow users to add manual notes.

## 5) Gemini Deep Research integration (how it runs)

### Key constraints (must design around)
- Deep Research is multi-step and often exceeds synchronous timeouts.
- It must run via the **Gemini Interactions API**.
- Long runs should use **background execution** and be **polled** until completion.

Implementation references:
- Deep Research docs: https://ai.google.dev/gemini-api/docs/deep-research
- Interactions API docs: https://ai.google.dev/gemini-api/docs/interactions
- Google blog (Deep Research agent): https://blog.google/technology/developers/deep-research-agent-gemini-api/
- Google blog (Interactions API): https://blog.google/technology/developers/interactions-api/

### Job pattern (backend + worker)
1) Backend receives request: `POST /strategies/{id}/landscape:run`
2) Backend enqueues a job (Redis queue)
3) Worker creates a new Interaction:
   - sets `background=true`
   - provides prompt + context pack + any tool constraints
4) Worker stores `interaction_id` and returns immediately
5) Backend/UI polls `GET /research-jobs/{id}`:
   - worker polls Gemini by `interaction_id`
   - state transitions: `queued → running → in_progress → completed/failed`
6) On completion:
   - store the generated artifacts (JSON + markdown report)
   - store citations and the evidence ledger
   - write normalized targets to the DB

## 6) Architecture (minimal, pragmatic)

### Recommended stack
- **Frontend**: Next.js (App Router) + Auth.js (Google/Microsoft SSO)
- **Backend API**: FastAPI (Python) *or* NestJS (TypeScript)
- **Queue/Workers**: Celery/RQ (Python) *or* BullMQ (TS) + Redis
- **DB**: Postgres
- **Object storage**: S3-compatible for raw HTML/PDF snapshots (optional for MVP; can store in DB early)
- **PDF generation**: server-side (later)

### Why a queue is required
Deep Research runs are long-lived; you need background execution and polling. The queue:
- prevents request timeouts
- provides retries + backoff
- enables scheduled monitoring jobs

### Deployment
- Docker Compose for dev
- Single cloud deployment later (Render/Fly/Cloud Run/etc.)
- Separate worker process from API process

## 7) Data model (high level)

### Tables
- **workspaces**: org container
- **users**, **workspace_members**
- **strategies**:
  - id, workspace_id, name
  - region_scope (EU/UK + countries)
  - intent (capability/vertical/geo)
  - seed_urls (array)
  - exclusions (json: BPO toggle, keywords)
  - created_by, timestamps
- **targets**:
  - id, workspace_id
  - canonical_name, website, countries
  - registry_ids (json: FR SIREN/SIRET, UK company_number, etc.)
  - status (candidate / shortlisted / rejected / watching)
  - tags (array)
  - bpo_score (0–100) + rationale
  - fit_score (0–100) + rationale
- **evidence_items**:
  - id, target_id
  - source_url, captured_at
  - content_type (html/pdf/registry/json)
  - storage_ref (s3 key or db blob reference)
  - excerpt / hash / citation_pointer
- **research_jobs**:
  - id, strategy_id, optional target_id
  - type (landscape / deep_profile / monitoring)
  - state, progress, error
  - provider (gemini_deep_research)
  - interaction_id (string)
  - started_at, finished_at
- **feedback_events**:
  - strategy_id, target_id
  - vote (+1/-1), reason_tag, comment
  - created_by, timestamp
- **signals**:
  - target_id, signal_type (news/officer_change/hiring_proxy)
  - occurred_at, summary, source_url, severity

## 8) Evidence ledger (the anti-hallucination layer)

Every important claim must be backed by one or more Evidence Items.

### What gets stored
- Source URL
- Timestamp captured
- Excerpt hash (or snippet)
- Citation pointer (for rendering “why we believe this”)

### UI requirement
On every profile page:
- show “Facts” (cited)
- show “Assumptions / hypotheses” (clearly labeled)
- show “Open questions” (to validate in diligence)

## 9) “Avoid BPO-heavy companies” — scoring and kill rules

### BPO score
Compute a **BPO likelihood score** (0–100) from evidence:
- Website language suggesting delegated processing / outsourced ops
- Service-heavy positioning (run ops for you) vs product-led (software platform)
- Hiring mix (ops roles vs engineering/product)
- Case studies emphasizing operational take-over

### Kill rule (default)
- If `bpo_score >= threshold` (e.g., 70), mark the target as:
  - `rejected` (unless user overrides)
  - record override justification

### Why this matters
You explicitly want to avoid acquiring a headcount-heavy BPO disguised as “software” in securities processing.

## 10) Landscape generation design (how to get good suggestions)

### Step 1 — Create the Context Pack (from seed URLs)
- Snapshot and clean the pages
- Extract:
  - modules/capabilities taxonomy
  - integration patterns
  - buyer personas and language
  - competitive claims

### Step 2 — Candidate discovery (Deep Research)
Prompt should request:
- 50–150 candidates in EU/UK
- citations for each
- a JSON output schema with fields:
  - name, website, country
  - “why fit” (short)
  - “software signals” (bullets)
  - “bpo signals” (bullets)
  - “evidence_links” (list)

### Step 3 — Deterministic filtering (your code)
- remove non-EU/UK
- remove duplicates / brands / holding-company noise
- apply BPO kill rule
- map registry IDs where possible

### Step 4 — Ranking + feedback loop
- Start with rule-based scoring
- Use thumbs up/down to tune ranking:
  - simple approach: promote “similar” targets to liked ones (embeddings)
  - keep it per-strategy: different strategies should yield different landscapes

## 11) Deep-dive target profile (template requirements)

### Sections
- Overview (what they do)
- Modules / capabilities (mapped to your taxonomy)
- ICP & go-to-market (cited)
- Delivery model (software vs services) + BPO score
- Team & org signals (engineering vs ops proxy)
- Tech & integration signals
- Regulatory / security posture (only cite what’s real)
- Risks and red flags
- Diligence questions
- “Fit to 4TPM” hypotheses (explicitly labeled)
- Evidence appendix (citations)

## 12) Monitoring (signals pipeline)

### Signal sources (recommended)
- News: periodic Deep Research delta runs (“what changed since last check?”)
- Registries:
  - UK officers list and filings (Companies House)
  - FR legal documents and leadership changes (Pappers)
- Hiring proxy:
  - count open roles from careers pages and job boards
  - track trend, not absolute “headcount”

### Cadence
- News: daily or weekly (depending on volume)
- Registry changes: weekly
- Hiring proxy: weekly

### Notifications
- Email or Slack later; start with in-app notifications

## 13) API endpoints (suggested)

### Strategies
- `POST /strategies`
- `GET /strategies`
- `GET /strategies/{id}`
- `POST /strategies/{id}/landscape:run`
- `POST /strategies/{id}/seed-pack:refresh`

### Targets
- `GET /targets?strategy_id=...`
- `POST /targets/{id}/profile:run`
- `POST /targets/{id}/watch`
- `POST /targets/{id}/unwatch`
- `POST /targets/{id}/reject`
- `POST /targets/{id}/shortlist`

### Jobs
- `GET /research-jobs/{id}`
- `GET /research-jobs?strategy_id=...`

### Feedback
- `POST /strategies/{id}/feedback`

## 14) MVP build order (to avoid “adding tech”)

1) Auth + workspace + strategy create
2) Seed URL snapshot + Context Pack generation
3) Research jobs framework (queue + state machine)
4) Gemini Deep Research integration (background + polling)
5) Landscape output (top 10) + thumbs feedback
6) Target deep profile generation + evidence ledger UI
7) Watchlist + monitoring jobs + signal timeline

Only after MVP:
- embeddings for “similar to liked targets”
- PDF exports and “IC memo generator”
- more data providers (paid databases, etc.)

## 15) Quality and safety checks

### Reliability checks
- refuse to show uncited “facts” (move to “hypothesis”)
- “customer list” must always be sourced
- never invent revenue numbers; store as unknown unless sourced

### Compliance / legal hygiene
- avoid LinkedIn scraping as a core feature
- store source URLs and timestamps for auditability
- honor robots.txt where applicable for your own crawling

---

## Appendix: Source links (for build references)
- Gemini Deep Research docs: https://ai.google.dev/gemini-api/docs/deep-research
- Gemini Interactions API docs: https://ai.google.dev/gemini-api/docs/interactions
- Deep Research agent announcement: https://blog.google/technology/developers/deep-research-agent-gemini-api/
- Interactions API announcement: https://blog.google/technology/developers/interactions-api/
- Companies House API catalogue: https://www.api.gov.uk/ch/companies-house/
- Companies House officers endpoint reference: https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/reference/officers/list
- Pappers (France) API docs: https://www.pappers.fr/api/documentation
- Pappers International API docs: https://www.pappers.in/api/documentation
