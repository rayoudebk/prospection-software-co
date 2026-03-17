Objective

Refactor the current “Strategy-based” app into a Workspace-based M&A Market Map product that replaces the analyst/consultant research phase. The product must be:

Evidence-based (no “vibe” summaries as the primary output)

Step-gated (later steps unlock only when earlier steps have enough context/evidence)

Able to support both similarity and complementarity value-creation opportunities

Focused on Europe (EU + UK) with ability to filter to narrower geos/verticals later

Built around a capability brick model and segmentation (vertical/buyer + geo)

Uses Gemini Deep Research in multiple narrow runs (discovery, enrichment, monitoring) with strict JSON schemas

Remove visible scoring and “BPO” wording from the UI entirely (we can keep internal “services-led” heuristics but not present it as a score).

Product: new core flows & screens

Replace “Create Strategy” with a Workspace onboarding stepper:

Screen 0: Workspaces

List workspaces

Create workspace (name, region scope default EU+UK)

Screen 1: Company Context (Context Pack)

Inputs:

buyer_company_url (single URL)

reference_vendor_urls[] (2–3 optional; these are “known competitors/adjacent vendors”)

geo_scope (default EU+UK with optional country include/exclude)

Optional: vertical_focus (multi-select: private bank / asset manager / broker / insurer / fund admin / wealth platform / other)
Actions:

“Refresh Context Pack” button triggers crawl and produces a Context Pack artifact + evidence ledger.
UX:

Clearly label:

“Your company” (buyer)

“Reference vendors” (competitors/peers)

Show crawl coverage: number of product pages found, number of PDFs, last refresh timestamp.
Gate:

Unlock Screen 2 only if context pack meets minimum coverage thresholds.

Screen 2: Brick Model (capability taxonomy)

Outputs:

A brick taxonomy (start from a default set, editable)

Mapping of buyer and reference vendors to bricks using cited evidence from the context pack
UX:

Let user rename, merge, split bricks

Let user mark 5–10 “priority bricks”
Gate:

Unlock Screen 3 only if user confirms taxonomy and there are citations backing at least N brick mappings.

Screen 3: Candidate Universe (Discovery)

Action:

“Run Discovery” triggers a Deep Research discovery job (broad) producing 50–150 candidates with minimal evidence.
UI:

Candidate list with:

name + website + HQ country (if found)

likely vertical/buyer tags (low confidence ok)

1–2 citations (must exist)

keep/remove toggle

Add manual “Add vendor” button (name + website)
Gate:

Unlock Screen 4 only when:

At least X candidates are kept

At least Y candidates have valid websites + evidence

User has reviewed at least Z candidates (track progress)

Screen 4: Segmentation (Market Map draft)

UI:

Market map view driven by filters:

geo (country include/exclude)

vertical/buyer

bricks coverage (rough, low confidence until enrichment)

User can correct tags (geo/vertical) and set focus:

choose 1–2 verticals and 3–5 bricks to deepen first
Gate:

Unlock Screen 5 when:

user has corrected tags for at least N vendors OR acknowledges coverage gaps

selected focus scope exists (vertical/bricks)

Screen 5: Enrichment (Dossiers, targeted and evidence-backed)

Action:

Select a batch of vendors (e.g., 10–30) and run enrichment jobs.
Output per vendor:

A structured dossier JSON with strict evidence references (modules, customers, integrations, hiring, signals).
UI:

Vendor detail page with tabs:

Modules/Bricks (exact module names + citations)

Customers (explicit references only)

Integrations & deployment claims

Hiring proxy (jobs categorized)

News/signals

Evidence (all cited URLs)
Gate:

Unlock Screen 6 when dossiers exist for at least N vendors in the chosen scope.

Screen 6: Lenses (Similarity & Complementarity)

UI:

Two computed views:

Similarity: overlap on bricks + vertical + geo

Complementarity: gap-fill bricks and adjacency
Constraints:

No numeric scoring in UI.

Show evidence counts and “proof bullets” instead:

“Bricks overlap: PMS, Reporting (citations: 7)”

“Explicit customer overlaps: 2”

“Adds bricks: Pre-trade compliance (citations: 4)”
Allow user to save filters as “Saved Views”.

Screen 7: Monitoring (deltas)

UI:

Watchlists: company watch + “market box” watch (saved view)

Timeline of deltas:

news delta

registry delta (if supported)

hiring delta (jobs page change, role mix shift)
Actions:

schedule monitoring jobs with cadence controls (weekly/daily)

notify in-app (email later)

Data model changes (replace Strategy entity)
Remove/Deprecate

Strategy entity and related routes/components

Any fitScore / bpoScore fields and UI usage

Add entities

Workspace

CompanyProfile (1 per workspace)

buyer_company_url

reference_vendor_urls[]

geo_scope { region: EU_UK, include_countries[], exclude_countries[] }

optional vertical_focus[]

BrickTaxonomy

bricks[] { id, name, description, synonyms[] }

priority_brick_ids[]

versioning metadata

Vendor

workspace_id

name, website

hq_country (nullable)

operating_countries[] (nullable)

tags_vertical[] (multi)

status: { candidate, kept, removed, enriched, watched }

VendorDossier (latest + versions)

vendor_id

dossier_json (strict schema)

created_at

EvidenceItem

workspace_id, vendor_id nullable

source_url, captured_at

content_type { html, pdf, registry, json }

storage_ref (file/db)

excerpt_hash

excerpt_text (optional, small)

Job

workspace_id, vendor_id nullable

type { context_pack, build_taxonomy, discovery_universe, enrich_modules, enrich_customers, enrich_hiring, enrich_integrations, monitoring_delta }

state { queued, running, polling, completed, failed }

provider { gemini_deep_research, crawler, registry }

interaction_id (nullable)

progress json

error text

SavedView

workspace_id

name

filters json (geo/vertical/bricks/lens)

Watch

workspace_id

kind { vendor, saved_view }

ref_id

cadence

Migration

Provide a migration plan that:

keeps old Strategy tables for now but no longer used by UI

adds new workspace tables

optionally “import” old strategies into workspaces by converting fields

Crawler: prioritize product proof

Update crawler to build Context Pack and vendor evidence using prioritized URLs:

URL ranking priority (highest first)

product/platform/solutions/modules/features/capabilities

integrations/api/documentation/security/architecture

case-studies/clients/press/news/resources

careers/jobs

Implementation details:

Attempt sitemap discovery first (/sitemap.xml, robots)

Cap pages per host (e.g., 60) with priority queue by URL score

Download PDFs found in priority sections and extract text

Store EvidenceItems for each captured resource

Context Pack must be structured:

sections per vendor site

list of “Top product pages used”

key extracted headings and module bullets with citations

Gemini Deep Research: break into narrow job types with strict JSON outputs

Replace single “deep research” prompt with multiple job prompts that output JSON only.

Job: discovery_universe

Inputs:

Brick taxonomy (confirmed)

Buyer + reference vendor context pack excerpts (product pages first)

Geo scope EU+UK
Outputs (JSON only):

candidates[] with:

name, website, hq_country (if found), operating_countries (if found)

likely vertical/buyer tags (with confidence)

“why relevant” bullets (each with citations)

minimum citations per candidate: 1–2
Hard rules:

No prose outside JSON.

If no evidence found, candidate must not be included.

Job: enrich_modules

For one vendor:

crawl vendor product pages + PDFs
Output JSON:

modules[] with:

module_name_as_marketed

mapped_brick_id

what_it_does (1–2 sentences max)

evidence_urls[] (required)

deployment_claims[] (saas/onprem/hybrid) with citations

integration_claims[] with citations

Job: enrich_customers

Output JSON:

explicit_customer_references[]:

customer_name

context (case study/press/partner)

evidence_url
Rule:

Only explicit customer names. If none, return empty array.

Job: enrich_hiring

Inputs:

careers/jobs pages captured by crawler
Output JSON:

job_postings[]:

title, location, function_category { engineering, product, implementation, ops, sales, compliance, other }

evidence_url

hiring_mix_summary:

counts per category

key signals (e.g., implementation-heavy) with citations to job list pages

Job: monitoring_delta

Inputs:

last_run_timestamp

vendor dossier snapshot
Output JSON:

deltas[] with:

type { news, registry, hiring, website_product }

summary

why_it_matters tag { competitive_move, acquisition_signal, execution_risk }

evidence_url
Rule:

Only deltas since last run.

General prompting constraints

Always require citations URLs for factual claims.

Separate:

facts (cited)

hypotheses (explicitly labeled)

unknowns (explicit list)
Store hypotheses but show them clearly in UI.

Computed views (no scores)

Implement set-logic lenses using dossier data:

Similarity lens (computed)

A vendor is “similar” if:

overlaps on bricks (based on modules mapped to bricks)

overlaps on vertical tags (user-corrected)

overlaps on geo
UI renders:

overlapping bricks + citation count

overlapping explicit customer refs (if any)

top 3 “proof bullets” with citations

Complementarity lens (computed)

A vendor is “complementary” if:

matches vertical/geo filters

adds bricks not covered by buyer/reference vendors strongly (based on brick coverage)
UI renders:

added bricks + evidence counts

integration signals suggesting coexistence

top 3 “proof bullets” with citations

No numeric score should be shown anywhere.

Step gating rules (must implement)

Implement gating checks per step:

Context Pack gate: minimum product pages + PDFs captured OR user manually supplies additional URLs.

Brick Model gate: taxonomy confirmed + N mapped bricks with citations.

Universe gate: X kept vendors with valid websites + evidence.

Segmentation gate: user reviewed N vendors and selected focus scope.

Enrichment gate: dossiers exist for at least N vendors in scope.

Monitoring gate: must have at least one dossier before watching vendor/view.

UI should show:

“What’s missing to proceed” checklist (transparent gating)

UI adjustments requested

Remove Fit Score and BPO Score everywhere.

Remove “Avoid BPO-heavy companies” text in UI.

Replace with a neutral advanced toggle: “Exclude services-led operators” (optional; can be hidden behind Advanced)

Make it explicit which URL is the buyer vs reference vendors.

During discovery, return 3–5 “New vendors to validate” by default when user wants a quick pass; provide a separate “Build full universe” action for deeper mapping.

Ensure every “Why relevant” and “Watchout” bullet is backed by a citation and has a “View evidence” affordance.

Implementation notes

Refactor code by introducing a workspace route namespace.

Replace old pages/routes using strategyId with workspaceId.

Jobs:

implement a background job runner with polling states

store raw Gemini JSON outputs and validate against schemas

fail jobs that don’t meet schema/citation rules

Add JSON schema validation (zod/ajv) for all job outputs.

Add minimal caching for crawls and Deep Research requests to avoid rework.

Acceptance criteria (definition of done)

User can create a workspace, add buyer URL + 2–3 reference vendors.

Context Pack shows product-first coverage and citations.

Brick Model is editable and maps buyer/reference vendors to bricks with citations.

Discovery builds a candidate universe with evidence, user can keep/remove, and gating works.

Enrichment produces vendor dossiers with modules/customers/hiring extracted into strict JSON with citations.

Similarity and Complementarity views work without scores, showing evidence-backed overlap/gap-fill.

Monitoring can run delta jobs and display a timeline of evidence-backed events.

If Cursor asks for “what bricks”, use this default taxonomy to start (editable): PMS, OMS, Pre-trade compliance, Post-trade compliance, Risk & limits, Performance & attribution, Client reporting, Data hub/IBOR/positions, Market data, Connectivity (FIX/SWIFT/custodians), Corporate actions, Fund admin/TA, Accounting/GL/reg reporting, Digital wealth channels, Workflow/approvals/audit.