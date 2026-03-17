# bug_fix.md

This backlog tracks product-critical defects and trust gaps for the static SME M&A radar.

## Build blockers

### P0 - Legacy route build break (`/strategies/[id]`, `/targets/[id]`)
- Owner: Frontend
- Status: Resolved
- Acceptance criteria:
1. `next build` completes without missing export errors from legacy hooks/api imports.
2. Legacy routes no longer depend on removed strategy/target runtime.

## Data correctness

### P0 - Discovery promise mismatch ("50-150 targets") vs actual behavior
- Owner: Product + Frontend
- Status: In progress
- Acceptance criteria:
1. UI copy no longer promises fixed volume when generation can return fewer/more results.
2. Universe/report language reflects curated longlist behavior.

### P1 - FR/UK filing metric coverage is implicit instead of explicit
- Owner: Backend
- Status: Resolved
- Acceptance criteria:
1. Cards show filing metrics only when source-backed facts exist.
2. Unsupported jurisdictions display explicit coverage note.

### P1 - Missing source labeling for weak claims
- Owner: Backend + Frontend
- Status: Resolved
- Acceptance criteria:
1. Claims without source are rendered as `hypothesis`.
2. Fact claims expose clickable source pills with URL + label.

## Lens correctness

### P0 - Complement lens placeholder/TODO logic
- Owner: Backend
- Status: Resolved
- Acceptance criteria:
1. Complement score uses fixed weighted components.
2. Non-trivial components without evidence contribute zero.
3. No TODO placeholder remains in complement route.

### P1 - Compete/complement scoring not persisted in immutable snapshots
- Owner: Backend
- Status: Resolved
- Acceptance criteria:
1. Snapshot generation persists compete and complement scores per vendor.
2. Regenerating creates a new snapshot; old snapshot rows are unchanged.

## Security/trust

### P0 - Unsafe export path write in context-pack export endpoint
- Owner: Backend
- Status: Resolved
- Acceptance criteria:
1. API no longer accepts arbitrary filesystem output path.
2. Export returns controlled payload only.

### P1 - Weak workspace guardrails for nested resources
- Owner: Backend
- Status: Resolved
- Acceptance criteria:
1. Report endpoints enforce workspace scoping for snapshot/item/vendor reads.
2. Nested vendor/report reads cannot leak across workspace IDs.

## UX/reporting

### P0 - Context form state hydration bug
- Owner: Frontend
- Status: Resolved
- Acceptance criteria:
1. Context form initializes from fetched profile using `useEffect`.
2. Saved values do not flicker/reset on render.

### P1 - Static report page missing in workspace flow
- Owner: Frontend
- Status: Resolved
- Acceptance criteria:
1. Navigation flow is `Context -> Bricks -> Universe -> Report`.
2. Report page supports snapshot generation, snapshot selection, compete/complement toggle, and export.

## Docs/debt

### P0 - README/API model mismatch (legacy strategies/targets docs)
- Owner: Docs + Backend
- Status: Resolved
- Acceptance criteria:
1. README reflects workspace/report architecture and current endpoints.
2. Legacy strategy/target API docs removed or clearly marked deprecated.

### P2 - Legacy map/lenses pages still present (not in primary flow)
- Owner: Frontend
- Status: Open
- Acceptance criteria:
1. Either remove deprecated pages or add explicit deprecation banners.
2. Ensure no primary navigation or onboarding copy points users there.
