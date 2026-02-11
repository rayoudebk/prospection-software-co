# Database Migrations

## migrate_canonical_entities.py

Adds canonical discovery entity support:

1. Creates `candidate_entities`, `candidate_entity_aliases`, `candidate_origin_edges`, and `registry_query_logs`
2. Adds `candidate_entity_id` to `vendor_screenings` (if missing)
3. Adds index on `vendor_screenings.candidate_entity_id`
4. Adds FK to `candidate_entities(id)` on PostgreSQL (best effort)

### How to run:

**From project root:**
```bash
cd backend
python -m migrations.migrate_canonical_entities
```

**Or with Docker:**
```bash
docker-compose exec backend python -m migrations.migrate_canonical_entities
```

## migrate_vertical_focus.py

Migrates `vertical_focus` data from `company_profiles` table to `brick_taxonomies` table.

### What it does:

1. Finds all `company_profiles` with `vertical_focus` data
2. Finds their corresponding `brick_taxonomies` (via `workspace_id`)
3. Copies `vertical_focus` from `company_profiles` to `brick_taxonomies`
4. Merges values if `brick_taxonomy` already has `vertical_focus` data
5. Optionally removes `vertical_focus` from `company_profiles` (commented out for safety)

### How to run:

**From project root:**
```bash
cd backend
python -m migrations.migrate_vertical_focus
```

**Or with Docker:**
```bash
docker-compose exec backend python -m migrations.migrate_vertical_focus
```

### Safety:

- The script does NOT remove data from `company_profiles` by default
- It merges existing values if both tables have data
- It skips workspaces that don't have a `brick_taxonomy`
- All changes are committed in a transaction (rolls back on error)

### After migration:

Once you've verified the migration worked correctly, you can optionally clean up by uncommenting the cleanup section at the end of the script to remove `vertical_focus` from `company_profiles`.

## migrate_decision_hygiene_v1.py

Adds decision hygiene schema fields across evidence, claims, and screening tables.

### What it does:

1. Adds evidence provenance/quality columns to `workspace_evidence`
2. Adds claim grouping, claim status, contradiction and freshness columns to `vendor_claims`
3. Adds decision classification, reason-code, rationale, and sufficiency fields to `vendor_screenings`
4. Backfills `source_tier` / `source_kind` heuristically from existing source URLs
5. Backfills `decision_classification` from legacy `screening_status`

### How to run:

```bash
cd backend
python -m migrations.migrate_decision_hygiene_v1
```

## migrate_report_item_decision_fields_v1.py

Adds decision metadata fields to report snapshot items.

### What it does:

1. Adds `decision_classification` to `report_snapshot_items`
2. Adds `reason_codes_json` to `report_snapshot_items`
3. Adds `evidence_summary_json` to `report_snapshot_items`
4. Backfills null JSON fields with empty objects

### How to run:

```bash
cd backend
python -m migrations.migrate_report_item_decision_fields_v1
```

## migrate_workspace_policy_v1.py

Adds workspace-level policy storage and P2 support tables.

### What it does:

1. Adds `decision_policy_json` to `workspaces` and backfills defaults
2. Syncs PostgreSQL `jobtype` enum values with current `JobType` definitions (including `monitoring_delta`)
3. Creates `workspace_feedback_events`
4. Creates claims graph tables:
   - `claim_graph_nodes`
   - `claim_graph_edges`
   - `claim_graph_edge_evidence`
5. Creates evaluation tables:
   - `evaluation_runs`
   - `evaluation_sample_results`

### How to run:

```bash
cd backend
python -m migrations.migrate_workspace_policy_v1
```

## migrate_directory_identity_hardening_v2.py

Adds directory-safe identity fields for mention/entity/screening records.

### What it does:

1. Adds identity split fields to `vendor_mentions`:
   - `profile_url`, `official_website_url`, `company_slug`, `solution_slug`, `entity_type`
2. Adds canonical entity extensions to `candidate_entities`:
   - `entity_type`, `first_party_domains_json`, `solutions_json`, `discovery_primary_url`
3. Adds screening identity/quality fields to `vendor_screenings`:
   - `candidate_discovery_url`, `candidate_official_website`, `top_claim_json`, `ranking_eligible`
4. Backfills profile/official URLs, slug fields, and defaults from existing data.

### How to run:

```bash
cd backend
python -m migrations.migrate_directory_identity_hardening_v2
```

## migrate_ranking_eligibility_v2.py

Backfills ranking eligibility and entity typing with directory-safe defaults.

### What it does:

1. Backfills `vendor_mentions.entity_type` from `/vendors/{company}/{solution}` URL patterns.
2. Backfills `vendor_screenings.ranking_eligible` based on:
   - entity type is company
   - official website is resolved and not directory-hosted
   - classification is not `not_good_target`
   - first-party source presence proxy in source summary

### How to run:

```bash
cd backend
python -m migrations.migrate_ranking_eligibility_v2
```

## migrate_company_profile_reference_evidence_v1.py

Adds explicit context evidence-link storage on company profiles.

### What it does:

1. Adds `reference_evidence_urls` to `company_profiles`
2. Backfills null rows to `[]`

### How to run:

```bash
cd backend
python -m migrations.migrate_company_profile_reference_evidence_v1
```
