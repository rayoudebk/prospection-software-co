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

## migrate_remove_legacy_taxonomy_v1.py

Removes the deprecated taxonomy schema after scope review is in place.

### What it does:

1. Ensures `company_context_packs` exist
2. Backfills missing company-context data for older workspaces
3. Normalizes `decision_policy_json` to use `scope_review`
4. Drops `brick_mappings`
5. Drops `brick_taxonomies`
6. Drops `vendors.tags_vertical`

### How to run:

```bash
cd backend
python -m migrations.migrate_remove_legacy_taxonomy_v1
```

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

## migrate_company_profile_context_split_v1.py

Adds explicit manual-vs-generated context fields on company profiles.

### What it does:

1. Adds `manual_brief_text` to `company_profiles`
2. Adds `generated_context_summary` to `company_profiles`
3. Backfills legacy `buyer_context_summary` into:
   - `manual_brief_text` when no `buyer_company_url` exists
   - `generated_context_summary` when `buyer_company_url` exists

### How to run:

```bash
cd backend
python -m migrations.migrate_company_profile_context_split_v1
```

## migrate_remove_legacy_buyer_context_summary_v1.py

Removes the deprecated `buyer_context_summary` column after the explicit split fields are available.

### What it does:

1. Ensures `manual_brief_text` exists
2. Ensures `generated_context_summary` exists
3. Backfills both fields from `buyer_context_summary` when needed
4. Drops `company_profiles.buyer_context_summary`

### How to run:

```bash
cd backend
python -m migrations.migrate_remove_legacy_buyer_context_summary_v1
```

## migrate_external_search_persistence_v1.py

Adds external search persistence tables:

- `external_search_runs`
- `external_search_results`

### How to run:

```bash
cd backend
python -m migrations.migrate_external_search_persistence_v1
```

## migrate_company_context_backfill_v1.py

Adds the company-context sourcing table and backfills existing workspaces.

### What it does:

1. Creates `company_context_packs`
2. Backfills missing company-context packs from `company_profiles`
3. Preserves existing company-context rows if they already contain data

### How to run:

```bash
cd backend
python -m migrations.migrate_company_context_backfill_v1
```
