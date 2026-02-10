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
