# Database Migrations

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
