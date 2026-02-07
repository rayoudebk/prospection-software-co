"""
Migration script to move vertical_focus from company_profiles to brick_taxonomies.

This script:
1. Finds all company_profiles with vertical_focus data
2. Finds their corresponding brick_taxonomies (via workspace_id)
3. Copies vertical_focus from company_profiles to brick_taxonomies
4. Removes vertical_focus from company_profiles (optional, commented out for safety)

Run with:
    python -m migrations.migrate_vertical_focus
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()


def migrate_vertical_focus():
    """Migrate vertical_focus from company_profiles to brick_taxonomies."""
    # Use sync engine for migration
    engine = create_engine(settings.database_url_sync, echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Step 0: Add vertical_focus column to brick_taxonomies if it doesn't exist
        print("Checking if vertical_focus column exists in brick_taxonomies...")
        try:
            # Check if column exists by querying information_schema
            col_check = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'brick_taxonomies' 
                AND column_name = 'vertical_focus'
            """))
            if col_check.fetchone():
                print("  ✓ Column already exists")
            else:
                raise Exception("Column not found")
        except Exception:
            print("  ⚠️  Column doesn't exist, adding it...")
            session.execute(text("""
                ALTER TABLE brick_taxonomies 
                ADD COLUMN IF NOT EXISTS vertical_focus JSON DEFAULT '[]'::json
            """))
            session.commit()
            print("  ✓ Column added successfully")
        
        # Step 1: Find all company_profiles with vertical_focus
        # Get all profiles and filter in Python to avoid JSON/JSONB type issues
        result = session.execute(text("""
            SELECT id, workspace_id, vertical_focus
            FROM company_profiles
            WHERE vertical_focus IS NOT NULL
        """))
        
        all_profiles = result.fetchall()
        
        # Filter out empty arrays in Python
        profiles_with_verticals = []
        for profile_id, workspace_id, vertical_focus in all_profiles:
            # Parse and check if it's a non-empty list
            try:
                if isinstance(vertical_focus, str):
                    parsed = json.loads(vertical_focus)
                else:
                    parsed = vertical_focus
                
                if isinstance(parsed, list) and len(parsed) > 0:
                    profiles_with_verticals.append((profile_id, workspace_id, parsed))
            except:
                # Skip if we can't parse it
                continue
        
        print(f"Found {len(profiles_with_verticals)} company profiles with vertical_focus data")

        migrated_count = 0
        skipped_count = 0

        for profile_id, workspace_id, vertical_focus in profiles_with_verticals:
            # Step 2: Find corresponding brick_taxonomy
            taxonomy_result = session.execute(text("""
                SELECT id, COALESCE(vertical_focus, '[]'::json) as vertical_focus
                FROM brick_taxonomies
                WHERE workspace_id = :workspace_id
            """), {"workspace_id": workspace_id})
            
            taxonomy = taxonomy_result.fetchone()
            
            if not taxonomy:
                print(f"  ⚠️  Workspace {workspace_id} has no brick_taxonomy, skipping")
                skipped_count += 1
                continue
            
            taxonomy_id, existing_verticals = taxonomy
            
            # Step 3: Check if taxonomy already has vertical_focus
            # If it does, we'll merge (union) the values
            if existing_verticals:
                # Parse existing verticals (could be JSON string or already parsed)
                if isinstance(existing_verticals, str):
                    try:
                        existing_list = json.loads(existing_verticals)
                    except:
                        existing_list = []
                else:
                    existing_list = existing_verticals if isinstance(existing_verticals, list) else []
                
                # vertical_focus is already parsed from the filter step above
                new_list = vertical_focus if isinstance(vertical_focus, list) else []
                
                # Merge (union) - keep unique values
                merged = list(set(existing_list + new_list))
                final_verticals = json.dumps(merged)
                
                if set(existing_list) == set(merged):
                    print(f"  ✓ Workspace {workspace_id}: vertical_focus already exists, skipping")
                    skipped_count += 1
                    continue
            else:
                # vertical_focus is already parsed from the filter step above
                final_verticals = json.dumps(vertical_focus) if isinstance(vertical_focus, list) else json.dumps([])
            
            # Step 4: Update brick_taxonomy with vertical_focus
            # Use JSON type (not JSONB) to match the column type
            # Cast the JSON string properly using CAST
            session.execute(text("""
                UPDATE brick_taxonomies
                SET vertical_focus = CAST(:vertical_focus AS JSON),
                    updated_at = NOW()
                WHERE id = :taxonomy_id
            """), {
                "taxonomy_id": taxonomy_id,
                "vertical_focus": final_verticals
            })
            
            print(f"  ✓ Migrated vertical_focus for workspace {workspace_id} (profile {profile_id} -> taxonomy {taxonomy_id})")
            migrated_count += 1
        
        # Step 5: Commit changes
        session.commit()
        print(f"\n✅ Migration complete!")
        print(f"   Migrated: {migrated_count}")
        print(f"   Skipped: {skipped_count}")
        
        # Optional: Remove vertical_focus from company_profiles
        # Uncomment the following lines if you want to clean up:
        # print("\n⚠️  Removing vertical_focus from company_profiles...")
        # session.execute(text("""
        #     UPDATE company_profiles
        #     SET vertical_focus = NULL
        #     WHERE vertical_focus IS NOT NULL
        # """))
        # session.commit()
        # print("✅ Cleanup complete!")
        
    except Exception as e:
        session.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print("Starting vertical_focus migration...")
    print("=" * 60)
    migrate_vertical_focus()
    print("=" * 60)
