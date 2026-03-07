"""
Migration script for workspace decision policy and P2 support tables.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text

from app.config import get_settings
from app.models.claims_graph import ClaimGraphNode, ClaimGraphEdge, ClaimGraphEdgeEvidence
from app.models.evaluation import EvaluationRun, EvaluationSampleResult
from app.models.job import JobType
from app.models.workspace_feedback import WorkspaceFeedbackEvent
from app.services.evidence_policy import DEFAULT_EVIDENCE_POLICY

settings = get_settings()


def _ensure_postgres_enum_values(conn, enum_type_name: str, values: list[str]) -> None:
    if conn.dialect.name != "postgresql":
        return
    rows = conn.execute(
        text(
            """
            SELECT e.enumlabel
            FROM pg_type t
            JOIN pg_enum e ON t.oid = e.enumtypid
            WHERE t.typname = :enum_name
            """
        ),
        {"enum_name": enum_type_name},
    ).fetchall()
    existing = {str(row[0]) for row in rows}
    for value in values:
        if value in existing:
            continue
        print(f"Adding enum value {enum_type_name}.{value}")
        conn.execute(text(f"ALTER TYPE {enum_type_name} ADD VALUE IF NOT EXISTS '{value}'"))


def migrate_workspace_policy_v1():
    engine = create_engine(settings.database_url_sync, echo=True)
    with engine.begin() as conn:
        inspector = inspect(conn)
        cols = {c["name"] for c in inspector.get_columns("workspaces")}
        if "decision_policy_json" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN decision_policy_json JSON"))
        conn.execute(
            text(
                "UPDATE workspaces SET decision_policy_json = COALESCE(decision_policy_json, :policy)"
            ),
            {"policy": json.dumps(DEFAULT_EVIDENCE_POLICY)},
        )
        _ensure_postgres_enum_values(conn, "jobtype", [job_type.value for job_type in JobType])

        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        if "workspace_feedback_events" not in tables:
            WorkspaceFeedbackEvent.__table__.create(bind=conn)
        if "claim_graph_nodes" not in tables:
            ClaimGraphNode.__table__.create(bind=conn)
        if "claim_graph_edges" not in tables:
            ClaimGraphEdge.__table__.create(bind=conn)
        if "claim_graph_edge_evidence" not in tables:
            ClaimGraphEdgeEvidence.__table__.create(bind=conn)
        if "evaluation_runs" not in tables:
            EvaluationRun.__table__.create(bind=conn)
        if "evaluation_sample_results" not in tables:
            EvaluationSampleResult.__table__.create(bind=conn)


if __name__ == "__main__":
    print("Starting workspace decision policy migration...")
    migrate_workspace_policy_v1()
    print("✅ Migration complete")
