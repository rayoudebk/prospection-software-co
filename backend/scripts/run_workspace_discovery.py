#!/usr/bin/env python3
from __future__ import annotations

import argparse

from app.config import get_settings
from app.models.job import Job, JobProvider, JobState, JobType
from app.workers.workspace_tasks import (
    SessionLocal,
    _run_discovery_universe_fixture,
    run_discovery_universe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run workspace discovery in the configured execution mode.")
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to run.")
    parser.add_argument(
        "--sync-fixture",
        action="store_true",
        help="When DISCOVERY_EXECUTION_MODE=fixture, run the fixture pipeline synchronously in-process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    execution_mode = str(getattr(settings, "discovery_execution_mode", "live") or "live").strip().lower() or "live"

    db = SessionLocal()
    try:
        job = Job(
            workspace_id=int(args.workspace_id),
            job_type=JobType.discovery_universe,
            state=JobState.queued,
            provider=JobProvider.gemini_flash,
            progress=0.0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        if execution_mode == "fixture" and args.sync_fixture:
            result = _run_discovery_universe_fixture(job.id)
            db.expire_all()
            refreshed = db.query(Job).filter(Job.id == job.id).first()
            print(
                {
                    "workspace_id": int(args.workspace_id),
                    "job_id": job.id,
                    "execution_mode": execution_mode,
                    "sync_fixture": True,
                    "result": result,
                    "state": refreshed.state.value if refreshed else None,
                    "run_quality_tier": (refreshed.result_json or {}).get("run_quality_tier") if refreshed else None,
                    "ranking_eligible_count": (refreshed.result_json or {}).get("ranking_eligible_count") if refreshed else None,
                }
            )
            return

        task = run_discovery_universe.delay(job.id)
        print(
            {
                "workspace_id": int(args.workspace_id),
                "job_id": job.id,
                "execution_mode": execution_mode,
                "sync_fixture": False,
                "task_id": str(task.id),
            }
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
