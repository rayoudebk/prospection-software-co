#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models.base import Base
import app.models  # noqa: F401
from app.models.company import Company
from app.models.intelligence import CandidateEntity, CompanyClaim, CompanyScreening, RegistryQueryLog
from app.models.job import Job, JobType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset discovery/universe state for a workspace.")
    parser.add_argument("--workspace-id", type=int, required=True, help="Workspace id to reset.")
    parser.add_argument(
        "--delete-manual",
        action="store_true",
        help="Also delete manually added companies. Default keeps them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    engine = create_engine(settings.database_url_sync)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as db:  # type: Session
        workspace_id = int(args.workspace_id)

        for row in db.query(CompanyClaim).filter(CompanyClaim.workspace_id == workspace_id).all():
            db.delete(row)
        for row in db.query(CompanyScreening).filter(CompanyScreening.workspace_id == workspace_id).all():
            db.delete(row)
        for row in db.query(RegistryQueryLog).filter(RegistryQueryLog.workspace_id == workspace_id).all():
            db.delete(row)
        for row in db.query(CandidateEntity).filter(CandidateEntity.workspace_id == workspace_id).all():
            db.delete(row)
        for row in (
            db.query(Job)
            .filter(Job.workspace_id == workspace_id, Job.job_type == JobType.discovery_universe)
            .all()
        ):
            db.delete(row)

        company_query = db.query(Company).filter(Company.workspace_id == workspace_id)
        if not args.delete_manual:
            company_query = company_query.filter(Company.is_manual.is_(False))
        for row in company_query.all():
            db.delete(row)

        db.commit()
        print(
            {
                "workspace_id": workspace_id,
                "deleted_manual_companies": bool(args.delete_manual),
                "status": "ok",
            }
        )


if __name__ == "__main__":
    main()
