from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.workers.celery_app import celery_app
from app.config import get_settings
from app.models.base import Base
from app.models.strategy import Strategy
from app.models.target import Target, TargetStatus
from app.models.research_job import ResearchJob, JobState
from app.models.evidence import EvidenceItem

# Sync engine for Celery workers (Celery doesn't support async)
settings = get_settings()
sync_engine = create_engine(settings.database_url_sync, echo=settings.debug)
SessionLocal = sessionmaker(bind=sync_engine)


def get_sync_db():
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Caller must close


@celery_app.task(name="app.workers.tasks.generate_context_pack")
def generate_context_pack(job_id: int):
    """Generate context pack from strategy seed URLs."""
    import asyncio
    from app.services.crawler import UnifiedCrawler

    db = SessionLocal()
    try:
        job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.state = JobState.running
        job.started_at = datetime.utcnow()
        db.commit()

        strategy = db.query(Strategy).filter(Strategy.id == job.strategy_id).first()
        if not strategy:
            job.state = JobState.failed
            job.error_message = "Strategy not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Strategy not found"}

        try:
            # Run async crawler in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            combined_markdown = []
            crawler = UnifiedCrawler(max_pages=30)
            
            for url in strategy.seed_urls:
                try:
                    context_pack_result = loop.run_until_complete(crawler.crawl_for_context(url))
                    combined_markdown.append(context_pack_result.raw_markdown)
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
                    continue
            
            loop.close()
            
            context_pack = "\n\n---\n\n".join(combined_markdown)
            strategy.context_pack = context_pack
            job.state = JobState.completed
            job.result_json = {"context_pack_length": len(context_pack)}
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"success": True, "context_pack_length": len(context_pack)}
        except Exception as e:
            job.state = JobState.failed
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(e)}
    finally:
        db.close()


def normalize_domain(url: str | None) -> str | None:
    """Extract and normalize domain from URL for deduplication."""
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url.lower() if url else None


def get_seed_domains(seed_urls: list) -> set:
    """Extract normalized domains from seed URLs."""
    domains = set()
    for url in seed_urls:
        domain = normalize_domain(url)
        if domain:
            domains.add(domain)
    return domains


@celery_app.task(name="app.workers.tasks.generate_landscape")
def generate_landscape(job_id: int):
    """Generate landscape (top candidates) using Gemini."""
    from app.services.gemini_client import GeminiClient
    from app.services.bpo_scorer import compute_bpo_score

    db = SessionLocal()
    try:
        job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.state = JobState.running
        job.started_at = datetime.utcnow()
        db.commit()

        strategy = db.query(Strategy).filter(Strategy.id == job.strategy_id).first()
        if not strategy:
            job.state = JobState.failed
            job.error_message = "Strategy not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Strategy not found"}

        try:
            client = GeminiClient()
            candidates = client.generate_landscape(
                context_pack=strategy.context_pack or "",
                region_scope=strategy.region_scope,
                intent=strategy.intent,
                exclusions=strategy.exclusions,
            )

            # Get existing target domains for this strategy to avoid duplicates
            existing_targets = db.query(Target).filter(Target.strategy_id == strategy.id).all()
            existing_domains = {normalize_domain(t.website) for t in existing_targets if t.website}
            
            # Get seed domains to mark seed companies
            seed_domains = get_seed_domains(strategy.seed_urls)

            # Create targets from candidates
            created_targets = []
            skipped_duplicates = 0
            
            for candidate in candidates:
                candidate_domain = normalize_domain(candidate.get("website"))
                
                # Skip duplicates
                if candidate_domain and candidate_domain in existing_domains:
                    skipped_duplicates += 1
                    continue
                
                # Track this domain to avoid duplicates within the same batch
                if candidate_domain:
                    existing_domains.add(candidate_domain)

                # Compute BPO score
                bpo_score, bpo_rationale = compute_bpo_score(candidate)

                # Apply kill rule
                status = TargetStatus.candidate
                if bpo_score >= 70 and strategy.exclusions.get("bpo_toggle", True):
                    status = TargetStatus.rejected
                
                # Check if this is a seed company
                is_seed = candidate_domain in seed_domains if candidate_domain else False

                target = Target(
                    strategy_id=strategy.id,
                    name=candidate.get("name", "Unknown"),
                    website=candidate.get("website"),
                    country=candidate.get("country"),
                    status=status,
                    is_seed=is_seed,
                    bpo_score=bpo_score,
                    bpo_rationale=bpo_rationale,
                    fit_score=candidate.get("fit_score", 50),
                    fit_rationale=candidate.get("why_fit", ""),
                    similarities=candidate.get("similarities"),
                    watchouts=candidate.get("watchouts"),
                )
                db.add(target)
                db.flush()

                # Add evidence items
                for link in candidate.get("evidence_links", []):
                    evidence = EvidenceItem(
                        target_id=target.id,
                        source_url=link,
                        excerpt=candidate.get("why_fit", ""),
                        content_type="web",
                    )
                    db.add(evidence)

                created_targets.append(target.id)

            job.state = JobState.completed
            job.result_json = {
                "targets_created": len(created_targets),
                "target_ids": created_targets,
                "duplicates_skipped": skipped_duplicates,
            }
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"success": True, "targets_created": len(created_targets), "duplicates_skipped": skipped_duplicates}

        except Exception as e:
            job.state = JobState.failed
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="app.workers.tasks.generate_deep_profile")
def generate_deep_profile(job_id: int):
    """Generate deep profile for a specific target."""
    from app.services.gemini_client import GeminiClient

    db = SessionLocal()
    try:
        job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.state = JobState.running
        job.started_at = datetime.utcnow()
        db.commit()

        target = db.query(Target).filter(Target.id == job.target_id).first()
        if not target:
            job.state = JobState.failed
            job.error_message = "Target not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": "Target not found"}

        strategy = db.query(Strategy).filter(Strategy.id == job.strategy_id).first()

        try:
            client = GeminiClient()
            profile_data = client.generate_deep_profile(
                target_name=target.name,
                target_website=target.website,
                context_pack=strategy.context_pack if strategy else "",
            )

            target.profile_markdown = profile_data.get("markdown", "")

            # Add new evidence items from profile research
            for link in profile_data.get("evidence_links", []):
                evidence = EvidenceItem(
                    target_id=target.id,
                    source_url=link,
                    excerpt=profile_data.get("excerpt", ""),
                    content_type="web",
                )
                db.add(evidence)

            job.state = JobState.completed
            job.result_json = {"profile_length": len(target.profile_markdown)}
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"success": True}

        except Exception as e:
            job.state = JobState.failed
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"error": str(e)}
    finally:
        db.close()

