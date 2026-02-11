from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "prospection",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks", "app.workers.workspace_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # Soft limit at 9 minutes
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.workers.workspace_tasks.run_discovery_universe": {"queue": "discovery.search"},
        "app.workers.workspace_tasks.stage_seed_ingest": {"queue": "discovery.search"},
        "app.workers.workspace_tasks.stage_llm_discovery_fanout": {"queue": "discovery.search"},
        "app.workers.workspace_tasks.stage_registry_identity_expand": {"queue": "discovery.registry"},
        "app.workers.workspace_tasks.stage_first_party_enrichment_parallel": {"queue": "discovery.crawl"},
        "app.workers.workspace_tasks.stage_scoring_claims_persist": {"queue": "discovery.score"},
        "app.workers.workspace_tasks.finalize_discovery_pipeline": {"queue": "discovery.score"},
        "app.workers.workspace_tasks.fail_discovery_pipeline": {"queue": "discovery.score"},
        "app.workers.workspace_tasks.discovery_watchdog": {"queue": "discovery.score"},
    },
    # Stage-specific overrides while the heavy monolith body still runs in stage_scoring_claims_persist.
    # Keep hard ceilings explicit so jobs terminate predictably, but avoid premature worker kills.
    task_annotations={
        "app.workers.workspace_tasks.stage_scoring_claims_persist": {
            "time_limit": max(900, int(settings.discovery_global_timeout_seconds)),
            "soft_time_limit": max(840, int(settings.discovery_global_timeout_seconds) - 60),
        },
        "app.workers.workspace_tasks.discovery_watchdog": {
            "time_limit": 120,
            "soft_time_limit": 90,
        },
        "app.workers.workspace_tasks.fail_discovery_pipeline": {
            "time_limit": 120,
            "soft_time_limit": 90,
        },
    },
)
