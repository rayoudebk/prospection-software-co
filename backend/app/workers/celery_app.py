from celery import Celery
from app.config import get_settings

settings = get_settings()
QUEUE_NAMESPACE = settings.resolved_celery_queue_namespace()


def discovery_queue_name(queue_suffix: str) -> str:
    suffix = str(queue_suffix or "").strip()
    if not suffix.startswith("discovery."):
        suffix = f"discovery.{suffix}"
    return f"{QUEUE_NAMESPACE}.{suffix}" if QUEUE_NAMESPACE else suffix


DISCOVERY_QUEUE_NAMES = {
    "crawl": discovery_queue_name("discovery.crawl"),
    "search": discovery_queue_name("discovery.search"),
    "registry": discovery_queue_name("discovery.registry"),
    "score": discovery_queue_name("discovery.score"),
}

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
    task_default_queue=DISCOVERY_QUEUE_NAMES["search"],
    task_routes={
        "app.workers.workspace_tasks.generate_context_pack_v2": {"queue": DISCOVERY_QUEUE_NAMES["crawl"]},
        "app.workers.workspace_tasks.run_company_context_refresh": {"queue": DISCOVERY_QUEUE_NAMES["crawl"]},
        "app.workers.workspace_tasks.run_enrich_company": {"queue": DISCOVERY_QUEUE_NAMES["crawl"]},
        "app.workers.workspace_tasks.run_monitoring_delta": {"queue": DISCOVERY_QUEUE_NAMES["crawl"]},
        "app.workers.workspace_tasks.run_discovery_universe": {"queue": DISCOVERY_QUEUE_NAMES["search"]},
        "app.workers.workspace_tasks.stage_seed_ingest": {"queue": DISCOVERY_QUEUE_NAMES["search"]},
        "app.workers.workspace_tasks.stage_llm_discovery_fanout": {"queue": DISCOVERY_QUEUE_NAMES["search"]},
        "app.workers.workspace_tasks.stage_registry_identity_expand": {"queue": DISCOVERY_QUEUE_NAMES["registry"]},
        "app.workers.workspace_tasks.stage_first_party_enrichment_parallel": {"queue": DISCOVERY_QUEUE_NAMES["crawl"]},
        "app.workers.workspace_tasks.stage_scoring_claims_persist": {"queue": DISCOVERY_QUEUE_NAMES["score"]},
        "app.workers.workspace_tasks.run_claims_graph_refresh": {"queue": DISCOVERY_QUEUE_NAMES["score"]},
        "app.workers.workspace_tasks.generate_static_report": {"queue": DISCOVERY_QUEUE_NAMES["score"]},
        "app.workers.workspace_tasks.finalize_discovery_pipeline": {"queue": DISCOVERY_QUEUE_NAMES["score"]},
        "app.workers.workspace_tasks.fail_discovery_pipeline": {"queue": DISCOVERY_QUEUE_NAMES["score"]},
        "app.workers.workspace_tasks.discovery_watchdog": {"queue": DISCOVERY_QUEUE_NAMES["score"]},
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
