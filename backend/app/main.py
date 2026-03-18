from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.models.base import init_db
from app.api import workspaces
from app.startup_migrations import run_startup_migrations
from app.services.discovery_readiness import (
    assert_critical_runtime_ready,
    compute_runtime_discovery_readiness,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    await init_db()
    run_startup_migrations()
    app.state.discovery_runtime_readiness = assert_critical_runtime_ready()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="M&A Market Map API",
    description="Workspace-based M&A research and due diligence platform",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - Workspace-based API
app.include_router(workspaces.router, prefix="/workspaces", tags=["workspaces"])


@app.get("/health")
async def health_check():
    readiness = compute_runtime_discovery_readiness(check_worker=True)
    status = "healthy" if readiness.get("db_schema_ok") and readiness.get("redis_available") else "degraded"
    return {"status": status, "discovery_readiness": readiness}


@app.get("/")
async def root():
    return {"message": "M&A Due Diligence API", "docs": "/docs"}
