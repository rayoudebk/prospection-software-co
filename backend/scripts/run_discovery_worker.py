#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import subprocess
import sys
from uuid import uuid4

from app.config import get_settings
from app.workers.celery_app import DISCOVERY_QUEUE_NAMES


def main() -> int:
    settings = get_settings()
    queue_list = ",".join(
        [
            DISCOVERY_QUEUE_NAMES["search"],
            DISCOVERY_QUEUE_NAMES["score"],
            DISCOVERY_QUEUE_NAMES["crawl"],
            DISCOVERY_QUEUE_NAMES["registry"],
        ]
    )
    cmd = [
        "celery",
        "-A",
        "app.workers.celery_app",
        "worker",
        "-n",
        f"discovery-{settings.resolved_celery_queue_namespace() or 'local'}-{os.getpid()}-{uuid4().hex[:6]}@%h",
        "-Q",
        queue_list,
        "--loglevel=warning",
    ]
    if platform.system().lower() == "darwin":
        cmd.extend(["--pool=solo", "--concurrency=1"])
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "backend")
    print({"command": cmd, "queue_list": queue_list})
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
