#!/usr/bin/env python3
from app.workers.celery_app import DISCOVERY_QUEUE_NAMES, QUEUE_NAMESPACE


if __name__ == "__main__":
    print({"queue_namespace": QUEUE_NAMESPACE, "queues": DISCOVERY_QUEUE_NAMES})
