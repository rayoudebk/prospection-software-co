from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import redis

from app.config import get_settings


class RetrievalCache:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._stats_key = "retrieval:stats:v1"

    @staticmethod
    def _key(namespace: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"retrieval:{namespace}:{digest}"

    def get_json(self, namespace: str, payload: str) -> Optional[Any]:
        key = self._key(namespace, payload)
        try:
            raw = self._client.get(key)
            if not raw:
                self._record_stat(namespace, "miss")
                return None
            self._record_stat(namespace, "hit")
            return json.loads(raw)
        except Exception:
            self._record_stat(namespace, "error")
            return None

    def set_json(self, namespace: str, payload: str, value: Any, ttl_seconds: int) -> None:
        key = self._key(namespace, payload)
        try:
            self._client.setex(key, max(1, int(ttl_seconds)), json.dumps(value))
        except Exception:
            self._record_stat(namespace, "set_error")
            return

    def stats_snapshot(self) -> dict[str, int]:
        try:
            raw = self._client.hgetall(self._stats_key) or {}
        except Exception:
            return {}
        snapshot: dict[str, int] = {}
        for key, value in raw.items():
            try:
                snapshot[str(key)] = int(value)
            except Exception:
                continue
        return snapshot

    def _record_stat(self, namespace: str, event: str) -> None:
        field = f"{str(namespace).strip().lower()}:{str(event).strip().lower()}"
        if not field:
            return
        try:
            self._client.hincrby(self._stats_key, field, 1)
        except Exception:
            return
