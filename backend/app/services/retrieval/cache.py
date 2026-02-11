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

    @staticmethod
    def _key(namespace: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"retrieval:{namespace}:{digest}"

    def get_json(self, namespace: str, payload: str) -> Optional[Any]:
        key = self._key(namespace, payload)
        try:
            raw = self._client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception:
            return None

    def set_json(self, namespace: str, payload: str, value: Any, ttl_seconds: int) -> None:
        key = self._key(namespace, payload)
        try:
            self._client.setex(key, max(1, int(ttl_seconds)), json.dumps(value))
        except Exception:
            return
