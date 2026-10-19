"""Optional Redis snapshots keyed by the committed database message version."""

from __future__ import annotations

import hashlib
import json
import logging

from pydantic import BaseModel, Field, model_validator

from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class SessionSnapshot(BaseModel):
    user_id: str
    session_id: str
    last_message_id: int = Field(ge=0)
    summary: str = ""
    summary_until_message_id: int = Field(default=0, ge=0)
    messages: list[ChatMessage] = Field(default_factory=list)
    message_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_messages(self) -> SessionSnapshot:
        if len(self.messages) != len(self.message_ids):
            raise ValueError("messages and IDs must align")
        if self.message_ids != sorted(set(self.message_ids)):
            raise ValueError("message IDs must increase")
        if any(i <= self.summary_until_message_id for i in self.message_ids):
            raise ValueError("messages must follow the summary cursor")
        last = self.message_ids[-1] if self.message_ids else self.summary_until_message_id
        if last != self.last_message_id:
            raise ValueError("snapshot is incomplete")
        if any(m.role not in {"user", "assistant"} for m in self.messages):
            raise ValueError("invalid persisted message role")
        return self


def session_cache_key(
    user_id: str,
    session_id: str,
    last_message_id: int,
    summary_until_message_id: int,
    summary: str,
) -> str:
    identity = json.dumps(
        [user_id, session_id, last_message_id, summary_until_message_id, summary],
        ensure_ascii=False,
    )
    return "chat:context:v1:" + hashlib.sha256(identity.encode()).hexdigest()


class SessionContextCache:
    enabled = False

    async def get(
        self,
        *,
        user_id: str,
        session_id: str,
        last_message_id: int,
        summary_until_message_id: int,
        summary: str,
    ) -> SessionSnapshot | None:
        return None

    async def set(self, snapshot: SessionSnapshot) -> None:
        return None


class RedisSessionContextCache(SessionContextCache):
    enabled = True

    def __init__(self, redis, *, ttl_seconds: int = 1800, max_bytes: int = 2_000_000) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._max_bytes = max_bytes

    async def get(
        self,
        *,
        user_id: str,
        session_id: str,
        last_message_id: int,
        summary_until_message_id: int,
        summary: str,
    ) -> SessionSnapshot | None:
        key = session_cache_key(
            user_id, session_id, last_message_id, summary_until_message_id, summary
        )
        try:
            raw = await self._redis.get(key)
            if not raw or len(raw.encode() if isinstance(raw, str) else raw) > self._max_bytes:
                return None
            snapshot = SessionSnapshot.model_validate_json(raw)
            if (
                snapshot.user_id != user_id
                or snapshot.session_id != session_id
                or snapshot.last_message_id != last_message_id
                or snapshot.summary_until_message_id != summary_until_message_id
                or snapshot.summary != summary
            ):
                return None
            return snapshot
        except Exception:
            logger.warning("session_cache_get_failed")
            return None

    async def set(self, snapshot: SessionSnapshot) -> None:
        try:
            raw = snapshot.model_dump_json()
            if len(raw.encode()) > self._max_bytes:
                return
            await self._redis.set(
                session_cache_key(
                    snapshot.user_id,
                    snapshot.session_id,
                    snapshot.last_message_id,
                    snapshot.summary_until_message_id,
                    snapshot.summary,
                ),
                raw,
                ex=self._ttl,
            )
        except Exception:
            logger.warning("session_cache_set_failed")
