import pytest
from pydantic import ValidationError

from app.memory.session_cache import RedisSessionContextCache, SessionSnapshot, session_cache_key
from app.schemas.chat import ChatMessage


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.now = 0
        self.broken = False
        self.get_calls = 0

    async def get(self, key):
        self.get_calls += 1
        if self.broken:
            raise ConnectionError("redis unavailable")
        item = self.store.get(key)
        return item[0] if item and item[1] > self.now else None

    async def set(self, key, value, ex):
        if self.broken:
            raise ConnectionError("redis unavailable")
        self.store[key] = (value, self.now + ex)


def snapshot(last=2):
    return SessionSnapshot(
        user_id="owner",
        session_id="s",
        last_message_id=last,
        messages=[
            ChatMessage(role="user", content="订单 ORD10001"),
            ChatMessage(role="assistant", content="好的"),
        ],
        message_ids=[last - 1, last],
    )


async def get(cache, snap):
    return await cache.get(
        user_id=snap.user_id,
        session_id=snap.session_id,
        last_message_id=snap.last_message_id,
        summary_until_message_id=snap.summary_until_message_id,
        summary=snap.summary,
    )


async def test_cache_roundtrip_and_expiration():
    redis = FakeRedis()
    cache = RedisSessionContextCache(redis, ttl_seconds=1800)
    snap = snapshot()
    assert await get(cache, snap) is None
    await cache.set(snap)
    assert await get(cache, snap) == snap
    redis.now = 1800
    assert await get(cache, snap) is None


async def test_old_write_cannot_replace_new_version():
    redis = FakeRedis()
    cache = RedisSessionContextCache(redis)
    old, new = snapshot(), snapshot(4)
    await cache.set(new)
    await cache.set(old)  # Delayed writer stores an older version after the new one.
    assert await get(cache, new) == new
    new.summary = "新的摘要"
    assert await get(cache, new) is None


@pytest.mark.parametrize("raw", ["invalid JSON", "{}", '{"user_id":"attacker"}'])
async def test_corrupt_payload_degrades_to_miss(raw):
    redis = FakeRedis()
    cache = RedisSessionContextCache(redis)
    redis.store[session_cache_key("owner", "s", 2, 0, "")] = (raw, 1800)
    assert await get(cache, snapshot()) is None


async def test_wrong_owner_or_version_in_payload_is_not_used():
    redis = FakeRedis()
    cache = RedisSessionContextCache(redis)
    snap = snapshot()
    wrong = snap.model_copy(update={"user_id": "other"})
    redis.store[session_cache_key("owner", "s", 2, 0, "")] = (wrong.model_dump_json(), 1800)
    assert await get(cache, snap) is None


async def test_redis_failure_and_oversize_payload_are_optional():
    redis = FakeRedis()
    cache = RedisSessionContextCache(redis, max_bytes=10)
    await cache.set(snapshot())
    assert redis.store == {}
    redis.broken = True
    assert await get(cache, snapshot()) is None
    await RedisSessionContextCache(redis).set(snapshot())


@pytest.mark.parametrize(
    "ids,last,cursor", [([2, 1], 1, 0), ([1, 1], 1, 0), ([1, 2], 3, 0), ([1, 2], 2, 1)]
)
def test_incomplete_or_invalid_snapshot_is_rejected(ids, last, cursor):
    with pytest.raises(ValidationError):
        SessionSnapshot(
            user_id="u",
            session_id="s",
            last_message_id=last,
            summary_until_message_id=cursor,
            message_ids=ids,
            messages=[
                ChatMessage(role="user", content="a"),
                ChatMessage(role="assistant", content="b"),
            ],
        )
