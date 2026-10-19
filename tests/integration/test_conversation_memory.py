import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.settings import Settings
from app.core.enums import CompressionLevel
from app.core.exceptions import ToolPermissionError
from app.core.request_scope import request_session
from app.memory.budget import estimate_tokens
from app.memory.context_builder import ContextBuilder
from app.memory.session_cache import RedisSessionContextCache
from app.memory.summarizer import ConversationSummarizer
from app.repositories.db import init_schema
from app.repositories.models import MessageRow, SessionRow
from app.repositories.session_repository import SessionRepository
from app.schemas.chat import ChatRequest
from app.services.container import build_container


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.broken = False
        self.reads = 0

    async def get(self, key):
        self.reads += 1
        if self.broken:
            raise ConnectionError("redis unavailable")
        return self.store.get(key)

    async def set(self, key, value, ex):
        if self.broken:
            raise ConnectionError("redis unavailable")
        self.store[key] = value


@pytest.fixture
async def container(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}",
        memory_recent_turns=2,
        token_budget=6000,
    )
    instance = await build_container(settings, seed=True)
    instance.llm.canned.update(
        {
            "citation_chunk_ids": '{"answer":"已记录订单信息","citation_chunk_ids":[]}',
            "新增历史": '{"summary":"用户订单 ORD10001，正在查询物流"}',
        }
    )
    yield instance
    await instance.aclose()


def answer_calls(llm):
    return [call for call in llm.calls if "citation_chunk_ids" in call[-1]["content"]]


async def test_followup_reaches_answer_and_planner_and_is_persisted(container):
    service = container.chat_service
    first = await service.chat(
        ChatRequest(message="我的订单是 ORD10001", user_id="u_demo"), trace_id="t1"
    )
    await service.chat(
        ChatRequest(message="查一下物流", user_id="u_demo", session_id=first.session_id),
        trace_id="t2",
    )
    call = answer_calls(container.llm)[-1]
    assert {"role": "user", "content": "我的订单是 ORD10001"} in call
    assert {"role": "assistant", "content": first.answer} in call
    planner_calls = [
        call
        for call in container.llm.calls
        if "查一下物流" in call[-1]["content"] and "步骤" in call[0]["content"]
    ]
    assert any("ORD10001" in str(call[:-1]) for call in planner_calls)
    assert any('"order_id": "ORD10001"' in str(call) for call in answer_calls(container.llm))
    judge_calls = [
        call for call in container.llm.calls if "hallucinated_spans" in call[-1]["content"]
    ]
    assert "我的订单是 ORD10001" in judge_calls[-1][-1]["content"]
    async with container.session_factory() as db:
        count = await db.scalar(
            select(func.count())
            .select_from(MessageRow)
            .where(MessageRow.session_id == first.session_id)
        )
        assert count == 4


async def test_summary_survives_new_service_and_streaming(container):
    service = container.chat_service
    first = await service.chat(
        ChatRequest(message="我的订单 ORD10001", user_id="u_demo"), trace_id="t0"
    )
    for i in range(3):
        await service.chat(
            ChatRequest(
                message=f"继续查询 {i}" + "长" * 3000, user_id="u_demo", session_id=first.session_id
            ),
            trace_id=f"t{i+1}",
        )
    async with container.session_factory() as db:
        row = await db.get(SessionRow, first.session_id)
        assert row.summary_until_message_id > 0
        old_cursor = row.summary_until_message_id
        assert "ORD10001" in row.summary
    # Reopen the same database through a fresh container to avoid relying on process memory.
    restarted = await build_container(container.settings)
    try:
        restarted.llm.canned.update(container.llm.canned)
        events = [
            event
            async for event in restarted.chat_service.stream(
                ChatRequest(message="查一下物流", user_id="u_demo", session_id=first.session_id),
                trace_id="stream",
            )
        ]
        assert events[-1]["event"] == "done"
        assert any(event["event"] == "answer" for event in events)
        call = answer_calls(restarted.llm)[-1]
        assert "历史摘要" in str(call) and "ORD10001" in str(call)
        assert '"order_id": "ORD10001"' in call[-1]["content"]
        assert not any(m["content"] == "我的订单 ORD10001" for m in call)
        async with restarted.session_factory() as db:
            row = await db.get(SessionRow, first.session_id)
            assert row.summary_until_message_id >= old_cursor
            count = await db.scalar(
                select(func.count())
                .select_from(MessageRow)
                .where(MessageRow.session_id == first.session_id)
            )
            assert count == 10
    finally:
        await restarted.aclose()


async def test_other_user_cannot_read_or_append_session(container):
    first = await container.chat_service.chat(
        ChatRequest(message="你好", user_id="owner"), trace_id="t"
    )
    request = ChatRequest(message="继续", user_id="other", session_id=first.session_id)
    with pytest.raises(ToolPermissionError):
        await container.chat_service.chat(request, trace_id="denied")
    with pytest.raises(ToolPermissionError):
        await container.chat_service.validate_session(request)
    with pytest.raises(ToolPermissionError):
        _ = [event async for event in container.chat_service.stream(request, trace_id="denied")]
    async with container.session_factory() as db:
        count = await db.scalar(
            select(func.count())
            .select_from(MessageRow)
            .where(MessageRow.session_id == first.session_id)
        )
        assert count == 2


async def test_existing_sqlite_schema_upgrades_without_losing_messages(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE sessions (session_id VARCHAR(64) PRIMARY KEY, user_id VARCHAR(64), summary TEXT, created_at DATETIME, updated_at DATETIME)"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO sessions (session_id,user_id,summary) VALUES ('old','owner','旧摘要')"
                )
            )
        await init_schema(engine)
        await init_schema(engine)
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT summary, summary_until_message_id FROM sessions WHERE session_id='old'"
                    )
                )
            ).one()
            assert row == ("旧摘要", 0)
    finally:
        await engine.dispose()


async def test_committed_cache_hit_skips_message_query_and_checks_owner(container, monkeypatch):
    redis = FakeRedis()
    service = container.chat_service
    service._session_cache = RedisSessionContextCache(redis)
    first = await service.chat(ChatRequest(message="你好", user_id="owner"), trace_id="t")
    assert redis.store

    async def no_message_query(*args, **kwargs):
        raise AssertionError("cache hit must skip the message contents query")

    monkeypatch.setattr(SessionRepository, "messages_after", no_message_query)
    async with request_session(container.session_factory) as db:
        state = await service._initial_state(
            db, ChatRequest(message="继续", user_id="owner"), first.session_id, "t2"
        )
        assert len(state["messages"]) == 2
    reads = redis.reads
    with pytest.raises(ToolPermissionError):
        await service.chat(
            ChatRequest(message="继续", user_id="other", session_id=first.session_id),
            trace_id="denied",
        )
    assert redis.reads == reads


async def test_db_version_change_bypasses_stale_cache_and_stream_refreshes_it(container):
    redis = FakeRedis()
    service = container.chat_service
    service._session_cache = RedisSessionContextCache(redis)
    first = await service.chat(ChatRequest(message="你好", user_id="owner"), trace_id="t")
    old_cache = dict(redis.store)
    # Another instance commits a turn without updating this cache.
    async with container.session_factory() as db:
        repo = SessionRepository(db)
        await repo.append_message(first.session_id, "user", "另一个实例的新消息")
        await repo.append_message(first.session_id, "assistant", "新的回答")
        await db.commit()
    async with request_session(container.session_factory) as db:
        state = await service._initial_state(
            db, ChatRequest(message="继续", user_id="owner"), first.session_id, "t2"
        )
        assert len(state["messages"]) == 4
        assert state["messages"][-2].content == "另一个实例的新消息"
    events = [
        event
        async for event in service.stream(
            ChatRequest(message="你好", user_id="owner", session_id=first.session_id),
            trace_id="stream",
        )
    ]
    assert events[-1]["event"] == "done"
    redis.store.update(old_cache)  # A delayed old write cannot overwrite the new version key.
    async with request_session(container.session_factory) as db:
        state = await service._initial_state(
            db, ChatRequest(message="继续", user_id="owner"), first.session_id, "t3"
        )
        assert len(state["messages"]) == 6


async def test_cache_outage_does_not_break_chat_or_committed_messages(container):
    redis = FakeRedis()
    redis.broken = True
    container.chat_service._session_cache = RedisSessionContextCache(redis)
    first = await container.chat_service.chat(
        ChatRequest(message="你好", user_id="owner"), trace_id="t"
    )
    result = await container.chat_service.chat(
        ChatRequest(message="继续", user_id="owner", session_id=first.session_id), trace_id="t2"
    )
    assert result.answer
    async with container.session_factory() as db:
        assert len(await SessionRepository(db).messages_after(first.session_id, 0)) == 4


async def test_rolled_back_turn_does_not_populate_cache(container, monkeypatch):
    redis = FakeRedis()
    container.chat_service._session_cache = RedisSessionContextCache(redis)

    async def fail(state):
        raise RuntimeError("graph failed")

    monkeypatch.setattr(container.chat_service, "_invoke", fail)
    with pytest.raises(RuntimeError):
        await container.chat_service.chat(
            ChatRequest(message="你好", user_id="owner", session_id="failed"), trace_id="t"
        )
    assert not redis.store
    async with container.session_factory() as db:
        assert await db.get(SessionRow, "failed") is None


@pytest.mark.parametrize(
    "ratio,level",
    [
        (0.55, CompressionLevel.SOFT_TRUNCATE),
        (0.75, CompressionLevel.HARD_COMPRESS),
        (0.90, CompressionLevel.AUTO_COMPACT),
    ],
)
async def test_live_graph_routes_history_through_three_compression_stages(container, ratio, level):
    service = container.chat_service
    request = ChatRequest(message="你好", user_id="owner", session_id="threshold")
    async with container.session_factory() as db:
        repo = SessionRepository(db)
        await repo.get_or_create("threshold", "owner")
        for _ in range(10):
            await repo.append_message("threshold", "user", "x")
            await repo.append_message("threshold", "assistant", "x")
        await db.commit()
    async with request_session(container.session_factory) as db:
        state = await service._initial_state(db, request, "threshold", "t")
        baseline = await ContextBuilder(
            container.settings, ConversationSummarizer(container.llm)
        ).build_answer(state)
        desired = (
            int(container.settings.token_budget * ratio) - baseline["context_tokens_before"] + 1
        )
        state["messages"][0].content = "x" * ((3 * desired + 1) // 2)
        assert estimate_tokens(state["messages"][0].content) == desired
        updates = [update async for update in service._graph.astream(state, stream_mode="updates")]
    prepared = next(update["prepare_memory"] for update in updates if "prepare_memory" in update)
    assert prepared["compression_level"] == level
    if level == CompressionLevel.SOFT_TRUNCATE:
        assert len(prepared["context_messages"]) == 20
    elif level == CompressionLevel.HARD_COMPRESS:
        assert len(prepared["context_messages"]) == 4
        assert len(prepared["messages"]) == 20
    else:
        assert prepared["summary_until_message_id"] > 0
        assert "ORD10001" in prepared["summary"]


async def test_corrupt_cache_falls_back_to_database_in_chat(container):
    redis = FakeRedis()
    service = container.chat_service
    service._session_cache = RedisSessionContextCache(redis)
    first = await service.chat(ChatRequest(message="你好", user_id="owner"), trace_id="t")
    for key in redis.store:
        redis.store[key] = "broken JSON"
    second = await service.chat(
        ChatRequest(message="你好", user_id="owner", session_id=first.session_id), trace_id="t2"
    )
    assert second.answer
    async with container.session_factory() as db:
        assert len(await SessionRepository(db).messages_after(first.session_id, 0)) == 4


async def test_container_wires_configured_redis_into_session_cache(container, monkeypatch):
    redis = FakeRedis()

    async def connect(settings):
        return redis

    monkeypatch.setattr("app.services.container.connect_redis", connect)
    instance = await build_container(container.settings)
    try:
        assert isinstance(instance.chat_service._session_cache, RedisSessionContextCache)
        assert instance.chat_service._session_cache._ttl == 1800
    finally:
        await instance.aclose()
