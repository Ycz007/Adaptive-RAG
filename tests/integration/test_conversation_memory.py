import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.settings import Settings
from app.core.exceptions import ToolPermissionError
from app.repositories.db import init_schema
from app.repositories.models import MessageRow, SessionRow
from app.schemas.chat import ChatRequest
from app.services.container import build_container


@pytest.fixture
async def container(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}",
        memory_recent_turns=2,
        memory_history_token_limit=2000,
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
            ChatRequest(message=f"继续查询 {i}", user_id="u_demo", session_id=first.session_id),
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
