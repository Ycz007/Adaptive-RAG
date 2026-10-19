import json
import re

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select

from app.config.settings import Settings
from app.core.enums import CompressionLevel
from app.core.request_scope import request_session
from app.memory.compression import compress_tool_results
from app.providers.base import LLMResponse
from app.providers.fake import FakeLLMProvider
from app.repositories.models import ToolOutputRow
from app.repositories.session_repository import SessionRepository
from app.repositories.tool_output_repository import ToolOutputRepository
from app.schemas.agent import ToolResult
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.container import build_container
from app.tools.base import Tool, ToolContext
from app.tools.tool_result_tools import LoadToolResultInput, LoadToolResultTool


class EmptyInput(BaseModel):
    pass


class LargeTool(Tool):
    name = "large_data"
    description = "查询商品的完整资料"
    input_model = EmptyInput

    def __init__(self):
        self.calls = 0

    async def execute(self, parsed, context):
        self.calls += 1
        return ToolResult(
            name=self.name, success=True, output={"intro": "x" * 100000, "detail": "TAIL_SECRET"}
        )


class RecoveryLLM(FakeLLMProvider):
    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        text = messages[-1]["content"]
        if "需要恢复的工具结果" in text or (
            '"steps"' in text and text.startswith("用户问题：加载")
        ):
            reference = re.search(r"tout_[0-9a-f]{32}", text).group(0)
            total = int(re.search(r'"total_chars":\s*(\d+)', text).group(1))
            arguments = {"result_ref": reference, "offset": max(0, total - 800), "limit": 1000}
            if "需要恢复的工具结果" in text:
                content = {"loads": [arguments]}
            else:
                content = {
                    "steps": [
                        {
                            "step_id": 1,
                            "goal": "恢复",
                            "tool_name": "load_tool_result",
                            "arguments": arguments,
                        }
                    ]
                }
        elif '"steps"' in text:
            content = {
                "steps": [
                    {"step_id": 1, "goal": "查询", "tool_name": "large_data", "arguments": {}}
                ]
            }
        elif '"complexity"' in text:
            content = {"complexity": "tool_required", "strategy": "agent", "confidence": 0.9}
        elif '"label"' in text:
            content = {"label": "faq_policy", "confidence": 0.9, "prompt_version": "test"}
        elif '"citation_chunk_ids"' in text:
            content = {
                "answer": "已恢复 TAIL_SECRET" if "TAIL_SECRET" in text else "缺少细节",
                "citation_chunk_ids": [],
            }
        elif '"supported"' in text:
            content = {"supported": True}
        else:
            content = {"summary": "历史商品资料"}
        return LLMResponse(content=json.dumps(content, ensure_ascii=False), model="fake")


async def make_container(settings):
    container = await build_container(settings)
    llm = RecoveryLLM()
    large = LargeTool()
    registry = container.chat_service.registry
    registry.register(large)
    container.chat_service = ChatService(
        settings=settings,
        llm=llm,
        retriever=container.retriever,
        reranker=container.chat_service._reranker,
        session_factory=container.session_factory,
        rate_limiter=container.chat_service._rate_limiter,
        registry=registry,
    )
    container.llm = llm
    return container, large


@pytest.fixture
async def container(tmp_path):
    instance, large = await make_container(
        Settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'outputs.db'}")
    )
    yield instance, large
    await instance.aclose()


async def save_output(instance, output=None):
    context = ToolContext(user_id="owner", session_id="s", trace_id="t")
    async with request_session(instance.session_factory) as db:
        await SessionRepository(db).get_or_create("s", "owner")
        saved = await ToolOutputRepository(db).save(
            ToolResult(
                name="original_tool", success=True, output=output or {"text": "原文" * 1000}
            ),
            context,
        )
        await db.commit()
    return saved, context


async def test_paged_loader_reconstructs_original_tool_message(container):
    instance, _ = container
    original = {"items": ["a", "中文", 'quoted"value'], "tail": "长内容" * 1000}
    saved, context = await save_output(instance, original)
    pages = []
    offset = 0
    async with request_session(instance.session_factory):
        while True:
            result = await LoadToolResultTool().run(
                {"result_ref": saved.result_ref, "offset": offset, "limit": 317}, context
            )
            assert result.success
            assert len(result.output["content"]) <= 317
            pages.append(result.output["content"])
            assert (
                compress_tool_results([result], CompressionLevel.AUTO_COMPACT)[0].output
                == result.output
            )
            if not result.output["has_more"]:
                break
            offset = result.output["next_offset"]
    restored = json.loads("".join(pages))
    assert restored["output"] == original
    assert restored["name"] == "original_tool"
    assert restored["success"] is True


@pytest.mark.parametrize("user_id,session_id", [("other", "s"), ("owner", "other_session")])
async def test_loader_cannot_cross_user_or_session(container, user_id, session_id):
    instance, _ = container
    saved, _ = await save_output(instance)
    context = ToolContext(user_id=user_id, session_id=session_id, trace_id="denied")
    async with request_session(instance.session_factory):
        result = await LoadToolResultTool().run({"result_ref": saved.result_ref}, context)
        assert not result.success
        assert result.output == {}
        assert result.error == "saved tool result not found"


async def test_unknown_ref_and_offset_past_end_are_bounded(container):
    instance, _ = container
    saved, context = await save_output(instance)
    async with request_session(instance.session_factory):
        missing = await LoadToolResultTool().run({"result_ref": "tout_" + "0" * 32}, context)
        assert not missing.success
        end = await LoadToolResultTool().run(
            {"result_ref": saved.result_ref, "offset": 10**9}, context
        )
        assert end.output["content"] == ""
        assert end.output["next_offset"] is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"result_ref": "../../secrets"},
        {"result_ref": "tout_" + "0" * 32, "limit": 4001},
        {"result_ref": "tout_" + "0" * 32, "offset": -1},
    ],
)
def test_loader_rejects_paths_and_unbounded_pages(arguments):
    with pytest.raises(ValidationError):
        LoadToolResultInput(**arguments)


async def test_agent_loads_truncated_tail_in_same_turn_without_rerunning_tool(container):
    instance, large = container
    response = await instance.chat_service.chat(
        ChatRequest(message="查询商品的完整资料", user_id="owner"), trace_id="t"
    )
    assert response.answer == "已恢复 TAIL_SECRET"
    assert large.calls == 1
    async with instance.session_factory() as db:
        records = list((await db.scalars(select(ToolOutputRow))).all())
        assert len(records) == 1  # Recovery pages are not archived recursively.
        assert "TAIL_SECRET" in records[0].output_json
        reference = records[0].result_ref
    answer_call = next(
        call for call in instance.llm.calls if '"citation_chunk_ids"' in call[-1]["content"]
    )
    assert reference in answer_call[-1]["content"]
    assert "TAIL_SECRET" in answer_call[-1]["content"]
    assert "x" * 10000 not in answer_call[-1]["content"]
    # A fresh process still discovers and loads the saved reference.
    restarted, new_large = await make_container(instance.settings)
    try:
        events = [
            event
            async for event in restarted.chat_service.stream(
                ChatRequest(
                    message=f"加载 {reference}", session_id=response.session_id, user_id="owner"
                ),
                trace_id="restart",
            )
        ]
        final = next(event["data"] for event in events if event["event"] == "answer")
        assert final["answer"] == "已恢复 TAIL_SECRET"
        assert new_large.calls == 0
    finally:
        await restarted.aclose()


async def test_failed_chat_rolls_back_archived_output(container, monkeypatch):
    instance, large = container

    async def failing_persist(*args, **kwargs):
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(instance.chat_service, "_persist", failing_persist)
    with pytest.raises(RuntimeError):
        await instance.chat_service.chat(
            ChatRequest(message="查询商品的完整资料", user_id="owner"), trace_id="fail"
        )
    assert large.calls == 1
    async with instance.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(ToolOutputRow)) == 0
