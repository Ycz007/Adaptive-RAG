import json

import pytest

from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError, LLMOutputError
from app.memory.context_builder import ContextBuilder, prompt_tokens
from app.memory.summarizer import ConversationSummarizer
from app.providers.fake import FakeLLMProvider
from app.schemas.agent import ToolResult
from app.schemas.chat import ChatMessage
from app.schemas.retrieval import RetrievedChunk


def builder(**kwargs):
    llm = FakeLLMProvider({"新增历史": '{"summary":"用户订单 ORD10001，正在查物流"}'})
    settings = Settings(token_budget=5000, context_safety_tokens=256, **kwargs)
    return ContextBuilder(settings, ConversationSummarizer(llm)), llm


def history(turns):
    return [
        ChatMessage(role=role, content=f"turn-{i}-{role}")
        for i in range(turns)
        for role in ("user", "assistant")
    ]


async def test_short_history_preserves_roles_without_summary_call():
    context, llm = builder()
    messages = history(2)
    result = await context.build_answer(
        {
            "user_query": "查一下物流",
            "messages": messages,
            "message_ids": [1, 2, 3, 4],
        }
    )
    assert result["answer_messages"][1:5] == [
        m.model_dump(include={"role", "content"}) for m in messages
    ]
    assert "查一下物流" in result["answer_messages"][-1]["content"]
    assert llm.calls == []


async def test_compaction_summarizes_evicted_turns_and_merges_old_summary():
    context, llm = builder(memory_recent_turns=2)
    messages = history(5)
    for message in messages[:6]:
        message.content += "长" * 1000
    result = await context.build_answer(
        {
            "user_query": "查一下物流",
            "messages": messages,
            "message_ids": list(range(11, 21)),
            "summary": "更早的订单 ORD10001",
            "summary_until_message_id": 10,
        }
    )
    assert result["message_ids"] == [17, 18, 19, 20]
    assert result["summary_until_message_id"] == 16
    summary_input = llm.calls[0][-1]["content"]
    assert "更早的订单 ORD10001" in summary_input
    assert "turn-0-user" in summary_input and "turn-2-assistant" in summary_input
    assert "turn-3-user" not in summary_input
    assert result["answer_messages"][1]["role"] == "user"
    assert "ORD10001" in result["answer_messages"][1]["content"]


async def test_large_sources_are_bounded_and_raw_state_is_preserved():
    context, _ = builder()
    tool = ToolResult(name="get_order", success=True, output={"blob": "大" * 50000})
    chunk = RetrievedChunk(
        chunk_id="c", document_id="d", title="政策", content="证据" * 50000, score=1
    )
    result = await context.build_answer(
        {
            "user_query": "政策是什么",
            "messages": history(1),
            "message_ids": [1, 2],
            "retrieved_chunks": [chunk],
            "tool_results": [tool],
        }
    )
    assert prompt_tokens(result["answer_messages"]) + 700 + 256 <= 5000
    assert len(tool.output["blob"]) == 50000
    assert len(chunk.content) == 100000
    assert "内容已截断" in result["answer_messages"][-1]["content"]


async def test_impossible_current_question_is_rejected_before_model_call():
    context, llm = builder()
    with pytest.raises(InvalidRequestError):
        await context.build_answer({"user_query": "很长" * 10000})
    assert llm.calls == []


async def test_failed_summary_does_not_mutate_history_or_cursor():
    llm = FakeLLMProvider({"新增历史": '{"summary":""}'})
    context = ContextBuilder(
        Settings(token_budget=5000, memory_recent_turns=1), ConversationSummarizer(llm)
    )
    state = {"user_query": "继续", "messages": history(3), "message_ids": list(range(1, 7))}
    state["messages"][0].content += "长" * 10000
    with pytest.raises(LLMOutputError):
        await context.build_answer(state)
    assert len(state["messages"]) == 6
    assert "summary_until_message_id" not in state


async def test_summarizer_processes_all_batches_within_budget():
    context, llm = builder()
    messages = [ChatMessage(role="user", content=f"marker-{i}:" + "长" * 900) for i in range(25)]
    await context.summarizer.summarize(
        messages, previous_summary="已有订单", token_budget=1800, safety_tokens=128
    )
    all_inputs = "".join(call[-1]["content"] for call in llm.calls)
    for i in range(25):
        assert f"marker-{i}:" in all_inputs
    assert "已有订单" in llm.calls[0][-1]["content"]
    assert all(prompt_tokens(call) + 256 + 128 <= 1800 for call in llm.calls)


async def test_small_tool_result_keeps_structured_output():
    context, _ = builder()
    result = await context.build_answer(
        {
            "user_query": "查物流",
            "tool_results": [
                ToolResult(name="track_shipment", success=True, output={"status": "shipped"})
            ],
        }
    )
    prompt = result["answer_messages"][-1]["content"]
    payload = json.loads(prompt.split("工具结果：\n")[1].split("\n\n输出 JSON")[0])
    assert payload[0]["output"] == {"status": "shipped"}
