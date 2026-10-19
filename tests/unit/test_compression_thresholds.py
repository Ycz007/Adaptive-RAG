import json

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.core.enums import CompressionLevel
from app.memory.budget import estimate_tokens
from app.memory.context_builder import ContextBuilder, history_messages, prompt_tokens
from app.memory.summarizer import ConversationSummarizer
from app.providers.fake import FakeLLMProvider
from app.schemas.agent import ToolResult
from app.schemas.chat import ChatMessage


def setup():
    llm = FakeLLMProvider({"新增历史": '{"summary":"订单 ORD10001"}'})
    settings = Settings()
    return ContextBuilder(settings, ConversationSummarizer(llm)), llm


def state_for_tokens(used):
    messages = [
        ChatMessage(role=role, content="x") for _ in range(10) for role in ("user", "assistant")
    ]
    current = (
        prompt_tokens(
            [
                {"role": "system", "content": "s"},
                *history_messages("", messages),
                {"role": "user", "content": "q"},
            ]
        )
        + 700
        + 512
    )
    desired = estimate_tokens(messages[0].content) + used - current
    messages[0].content = "x" * ((3 * desired + 1) // 2)
    return {"user_query": "q", "messages": messages, "message_ids": list(range(1, 21))}


@pytest.mark.parametrize(
    "used,expected",
    [
        (63999, CompressionLevel.NONE),
        (64000, CompressionLevel.SOFT_TRUNCATE),
        (64001, CompressionLevel.SOFT_TRUNCATE),
        (89599, CompressionLevel.SOFT_TRUNCATE),
        (89600, CompressionLevel.HARD_COMPRESS),
        (89601, CompressionLevel.HARD_COMPRESS),
        (108799, CompressionLevel.HARD_COMPRESS),
        (108800, CompressionLevel.AUTO_COMPACT),
        (108801, CompressionLevel.AUTO_COMPACT),
    ],
)
async def test_live_builder_exact_128k_threshold_boundaries(used, expected):
    context, llm = setup()
    source = state_for_tokens(used)
    result = await context.prepare(source, system="s", current="q", output_tokens=700)
    assert result["context_tokens_before"] == used
    assert result["compression_level"] == expected
    assert result["context_tokens_after"] <= 128000
    if expected in {CompressionLevel.NONE, CompressionLevel.SOFT_TRUNCATE}:
        assert len(result["context_messages"]) == 20
        assert not llm.calls
    elif expected == CompressionLevel.HARD_COMPRESS:
        assert len(result["context_messages"]) == 16
        assert len(result["messages"]) == 20  # Old source survives the sliding window.
        assert result["summary_until_message_id"] == 0
        assert not llm.calls
    else:
        assert len(result["messages"]) == 16
        assert result["summary_until_message_id"] == 4
        assert "ORD10001" in result["summary"]
        assert llm.calls


async def test_tool_compression_triggers_from_original_total_and_does_not_hide_85_percent():
    context, llm = setup()
    source = state_for_tokens(50000)
    source["tool_results"] = [
        ToolResult(name="get_order", success=True, output={"blob": "长" * 100000})
    ]
    result = await context.build_answer(source)
    assert result["compression_level"] == CompressionLevel.AUTO_COMPACT
    assert result["summary_until_message_id"] > 0
    assert llm.calls
    payload = json.loads(
        result["answer_messages"][-1]["content"].split("工具结果：\n")[1].split("\n\n输出 JSON")[0]
    )
    assert payload[0]["truncated"]
    assert len(payload[0]["output"]["preview"]) <= 400
    assert len(source["tool_results"][0].output["blob"]) == 100000


async def test_no_early_compaction_merely_for_more_than_eight_turns_or_large_tool():
    context, llm = setup()
    source = state_for_tokens(20000)
    source["tool_results"] = [
        ToolResult(name="get_order", success=True, output={"blob": "a" * 12000})
    ]
    result = await context.build_answer(source)
    assert result["compression_level"] == CompressionLevel.NONE
    assert len(result["context_messages"]) == 20
    assert "a" * 12000 in result["answer_messages"][-1]["content"]
    assert not llm.calls


@pytest.mark.parametrize(
    "target,expected,preview_limit",
    [
        (63999, CompressionLevel.NONE, None),
        (64000, CompressionLevel.SOFT_TRUNCATE, 400),
        (89600, CompressionLevel.HARD_COMPRESS, 160),
    ],
)
async def test_answer_tool_payload_obeys_50_and_70_percent_thresholds(
    target, expected, preview_limit
):
    context, llm = setup()
    source = state_for_tokens(20000)
    source["tool_results"] = [
        ToolResult(name="get_order", success=True, output={"blob": "a" * 2000})
    ]
    baseline = await context.build_answer(source)
    desired = (
        estimate_tokens(source["messages"][0].content) + target - baseline["context_tokens_before"]
    )
    source["messages"][0].content = "x" * ((3 * desired + 1) // 2)
    result = await context.build_answer(source)
    assert result["context_tokens_before"] == target
    assert result["compression_level"] == expected
    payload = json.loads(
        result["answer_messages"][-1]["content"].split("工具结果：\n")[1].split("\n\n输出 JSON")[0]
    )
    if preview_limit is None:
        assert payload[0]["output"] == {"blob": "a" * 2000}
    else:
        assert payload[0]["truncated"]
        assert len(payload[0]["output"]["preview"]) == preview_limit
    assert not llm.calls


async def test_70_percent_hidden_history_is_included_when_later_reaching_85_percent():
    context, llm = setup()
    state = state_for_tokens(89600)
    windowed = await context.prepare(state, system="s", current="q", output_tokens=700)
    assert len(windowed["messages"]) == 20
    state.update(windowed)
    result = await context.prepare(state, system="s", current="q" + "z" * 40000, output_tokens=700)
    assert result["summary_until_message_id"] == 4
    assert any("x" * 100 in call[-1]["content"] for call in llm.calls)


@pytest.mark.parametrize(
    "values",
    [
        {"compress_soft_ratio": 0.9},
        {"compress_hard_ratio": 0.85},
        {"auto_compact_ratio": 1},
        {"token_budget": 0},
    ],
)
def test_invalid_budget_configuration_is_rejected(values):
    with pytest.raises(ValidationError):
        Settings(**values)
