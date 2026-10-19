import pytest

from app.agents.nodes.faithfulness import faithfulness_node, self_correct_answer
from app.config.settings import Settings
from app.guardrails.hallucination import FaithfulnessChecker
from app.memory.context_builder import ContextBuilder, prompt_tokens
from app.memory.summarizer import ConversationSummarizer
from app.providers.base import LLMResponse
from app.providers.fake import FakeLLMProvider
from app.schemas.agent import ToolResult
from app.schemas.common import TokenUsage
from app.schemas.retrieval import RetrievedChunk


class _SequenceLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts = []

    async def ping(self) -> bool:
        return True

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        self.prompts.append(messages)
        if self.calls == 1:
            content = '{"answer":"七天无理由不含定制","citation_chunk_ids":[]}'
        else:
            content = '{"supported":true,"hallucinated_spans":[],"rationale":"ok"}'
        return LLMResponse(content=content, usage=TokenUsage(total_tokens=4), model="fake")


@pytest.mark.asyncio
async def test_self_correct_rewrites_unsupported_answer() -> None:
    llm = _SequenceLLM()
    checker = FaithfulnessChecker(llm)
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="return_policy_0",
        title="退货",
        content="定制商品不适用七天无理由。",
        score=1.0,
        rank=1,
        source="bm25",
    )
    from app.guardrails.hallucination import FaithfulnessResult

    first = FaithfulnessResult(
        supported=False, hallucinated_spans=["随时退"], rationale="unsupported"
    )
    answer, judged, corrected = await self_correct_answer(
        llm=llm,
        checker=checker,
        query="定制杯能七天无理由吗",
        answer="都可以随时退",
        chunks=[chunk],
        tool_results_text="",
        first_result=first,
        conversation_text="用户此前提供订单 ORD10001",
    )
    assert corrected is True
    assert judged.supported is True
    assert "定制" in answer
    assert all("用户此前提供订单 ORD10001" in prompt[-1]["content"] for prompt in llm.prompts)


async def test_self_correction_and_recheck_respect_budget_with_large_references():
    from app.guardrails.hallucination import FaithfulnessResult

    llm = _SequenceLLM()
    settings = Settings(token_budget=3000, context_safety_tokens=256)
    checker = FaithfulnessChecker(llm, settings=settings)
    await self_correct_answer(
        llm=llm,
        checker=checker,
        query="政策是什么",
        answer="未经验证的回答",
        chunks=[
            RetrievedChunk(
                chunk_id="c", document_id="d", title="政策", content="证据" * 50000, score=1
            )
        ],
        tool_results_text="工具结果" * 50000,
        conversation_text="历史消息" * 50000,
        first_result=FaithfulnessResult(supported=False),
    )
    assert prompt_tokens(llm.prompts[0]) + 500 + 256 <= 3000
    assert all(prompt_tokens(prompt) + 256 + 256 <= 3000 for prompt in llm.prompts[1:])


async def test_judge_uses_same_compressed_tool_view_as_answer():
    llm = FakeLLMProvider({"hallucinated_spans": '{"supported":true}'})
    settings = Settings(token_budget=5000)
    state = {
        "user_query": "查订单",
        "tool_results": [
            ToolResult(name="get_order", success=True, output={"blob": "原始" * 50000})
        ],
    }
    context = await ContextBuilder(settings, ConversationSummarizer(llm)).build_answer(state)
    state.update(context)
    state["final_answer"] = "已查询"
    await faithfulness_node(state, FaithfulnessChecker(llm, settings=settings), llm)
    assert "原始" * 1000 not in llm.calls[-1][-1]["content"]
    assert "truncated" in llm.calls[-1][-1]["content"]
    assert prompt_tokens(llm.calls[-1]) + 256 + 512 <= 5000
