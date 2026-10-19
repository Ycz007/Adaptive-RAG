"""Self-RAG faithfulness check and answer self-correction."""

from __future__ import annotations

from app.agents.nodes.generate import AnswerPayload, _extractive_fallback
from app.agents.state import AgentState
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.guardrails.hallucination import FaithfulnessChecker, FaithfulnessResult
from app.memory.context_builder import bounded_reference_messages, history_messages
from app.observability.metrics import HALLUCINATION_FLAGS
from app.prompts.faithfulness import SELF_CORRECT_SYSTEM, SELF_CORRECT_USER
from app.providers.base import LLMProvider
from app.providers.json_parser import parse_model
from app.schemas.retrieval import RetrievedChunk


async def self_correct_answer(
    *,
    llm: LLMProvider,
    checker: FaithfulnessChecker,
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    tool_results_text: str,
    first_result: FaithfulnessResult,
    conversation_text: str = "",
    evidence_text: str | None = None,
) -> tuple[str, FaithfulnessResult, bool]:
    """LLM rewrite once, then extractive fallback. Returns (answer, judge, corrected)."""
    evidence = (
        "\n\n".join(f"[{c.chunk_id}] {c.title}\n{c.content}" for c in chunks[:6]) or "(无检索结果)"
    )
    if evidence_text is not None:
        evidence = evidence_text
    messages = bounded_reference_messages(
        checker.settings,
        system=SELF_CORRECT_SYSTEM,
        render=lambda refs: SELF_CORRECT_USER.format(
            query=query,
            answer=answer,
            critique=first_result.rationale or ",".join(first_result.hallucinated_spans),
            **refs,
        ),
        references={
            "evidence": evidence,
            "tool_results": tool_results_text or "(无)",
            "conversation": conversation_text or "(无)",
        },
        output_tokens=500,
    )
    try:
        response = await llm.generate(
            messages,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        rewritten = parse_model(response.content, AnswerPayload).answer
        judged = await checker.check(
            query=query,
            answer=rewritten,
            chunks=chunks,
            tool_results_text=tool_results_text,
            conversation_text=conversation_text,
            evidence_text=evidence_text,
        )
        if judged.supported:
            return rewritten, judged, True
    except (LLMProviderError, LLMOutputError):
        pass
    patched = _extractive_fallback(query, evidence, tool_results_text)
    judged = await checker.check(
        query=query,
        answer=patched,
        chunks=chunks,
        tool_results_text=tool_results_text,
        conversation_text=conversation_text,
        evidence_text=evidence_text,
    )
    return patched, judged, True


async def faithfulness_node(
    state: AgentState, checker: FaithfulnessChecker, llm: LLMProvider
) -> dict:
    answer = state.get("final_answer") or ""
    if state.get("refused") or state.get("pending_action"):
        return {}
    chunks = state.get("retrieved_chunks") or []
    tools = state.get("tool_results") or []
    tool_text = state.get("context_tool_results", str([t.model_dump() for t in tools]))
    evidence_text = state.get("context_evidence")
    conversation_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history_messages(
            state.get("summary") or "", state.get("context_messages", state.get("messages")) or []
        )
    )
    result = await checker.check(
        query=state["user_query"],
        answer=answer,
        chunks=chunks,
        tool_results_text=tool_text,
        conversation_text=conversation_text,
        evidence_text=evidence_text,
    )
    extra = {"faithfulness": result.model_dump(), "self_corrected": False}
    if result.supported:
        return {"extra": extra}
    HALLUCINATION_FLAGS.inc()
    patched, judged, corrected = await self_correct_answer(
        llm=llm,
        checker=checker,
        query=state["user_query"],
        answer=answer,
        chunks=chunks,
        tool_results_text=tool_text,
        first_result=result,
        conversation_text=conversation_text,
        evidence_text=evidence_text,
    )
    extra = {"faithfulness": judged.model_dump(), "self_corrected": corrected}
    return {"final_answer": patched, "extra": extra}
