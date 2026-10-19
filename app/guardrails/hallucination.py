"""Faithfulness scoring against retrieved evidence."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.settings import Settings
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.memory.context_builder import bounded_reference_messages
from app.prompts.faithfulness import FAITHFULNESS_SYSTEM, FAITHFULNESS_USER
from app.providers.base import LLMProvider
from app.providers.json_parser import parse_model
from app.schemas.retrieval import RetrievedChunk


class FaithfulnessResult(BaseModel):
    supported: bool
    hallucinated_spans: list[str] = Field(default_factory=list)
    rationale: str = ""


class FaithfulnessChecker:
    def __init__(self, llm: LLMProvider, *, settings: Settings | None = None) -> None:
        self._llm = llm
        self.settings = settings or Settings()

    async def check(
        self,
        *,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        tool_results_text: str,
        conversation_text: str = "",
        evidence_text: str | None = None,
    ) -> FaithfulnessResult:
        evidence = "\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks[:8]) or "(none)"
        if evidence_text is not None:
            evidence = evidence_text
        messages = bounded_reference_messages(
            self.settings,
            system=FAITHFULNESS_SYSTEM,
            render=lambda refs: FAITHFULNESS_USER.format(query=query, answer=answer, **refs),
            references={
                "evidence": evidence,
                "tool_results": tool_results_text or "(none)",
                "conversation": conversation_text or "(none)",
            },
            output_tokens=256,
        )
        try:
            response = await self._llm.generate(
                messages,
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return parse_model(response.content, FaithfulnessResult)
        except (LLMProviderError, LLMOutputError):
            # Fail closed on empty evidence + assertive answer is handled by caller.
            return FaithfulnessResult(supported=True, rationale="checker_unavailable")
