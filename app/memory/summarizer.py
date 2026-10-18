"""LLM-backed conversation summarisation."""

from app.core.exceptions import InvalidRequestError, LLMOutputError
from app.memory.budget import estimate_tokens
from app.prompts.summary import SUMMARY_SYSTEM, SUMMARY_USER
from app.providers.base import LLMProvider
from app.providers.json_parser import extract_json_object
from app.schemas.chat import ChatMessage


class ConversationSummarizer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def summarize(
        self,
        messages: list[ChatMessage],
        *,
        previous_summary: str = "",
        token_budget: int = 128_000,
        safety_tokens: int = 512,
        max_chars: int = 800,
    ) -> str:
        summary = previous_summary
        history = "\n".join(f"{m.role}: {m.content}" for m in messages)
        # Every evicted message is processed, in bounded batches with the previous summary.
        while history:
            prefix = f"已有摘要：\n{summary}\n\n新增历史：\n"
            overhead = (
                estimate_tokens(SUMMARY_SYSTEM)
                + estimate_tokens(SUMMARY_USER.format(history=prefix))
                + 32
            )
            capacity = min(4096, token_budget - safety_tokens - 256 - overhead)
            if capacity <= 0:
                raise InvalidRequestError("摘要输入超过上下文预算，请增大 TOKEN_BUDGET")
            size = max(1, int(capacity * 1.5))
            batch, history = history[:size], history[size:]
            response = await self._llm.generate(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": SUMMARY_USER.format(history=prefix + batch)},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            payload = extract_json_object(response.content)
            value = payload.get("summary")
            if not isinstance(value, str) or not value.strip():
                raise LLMOutputError("conversation summary is empty or invalid")
            summary = value.strip()[:max_chars]
        return summary
