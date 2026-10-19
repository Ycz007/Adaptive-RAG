"""Chat orchestration: graph execution, streaming and persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.exceptions import AppError, GuardrailError, RateLimitError
from app.core.ids import new_id
from app.core.request_scope import request_session
from app.core.streaming import reset_token_sink, set_token_sink
from app.infra.rate_limit import RateLimiter
from app.memory.session_cache import SessionContextCache, SessionSnapshot
from app.observability.metrics import REQUEST_LATENCY, REQUESTS, ROUTING, STAGE_LATENCY, TOKEN_USAGE
from app.providers.base import LLMProvider
from app.repositories.models import SessionRow
from app.repositories.session_repository import SessionRepository
from app.repositories.tool_output_repository import ToolOutputRepository
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import DocumentReranker
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from app.schemas.common import LatencyBreakdown, TokenUsage
from app.tools.factory import build_tool_registry
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMProvider,
        retriever: HybridRetriever,
        reranker: DocumentReranker,
        session_factory: async_sessionmaker[AsyncSession],
        rate_limiter: RateLimiter,
        registry: ToolRegistry | None = None,
        session_cache: SessionContextCache | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._retriever = retriever
        self._reranker = reranker
        self._session_factory = session_factory
        self._rate_limiter = rate_limiter
        self._session_cache = session_cache or SessionContextCache()
        self.registry = registry or build_tool_registry(retriever)
        self._graph = build_graph(
            llm=llm,
            settings=settings,
            retriever=retriever,
            reranker=reranker,
            registry=self.registry,
        )

    async def chat(self, request: ChatRequest, *, trace_id: str) -> ChatResponse:
        await self._enforce_rate_limit(request.user_id)
        started = time.perf_counter()
        session_id = request.session_id or new_id("sess_")
        async with request_session(self._session_factory) as db:
            state = await self._initial_state(db, request, session_id, trace_id)
            result = await self._invoke(state)
            response = self._to_response(result, session_id=session_id, trace_id=trace_id)
            await self._persist(
                db,
                request,
                response,
                summary=result.get("summary") or "",
                summary_until_message_id=result.get("summary_until_message_id") or 0,
            )
            await db.commit()
        await self._refresh_session_cache(request.user_id, session_id)
        self._observe(response, started)
        return response

    async def stream(self, request: ChatRequest, *, trace_id: str) -> AsyncIterator[dict]:
        """SSE: node events plus token events from DeepSeek stream during generate_answer."""
        await self._enforce_rate_limit(request.user_id)
        yield {"event": "started", "trace_id": trace_id}
        started = time.perf_counter()
        session_id = request.session_id or new_id("sess_")
        async with request_session(self._session_factory) as db:
            state = await self._initial_state(db, request, session_id, trace_id)
            queue: asyncio.Queue[dict | None] = asyncio.Queue()
            holder: dict = {"merged": dict(state), "guardrail": None, "error": None}

            async def emit(token: str) -> None:
                await queue.put({"event": "token", "text": token})

            sink_token = set_token_sink(emit)

            async def _run_graph() -> None:
                try:
                    async for update in self._graph.astream(state, stream_mode="updates"):
                        for node, payload in update.items():
                            if isinstance(payload, dict):
                                holder["merged"].update(payload)
                            event: dict = {"event": "node", "node": node}
                            if isinstance(payload, dict) and payload.get("intent"):
                                event["intent"] = payload["intent"].label.value
                            if isinstance(payload, dict) and payload.get("strategy"):
                                event["strategy"] = payload["strategy"].value
                            await queue.put(event)
                except GuardrailError as exc:
                    holder["guardrail"] = exc
                except Exception as exc:  # graph failures surface after the queue drains
                    holder["error"] = exc
                finally:
                    await queue.put(None)

            task = asyncio.create_task(_run_graph())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
                await task
            finally:
                reset_token_sink(sink_token)
                if not task.done():
                    task.cancel()

            if holder["guardrail"] is not None:
                exc = holder["guardrail"]
                await db.commit()
                yield {
                    "event": "answer",
                    "data": ChatResponse(
                        answer=exc.message,
                        session_id=session_id,
                        trace_id=trace_id,
                        refused=True,
                        error_code=exc.error_code.value,
                    ).model_dump(mode="json"),
                }
                yield {"event": "done", "trace_id": trace_id}
                return
            if holder["error"] is not None:
                raise holder["error"]
            merged: AgentState = holder["merged"]
            response = self._to_response(merged, session_id=session_id, trace_id=trace_id)
            await self._persist(
                db,
                request,
                response,
                summary=merged.get("summary") or "",
                summary_until_message_id=merged.get("summary_until_message_id") or 0,
            )
            await db.commit()
        await self._refresh_session_cache(request.user_id, session_id)
        self._observe(response, started)
        yield {"event": "answer", "data": response.model_dump(mode="json")}
        yield {"event": "done", "trace_id": trace_id}

    async def _enforce_rate_limit(self, user_id: str) -> None:
        allowed = await self._rate_limiter.allow(f"user:{user_id}")
        if not allowed:
            raise RateLimitError()

    async def validate_session(self, request: ChatRequest) -> None:
        """Check ownership before a streaming response sends HTTP headers."""
        if request.session_id:
            async with self._session_factory() as db:
                await SessionRepository(db).check_owner(request.session_id, request.user_id)

    async def _initial_state(
        self,
        db: AsyncSession,
        request: ChatRequest,
        session_id: str,
        trace_id: str,
    ) -> AgentState:
        sessions = SessionRepository(db)
        session = await sessions.get_or_create(session_id, request.user_id)
        snapshot = await self._session_snapshot(sessions, session, use_cache=True)
        return {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": request.user_id,
            "user_query": request.message,
            "messages": snapshot.messages,
            "message_ids": snapshot.message_ids,
            "summary": snapshot.summary,
            "summary_until_message_id": snapshot.summary_until_message_id,
            "retrieved_chunks": [],
            "tool_results": [],
            "tool_result_refs": await ToolOutputRepository(db).recent(request.user_id, session_id),
            "citations": [],
            "refused": False,
            "step_count": 0,
            "token_usage": TokenUsage(),
            "latency": LatencyBreakdown(),
            "confirm_action_id": request.confirm_action_id,
            "idempotency_key": request.idempotency_key,
        }

    async def _session_snapshot(
        self, sessions: SessionRepository, session: SessionRow, *, use_cache: bool
    ) -> SessionSnapshot:
        # Ownership is checked before this method. Validate the message version
        # against the DB even on cache hits; old writers cannot resurrect old state.
        cursor = session.summary_until_message_id
        summary = session.summary if cursor else ""
        latest = (
            await sessions.latest_message_id(session.session_id)
            if self._session_cache.enabled
            else None
        )
        if use_cache and latest is not None:
            snapshot = await self._session_cache.get(
                user_id=session.user_id,
                session_id=session.session_id,
                last_message_id=latest,
                summary_until_message_id=cursor,
                summary=summary,
            )
            if snapshot is not None:
                return snapshot
        rows = await sessions.messages_after(session.session_id, cursor, until_message_id=latest)
        return SessionSnapshot(
            user_id=session.user_id,
            session_id=session.session_id,
            last_message_id=latest if latest is not None else (rows[-1].id if rows else cursor),
            summary=summary,
            summary_until_message_id=cursor,
            messages=[ChatMessage(role=row.role, content=row.content) for row in rows],  # type: ignore[arg-type]
            message_ids=[row.id for row in rows],
        )

    async def _refresh_session_cache(self, user_id: str, session_id: str) -> None:
        if not self._session_cache.enabled:
            return
        try:
            # Only committed data is read and cached. Cache failure never turns a
            # successfully committed answer into an API error.
            async with self._session_factory() as db:
                sessions = SessionRepository(db)
                session = await sessions.check_owner(session_id, user_id)
                if session is not None:
                    snapshot = await self._session_snapshot(sessions, session, use_cache=False)
                    await self._session_cache.set(snapshot)
        except Exception:
            logger.warning("session_cache_refresh_failed")

    async def _invoke(self, state: AgentState) -> AgentState:
        try:
            # LangGraph widens the state to dict[str, Any]; the graph contract is AgentState.
            return cast(AgentState, await self._graph.ainvoke(state))
        except GuardrailError as exc:
            return {
                **state,
                "final_answer": exc.message,
                "refused": True,
                "error": exc.error_code.value,
            }
        except AppError:
            raise

    async def _persist(
        self,
        db: AsyncSession,
        request: ChatRequest,
        response: ChatResponse,
        *,
        summary: str = "",
        summary_until_message_id: int = 0,
    ) -> None:
        sessions = SessionRepository(db)
        await sessions.append_message(response.session_id, "user", request.message)
        await sessions.append_message(response.session_id, "assistant", response.answer)
        if summary:
            await sessions.save_summary(
                response.session_id, summary, until_message_id=summary_until_message_id
            )

    def _to_response(
        self,
        result: AgentState,
        *,
        session_id: str,
        trace_id: str,
    ) -> ChatResponse:
        answer = result.get("final_answer") or "暂时无法回答，请稍后重试。"
        strategy = result.get("strategy")
        intent = result.get("intent")
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            trace_id=trace_id,
            intent=intent.label if intent else None,
            complexity=result.get("query_complexity"),
            strategy=strategy,
            citations=result.get("citations") or [],
            pending_action=result.get("pending_action"),
            refused=bool(result.get("refused")),
            error_code=result.get("error"),
            token_usage=result.get("token_usage") or TokenUsage(),
            latency=result.get("latency") or LatencyBreakdown(),
        )

    def _observe(self, response: ChatResponse, started: float) -> None:
        total_ms = (time.perf_counter() - started) * 1000
        response.latency.total_ms = total_ms
        REQUESTS.labels(
            strategy=(response.strategy.value if response.strategy else "unknown"),
            intent=(response.intent.value if response.intent else "unknown"),
            status="ok" if not response.refused else "refused",
        ).inc()
        if response.strategy:
            ROUTING.labels(strategy=response.strategy.value).inc()
        REQUEST_LATENCY.observe(total_ms / 1000)
        STAGE_LATENCY.labels(stage="intent").observe(response.latency.intent_ms / 1000)
        STAGE_LATENCY.labels(stage="retrieval").observe(response.latency.retrieval_ms / 1000)
        STAGE_LATENCY.labels(stage="llm").observe(response.latency.llm_ms / 1000)
        TOKEN_USAGE.labels(kind="total").inc(response.token_usage.total_tokens)
        logger.info(
            "chat_completed",
            extra={
                "trace_id": response.trace_id,
                "strategy": response.strategy.value if response.strategy else None,
                "intent": response.intent.value if response.intent else None,
                "latency_ms": round(total_ms, 1),
            },
        )
