"""Intent and complexity routing nodes."""

from __future__ import annotations

import re
import time

from app.agents.nodes.intent import IntentClassifier
from app.agents.state import AgentState
from app.core.enums import QueryComplexity, RetrievalStrategy
from app.retrieval.adaptive import AdaptiveRouter
from app.schemas.common import LatencyBreakdown


def _latency(state: AgentState) -> LatencyBreakdown:
    return state.get("latency") or LatencyBreakdown()


async def intent_node(state: AgentState, classifier: IntentClassifier) -> dict:
    started = time.perf_counter()
    intent = await classifier.classify(state["user_query"])
    latency = _latency(state)
    latency.intent_ms = (time.perf_counter() - started) * 1000
    return {"intent": intent, "latency": latency}


async def router_node(state: AgentState, router: AdaptiveRouter) -> dict:
    started = time.perf_counter()
    intent = state["intent"]
    assert intent is not None
    complexity = await router.route(state["user_query"], intent)
    if complexity.strategy != RetrievalStrategy.REFUSE and re.search(
        r"tout_[0-9a-f]{32}", state["user_query"]
    ):
        complexity = complexity.model_copy(
            update={
                "strategy": RetrievalStrategy.AGENT,
                "complexity": QueryComplexity.TOOL_REQUIRED,
            }
        )
    latency = _latency(state)
    latency.routing_ms = (time.perf_counter() - started) * 1000
    return {
        "complexity": complexity,
        "strategy": complexity.strategy,
        "query_complexity": complexity.complexity,
        "latency": latency,
    }
