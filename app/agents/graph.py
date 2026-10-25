"""LangGraph state machine that wires the agent nodes together."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.agent_loop import agent_act_node, agent_decide_node
from app.agents.nodes.faithfulness import faithfulness_node
from app.agents.nodes.generate import generate_node, refuse_node
from app.agents.nodes.guardrail import guardrail_node
from app.agents.nodes.intent import IntentClassifier
from app.agents.nodes.memory import memory_node
from app.agents.nodes.retrieval import retrieval_node
from app.agents.nodes.router import intent_node, router_node
from app.agents.nodes.tool_recovery import tool_recovery_node
from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.enums import RetrievalStrategy
from app.guardrails.hallucination import FaithfulnessChecker
from app.memory.summarizer import ConversationSummarizer
from app.providers.base import LLMProvider
from app.retrieval.adaptive import AdaptiveRouter
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.retrieval.self_rag import SelfRAGLoop
from app.tools.registry import ToolRegistry


def _route_after_complexity(state: AgentState) -> str:
    strategy = state.get("strategy") or RetrievalStrategy.SINGLE
    mapping = {
        RetrievalStrategy.DIRECT: "compact_memory",
        RetrievalStrategy.REFUSE: "refuse_answer",
        RetrievalStrategy.SINGLE: "retrieve_docs",
        RetrievalStrategy.MULTI: "retrieve_docs",
        RetrievalStrategy.AGENT: "agent_decide",
    }
    return mapping[strategy]


def build_graph(
    *,
    llm: LLMProvider,
    settings: Settings,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    registry: ToolRegistry,
) -> CompiledStateGraph:
    classifier = IntentClassifier(llm)  # 意图分类器
    adaptive = AdaptiveRouter(llm)  # 自适应路由器（判断复杂度）
    rewriter = QueryRewriter(llm)  # 查询改写器
    self_rag = SelfRAGLoop(retriever, reranker, rewriter, llm, settings)  # Self-RAG 自反思检索循环
    checker = FaithfulnessChecker(llm, settings=settings)  # 忠诚度检查器（防幻觉）
    summarizer = ConversationSummarizer(llm)  # 对话摘要器

    graph = StateGraph(AgentState)
    graph.add_node("apply_guardrail", guardrail_node)
    graph.add_node("classify_intent", partial(intent_node, classifier=classifier))
    graph.add_node("route_complexity", partial(router_node, router=adaptive))
    graph.add_node(
        "retrieve_docs",
        partial(
            retrieval_node,
            retriever=retriever,
            reranker=reranker,
            rewriter=rewriter,
            self_rag=self_rag,
            settings=settings,
        ),
    )
    graph.add_node(
        "agent_decide", partial(agent_decide_node, llm=llm, registry=registry, settings=settings)
    )
    graph.add_node("agent_act", partial(agent_act_node, registry=registry, settings=settings))
    graph.add_node("generate_answer", partial(generate_node, llm=llm))
    graph.add_node("refuse_answer", refuse_node)
    graph.add_node("check_faithfulness", partial(faithfulness_node, checker=checker, llm=llm))
    graph.add_node("compact_memory", partial(memory_node, settings=settings, summarizer=summarizer))
    graph.add_node("prepare_memory", partial(memory_node, settings=settings, summarizer=summarizer))
    graph.add_node(
        "recover_tool_results",
        partial(tool_recovery_node, llm=llm, registry=registry, settings=settings),
    )

    def _after_guardrail(state: AgentState) -> str:
        return "refuse_answer" if state.get("refused") else "prepare_memory"

    graph.add_edge(START, "apply_guardrail")
    graph.add_conditional_edges(
        "apply_guardrail",
        _after_guardrail,
        {"refuse_answer": "refuse_answer", "prepare_memory": "prepare_memory"},
    )
    graph.add_edge("prepare_memory", "classify_intent")
    graph.add_edge("classify_intent", "route_complexity")
    graph.add_conditional_edges(
        "route_complexity",
        _route_after_complexity,
        {
            "compact_memory": "compact_memory",
            "refuse_answer": "refuse_answer",
            "retrieve_docs": "retrieve_docs",
            "agent_decide": "agent_decide",
        },
    )
    graph.add_edge("retrieve_docs", "compact_memory")
    graph.add_conditional_edges(
        "agent_decide",
        lambda state: (
            "compact_memory"
            if state.get("agent_done")
            else "agent_act"
            if state.get("agent_tool_name")
            else "agent_decide"
        ),
        {
            "compact_memory": "compact_memory",
            "agent_act": "agent_act",
            "agent_decide": "agent_decide",
        },
    )
    graph.add_edge("agent_act", "agent_decide")
    graph.add_conditional_edges(
        "compact_memory",
        lambda state: "generate_answer"
        if state.get("strategy") == RetrievalStrategy.AGENT
        else "recover_tool_results",
        {"generate_answer": "generate_answer", "recover_tool_results": "recover_tool_results"},
    )
    graph.add_edge("recover_tool_results", "generate_answer")
    graph.add_edge("generate_answer", "check_faithfulness")
    graph.add_edge("refuse_answer", END)
    graph.add_edge("check_faithfulness", END)
    # Includes routing, all decision/action rounds, final generation and checking.
    return graph.compile().with_config(recursion_limit=2 * settings.max_agent_steps + 16)
