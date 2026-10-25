"""One model decision per graph iteration, followed by one observed tool call."""

from __future__ import annotations

import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.nodes.executor import _fingerprint, execute_calls
from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError, LLMOutputError, LLMProviderError
from app.core.ids import new_id
from app.memory.context_builder import ContextBuilder
from app.memory.summarizer import ConversationSummarizer
from app.prompts.agent_loop import AGENT_LOOP_SYSTEM
from app.providers.base import LLMProvider
from app.providers.json_parser import parse_model
from app.schemas.agent import ToolCall
from app.schemas.chat import ActionProposal
from app.schemas.common import LatencyBreakdown, TokenUsage
from app.skills.loader import match_skill
from app.tools.permissions import WRITE_RISKS
from app.tools.registry import ToolRegistry


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "finish"]
    tool_name: str | None = None
    arguments: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "tool" and not self.tool_name:
            raise ValueError("tool action requires tool_name")
        if self.action == "finish" and (self.tool_name or self.arguments):
            raise ValueError("finish cannot contain a tool call")
        return self


def _observation(state: AgentState, message: str, *, stop: bool = False) -> dict:
    return {
        "agent_feedback": message,
        "agent_done": stop,
        "agent_stop_reason": message if stop else "",
        "agent_tool_name": None,
        "agent_tool_arguments": {},
    }


async def agent_decide_node(
    state: AgentState, *, llm: LLMProvider, registry: ToolRegistry, settings: Settings
) -> dict:
    iterations = state.get("agent_iterations", 0)
    if iterations >= settings.max_agent_steps:
        return _observation(state, "已达到 Agent 决策轮数上限，请说明未完成事项。", stop=True)

    intent = state.get("intent")
    skill = match_skill(state["user_query"], intent.label.value if intent else "faq_policy")
    allowed = set(registry.names())
    if skill:
        allowed &= set(skill.get("tools", [])) | {
            "search_knowledge",
            "escalate_to_human",
            "load_tool_result",
        }
    catalog = [tool.openai_schema() for tool in registry.all() if tool.name in allowed]
    system = AGENT_LOOP_SYSTEM.format(
        tools=json.dumps(catalog, ensure_ascii=False),
        skill=json.dumps(skill, ensure_ascii=False) if skill else "通用工具集",
    )
    builder = ContextBuilder(settings, ConversationSummarizer(llm))

    def render(evidence: str, observations: str) -> str:
        return (
            f"用户问题：{state['user_query']}\n"
            f"决策轮数：{iterations + 1}/{settings.max_agent_steps}；"
            f"已调用工具：{state.get('step_count', 0)}/{settings.max_tool_calls}\n"
            f"执行反馈：{state.get('agent_feedback') or '(无)'}\n"
            f"工具观察（按调用顺序）：{observations or '(无)'}\n"
            f"已执行的调用：{json.dumps(state.get('agent_calls') or [], ensure_ascii=False)}\n"
            f"历史结果目录：{json.dumps(state.get('tool_result_refs') or [], ensure_ascii=False)}\n"
            f"其他检索证据：{evidence}\n"
            "现在输出一个 action 决策。"
        )

    # Count the actual decision prompt, including tool schemas and raw observations,
    # before choosing the same 50% / 70% / 85% stages used by answer generation.
    context = await builder.build_answer(
        state,
        system=system,
        render_current=render,
        output_tokens=512,
    )
    update = {**context, "agent_iterations": iterations + 1}
    started = time.perf_counter()
    try:
        response = await llm.generate(
            context["answer_messages"],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
    except LLMProviderError:
        return {
            **update,
            **_observation(state, "决策模型暂时不可用，任务可能尚未完成。", stop=True),
        }
    latency = state.get("latency") or LatencyBreakdown()
    latency.llm_ms += (time.perf_counter() - started) * 1000
    previous = state.get("token_usage") or TokenUsage()
    update.update(
        {
            "latency": latency,
            "token_usage": TokenUsage(
                prompt_tokens=previous.prompt_tokens + response.usage.prompt_tokens,
                completion_tokens=previous.completion_tokens + response.usage.completion_tokens,
                total_tokens=previous.total_tokens + response.usage.total_tokens,
            ),
        }
    )
    try:
        decision = parse_model(response.content, AgentDecision)
    except LLMOutputError:
        return {**update, **_observation(state, "决策格式错误，请按 action 协议重新输出。")}
    if decision.action == "finish":
        return {**update, **_observation(state, "Agent 已结束取证。", stop=True)}
    if state.get("step_count", 0) >= settings.max_tool_calls:
        return {
            **update,
            **_observation(state, "已达到工具调用上限，请说明未完成事项。", stop=True),
        }
    if decision.tool_name not in allowed:
        return {**update, **_observation(state, "工具不在当前可用目录，请重新选择。")}
    tool = registry.get(decision.tool_name)
    try:
        decision.arguments = tool.parse(decision.arguments).model_dump(mode="json")
    except InvalidRequestError as exc:
        return {**update, **_observation(state, str(exc))}
    fingerprint = _fingerprint(tool.name, decision.arguments)
    if fingerprint in (state.get("agent_call_fingerprints") or []):
        return {**update, **_observation(state, "相同工具和参数已执行，请使用已有结果或调整调用。")}
    if tool.risk in WRITE_RISKS and (
        not state.get("confirm_action_id") or state.get("agent_write_attempted")
    ):
        pending = ActionProposal(
            action_id=new_id("act_"),
            tool_name=tool.name,
            arguments=decision.arguments,
            risk=tool.risk,
            summary=f"即将执行 {tool.name}，需要你确认后才会真正提交。",
        )
        return {
            **update,
            **_observation(state, "等待用户确认写操作。", stop=True),
            "pending_action": pending,
        }
    return {
        **update,
        "agent_done": False,
        "agent_feedback": "",
        "agent_tool_name": tool.name,
        "agent_tool_arguments": decision.arguments,
    }


async def agent_act_node(state: AgentState, *, registry: ToolRegistry, settings: Settings) -> dict:
    name = state["agent_tool_name"]
    if name is None:
        return _observation(state, "没有待执行工具。", stop=True)
    arguments = state.get("agent_tool_arguments") or {}
    # Share the execution safeguards without generating or executing a plan.
    result = await execute_calls(
        state,
        calls_to_run=[ToolCall(name=name, arguments=arguments)],
        registry=registry,
        settings=settings,
        allow_fallback=False,
    )
    return {
        **result,
        "tool_results": [*(state.get("tool_results") or []), *result["tool_results"]],
        "agent_tool_name": None,
        "agent_tool_arguments": {},
        "agent_calls": [*(state.get("agent_calls") or []), {"tool": name, "arguments": arguments}],
        "agent_call_fingerprints": [
            *(state.get("agent_call_fingerprints") or []),
            _fingerprint(name, arguments),
        ],
        "agent_write_attempted": bool(
            state.get("agent_write_attempted") or registry.get(name).risk in WRITE_RISKS
        ),
    }
