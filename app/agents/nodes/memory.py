"""Conversation memory compaction node."""

from app.agents.state import AgentState
from app.config.settings import Settings
from app.memory.context_builder import ContextBuilder
from app.memory.summarizer import ConversationSummarizer


async def memory_node(
    state: AgentState,
    *,
    settings: Settings,
    summarizer: ConversationSummarizer,
) -> dict:
    return await ContextBuilder(settings, summarizer).build_answer(state)
