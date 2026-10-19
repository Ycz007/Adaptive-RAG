"""Bounded, read-only recovery of a saved tool output."""

from pydantic import BaseModel, Field

from app.core.request_scope import get_db_session
from app.repositories.tool_output_repository import ToolOutputRepository
from app.schemas.agent import ToolResult
from app.tools.base import Tool, ToolContext


class LoadToolResultInput(BaseModel):
    result_ref: str = Field(
        pattern=r"^tout_[0-9a-f]{32}$", description="来自截断预览或历史结果目录的引用 ID"
    )
    offset: int = Field(default=0, ge=0, description="JSON 原文字符偏移")
    limit: int = Field(default=1000, ge=1, le=4000, description="本页字符数，最多 4000")


class LoadToolResultTool(Tool):
    name = "load_tool_result"
    description = (
        "按 result_ref 分页加载当前用户、当前会话已保存的工具结果原文。"
        "用于截断预览缺少细节的情况；传 offset、limit，按 next_offset 继续读取。"
        "读取历史快照，不重新执行原业务操作；历史状态不能作为当前业务状态。"
    )
    input_model = LoadToolResultInput

    async def execute(self, parsed: LoadToolResultInput, context: ToolContext) -> ToolResult:
        row = await ToolOutputRepository(get_db_session()).get_visible(parsed.result_ref, context)
        if row is None:
            return ToolResult(name=self.name, success=False, error="saved tool result not found")
        offset = min(parsed.offset, row.total_chars)
        end = min(offset + parsed.limit, row.total_chars)
        return ToolResult(
            name=self.name,
            success=True,
            output={
                "result_ref": row.result_ref,
                "tool_name": row.tool_name,
                "content": row.output_json[offset:end],
                "offset": offset,
                "next_offset": end if end < row.total_chars else None,
                "has_more": end < row.total_chars,
                "total_chars": row.total_chars,
                "created_at": row.created_at.isoformat(),
                "historical_snapshot": True,
            },
        )
