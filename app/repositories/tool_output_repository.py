"""Persist immutable tool outputs in the current chat transaction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ToolPermissionError
from app.core.ids import new_id
from app.repositories.models import SessionRow, ToolOutputRow
from app.schemas.agent import ToolResult

if TYPE_CHECKING:
    from app.tools.base import ToolContext


class ToolOutputRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(self, result: ToolResult, context: ToolContext) -> ToolResult:
        session = await self.db.get(SessionRow, context.session_id)
        if session is None or session.user_id != context.user_id:
            raise ToolPermissionError("session not found or access denied")
        text = json.dumps(
            result.model_dump(mode="json", exclude={"result_ref", "output_chars"}),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        row = ToolOutputRow(
            result_ref=new_id("tout_"),
            user_id=context.user_id,
            session_id=context.session_id,
            trace_id=context.trace_id,
            tool_name=result.name,
            output_json=text,
            preview=text[:160],
            total_chars=len(text),
        )
        self.db.add(row)
        await self.db.flush()
        return result.model_copy(update={"result_ref": row.result_ref, "output_chars": len(text)})

    async def get_visible(self, result_ref: str, context: ToolContext) -> ToolOutputRow | None:
        return await self.db.scalar(
            select(ToolOutputRow)
            .join(SessionRow, SessionRow.session_id == ToolOutputRow.session_id)
            .where(
                ToolOutputRow.result_ref == result_ref,
                ToolOutputRow.user_id == context.user_id,
                ToolOutputRow.session_id == context.session_id,
                SessionRow.user_id == context.user_id,
            )
        )

    async def recent(self, user_id: str, session_id: str, *, limit: int = 10) -> list[dict]:
        # Deliberately exclude output_json so the discovery query stays small.
        rows = await self.db.execute(
            select(
                ToolOutputRow.result_ref,
                ToolOutputRow.tool_name,
                ToolOutputRow.preview,
                ToolOutputRow.total_chars,
                ToolOutputRow.created_at,
            )
            .join(SessionRow, SessionRow.session_id == ToolOutputRow.session_id)
            .where(
                ToolOutputRow.user_id == user_id,
                ToolOutputRow.session_id == session_id,
                SessionRow.user_id == user_id,
            )
            .order_by(ToolOutputRow.created_at.desc(), ToolOutputRow.result_ref.desc())
            .limit(limit)
        )
        return [dict(row._mapping) | {"created_at": row.created_at.isoformat()} for row in rows]
