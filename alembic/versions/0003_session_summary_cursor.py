"""Track the last message covered by a conversation summary.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("sessions")}
    if "summary_until_message_id" not in columns:
        op.add_column(
            "sessions",
            sa.Column("summary_until_message_id", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    op.drop_column("sessions", "summary_until_message_id")
