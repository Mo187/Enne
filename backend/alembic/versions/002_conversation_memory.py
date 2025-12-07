"""Add conversation memory tables for persistent context

Revision ID: 002
Revises: 001
Create Date: 2025-11-25

This migration adds:
- conversations: Master conversation record per user
- conversation_messages: Individual messages with importance scoring
- entity_tracking: Track mentioned entities with positions for "the second one" references
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create conversations table - Master conversation record per user
    op.create_table('conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('message_count', sa.Integer(), default=0, nullable=False),
        sa.Column('total_tokens', sa.Integer(), default=0, nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('title', sa.String(length=500), nullable=True),  # Auto-generated title
        sa.Column('summary', sa.Text(), nullable=True),  # AI-generated summary of older messages
        sa.Column('meta_data', JSON, nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conversations_user_active', 'conversations', ['user_id', 'is_active'], unique=False)
    op.create_index('ix_conversations_last_activity', 'conversations', ['last_activity_at'], unique=False)

    # Create conversation_messages table - Individual messages with importance scoring
    op.create_table('conversation_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),  # 'user', 'assistant', 'system'
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('importance_score', sa.Float(), default=0.5, nullable=False),  # 0.0-1.0 for prioritization
        sa.Column('has_data_payload', sa.Boolean(), default=False, nullable=False),  # Contains query results
        sa.Column('has_tool_call', sa.Boolean(), default=False, nullable=False),  # Contains tool execution
        sa.Column('intent', sa.String(length=100), nullable=True),  # Detected intent for this message
        sa.Column('entities_mentioned', JSON, nullable=True),  # Entity types mentioned
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('meta_data', JSON, nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conv_messages_conversation', 'conversation_messages', ['conversation_id', 'created_at'], unique=False)
    op.create_index('ix_conv_messages_importance', 'conversation_messages', ['conversation_id', 'importance_score'], unique=False)

    # Create entity_tracking table - Track ALL mentioned entities with positions
    op.create_table('entity_tracking',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=False),  # 'email', 'contact', 'project', 'task', 'organization'
        sa.Column('entity_id', sa.String(length=255), nullable=True),  # Internal ID (e.g., contact.id)
        sa.Column('external_id', sa.String(length=500), nullable=True),  # External ID (e.g., email ID from Microsoft)
        sa.Column('entity_name', sa.String(length=500), nullable=True),  # Name/subject for display
        sa.Column('entity_data', JSON, nullable=True),  # Full entity data for reference
        sa.Column('list_position', sa.Integer(), nullable=True),  # Position in list (0-indexed) for "the second one"
        sa.Column('is_current_focus', sa.Boolean(), default=False, nullable=False),  # Currently being discussed
        sa.Column('mention_count', sa.Integer(), default=1, nullable=False),
        sa.Column('first_mentioned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_mentioned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=True),  # Which message mentioned this entity
        sa.Column('meta_data', JSON, nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['message_id'], ['conversation_messages.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_entity_tracking_conv_type', 'entity_tracking', ['conversation_id', 'entity_type'], unique=False)
    op.create_index('ix_entity_tracking_focus', 'entity_tracking', ['conversation_id', 'is_current_focus'], unique=False)
    op.create_index('ix_entity_tracking_position', 'entity_tracking', ['conversation_id', 'entity_type', 'list_position'], unique=False)
    op.create_index('ix_entity_tracking_last_mentioned', 'entity_tracking', ['conversation_id', 'last_mentioned_at'], unique=False)


def downgrade() -> None:
    op.drop_table('entity_tracking')
    op.drop_table('conversation_messages')
    op.drop_table('conversations')
