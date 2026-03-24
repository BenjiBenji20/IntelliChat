import uuid
from sqlalchemy import CheckConstraint, Column, ForeignKey, Integer, String, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String, nullable=False)
    role = Column(String(9), nullable=True)
    summary = Column(String, nullable=True)
    token_count = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_conversation_summaries_chatbot_id", "chatbot_id"),
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="conversation_summaries_role_check"
        )
    )

    # relationships
    chatbot = relationship("Chatbot", back_populates="conversation_summaries", uselist=False)
    