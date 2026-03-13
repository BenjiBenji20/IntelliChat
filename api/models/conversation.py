import uuid
from sqlalchemy import Column, ForeignKey, String, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(100), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_conversations_chatbot_id", "chatbot_id"),
        Index("idx_conversations_user_id", "user_id"),
    )

    # relationships
    chatbot = relationship("Chatbot", back_populates="conversations", uselist=False)
    profile = relationship("Profile", back_populates="conversations", uselist=False)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete")