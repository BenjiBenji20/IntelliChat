import uuid
from sqlalchemy import Column, ForeignKey, String, Text, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base

class ChatbotBehavior(Base):
    __tablename__ = "chatbot_behavior"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)

    # nullable fields
    category = Column(String(255), nullable=True)
    target_audience = Column(String(255), nullable=True)
    description = Column(String(500), nullable=True)
    tone = Column(String(50), nullable=True)
    language = Column(String(255), nullable=True)
    response_style = Column(String(255), nullable=True)
    fallback_message = Column(String(500), nullable=True)
    policy_restriction = Column(String(255), nullable=True)
    system_prompt = Column(Text, nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_chatbot_behavior_user_id", "user_id"),
        Index("idx_chatbot_behavior_chatbot_id", "chatbot_id"),
    )

    # relationships
    profile = relationship("Profile", back_populates="chatbot_behaviors", uselist=False)
    chatbot = relationship("Chatbot", back_populates="chatbot_behavior", uselist=False)