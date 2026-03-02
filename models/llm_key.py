import uuid
from sqlalchemy import Column, ForeignKey, Numeric, String, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base


class LlmKey(Base):
    __tablename__ = "llm_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(255), nullable=False, default="Groq")
    api_key_encrypted = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    temperature = Column(Numeric(3, 2), nullable=False, default=0.70)
    llm_name = Column(String(100), nullable=False, default="openai/gpt-oss-120b")

    __table_args__ = (
        Index("idx_llm_keys_user_id", "user_id"),
        Index("idx_llm_keys_chatbot_id", "chatbot_id"),
        UniqueConstraint("chatbot_id", name="unique_chatbot_llm")
    )

    # relationships
    profile = relationship("Profile", back_populates="llm_keys", uselist=False)
    chatbot = relationship("Chatbot", back_populates="llm_keys", uselist=False)
    