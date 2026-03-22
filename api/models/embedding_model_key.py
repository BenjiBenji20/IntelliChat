import uuid
from sqlalchemy import Column, ForeignKey, String, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class EmbeddingModelKey(Base):
    __tablename__ = "embedding_model_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(255), nullable=False)
    api_key_encrypted = Column(String, nullable=False)
    embedding_model_name = Column(String(100), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_embedding_model_keys_user_id", "user_id"),
        Index("idx_embedding_model_keys_chatbot_id", "chatbot_id"),
        CheckConstraint(
            "provider IN ('google ai studio', 'openai', 'anthropic', 'azure openai')",
            name="embedding_model_keys_provider_check"
        ),
        CheckConstraint(
            "embedding_model_name IN ("
                "'gemini-embedding-001','text-embedding-004', 'gemini-embedding-2-preview'"
            ")",
            name="embedding_model_keys_embedding_model_check"
        ),
    )

    # relationships
    profile = relationship("Profile", back_populates="embedding_model_keys", uselist=False)
    chatbot = relationship("Chatbot", back_populates="embedding_model_keys", uselist=False)
    