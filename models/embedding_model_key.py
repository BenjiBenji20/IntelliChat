import uuid
from sqlalchemy import Column, ForeignKey, String, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base


class EmbeddingModelKey(Base):
    __tablename__ = "embedding_model_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(255), nullable=False, default="Google Studio AI")
    api_key_encrypted = Column(String, nullable=False)
    embedding_model_name = Column(String(100), nullable=False, default="models/gemini-embedding-001")
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_embedding_model_keys_user_id", "user_id"),
        Index("idx_embedding_model_keys_chatbot_id", "chatbot_id"),
        # UniqueConstraint("chatbot_id", "unique_chatbot_embedding")
    )

    # relationships
    profile = relationship("Profile", back_populates="embedding_model_keys", uselist=False)
    chatbot = relationship("Chatbot", back_populates="embedding_model_keys", uselist=False)
    