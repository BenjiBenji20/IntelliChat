import uuid
from sqlalchemy import Column, ForeignKey, String, Boolean, Numeric, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base


class Chatbot(Base):
    __tablename__ = "chatbots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    application_name = Column(String(100), nullable=False)
    has_memory = Column(Boolean, nullable=False, default=False)
    system_prompt = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "application_name", name="uq_chatbots_user_name"),
        Index("idx_chatbots_user_id", "user_id"),
        Index("idx_chatbots_project_id", "project_id")
    )

    # relationships
    profile = relationship("Profile", back_populates="chatbots", uselist=False)
    documents = relationship("Document", back_populates="chatbot", cascade="all, delete")
    embeddings_metadata = relationship("EmbeddingMetadata", back_populates="chatbot", cascade="all, delete")
    conversations = relationship("Conversation", back_populates="chatbot", cascade="all, delete")
    api_keys = relationship("ApiKey", back_populates="chatbot", cascade="all, delete")
    redis_keys = relationship("RedisKey", back_populates="chatbot", cascade="all, delete")
    embedding_model_keys = relationship("EmbeddingModelKey", back_populates="chatbot", cascade="all, delete")
    llm_keys = relationship("LlmKey", back_populates="chatbot", cascade="all, delete")
    project = relationship("Project", back_populates="chatbots", uselist=False)
    