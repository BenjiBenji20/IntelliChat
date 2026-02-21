from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(UUID(as_uuid=True), primary_key=True)  # references auth.users, no default
    full_name = Column(String(100), nullable=False, index=True)
    user_type = Column(String(20), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    plan = Column(String(20), nullable=False, default="free")
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # relationships
    chatbots = relationship("Chatbot", back_populates="profile", cascade="all, delete")
    documents = relationship("Document", back_populates="profile", cascade="all, delete")
    embeddings_metadata = relationship("EmbeddingMetadata", back_populates="profile", cascade="all, delete")
    conversations = relationship("Conversation", back_populates="profile", cascade="all, delete")
    messages = relationship("Message", back_populates="profile", cascade="all, delete")
    api_keys = relationship("ApiKey", back_populates="profile", cascade="all, delete")
    redis_keys = relationship("RedisKey", back_populates="profile", cascade="all, delete")
    embedding_model_keys = relationship("EmbeddingModelKey", back_populates="profile", cascade="all, delete")
    llm_keys = relationship("LlmKey", back_populates="profile", cascade="all, delete")
    