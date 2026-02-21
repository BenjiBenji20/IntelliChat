import uuid
from sqlalchemy import Column, ForeignKey, String, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base


class RedisKey(Base):
    __tablename__ = "redis_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(255), nullable=False, default="Upstash")
    redis_url = Column(String, nullable=False)
    redis_token_encrypted = Column(String, nullable=False)
    redis_port = Column(Integer, default=6379)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_redis_keys_user_id", "user_id"),
        Index("idx_redis_keys_chatbot_id", "chatbot_id"),
    )

    # relationships
    profile = relationship("Profile", back_populates="redis_keys", uselist=False)
    chatbot = relationship("Chatbot", back_populates="redis_keys", uselist=False)
    