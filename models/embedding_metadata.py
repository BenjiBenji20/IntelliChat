import uuid
from sqlalchemy import Column, ForeignKey, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base


class EmbeddingMetadata(Base):
    __tablename__ = "embeddings_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_embeddings_metadata_chatbot_id", "chatbot_id"),
        Index("idx_embeddings_metadata_user_id", "user_id"),
        Index("idx_embeddings_metadata_document_id", "document_id"),
    )

    # relationships
    chatbot = relationship("Chatbot", back_populates="embeddings_metadata", uselist=False)
    profile = relationship("Profile", back_populates="embeddings_metadata", uselist=False)
    document = relationship("Document", back_populates="embeddings_metadata", uselist=False)
    