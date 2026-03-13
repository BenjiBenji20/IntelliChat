import uuid
from sqlalchemy import CheckConstraint, Column, ForeignKey, Integer, String, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    storage_path = Column(String, nullable=False)
    status = Column(String(20), nullable=False, default="processing")
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_documents_chatbot_id", "chatbot_id"),
        Index("idx_documents_user_id", "user_id"),
        CheckConstraint(
            "file_size <= 52428800", name="documents_file_size_check"
        ),
        CheckConstraint(
            "status = ANY(ARRAY['pending', 'uploaded', 'processing', 'indexed', 'failed', 'inactive'])",
            name="documents_status_check"
        )
    )

    # relationships
    chatbot = relationship("Chatbot", back_populates="documents", uselist=False)
    profile = relationship("Profile", back_populates="documents", uselist=False)
    embeddings_metadata = relationship("EmbeddingMetadata", back_populates="document", cascade="all, delete")
    