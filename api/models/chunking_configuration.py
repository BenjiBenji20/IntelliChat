import uuid
from sqlalchemy import Column, ForeignKey, String, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class ChunkingConfiguration(Base):
    __tablename__ = "chunking_configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_size = Column(Integer, default=500)
    chunk_overlap = Column(Integer, default=50)
    separator = Column(String(50), default='\n\n')
    content_type = Column(String(100), default='knowledge')
    document_type = Column(String(100), default='knowledge_base')
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_chunking_configurations_document_id", "document_id"),
        Index("idx_chunking_configurations_chatbot_id", "chatbot_id"),
    )

    # relationships
    chatbot = relationship("Chatbot", back_populates="chunking_configurations", uselist=False)
    document = relationship("Document", back_populates="chunking_configurations")
    