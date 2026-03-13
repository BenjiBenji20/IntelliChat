import uuid
from sqlalchemy import Column, ForeignKey, String, Boolean, Index, UniqueConstraint, Text
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    hashed_api_key = Column(String(64), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_projects_owner_name"),
        Index("idx_projects_owner_id", "owner_id"),
    )

    # relationships
    owner = relationship("Profile", foreign_keys=[owner_id], back_populates="owned_projects", uselist=False)
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete")
    invitations = relationship("ProjectInvitation", back_populates="project", cascade="all, delete")
    chatbots = relationship("Chatbot", back_populates="project", cascade="all, delete")
    