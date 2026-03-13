import uuid
from sqlalchemy import Column, ForeignKey, String, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.base import Base


class ProjectInvitation(Base):
    __tablename__ = "project_invitations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    invited_username = Column(String(50), ForeignKey("profiles.username"), nullable=False)
    invited_by = Column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("project_id", "invited_username", name="uq_project_invitations_project_username"),
        Index("idx_project_invitations_project_id", "project_id"),
        Index("idx_project_invitations_invited_username", "invited_username")
    )

    # relationships
    project = relationship("Project", back_populates="invitations", uselist=False)
    invited_user = relationship("Profile", foreign_keys=[invited_username], back_populates="received_invitations", uselist=False)
    inviter = relationship("Profile", foreign_keys=[invited_by], back_populates="sent_invitations", uselist=False)
    