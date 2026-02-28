from uuid import UUID

from pydantic import BaseModel


class CreateProjectInvitationSchema(BaseModel):
    project_id: UUID
    invited_by: UUID
    invited_usernames: list[str]
    
    
class FailedInvitations(BaseModel):
    username: str
    reason: str = None
    
class ResponseCreateProjectInvitationSchema(BaseModel):
  invited_users: list[str] = None
  failed_invitations: list[FailedInvitations] = None
  