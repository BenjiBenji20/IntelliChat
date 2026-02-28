from datetime import datetime, timedelta, timezone
from uuid import UUID
from sqlalchemy import and_, select
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from base.base_crud_repository import BaseCrudRepository
from models.project import Project
from models.project_invitation import ProjectInvitation
from models.profile import Profile
from models.project_member import ProjectMember
from modules.projects.project.project_repository import ProjectRepository


class ProjectInvitationRepository(BaseCrudRepository[ProjectInvitation]):
    """
    ProjectInvitation Repository for direct DB operations.
    
    Inherits standardized CRUD from BaseCrudRepository.
    """
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ProjectInvitation, db)
        self.project_repo = ProjectRepository(db)

    async def create_project_invitations(
        self, 
        project_id: UUID,
        invited_by: UUID, 
        invited_usernames: list[str]                   
    ) -> dict:
        """
        Bulk user invitation
        
        Validations:
            - project 
            - cannot invite if project deleted
            - owner cannot self invite
            - cannot invite existing 
            - username exists
            - caller must be a project owner
            - project owner only role
            - cannot invite already pending invitation
        """
        try:
            # 1. Verify project exists and caller is the owner
            project: Project = await self.project_repo.get_by_id(project_id)
            if project is None:
                raise HTTPException(status_code=404, detail=f'Project {project_id} not found.')
            
            if project.owner_id != invited_by:
                raise HTTPException(status_code=403, detail='Only the project owner can invite members.')

            # 2. Remove duplicates from input
            invited_usernames = list(set(invited_usernames))

            # 3. Fetch caller's profile to get their username (for self-invite check)
            caller_profile = await self.db.scalar(
                select(Profile).where(Profile.id == invited_by)
            )
            if caller_profile is None:
                raise HTTPException(status_code=404, detail='Caller profile not found.')

            # 4. Validate all provided usernames exist
            existing_profiles = await self.db.scalars(
                select(Profile).where(Profile.username.in_(invited_usernames))
            )
            existing_profiles = existing_profiles.all()
            existing_usernames = {p.username for p in existing_profiles}

            not_found_usernames = set(invited_usernames) - existing_usernames
            
            # 5. Fetch existing members of the project (by user_id)
            existing_members = await self.db.scalars(
                select(ProjectMember).where(ProjectMember.project_id == project_id)
            )
            existing_member_user_ids = {m.user_id for m in existing_members.all()}
            
            # 6. Fetch pending invitations for this project
            pending_invitations = await self.db.scalars(
                select(ProjectInvitation).where(
                    and_(
                        ProjectInvitation.project_id == project_id,
                        ProjectInvitation.status == 'pending',
                        ProjectInvitation.invited_username.in_(invited_usernames)
                    )
                )
            )
            already_pending_usernames = {inv.invited_username for inv in pending_invitations.all()}

            # 7. Build results — track what succeeded and what didn't
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=7)

            results = {
                'invited_users': [],
                'failed_invitations': []
            }

            invitations_to_insert = []

            for username in invited_usernames:
                # Self-invite check
                if username == caller_profile.username:
                    results['failed_invitations'].append({'username': username, 'reason': 'Cannot invite yourself.'})
                    continue

                # Username doesn't exist
                if username in not_found_usernames:
                    results['failed_invitations'].append({'username': username, 'reason': 'User not found.'})
                    continue

                # Already a member
                profile = next((p for p in existing_profiles if p.username == username), None)
                if profile and profile.id in existing_member_user_ids:
                    results['failed_invitations'].append({'username': username, 'reason': 'User is already a member.'})
                    continue

                # Already has a pending invitation
                if username in already_pending_usernames:
                    results['failed_invitations'].append({'username': username, 'reason': 'Invitation already pending.'})
                    continue

                invitations_to_insert.append(ProjectInvitation(
                    project_id=project_id,
                    invited_username=username,
                    invited_by=invited_by,
                    status='pending',
                    expires_at=expires_at,
                    created_at=now,
                    updated_at=now,
                ))
                results['invited_users'].append(username)

            # 8. Bulk insert valid invitations
            if invitations_to_insert:
                self.db.add_all(invitations_to_insert)
                await self.db.commit()

            return results

        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            raise e
        
        
        
        
        
        