from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from uuid import UUID
from typing import Optional

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UUID:
    """
    Extracts user_id from Supabase JWT.
    
    Flow:
    1. Extract Bearer token from Authorization header.
    2. Parse JWT claims (unverified as secret is managed by Supabase/RLS).
    3. Return the 'sub' claim as the user's UUID.
    """
    token = credentials.credentials
    try:
        # Note: Signature verification is skipped as the secret is not stored 
        # on this server. This assumes the API is behind a gateway or 
        # primarily uses RLS at the DB level.
        payload = jwt.get_unverified_claims(token)
        user_id: Optional[str] = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
            
        return UUID(user_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
