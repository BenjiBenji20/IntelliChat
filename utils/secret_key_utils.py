import base64
import hashlib
import os
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, status
from configs.settings import settings


def _get_fernet() -> Fernet:
    """
    Derive a consistent Fernet key from the app's ENCRYPTION_KEY setting.
    Uses SHA-256 to ensure the key is exactly 32 bytes, then base64url encodes it.
    """
    if not settings.ENCRYPTION_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Encryption key not configured."
        )
    raw = hashlib.sha256(settings.ENCRYPTION_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(raw)
    return Fernet(fernet_key)


def encrypt_secret(raw_key: str) -> str:
    """
    Encrypt a raw secret key string.
    Returns a base64url-encoded encrypted string safe for DB storage.
    """
    if not settings.ENCRYPTION_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Encryption key not configured."
        )
    cipher_suite = Fernet(settings.ENCRYPTION_KEY.encode())
    return cipher_suite.encrypt(raw_key.encode()).decode()


def decrypt_secret(encrypted_key: str) -> str:
    """
    Decrypt an encrypted secret key string.
    Raises 401 if the token is invalid or tampered with.
    """
    try:
        if not settings.ENCRYPTION_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Encryption key not configured."
            )
        cipher_suite = Fernet(settings.ENCRYPTION_KEY.encode())
        return cipher_suite.decrypt(encrypted_key.encode()).decode()
    except InvalidToken:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or corrupted encrypted key."
        )


def hash_secret(raw_key: str) -> str:
    """
    One-way SHA-256 hash of a raw secret key.
    For storing API keys for fast lookup/comparison.
    """
    return hashlib.sha256(raw_key.encode()).hexdigest()
