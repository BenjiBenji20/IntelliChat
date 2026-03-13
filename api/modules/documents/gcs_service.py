import datetime
import os
from typing import Optional

from google.cloud import storage

from api.configs.gcp_client import get_bucket, get_storage_client
from api.configs.settings import settings


SIGNED_URL_UPLOAD_EXPIRY_SECONDS = settings.GCS_UPLOAD_URL_EXPIRY_SECONDS
SIGNED_URL_DOWNLOAD_EXPIRY_SECONDS = settings.GCS_DOWNLOAD_URL_EXPIRY_SECONDS

class GCSService:
    """
    Wraps all Google Cloud Storage operations.
    All methods are synchronous because the GCS SDK is not async-native.
    Run these in a thread pool executor when calling from async FastAPI routes
    (handled in the FileService layer via asyncio.to_thread).
    """

    def __init__(self) -> None:
        self._client: storage.Client = get_storage_client()
        self._bucket: storage.Bucket = get_bucket(self._client)

    def generate_signed_upload_url(
        self,
        object_key: str,
        content_type: str,
        file_size: int,
    ) -> str:
        """
        Generates a V4 signed URL for a direct PUT upload from the client.

        The signed URL is scoped to:
          - a specific object key
          - a specific content-type
          - a max Content-Length via condition
        """
        blob = self._bucket.blob(object_key)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=SIGNED_URL_UPLOAD_EXPIRY_SECONDS),
            method="PUT",
            content_type=content_type
        )
        return url


    def generate_signed_download_url(self, object_key: str) -> str:
        """
        Generates a V4 signed URL for a GET download.
        """
        blob = self._bucket.blob(object_key)
        file_name = object_key.split("/")[-1]
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=SIGNED_URL_DOWNLOAD_EXPIRY_SECONDS),
            method="GET",
            response_disposition=f'attachment; filename="{file_name}"',
            response_type="application/octet-stream",
        )
        return url


    def delete_object(self, object_key: str) -> None:
        """
        Deletes an object from GCS. Silent if already missing.
        """
        blob = self._bucket.blob(object_key)
        try:
            blob.delete()
        except Exception:
            # Object may already be gone (e.g. lifecycle-deleted). That's fine.
            pass


    def object_exists(self, object_key: str) -> bool:
        """
        Returns True if the object exists in the bucket.
        Used to validate that the client actually completed the upload.
        """
        blob = self._bucket.blob(object_key)
        return blob.exists()


    def get_object_metadata(self, object_key: str) -> Optional[dict]:
        """
        Returns size, content_type, md5_hash of an object.
        Returns None if the object doesn't exist.
        """
        from google.cloud.exceptions import NotFound
        blob = self._bucket.blob(object_key)
        try:
            blob.reload()
        except NotFound:
            return None
        return {
            "size": blob.size,
            "content_type": blob.content_type,
            "md5_hash": blob.md5_hash,
            "updated": blob.updated,
        }

gcs_service = GCSService()
