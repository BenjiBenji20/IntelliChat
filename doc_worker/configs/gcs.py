# doc_worker/configs/gcs.py
import os
from google.cloud import storage
from google.oauth2 import service_account
from doc_worker.configs.settings import settings

_gcs_client: storage.Client = None
_gcs_bucket: storage.Bucket = None

def init_gcs_client():
    global _gcs_client, _gcs_bucket

    credentials = service_account.Credentials.from_service_account_info(
        settings.get_gcs_credentials  
    )

    _gcs_client = storage.Client(
        credentials=credentials,
        project=settings.GCP_PROJECT_ID
    )

    _gcs_bucket = _gcs_client.bucket(settings.GCS_BUCKET_NAME)


def get_gcs_bucket() -> storage.Bucket:
    return _gcs_bucket