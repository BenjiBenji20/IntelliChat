from google.cloud import storage
from doc_worker.configs.settings import settings

_gcs_client: storage.Client = None
_gcs_bucket: storage.Bucket = None

def init_gcs_client():
    global _gcs_client, _gcs_bucket
    _gcs_client = storage.Client()
    _gcs_bucket = _gcs_client.bucket(settings.GCS_BUCKET_NAME)

def get_gcs_bucket() -> storage.Bucket:
    return _gcs_bucket
