from functools import lru_cache

from google.cloud import tasks_v2
from google.oauth2 import service_account

from api.configs.settings import settings


@lru_cache(maxsize=1)
def get_cloud_tasks_client() -> tasks_v2.CloudTasksClient:
    """
    Returns a singleton Cloud Tasks client.
    """
    if settings.get_gcs_credentials:
        info: dict = settings.get_gcs_credentials

        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")

        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return tasks_v2.CloudTasksClient(credentials=credentials)

    return tasks_v2.CloudTasksClient()
