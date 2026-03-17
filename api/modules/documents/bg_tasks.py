import json

from api.configs.settings import settings
from api.configs.cloud_tasks import get_cloud_tasks_client
from google.cloud import tasks_v2
import logging

logger = logging.getLogger(__name__)

def _enqueue_tasks(task_payloads: list[dict], endpoint: str) -> None:
    """
    Sync — runs in BackgroundTasks threadpool after response is sent.
 
    Each payload maps 1:1 to the worker's ProcessDocumentRequestSchema.
    Failures are logged, never raised — a missed enqueue doesn't corrupt
    data. Document stays 'uploaded' and can be re-confirmed to retry.
    """
    client = get_cloud_tasks_client()
 
    for payload in task_payloads:
        try:
            task = tasks_v2.Task(
                http_request=tasks_v2.HttpRequest(
                    http_method=tasks_v2.HttpMethod.POST,
                    url=settings.WORKER_URL + endpoint,
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(payload).encode("utf-8"),
                    oidc_token=tasks_v2.OidcToken(
                        service_account_email=settings.CLOUD_TASKS_BUCKET_SA_EMAIL
                    ),
                )
            )
            client.create_task(
                parent=settings.CLOUD_TASKS_QUEUE_PATH,
                task=task,
            )
            
            logger.info(
                "Task enqueued | document_id=%s chatbot_id=%s",
                payload["document_id"],
                payload["chatbot_id"],
            )
        except Exception as e:
            logger.error(
                "Failed to enqueue task | document_id=%s error=%s",
                payload["document_id"],
                str(e),
            )
            