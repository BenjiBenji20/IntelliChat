import os
import sys
import time
import asyncio
import requests
from uuid import UUID, uuid4

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Must import all models so SQLAlchemy mapper can resolve all relationships
from api.db.base import Base
from api.models.profile import Profile
from api.models.chatbot import Chatbot
from api.models.document import Document
from api.models.embedding_metadata import EmbeddingMetadata
from api.models.conversation import Conversation
from api.models.message import Message
from api.models.api_key import ApiKey
from api.models.redis_key import RedisKey
from api.models.embedding_model_key import EmbeddingModelKey
from api.models.llm_key import LlmKey
from api.models.project import Project
from api.models.project_member import ProjectMember
from api.models.project_invitation import ProjectInvitation
from api.models.chatbot_behavior import ChatbotBehavior

from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from api.configs.settings import settings
from api.dependencies.auth import get_current_user
from api.dependencies.rate_limit import rate_limit_by_user
from api.db.db_session import get_async_db
from main import app


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_ID    = UUID("ac8bcd95-8a77-4ea1-8286-789dd27c313b")
CHATBOT_ID = UUID("aa73d3e2-a04d-463a-b2e4-58edb93ab871")
FAKE_CHATBOT_ID = uuid4()

TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'test_files')

engine = create_async_engine(settings.DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Dependency overrides — bypass auth and rate limiting
# ---------------------------------------------------------------------------

app.dependency_overrides[get_current_user] = lambda: USER_ID

# rate_limit_by_user() returns a dependency function — override the returned
# function directly by patching the factory to always return a no-op
def _noop_rate_limit(*args, **kwargs):
    async def _noop(request=None):
        return None
    return _noop

app.dependency_overrides[rate_limit_by_user] = _noop_rate_limit


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

results_dir = os.path.join(project_root, 'tests/results/module')
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, 'document_router.txt')

with open(results_file, "w", encoding="utf-8") as f:
    f.write("=== Document Routes + Service Integration Test Results ===\n\n")

def log_result(test_name: str, result: str):
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"--- {test_name} ---\n")
        f.write(result + "\n\n")
    print(f"[{test_name}] executed.")

def log_failed(test_name: str, error):
    msg = f"FAILED: {error}"
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"--- {test_name} ---\n")
        f.write(msg + "\n\n")
    print(f"[{test_name}] FAILED: {error}")


# ---------------------------------------------------------------------------
# DB cleanup helper
# ---------------------------------------------------------------------------

async def cleanup_documents(document_ids: list[UUID]):
    if not document_ids:
        return
    async with TestingSessionLocal() as db:
        await db.execute(
            text("DELETE FROM documents WHERE id = ANY(:ids)"),
            {"ids": document_ids}
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

async def test_happy_path_full_flow(client: AsyncClient):
    """
    Full upload flow:
    POST /bulk-upload-url → PUT to GCS → POST /bulk-confirm → GET / → GET download → DELETE
    """
    test_name = "Test 1: Happy Path — full upload flow"
    document_ids = []

    try:
        # --- Step 1: Generate signed upload URLs ---
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "route_test_1.txt", "file_type": "text/plain", "file_size": 1024},
                {"chatbot_id": str(CHATBOT_ID), "file_name": "route_test_2.txt", "file_type": "text/plain", "file_size": 1024},
            ]
        }
        response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"

        data = response.json()
        assert data["total"] == 2, f"Expected 2 results, got {data['total']}"
        assert len(data["failed"]) == 0, f"Expected no failures, got {data['failed']}"

        document_ids = [UUID(r["document_id"]) for r in data["results"]]
        upload_items = data["results"]

        # --- Step 2: PUT files directly to GCS ---
        test_file_path = os.path.join(TEST_FILES_DIR, "happy_path.txt")
        with open(test_file_path, "rb") as f:
            file_data = f.read()

        for item in upload_items:
            put_response = requests.put(
                item["upload_url"],
                data=file_data,
                headers={"Content-Type": "text/plain"}
            )
            assert put_response.status_code == 200, f"GCS PUT failed: {put_response.status_code} {put_response.text}"

        # --- Step 3: Wait so files are visible in GCS console ---
        print(f"\n[{test_name}] Files uploaded to GCS. Waiting 30 seconds — check your bucket now...")
        time.sleep(30)

        # --- Step 4: Confirm uploads ---
        confirm_payload = {
            "chatbot_id": str(CHATBOT_ID),
            "document_ids": [str(did) for did in document_ids]
        }
        confirm_response = await client.post("/api/documents/bulk-confirm", json=confirm_payload)
        assert confirm_response.status_code == 200, f"Expected 200, got {confirm_response.status_code}: {confirm_response.text}"

        confirm_data = confirm_response.json()
        assert len(confirm_data["confirmed"]) == 2, f"Expected 2 confirmed, got {len(confirm_data['confirmed'])}"
        assert len(confirm_data["failed"]) == 0, f"Unexpected failures: {confirm_data['failed']}"

        # --- Step 5: List documents ---
        list_response = await client.get(f"/api/documents/?chatbot_id={CHATBOT_ID}&limit=20&offset=0")
        assert list_response.status_code == 200, f"Expected 200, got {list_response.status_code}"

        list_data = list_response.json()
        listed_ids = [UUID(d["document_id"]) for d in list_data["documents"]]
        assert all(did in listed_ids for did in document_ids), "Not all uploaded docs appear in list"

        # --- Step 6: Generate download URL for first document ---
        first_doc_id = document_ids[0]
        download_response = await client.get(f"/api/documents/download/{CHATBOT_ID}/{first_doc_id}")
        assert download_response.status_code == 200, f"Expected 200, got {download_response.status_code}"

        download_data = download_response.json()
        assert "download_url" in download_data, "No download_url in response"
        assert download_data["download_url"].startswith("https://"), "download_url should be a valid URL"

        # Verify the download URL actually works
        dl = requests.get(download_data["download_url"])
        assert dl.status_code == 200, f"Download URL returned {dl.status_code}"
        assert dl.text == file_data.decode("utf-8"), "Downloaded content does not match uploaded content"

        # --- Step 7: Delete both documents ---
        for did in document_ids:
            del_response = await client.delete(f"/api/documents/delete/{CHATBOT_ID}/{did}")
            assert del_response.status_code == 200, f"Delete failed: {del_response.status_code}"

        document_ids = []  # already cleaned up

        log_result(test_name,
            f"POST /bulk-upload-url: 201 ✓\n"
            f"GCS PUT (2 files): 200 ✓\n"
            f"POST /bulk-confirm: 200 ✓\n"
            f"Confirmed count: 2 ✓\n"
            f"GET /: 200, documents listed ✓\n"
            f"GET /download: 200, URL valid ✓\n"
            f"Downloaded content matches: True ✓\n"
            f"DELETE (2 files): 200 ✓"
        )

    except Exception as e:
        log_failed(test_name, e)
    finally:
        await cleanup_documents(document_ids)


async def test_validation_invalid_file_type(client: AsyncClient):
    """All files with unsupported type — should all land in 'failed', no DB rows created."""
    test_name = "Test 2: Validation — invalid file type"
    try:
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "image.png", "file_type": "image/png", "file_size": 1024},
                {"chatbot_id": str(CHATBOT_ID), "file_name": "video.mp4", "file_type": "video/mp4", "file_size": 1024},
            ]
        }
        response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"

        log_result(test_name,
            f"Status code: {response.status_code} ✓\n"
            f"All invalid types correctly rejected: True ✓"
        )
    except Exception as e:
        log_failed(test_name, e)


async def test_validation_oversized_file(client: AsyncClient):
    """File exceeding 50MB limit — should land in 'failed'."""
    test_name = "Test 3: Validation — oversized file"
    try:
        oversized = 50 * 1024 * 1024 + 1  # 1 byte over limit
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "huge.txt", "file_type": "text/plain", "file_size": oversized},
            ]
        }
        response = await client.post("/api/documents/bulk-upload-url", json=payload)
        data = response.json()

        # Oversized file should be in failed, not in results
        assert len(data.get("failed", [])) > 0 or data.get("total", 0) == 0, \
            "Oversized file should be rejected"

        log_result(test_name,
            f"Status code: {response.status_code}\n"
            f"Failed list: {data.get('failed')}\n"
            f"Results count: {data.get('total', 0)}\n"
            f"Oversized file correctly handled: True ✓"
        )
    except Exception as e:
        log_failed(test_name, e)


async def test_partial_validation_failure(client: AsyncClient):
    """Mix of valid and invalid files — valid ones proceed, invalid go to 'failed'."""
    test_name = "Test 4: Validation — partial failure (mixed valid and invalid)"
    document_ids = []
    try:
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "valid.txt", "file_type": "text/plain", "file_size": 1024},
                {"chatbot_id": str(CHATBOT_ID), "file_name": "invalid.exe", "file_type": "application/x-msdownload", "file_size": 1024},
            ]
        }
        response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"

        data = response.json()
        assert data["total"] == 1, f"Expected 1 valid result, got {data['total']}"
        assert len(data["failed"]) == 1, f"Expected 1 failed, got {data['failed']}"
        assert data["failed"][0] == "invalid.exe"

        document_ids = [UUID(r["document_id"]) for r in data["results"]]

        log_result(test_name,
            f"Total valid: {data['total']} ✓\n"
            f"Failed: {data['failed']} ✓\n"
            f"Valid file proceeded, invalid rejected: True ✓"
        )
    except Exception as e:
        log_failed(test_name, e)
    finally:
        await cleanup_documents(document_ids)


async def test_confirm_without_gcs_upload(client: AsyncClient):
    """Confirm a document that was never PUT to GCS — should land in 'failed'."""
    test_name = "Test 5: Confirm — file not in GCS"
    document_ids = []
    try:
        # Create the DB record but skip the GCS PUT
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "never_uploaded.txt", "file_type": "text/plain", "file_size": 1024},
            ]
        }
        upload_response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert upload_response.status_code == 201

        data = upload_response.json()
        document_ids = [UUID(r["document_id"]) for r in data["results"]]

        # Skip the GCS PUT intentionally — go straight to confirm
        confirm_payload = {
            "chatbot_id": str(CHATBOT_ID),
            "document_ids": [str(did) for did in document_ids]
        }
        confirm_response = await client.post("/api/documents/bulk-confirm", json=confirm_payload)
        assert confirm_response.status_code == 200

        confirm_data = confirm_response.json()
        assert len(confirm_data["confirmed"]) == 0, "Should not confirm file that was never uploaded"
        assert len(confirm_data["failed"]) == 1, "Should have 1 failed entry"
        assert "not found in storage" in confirm_data["failed"][0]["reason"].lower()

        log_result(test_name,
            f"Confirmed (should be 0): {len(confirm_data['confirmed'])} ✓\n"
            f"Failed (should be 1): {len(confirm_data['failed'])} ✓\n"
            f"Reason: {confirm_data['failed'][0]['reason']} ✓"
        )
    except Exception as e:
        log_failed(test_name, e)
    finally:
        await cleanup_documents(document_ids)


async def test_confirm_nonexistent_document(client: AsyncClient):
    """Confirm a document_id that doesn't exist in DB — should land in 'failed'."""
    test_name = "Test 6: Confirm — non-existent document_id"
    try:
        fake_id = uuid4()
        confirm_payload = {
            "chatbot_id": str(CHATBOT_ID),
            "document_ids": [str(fake_id)]
        }
        response = await client.post("/api/documents/bulk-confirm", json=confirm_payload)
        assert response.status_code == 200

        data = response.json()
        assert len(data["confirmed"]) == 0
        assert len(data["failed"]) == 1
        assert "not found" in data["failed"][0]["reason"].lower()

        log_result(test_name,
            f"Confirmed (should be 0): {len(data['confirmed'])} ✓\n"
            f"Failed (should be 1): {len(data['failed'])} ✓\n"
            f"Reason: {data['failed'][0]['reason']} ✓"
        )
    except Exception as e:
        log_failed(test_name, e)


async def test_list_pagination(client: AsyncClient):
    """GET / — verify limit/offset and total."""
    test_name = "Test 7: List — pagination"
    document_ids = []
    try:
        # Insert 5 documents
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": f"list_test_{i}.txt", "file_type": "text/plain", "file_size": 1024}
                for i in range(5)
            ]
        }
        upload_response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert upload_response.status_code == 201
        document_ids = [UUID(r["document_id"]) for r in upload_response.json()["results"]]

        # Page 1
        r1 = await client.get(f"/api/documents/?chatbot_id={CHATBOT_ID}&limit=3&offset=0")
        assert r1.status_code == 200
        d1 = r1.json()
        assert len(d1["documents"]) == 3
        assert d1["total"] >= 5

        # Page 2
        r2 = await client.get(f"/api/documents/?chatbot_id={CHATBOT_ID}&limit=3&offset=3")
        assert r2.status_code == 200
        d2 = r2.json()
        assert len(d2["documents"]) >= 1

        # No overlap
        page1_ids = {d["document_id"] for d in d1["documents"]}
        page2_ids = {d["document_id"] for d in d2["documents"]}
        assert page1_ids.isdisjoint(page2_ids), "Pages should not overlap"

        log_result(test_name,
            f"Total: {d1['total']} ✓\n"
            f"Page 1 (limit=3): {len(d1['documents'])} results ✓\n"
            f"Page 2 (offset=3): {len(d2['documents'])} results ✓\n"
            f"No overlap between pages: True ✓"
        )
    except Exception as e:
        log_failed(test_name, e)
    finally:
        await cleanup_documents(document_ids)


async def test_delete_removes_from_gcs_and_db(client: AsyncClient):
    """DELETE — verify document is gone from both GCS and DB after deletion."""
    test_name = "Test 8: Delete — removed from GCS and DB"
    document_ids = []
    try:
        from api.modules.documents.gcs_service import gcs_service

        # Upload one file fully
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "delete_test.txt", "file_type": "text/plain", "file_size": 1024}
            ]
        }
        upload_response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert upload_response.status_code == 201

        result = upload_response.json()["results"][0]
        doc_id = UUID(result["document_id"])
        storage_path = result["storage_path"]
        document_ids = [doc_id]

        # PUT to GCS
        test_file_path = os.path.join(TEST_FILES_DIR, "happy_path.txt")
        with open(test_file_path, "rb") as f:
            file_data = f.read()

        put_response = requests.put(result["upload_url"], data=file_data, headers={"Content-Type": "text/plain"})
        assert put_response.status_code == 200

        # Confirm
        confirm_response = await client.post("/api/documents/bulk-confirm", json={
            "chatbot_id": str(CHATBOT_ID),
            "document_ids": [str(doc_id)]
        })
        assert confirm_response.status_code == 200

        print(f"\n[{test_name}] File uploaded. Waiting 30 seconds — check your bucket now...")
        time.sleep(30)

        # Verify exists in GCS before delete
        exists_before = gcs_service.object_exists(storage_path)
        assert exists_before, "File should exist in GCS before delete"

        # Delete via route
        del_response = await client.delete(f"/api/documents/delete/{CHATBOT_ID}/{doc_id}")
        assert del_response.status_code == 200

        document_ids = []  # already deleted

        # Verify gone from GCS
        exists_after = gcs_service.object_exists(storage_path)
        assert not exists_after, "File should be gone from GCS after delete"

        # Verify gone from DB via list
        list_response = await client.get(f"/api/documents/?chatbot_id={CHATBOT_ID}&limit=100&offset=0")
        listed_ids = [d["document_id"] for d in list_response.json()["documents"]]
        assert str(doc_id) not in listed_ids, "Document should not appear in list after delete"

        log_result(test_name,
            f"File existed in GCS before delete: {exists_before} ✓\n"
            f"DELETE /delete response: {del_response.status_code} ✓\n"
            f"File exists in GCS after delete: {exists_after} ✓\n"
            f"Document removed from DB list: True ✓"
        )
    except Exception as e:
        log_failed(test_name, e)
    finally:
        await cleanup_documents(document_ids)


async def test_download_wrong_chatbot(client: AsyncClient):
    """GET /download with wrong chatbot_id — should return 404."""
    test_name = "Test 9: Download — wrong chatbot_id returns 404"
    document_ids = []
    try:
        # Create a document under real chatbot
        payload = {
            "chatbot_id": str(CHATBOT_ID),
            "files": [
                {"chatbot_id": str(CHATBOT_ID), "file_name": "auth_test.txt", "file_type": "text/plain", "file_size": 1024}
            ]
        }
        upload_response = await client.post("/api/documents/bulk-upload-url", json=payload)
        assert upload_response.status_code == 201

        doc_id = UUID(upload_response.json()["results"][0]["document_id"])
        document_ids = [doc_id]

        # Try to download with wrong chatbot_id
        response = await client.get(f"/api/documents/download/{FAKE_CHATBOT_ID}/{doc_id}")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"

        log_result(test_name,
            f"Download with wrong chatbot_id returned: {response.status_code} ✓\n"
            f"Cross-chatbot access correctly blocked: True ✓"
        )
    except Exception as e:
        log_failed(test_name, e)
    finally:
        await cleanup_documents(document_ids)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all_tests():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        await test_happy_path_full_flow(client)
        await test_validation_invalid_file_type(client)
        await test_validation_oversized_file(client)
        await test_partial_validation_failure(client)
        await test_confirm_without_gcs_upload(client)
        await test_confirm_nonexistent_document(client)
        await test_list_pagination(client)
        await test_delete_removes_from_gcs_and_db(client)
        await test_download_wrong_chatbot(client)

    print(f"\nAll tests completed. Results written to: {results_file}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())