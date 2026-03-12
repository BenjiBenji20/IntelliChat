import time
import os
import sys
import asyncio
from uuid import UUID, uuid4

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from configs.settings import settings
from modules.documents.document_repository import DocumentRepository

# Must import all models so SQLAlchemy mapper can resolve all relationships
from db.base import Base
from models.profile import Profile
from models.chatbot import Chatbot
from models.document import Document
from models.embedding_metadata import EmbeddingMetadata
from models.conversation import Conversation
from models.message import Message
from models.api_key import ApiKey
from models.redis_key import RedisKey
from models.embedding_model_key import EmbeddingModelKey
from models.llm_key import LlmKey
from models.project import Project
from models.project_member import ProjectMember
from models.project_invitation import ProjectInvitation
from models.chatbot_behavior import ChatbotBehavior


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_ID    = UUID("ac8bcd95-8a77-4ea1-8286-789dd27c313b")
CHATBOT_ID = UUID("aa73d3e2-a04d-463a-b2e4-58edb93ab871")
FAKE_CHATBOT_ID = uuid4()  # random — should never match anything

engine = create_async_engine(settings.DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

results_dir = os.path.join(project_root, 'tests/results/module')
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, 'document_repository.txt')

with open(results_file, "w", encoding="utf-8") as f:
    f.write("=== Document Repository Integration Test Results ===\n\n")

def log_result(test_name: str, result: str):
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"--- {test_name} ---\n")
        f.write(result + "\n\n")
    print(f"[{test_name}] executed.")

def log_failed(test_name: str, error: Exception):
    log_result(test_name, f"FAILED: {error}")
    print(f"[{test_name}] FAILED: {error}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_files(n: int) -> list[dict]:
    return [
        {
            "file_name": f"test_file_{i+1}.txt",
            "file_type": "text/plain",
            "file_size": 1024,
            "storage_path": f"test-uploads/test_file_{i+1}.txt",
            "status": "pending",
        }
        for i in range(n)
    ]


async def cleanup(db: AsyncSession, document_ids: list[UUID]):
    """Hard delete inserted test rows so dev DB stays clean."""
    if not document_ids:
        return
    await db.execute(
        text("DELETE FROM documents WHERE id = ANY(:ids)"),
        {"ids": document_ids}
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

async def test_bulk_create_single(db: AsyncSession):
    """Insert 1 document — verify all fields persisted correctly."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(1),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]
        
        time.sleep(30)  # 30sec interuption to check file upload in GCS and supabase record before file and record deletion 

        doc = docs[0]
        assert len(docs) == 1, "Should return exactly 1 document"
        assert doc.id is not None, "ID should be populated"
        assert doc.user_id == USER_ID
        assert doc.chatbot_id == CHATBOT_ID
        assert doc.file_name == "test_file_1.txt"
        assert doc.file_type == "text/plain"
        assert doc.status == "pending"
        assert doc.created_at is not None
        assert doc.updated_at is not None

        log_result(
            "Test 1: bulk_create_documents — single insert",
            f"Document created successfully.\n"
            f"ID: {doc.id}\n"
            f"file_name: {doc.file_name}\n"
            f"file_type: {doc.file_type}\n"
            f"status: {doc.status}\n"
            f"created_at populated: {doc.created_at is not None}\n"
            f"All field assertions passed: True"
        )
    except Exception as e:
        log_failed("Test 1: bulk_create_documents — single insert", e)
    finally:
        await cleanup(db, inserted_ids)

    return inserted_ids[0] if inserted_ids else None


async def test_bulk_create_multiple(db: AsyncSession):
    """Insert 5 documents in one call — verify count and unique IDs."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(5),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        assert len(docs) == 5, f"Expected 5 documents, got {len(docs)}"
        unique_ids = set(d.id for d in docs)
        assert len(unique_ids) == 5, "All document IDs should be unique"

        log_result(
            "Test 2: bulk_create_documents — 5 documents",
            f"Documents created: {len(docs)}\n"
            f"All IDs unique: {len(unique_ids) == 5}\n"
            f"IDs: {[str(d.id) for d in docs]}"
        )
    except Exception as e:
        log_failed("Test 2: bulk_create_documents — 5 documents", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_bulk_create_duplicate_filenames(db: AsyncSession):
    """Duplicate file names under same chatbot should succeed (not unique-constrained)."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        files = [
            {"file_name": "duplicate.txt", "file_type": "text/plain", "file_size": 1024, "storage_path": "test-uploads/dup1.txt", "status": "pending"},
            {"file_name": "duplicate.txt", "file_type": "text/plain", "file_size": 1024, "storage_path": "test-uploads/dup2.txt", "status": "pending"},
        ]
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=files,
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]
        
        time.sleep(30)  # 30sec interuption to check file upload in GCS and supabase record before file and record deletion 

        assert len(docs) == 2, "Both duplicate-named docs should insert"
        assert docs[0].file_name == docs[1].file_name == "duplicate.txt"

        log_result(
            "Test 3: bulk_create_documents — duplicate file names",
            f"Both documents inserted successfully.\n"
            f"file_name both: {docs[0].file_name}\n"
            f"IDs are different: {docs[0].id != docs[1].id}"
        )
    except Exception as e:
        log_failed("Test 3: bulk_create_documents — duplicate file names", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_get_by_ids_correct_chatbot(db: AsyncSession, document_id: UUID):
    """Fetch documents that exist under the correct chatbot."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        # Insert fresh docs to fetch
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(3),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        fetched = await repo.get_by_ids_and_chatbot_id(inserted_ids, CHATBOT_ID)

        assert len(fetched) == 3, f"Expected 3, got {len(fetched)}"
        fetched_ids = set(d.id for d in fetched)
        assert fetched_ids == set(inserted_ids), "Fetched IDs should match inserted IDs"

        log_result(
            "Test 4: get_by_ids_and_chatbot_id — correct chatbot",
            f"Requested: {len(inserted_ids)} documents\n"
            f"Fetched: {len(fetched)} documents\n"
            f"All IDs match: True"
        )
    except Exception as e:
        log_failed("Test 4: get_by_ids_and_chatbot_id — correct chatbot", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_get_by_ids_wrong_chatbot(db: AsyncSession):
    """Request IDs that belong to a different chatbot — should return empty."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(2),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        # Fetch using a completely different chatbot_id
        fetched = await repo.get_by_ids_and_chatbot_id(inserted_ids, FAKE_CHATBOT_ID)

        assert len(fetched) == 0, f"Expected 0 results for wrong chatbot, got {len(fetched)}"

        log_result(
            "Test 5: get_by_ids_and_chatbot_id — wrong chatbot",
            f"Fetched with wrong chatbot_id: {len(fetched)} results\n"
            f"Correctly returned empty: True"
        )
    except Exception as e:
        log_failed("Test 5: get_by_ids_and_chatbot_id — wrong chatbot", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_get_by_ids_mixed_chatbot(db: AsyncSession):
    """Mix of valid and foreign chatbot IDs — should return only matching ones."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(2),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        # Add a random UUID that doesn't exist at all
        mixed_ids = inserted_ids + [uuid4(), uuid4()]
        fetched = await repo.get_by_ids_and_chatbot_id(mixed_ids, CHATBOT_ID)

        assert len(fetched) == 2, f"Expected 2 real matches, got {len(fetched)}"

        log_result(
            "Test 6: get_by_ids_and_chatbot_id — mixed valid and fake IDs",
            f"Requested: {len(mixed_ids)} IDs (2 real, 2 fake)\n"
            f"Fetched: {len(fetched)} documents\n"
            f"Only real IDs returned: True"
        )
    except Exception as e:
        log_failed("Test 6: get_by_ids_and_chatbot_id — mixed valid and fake IDs", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_bulk_update_status(db: AsyncSession):
    """Update 3 documents to 'uploaded' — verify all 3 changed."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(3),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        await repo.bulk_update_status(inserted_ids, "uploaded")
        await db.commit()

        # Verify by re-fetching
        fetched = await repo.get_by_ids_and_chatbot_id(inserted_ids, CHATBOT_ID)
        statuses = [d.status for d in fetched]

        assert all(s == "uploaded" for s in statuses), f"Not all statuses updated: {statuses}"

        log_result(
            "Test 7: bulk_update_status — update 3 to uploaded",
            f"Documents updated: {len(fetched)}\n"
            f"All statuses are 'uploaded': {all(s == 'uploaded' for s in statuses)}\n"
            f"Statuses: {statuses}"
        )
    except Exception as e:
        log_failed("Test 7: bulk_update_status — update 3 to uploaded", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_bulk_update_status_with_nonexistent_id(db: AsyncSession):
    """Update with a mix of real and non-existent IDs — real ones should still update."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(2),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        # Mix real IDs with fake ones
        mixed_ids = inserted_ids + [uuid4(), uuid4()]
        await repo.bulk_update_status(mixed_ids, "uploaded")
        await db.commit()

        fetched = await repo.get_by_ids_and_chatbot_id(inserted_ids, CHATBOT_ID)
        statuses = [d.status for d in fetched]

        assert all(s == "uploaded" for s in statuses), f"Real docs not updated: {statuses}"

        log_result(
            "Test 8: bulk_update_status — with non-existent IDs mixed in",
            f"Real documents updated correctly: {all(s == 'uploaded' for s in statuses)}\n"
            f"No error raised for fake IDs: True"
        )
    except Exception as e:
        log_failed("Test 8: bulk_update_status — with non-existent IDs mixed in", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_get_all_paginated(db: AsyncSession):
    """Pagination — verify limit/offset and total count."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(7),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]

        # Page 1: first 3
        page1, total = await repo.get_all_by_chatbot_id(CHATBOT_ID, limit=3, offset=0)
        # Page 2: next 3
        page2, _ = await repo.get_all_by_chatbot_id(CHATBOT_ID, limit=3, offset=3)
        # Page 3: last 1
        page3, _ = await repo.get_all_by_chatbot_id(CHATBOT_ID, limit=3, offset=6)

        assert len(page1) == 3, f"Page 1 should have 3, got {len(page1)}"
        assert len(page2) == 3, f"Page 2 should have 3, got {len(page2)}"
        assert len(page3) >= 1, f"Page 3 should have at least 1, got {len(page3)}"
        assert total >= 7, f"Total should be at least 7, got {total}"

        # No overlap between pages
        page1_ids = set(d.id for d in page1)
        page2_ids = set(d.id for d in page2)
        assert page1_ids.isdisjoint(page2_ids), "Pages should not overlap"

        log_result(
            "Test 9: get_all_by_chatbot_id — pagination",
            f"Total count: {total}\n"
            f"Page 1 (limit=3, offset=0): {len(page1)} results\n"
            f"Page 2 (limit=3, offset=3): {len(page2)} results\n"
            f"Page 3 (limit=3, offset=6): {len(page3)} results\n"
            f"No overlap between pages: True"
        )
    except Exception as e:
        log_failed("Test 9: get_all_by_chatbot_id — pagination", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_get_all_empty_chatbot(db: AsyncSession):
    """Chatbot with no documents should return empty list and total=0."""
    repo = DocumentRepository(db)
    try:
        docs, total = await repo.get_all_by_chatbot_id(FAKE_CHATBOT_ID, limit=20, offset=0)

        assert len(docs) == 0, f"Expected 0 docs, got {len(docs)}"
        assert total == 0, f"Expected total=0, got {total}"

        log_result(
            "Test 10: get_all_by_chatbot_id — empty chatbot",
            f"Documents returned: {len(docs)}\n"
            f"Total: {total}\n"
            f"Correctly returned empty: True"
        )
    except Exception as e:
        log_failed("Test 10: get_all_by_chatbot_id — empty chatbot", e)


async def test_delete_cascade(db: AsyncSession):
    """
    Delete a document — verify the row is gone.
    FK cascade to embeddings_metadata is enforced by the DB schema (ondelete=CASCADE).
    We verify the document itself is deleted; cascade is trusted to the DB constraint.
    """
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(1),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]
        doc_id = docs[0].id

        # Delete via base repo method
        await repo.delete(doc_id)
        await db.commit()
        inserted_ids = []  # already deleted, skip cleanup

        # Verify gone
        fetched = await repo.get_by_ids_and_chatbot_id([doc_id], CHATBOT_ID)
        assert len(fetched) == 0, "Document should be gone after delete"

        log_result(
            "Test 11: delete — document removed",
            f"Document {doc_id} deleted.\n"
            f"Fetch after delete returned: {len(fetched)} results\n"
            f"Document successfully removed: True\n"
            f"FK cascade to embeddings_metadata enforced by DB ondelete=CASCADE constraint."
        )
    except Exception as e:
        log_failed("Test 11: delete — document removed", e)
    finally:
        await cleanup(db, inserted_ids)


async def test_get_by_document_and_chatbot_id(db: AsyncSession):
    """Single document fetch scoped to chatbot_id."""
    repo = DocumentRepository(db)
    inserted_ids = []
    try:
        docs = await repo.bulk_create_documents(
            user_id=USER_ID,
            chatbot_id=CHATBOT_ID,
            files=make_files(1),
        )
        await db.commit()
        inserted_ids = [d.id for d in docs]
        doc_id = docs[0].id

        # Correct chatbot — should find it
        found = await repo.get_by_document_and_chatbot_id(doc_id, CHATBOT_ID)
        assert found is not None, "Should find document with correct chatbot_id"
        assert found.id == doc_id

        # Wrong chatbot — should return None
        not_found = await repo.get_by_document_and_chatbot_id(doc_id, FAKE_CHATBOT_ID)
        assert not_found is None, "Should return None for wrong chatbot_id"

        log_result(
            "Test 12: get_by_document_and_chatbot_id",
            f"Found with correct chatbot_id: {found is not None}\n"
            f"Returned None for wrong chatbot_id: {not_found is None}"
        )
    except Exception as e:
        log_failed("Test 12: get_by_document_and_chatbot_id", e)
    finally:
        await cleanup(db, inserted_ids)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all_tests():
    """Per session test"""
    async with TestingSessionLocal() as db:
        await test_bulk_create_single(db)
    async with TestingSessionLocal() as db:
        await test_bulk_create_multiple(db)
    async with TestingSessionLocal() as db:
        await test_bulk_create_duplicate_filenames(db)
    async with TestingSessionLocal() as db:
        await test_get_by_ids_correct_chatbot(db, None)
    async with TestingSessionLocal() as db:
        await test_get_by_ids_wrong_chatbot(db)
    async with TestingSessionLocal() as db:
        await test_get_by_ids_mixed_chatbot(db)
    async with TestingSessionLocal() as db:
        await test_bulk_update_status(db)
    async with TestingSessionLocal() as db:
        await test_bulk_update_status_with_nonexistent_id(db)
    async with TestingSessionLocal() as db:
        await test_get_all_paginated(db)
    async with TestingSessionLocal() as db:
        await test_get_all_empty_chatbot(db)
    async with TestingSessionLocal() as db:
        await test_delete_cascade(db)
    async with TestingSessionLocal() as db:
        await test_get_by_document_and_chatbot_id(db)

    print(f"\nAll tests completed. Results written to: {results_file}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())