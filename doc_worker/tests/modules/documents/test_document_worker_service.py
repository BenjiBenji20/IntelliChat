import os
import sys
import asyncio
import asyncpg
import pytest
import pytest_asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

# ============================================================
# PATH SETUP — ensures doc_worker imports resolve correctly
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from doc_worker.modules.documents.document_worker_service import DocumentWorkerService
from doc_worker.modules.documents.document_worker_schema import (
    ProcessDocumentRequestSchema,
    ProcessDocumentResponseSchema,
)
from doc_worker.configs.gcs import init_gcs_client
from doc_worker.configs.qdrant import init_qdrant_client, close_qdrant_client
from doc_worker.configs.settings import settings

# ============================================================
# CONSTANTS
# ============================================================
CHATBOT_ID = UUID("0a95d2a8-ee55-49e1-ab55-f830d19f9fe1")

TEST_FILES = {
    "txt": {
        "document_id": UUID("d7ba9bad-3b9e-4780-bc96-22d203bac5f4"),
        "file_name": "i_love_about_re_zero.txt",
    },
    "md": {
        "document_id": UUID("2b1d719b-03d0-46c5-af3a-9fd53d12b5d4"),
        "file_name": "cvms-narrative-knowledge-source.md",
    },
    "json": {
        "document_id": UUID("8d5a1f2b-22b7-4e0c-986f-c849c8c35873"),
        "file_name": "cvms-structured-data.json",
    },
    "jsonl": {
        "document_id": UUID("f36832a8-d442-4a3b-bf9c-993e0605665f"),
        "file_name": "cvms-qa-structured-data.jsonl",
    },
    "pdf": {
        "document_id": UUID("f77104f0-c2fa-4379-bbf3-4bd931c8f050"),
        "file_name": "CVMSM-Info.pdf",
    },
}

TEST_FILES_DIR = Path(__file__).resolve().parents[3] / "doc_worker" / "test_files"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATOR = "\n\n"

# ============================================================
# RESULTS WRITER
# ============================================================
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "modules" / "documents"
RESULTS_FILE = RESULTS_DIR / "document_worker_service_test_results.txt"

def write_result(test_name: str, passed: bool, detail: str = ""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    status = "✅ PASS" if passed else "❌ FAIL"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {status} | {test_name}")
        if detail:
            f.write(f" | {detail}")
        f.write("\n")


def write_section(title: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


# ============================================================
# FIXTURES
# ============================================================
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def db_connection():
    """Real Supabase asyncpg connection."""
    conn = await asyncpg.connect(dsn=settings.DATABASE_URL)
    yield conn
    await conn.close()


@pytest_asyncio.fixture(scope="session", autouse=True, loop_scope="session")
async def setup_clients():
    """Initialize GCS and Qdrant clients per test to avoid loop match errors."""
    init_gcs_client()
    await init_qdrant_client()
    yield
    await close_qdrant_client()


@pytest.fixture
def mock_gcs_bucket():
    """
    Mock GCS bucket — returns real file bytes from test_files/ directory.
    Simulates GCS without needing a real bucket connection in tests.
    """
    def make_mock_bucket(file_type: str):
        file_info = TEST_FILES[file_type]
        file_path = TEST_FILES_DIR / file_info["file_name"]

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = file_path.read_bytes()

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        return mock_bucket

    return make_mock_bucket


@pytest.fixture(scope="session")
def mock_gcs_bucket_not_found():
    """Mock GCS bucket that simulates file not found."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    return mock_bucket


def build_payload(file_type: str, override_status: str = None) -> ProcessDocumentRequestSchema:
    file_info = TEST_FILES[file_type]
    return ProcessDocumentRequestSchema(
        document_id=file_info["document_id"],
        chatbot_id=CHATBOT_ID,
        file_name=file_info["file_name"],
        file_type=file_type,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=SEPARATOR,
    )


# ============================================================
# SETUP RESULTS FILE
# ============================================================
def pytest_configure(config):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"DocumentWorkerService Test Results\n")
        f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n")


# ============================================================
# HAPPY PATH TESTS — one per file type
# ============================================================
class TestHappyPath:

    @pytest.mark.asyncio(loop_scope="session")
    async def test_process_txt_document(self, db_connection):
        write_section("HAPPY PATH TESTS")
        payload = build_payload("txt")
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = (
            isinstance(result, ProcessDocumentResponseSchema)
            and result.status == "indexed"
            and result.document_id == payload.document_id
        )
        write_result(
            "test_process_txt_document",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "indexed", f"Expected indexed, got: {result.status}"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_process_md_document(self, db_connection):
        payload = build_payload("md")
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = result.status == "indexed"
        write_result(
            "test_process_md_document",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "indexed", f"Expected indexed, got: {result.status}"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_process_json_document(self, db_connection):
        payload = build_payload("json")
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = result.status == "indexed"
        write_result(
            "test_process_json_document",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "indexed", f"Expected indexed, got: {result.status}"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_process_jsonl_document(self, db_connection):
        payload = build_payload("jsonl")
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = result.status == "indexed"
        write_result(
            "test_process_jsonl_document",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "indexed", f"Expected indexed, got: {result.status}"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_process_pdf_document(self, db_connection):
        payload = build_payload("pdf")
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = result.status == "indexed"
        write_result(
            "test_process_pdf_document",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "indexed", f"Expected indexed, got: {result.status}"


# ============================================================
# NEGATIVE / GUARDRAIL TESTS
# ============================================================
class TestNegativeCases:

    @pytest.mark.asyncio(loop_scope="session")
    async def test_unsupported_file_type(self, db_connection):
        """Unsupported file type → status=failed, return 200, no retry."""
        write_section("NEGATIVE / GUARDRAIL TESTS")
        payload = build_payload("txt")
        payload.file_type = "xlsx"  # unsupported
        service = DocumentWorkerService(db=db_connection)

        # reset status so idempotency guard doesn't trigger first
        await service.repo.db.execute(
            "UPDATE documents SET status = 'pending' WHERE id = $1",
            payload.document_id
        )

        result = await service.process_document(payload)

        passed = result.status == "failed"
        write_result(
            "test_unsupported_file_type",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "failed"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_file_not_found_in_gcs(self, db_connection):
        """GCS file not found → status=failed, return 200, no retry."""
        payload = build_payload("txt")
        payload.file_name = "does_not_exist_in_gcs.txt"
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = result.status == "failed"
        write_result(
            "test_file_not_found_in_gcs",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "failed"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_document_not_found_in_db(self, db_connection):
        """Document ID not in DB → status=failed, return 200, no retry."""
        payload = build_payload("txt")
        payload.document_id = UUID("00000000-0000-0000-0000-000000000000")  # non-existent
        service = DocumentWorkerService(db=db_connection)

        result = await service.process_document(payload)

        passed = result.status == "failed"
        write_result(
            "test_document_not_found_in_db",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "failed"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_idempotency_already_indexed(self, db_connection):
        """
        Idempotency check — if document is already indexed,
        worker must return 200 immediately without reprocessing.
        Run AFTER a successful happy path test on the same document_id.
        """
        payload = build_payload("txt")
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'indexed' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = (
            result.status == "indexed"
            and result.message == "Document already indexed."
        )
        write_result(
            "test_idempotency_already_indexed",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "indexed"
        assert result.message == "Document already indexed."

    @pytest.mark.asyncio(loop_scope="session")
    async def test_embedding_model_not_configured(self, db_connection):
        """
        Chatbot with no embedding model configured →
        status=failed, return 200, no retry.
        """
        payload = build_payload("txt")
        payload.chatbot_id = UUID("00000000-0000-0000-0000-000000000000")  # no model configured
        service = DocumentWorkerService(db=db_connection)

        await service.repo.db.execute("UPDATE documents SET status = 'pending' WHERE id = $1", payload.document_id)

        result = await service.process_document(payload)

        passed = result.status == "failed"
        write_result(
            "test_embedding_model_not_configured",
            passed,
            f"status={result.status} | message={result.message}"
        )
        assert result.status == "failed"


# ============================================================
# HOW TO RUN
# ============================================================
# Install dependencies first:
#   pip install pytest pytest-asyncio
#
# Run all tests:
#   pytest doc_worker/tests/modules/documents/test_document_worker_service.py -v
#
# Run only happy path:
#   pytest doc_worker/tests/modules/documents/test_document_worker_service.py::TestHappyPath -v
#
# Run only negative cases:
#   pytest doc_worker/tests/modules/documents/test_document_worker_service.py::TestNegativeCases -v
#
# Results written to:
#   doc_worker/tests/results/modules/documents/document_worker_service_test_results.txt
