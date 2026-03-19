from uuid import UUID


def construct_file_path(
    chatbot_id: UUID, document_id: UUID, file_name: str
) -> str:
    """
    Deterministic, user-scoped GCS object key.
    Pattern: api/uploads/{chatbot_id}/{document_id}/{file_name}

    Keeping user_id at the top level makes it trivial to:
      - List all files for a user
      - Apply lifecycle rules per prefix
      - Audit access patterns
    """
    return f"api/uploads/{chatbot_id}/{document_id}/{file_name}"
