from uuid import UUID

def create_collection_name(chatbot_id: UUID) -> str:
    return f"chatbot_{chatbot_id}"
    