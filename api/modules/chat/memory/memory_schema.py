from typing import TypedDict, Optional, List

class Turn(TypedDict):
    role: str
    content: str

class MemoryResult(TypedDict):
    turns: Optional[List[Turn]]
    summary: Optional[str]
