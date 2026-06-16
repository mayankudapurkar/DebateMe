from pydantic import BaseModel
from typing import Optional, List


class StartDebateRequest(BaseModel):
    topic: str
    user_position: str
    session_id: Optional[str] = None


class ContinueDebateRequest(BaseModel):
    session_id: str
    user_argument: str


class EndDebateRequest(BaseModel):
    session_id: str


class FallacyDetail(BaseModel):
    name: str
    explanation: str


class FallacyResult(BaseModel):
    fallacies_found: bool
    fallacies: List[FallacyDetail]
    clean_argument: bool


class DebateResponse(BaseModel):
    session_id: str
    ai_response: str
    turn: int
    chunks_indexed: Optional[int] = None
    fallacies: Optional[FallacyResult] = None


class ScoreResponse(BaseModel):
    session_id: str
    scores: dict