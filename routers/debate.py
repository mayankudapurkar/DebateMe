from fastapi import APIRouter, HTTPException
from models.schemas import (
    StartDebateRequest, ContinueDebateRequest,
    EndDebateRequest, DebateResponse, ScoreResponse
)
from services.debate_engine import DebateEngine
import uuid

router = APIRouter()

# In-memory session store (use Redis in production)
sessions: dict[str, DebateEngine] = {}


@router.post("/start", response_model=DebateResponse)
async def start_debate(req: StartDebateRequest):
    """Start a new debate session"""
    session_id = str(uuid.uuid4())
    engine = DebateEngine()

    try:
        result = engine.start_debate(req.topic, req.user_position)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start debate: {str(e)}")

    sessions[session_id] = engine

    return DebateResponse(
        session_id=session_id,
        ai_response=result["ai_response"],
        turn=result["turn"],
        chunks_indexed=result["chunks_indexed"]
    )


@router.post("/continue", response_model=DebateResponse)
async def continue_debate(req: ContinueDebateRequest):
    """Continue an existing debate"""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = sessions[req.session_id]

    try:
        result = engine.continue_debate(req.user_argument)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DebateResponse(
    session_id=req.session_id,
    ai_response=result["ai_response"],
    turn=result["turn"],
    fallacies=result.get("fallacies")
    )


@router.post("/end", response_model=ScoreResponse)
async def end_debate(req: EndDebateRequest):
    """End debate and get scores"""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = sessions[req.session_id]

    try:
        scores = engine.end_debate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Clean up session
    del sessions[req.session_id]

    return ScoreResponse(session_id=req.session_id, scores=scores)


@router.get("/sessions")
async def list_sessions():
    """Dev endpoint - list active sessions"""
    return {"active_sessions": list(sessions.keys())}
