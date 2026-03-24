"""routes/memory_routes.py — User memory and profile inspection endpoints."""

from fastapi import APIRouter, HTTPException

from config import USER_CODES
from deps import REDIS_CLIENT
from memory import (
    get_emotional_state,
    get_recent_conversations,
    get_self_memory,
    get_user_preferences,
    get_user_profile,
    get_user_projects,
)

router = APIRouter(tags=["memory"])


@router.get("/memory/profile/{user_code}")
async def memory_profile(user_code: str):
    return {
        "profile":     get_user_profile(user_code),
        "projects":    get_user_projects(user_code),
        "preferences": get_user_preferences(user_code),
    }


@router.get("/memory/emotional-state")
async def memory_emotion():
    return get_emotional_state()


@router.get("/memory/recent/{user_code}")
async def memory_recent(user_code: str, hours: int = 24):
    if user_code not in USER_CODES:
        raise HTTPException(403)
    return {"conversations": get_recent_conversations(user_code, hours)}


@router.get("/memory/self")
async def memory_self():
    return get_self_memory()


@router.delete("/memory/reset")
async def memory_reset():
    r = REDIS_CLIENT
    for key in r.scan_iter("working:*"):
        r.delete(key)
    for key in r.scan_iter("user:*"):
        r.delete(key)
    for key in r.scan_iter("episodic:*"):
        r.delete(key)
    r.delete("jarvis:emotional_state")
    return {"status": "memory reset"}
