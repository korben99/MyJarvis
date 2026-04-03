"""routes/briefing.py — Morning briefing endpoints + scheduler job."""

from fastapi import APIRouter, Header, HTTPException

from briefing import deliver_briefing, gather_briefing, get_stored_briefing, store_briefing
from config import USER_CODES, USERS
from helpers import get_logger

logger = get_logger("jarvis-briefing")
router = APIRouter(tags=["briefing"])


async def run_morning_briefings():
    """Scheduled job: generate and deliver briefings for users with briefing_enabled=true."""
    logger.info("Morning briefing job started")
    for user_code, user in USERS.items():
        if not user.get("briefing_enabled", False):
            continue
        try:
            result = await gather_briefing(user_code)
            store_briefing(user_code, result)
            deliver_briefing(user_code, result)
        except Exception as exc:
            logger.error("Briefing failed for %s: %s", user_code, type(exc).__name__)
    logger.info("Morning briefing job complete")


@router.post("/briefing/generate/{user_code}")
async def briefing_generate(user_code: str, authorization: str = Header(default=None)):
    """Generate (or regenerate) the morning briefing on demand."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    result = await gather_briefing(user_code)
    store_briefing(user_code, result)
    deliver_briefing(user_code, result)
    return {
        "status": "ok",
        "user": USER_CODES[user_code],
        "generated_at": result.generated_at,
        "preview": result.text[:200],
    }


@router.get("/briefing/{user_code}")
async def briefing_get(user_code: str, authorization: str = Header(default=None)):
    """Return the stored morning briefing for a user."""
    requesting_code = None
    if authorization and authorization.startswith("Bearer "):
        requesting_code = authorization[7:].strip()
    if not requesting_code or requesting_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")

    stored = get_stored_briefing(user_code)
    if not stored:
        raise HTTPException(404, "No briefing available — call /briefing/generate first")
    return {
        "user": stored.user_name,
        "generated_at": stored.generated_at,
        "text": stored.text,
        "html": stored.html,
    }
