"""routes/device.py — iOS device registration and push notification polling."""

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import USER_CODES
from deps import REDIS_CLIENT
from helpers import get_logger
from self import generate_proactive_push

logger = get_logger("jarvis-device")
router = APIRouter(tags=["device"])


class DeviceRegisterRequest(BaseModel):
    user_code:    str
    device_token: str


@router.post("/device/register")
async def device_register(req: DeviceRegisterRequest):
    """
    Register an iOS device token for a user.
    Stored in Redis under jarvis:device:token:{user_code}.
    """
    if req.user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    if not req.device_token:
        raise HTTPException(400, "device_token required")

    token_key = f"jarvis:device:token:{req.user_code}"
    existing  = REDIS_CLIENT.get(token_key)
    if existing and existing == req.device_token:
        logger.debug("Device re-registered (no-op) for %s", req.user_code)
    else:
        REDIS_CLIENT.set(token_key, req.device_token)
        logger.info("Device registered for %s", req.user_code)
    return {"status": "ok", "user_code": req.user_code}


@router.get("/device/pending/{user_code}")
async def device_pending(user_code: str):
    """
    Poll for pending push notifications. Returns and clears the queue.
    Called by the iOS app every ~15 min via BGAppRefreshTask.
    """
    if user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")

    pending_key = f"jarvis:push:pending:{user_code}"
    messages: list[dict] = []
    while True:
        raw = REDIS_CLIENT.lpop(pending_key)
        if raw is None:
            break
        try:
            messages.append(json.loads(raw))
        except Exception:
            pass

    return {"user_code": user_code, "messages": messages, "count": len(messages)}


@router.post("/device/push/test/{user_code}")
async def device_push_test(user_code: str):
    """Manually trigger a proactive push generation for one user (dev/test)."""
    if user_code not in USER_CODES:
        raise HTTPException(403, "Invalid user code")
    result = await generate_proactive_push(user_code)
    return {"status": result}
