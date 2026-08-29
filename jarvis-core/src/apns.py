"""
apns.py — Apple Push Notification service (HTTP/2 JWT auth)
===========================================================
Requires: pip install "httpx[http2]" cryptography

APNs JWT flow:
  1. Sign a short-lived JWT with the p8 private key (ES256).
  2. POST to https://api.push.apple.com/3/device/{token} over HTTP/2.
  3. Refresh the JWT every 50 min (APNs invalidates tokens after 60 min).

Configuration (.env):
  APNS_KEY_ID      — 10-char key ID shown in Apple Developer → Keys
  APNS_TEAM_ID     — 10-char team ID (top-right of developer.apple.com)
  APNS_BUNDLE_ID   — bundle ID of the iOS app (e.g. com.example.JarvisApp)
  APNS_KEY_PATH    — absolute path to the downloaded AuthKey_XXXXXXXXXX.p8
  APNS_ENV         — "production" (TestFlight/App Store) or "sandbox" (Xcode direct)
"""

import base64
import json
import time
from typing import Optional

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from config import APNS_BUNDLE_ID, APNS_ENV, APNS_KEY_ID, APNS_KEY_PATH, APNS_TEAM_ID
from helpers import get_logger

logger = get_logger("jarvis-apns")

_APNS_HOST = (
    "api.push.apple.com"
    if APNS_ENV == "production"
    else "api.sandbox.push.apple.com"
)

# JWT cache: one token per process, refreshed every 50 min.
_jwt_token: str = ""
_jwt_issued_at: int = 0


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _build_jwt() -> str:
    """Generate a signed APNs JWT. Caches for 50 min."""
    global _jwt_token, _jwt_issued_at

    now = int(time.time())
    if _jwt_token and now - _jwt_issued_at < 3000:  # 50 min
        return _jwt_token

    if not APNS_KEY_PATH or not APNS_KEY_ID or not APNS_TEAM_ID:
        raise RuntimeError(
            "APNs not configured — set APNS_KEY_PATH, APNS_KEY_ID, APNS_TEAM_ID in .env"
        )

    header  = _b64url(json.dumps({"alg": "ES256", "kid": APNS_KEY_ID}).encode())
    payload = _b64url(json.dumps({"iss": APNS_TEAM_ID, "iat": now}).encode())
    message = f"{header}.{payload}".encode()

    with open(APNS_KEY_PATH, "rb") as f:
        private_key = load_pem_private_key(f.read(), password=None)

    der_sig      = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    r, s         = decode_dss_signature(der_sig)
    raw_sig      = r.to_bytes(32, "big") + s.to_bytes(32, "big")

    _jwt_token    = f"{header}.{payload}.{_b64url(raw_sig)}"
    _jwt_issued_at = now
    return _jwt_token


def is_real_apns_token(token: str) -> bool:
    """True when token looks like a real 64-hex APNs token (not a polling placeholder)."""
    return len(token) == 64 and all(c in "0123456789abcdef" for c in token.lower())


async def send_apns_push(
    device_token: str,
    body: str,
    title: str = "Jarvis",
    data: Optional[dict] = None,
) -> bool:
    """
    Send an APNs alert push. Returns True on success.
    Silently returns False when APNs is not configured or token is a polling placeholder.
    """
    if not is_real_apns_token(device_token):
        return False
    if not APNS_BUNDLE_ID:
        logger.debug("APNs skipped — APNS_BUNDLE_ID not set")
        return False

    try:
        jwt = _build_jwt()
    except Exception as exc:
        logger.error("APNs JWT error: %s", exc)
        return False

    payload: dict = {
        "aps": {
            "alert": {"title": title, "body": body},
            "sound": "default",
        }
    }
    if data:
        payload.update(data)

    url = f"https://{_APNS_HOST}/3/device/{device_token}"
    headers = {
        "authorization": f"bearer {jwt}",
        "apns-topic":    APNS_BUNDLE_ID,
        "apns-push-type": "alert",
        "apns-priority": "10",
    }

    try:
        async with httpx.AsyncClient(http2=True, timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            logger.info("APNs push sent → %s…", device_token[:8])
            return True
        logger.warning(
            "APNs push rejected: HTTP %d — %s", resp.status_code, resp.text[:200]
        )
        return False
    except Exception as exc:
        logger.error("APNs push error: %s", exc)
        return False
