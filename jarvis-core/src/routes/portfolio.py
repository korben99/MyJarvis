"""routes/portfolio.py — Trading portfolio endpoints."""

import asyncio
import os
import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from config import (
    DEFAULT_TEMP,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    USER_CODES,
    USER_TRADING,
)
from deps import REDIS_CLIENT
from helpers import call_llm_async, get_logger
from trading import (
    get_portfolio,
    get_portfolio_summary_text,
    import_csv_to_redis,
    suggest_thresholds_llm,
)
from trading.core import TRADE_DATA_DIR

logger = get_logger("jarvis-portfolio")
router = APIRouter(tags=["portfolio"])


def _auth(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:].strip()
    return None


def _garde(authorization: str | None, user_code: str) -> None:
    """Autorise l'accès au portefeuille de `user_code`, ou lève. Trois conditions :

    1. Le jeton porte le code de CE utilisateur. Vérifier seulement qu'il désigne un
       utilisateur connu laisse n'importe quel jeton valide lire, importer ou modifier le
       portefeuille de n'importe qui — les données financières d'un membre du foyer sont
       lisibles par un autre.
    2. L'utilisateur existe.
    3. Le trading lui est activé. `USER_TRADING` est le drapeau qui décide qu'un compte a
       un portefeuille ; sans cette garde, un import ouvre un portefeuille à quelqu'un qui
       n'en a pas, et le briefing le lui sert ensuite.
    """
    if _auth(authorization) != user_code:
        raise HTTPException(403, "Invalid user code")
    if user_code not in USER_CODES:
        raise HTTPException(404, "Unknown user code")
    if user_code not in USER_TRADING:
        raise HTTPException(403, "Trading not enabled for this user")


@router.get("/portfolio/{user_code}")
async def portfolio_get(user_code: str, authorization: str = Header(default=None)):
    """Return the full portfolio with live P&L for a user."""
    _garde(authorization, user_code)
    return {"user": USER_CODES[user_code], "positions": get_portfolio(user_code)}


@router.post("/portfolio/import/{user_code}")
async def portfolio_import(user_code: str, authorization: str = Header(default=None)):
    """Force a re-parse of the latest CSV in the user's TRADE_DATA_DIR subfolder."""
    _garde(authorization, user_code)

    REDIS_CLIENT.delete(f"trade:{user_code}:last_import_ts")
    count = await asyncio.to_thread(import_csv_to_redis, user_code)
    return {"status": "ok", "positions_imported": count}


@router.post("/portfolio/upload/{user_code}")
async def portfolio_upload(
    user_code: str,
    file: UploadFile = File(...),
    authorization: str = Header(default=None),
):
    """Upload a Boursorama CSV export directly and import it immediately."""
    _garde(authorization, user_code)
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted")

    # Dépôt dans le sous-dossier de l'utilisateur, et `TRADE_DATA_DIR` importé de
    # trading.core plutôt que relu ici : les deux lectures avaient des valeurs par défaut
    # différentes, donc l'écriture n'atterrissait au bon endroit que parce que .env fixe
    # la variable.
    trade_dir = os.path.join(TRADE_DATA_DIR, user_code)
    os.makedirs(trade_dir, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(trade_dir, f"positions_{ts}.csv")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    REDIS_CLIENT.delete(f"trade:{user_code}:last_import_ts")
    count = await asyncio.to_thread(import_csv_to_redis, user_code)
    return {"status": "ok", "saved_as": dest, "positions_imported": count}


@router.put("/portfolio/position/{user_code}/{isin}")
async def portfolio_patch_position(
    user_code: str,
    isin: str,
    body: dict,
    authorization: str = Header(default=None),
):
    """Patch Jarvis-managed fields on a position."""
    _garde(authorization, user_code)

    _ALLOWED = {"threshold_high", "threshold_low", "dividend_eur", "dividend_date", "notes", "yahoo_ticker"}
    to_set = {k: str(v) for k, v in body.items() if k in _ALLOWED}
    if not to_set:
        raise HTTPException(400, f"No valid fields. Allowed: {_ALLOWED}")

    key = f"trade:{user_code}:pos:{isin}"
    if not REDIS_CLIENT.exists(key):
        raise HTTPException(404, f"Position {isin} not found for user {user_code}")

    REDIS_CLIENT.hset(key, mapping=to_set)
    return {"status": "ok", "updated": to_set}


@router.get("/portfolio/analysis/{user_code}")
async def portfolio_analysis(user_code: str, authorization: str = Header(default=None)):
    """Trigger an on-demand AI analysis of the portfolio."""
    _garde(authorization, user_code)

    summary = get_portfolio_summary_text(user_code)
    if not summary:
        raise HTTPException(404, "No portfolio data — import a CSV first")

    try:
        analysis = await call_llm_async(
            [
                {"role": "system", "content": "Tu es Jarvis, conseiller financier personnel. Analyse le portefeuille boursier de l'utilisateur de façon factuelle et constructive."},
                {"role": "user",   "content": f"Analyse ce portefeuille et donne tes observations :\n\n{summary}"},
            ],
            model=PRIMARY_MODEL, api_url=PRIMARY_API_URL, api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP, max_tokens=2000, json_response=False, no_think=True, timeout=40.0,
        )
        return {"user": USER_CODES[user_code], "analysis": analysis, "portfolio_snapshot": summary}
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")


@router.post("/portfolio/suggest-thresholds/{user_code}")
async def portfolio_suggest_thresholds(user_code: str, authorization: str = Header(default=None)):
    """Ask the LLM to suggest threshold_high / threshold_low for all positions."""
    _garde(authorization, user_code)

    suggestions = await suggest_thresholds_llm(user_code)
    return {"status": "ok", "updated": len(suggestions), "suggestions": suggestions}
