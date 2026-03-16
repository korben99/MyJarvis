"""
Jarvis Trading Module
======================
Monitors a Boursorama stock portfolio from CSV exports.

Data flow:
  1. User drops a Boursorama CSV export in TRADE_DATA_DIR (default /app/trade_data).
  2. Every hour the scheduler calls run_trade_check():
       - import_csv_to_redis()  : parses the most recent CSV (mtime-gated, skips if unchanged)
       - fetch_live_prices()    : yfinance per ISIN → current price + intraday %
       - update_prices_in_redis(): writes prices back into position hashes
       - evaluate_alerts()      : PRIMARY_MODEL decides if an alert is worth sending
  3. Fired alerts are stored as pending in Redis and injected into the next /chat response.

Redis layout:
  trade:{user_code}:index          SET  — ISINs in the portfolio
  trade:{user_code}:pos:{isin}     HASH — all fields for one position
  trade:{user_code}:last_import_ts STR  — mtime of last imported CSV (float epoch)
  trade:{user_code}:pending_alerts STR  — JSON list of queued alert dicts
  trade:price_cache:{isin}         STR  — JSON {price, intraday_var_pct}  TTL 55 min

Position hash fields:
  CSV-sourced (refreshed on each import):
    name, isin, quantity, buying_price, last_price_csv,
    total_var_pct_csv, last_movement_date, imported_at
  Price-feed (refreshed hourly by yfinance):
    last_price, intraday_var_pct, price_updated_at
  Jarvis-managed (never overwritten by import or price feed):
    yahoo_ticker, dividend_eur, dividend_date,
    threshold_high, threshold_low,
    last_alert_at, last_alert_reason, notes
"""

import asyncio
import csv
import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
import pytz
import redis as redis_lib

from config import (
    BRIEFING_TIMEZONE,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    REDIS_URL,
    no_think_suffix,
)

logger = logging.getLogger("jarvis-trading")

TRADE_DATA_DIR = os.getenv("TRADE_DATA_DIR", "/app/trade_data")
PRICE_CACHE_TTL = 55 * 60  # 55 minutes
_ALERT_MIN_INTERVAL_HOURS = 4  # Don't re-alert same position within 4 h
_ALERT_QUEUE_TTL = 86400       # Pending alerts expire after 24 h

# ── Known ISIN → Yahoo Finance ticker ─────────────────────────────────────
# Covers the user's current portfolio.  Add new lines when you buy a new stock.
# Euronext Paris: .PA  |  Euronext Brussels: .BR  |  US: no suffix
_ISIN_TICKER_MAP: dict[str, str] = {
    "FR0000120578": "SAN.PA",    # Sanofi
    "FR0010667147": "COFA.PA",   # Coface
    "FR0011871110": "PANX.PA",   # Amundi PEA Nasdaq-100 UCITS ETF
    "FR0000053225": "MMT.PA",    # M6 Metropole Television
    "FR0000120628": "CS.PA",     # AXA
    "FR0000124141": "VIE.PA",    # Veolia
    "BE0003470755": "SOLB.BR",   # Solvay (Brussels)
    "FR0000125338": "CAP.PA",    # Capgemini
    "IE0002XZSHO1": "WSRP.PA",   # iShares MSCI World Swap PEA ETF
}


# ── Redis helpers ──────────────────────────────────────────────────────────

_redis_client: redis_lib.Redis | None = None


def _get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _idx_key(user_code: str) -> str:
    return f"trade:{user_code}:index"


def _pos_key(user_code: str, isin: str) -> str:
    return f"trade:{user_code}:pos:{isin}"


def _import_ts_key(user_code: str) -> str:
    return f"trade:{user_code}:last_import_ts"


def _price_cache_key(isin: str) -> str:
    return f"trade:price_cache:{isin}"


def _alert_queue_key(user_code: str) -> str:
    return f"trade:{user_code}:pending_alerts"


# ── CSV parsing ────────────────────────────────────────────────────────────

def _parse_french_float(val: str) -> float:
    """Convert French-format number to float.  '7 944,56' → 7944.56"""
    return float(val.strip().strip('"').replace("\u202f", "").replace("\xa0", "").replace(" ", "").replace(",", "."))


def _find_latest_csv() -> str | None:
    """Return path of the most recently modified CSV in TRADE_DATA_DIR."""
    files = glob.glob(os.path.join(TRADE_DATA_DIR, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_boursorama_csv(path: str) -> list[dict]:
    """
    Parse a Boursorama positions export CSV.
    Columns: name;isin;quantity;buyingPrice;lastPrice;intradayVariation;
             amount;amountVariation;variation;lastMovementDate;compensation
    Returns list of normalised position dicts.
    """
    positions = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row.get("name", "").strip().strip('"')
            isin = row.get("isin", "").strip().strip('"')
            if not isin or not name:
                continue
            try:
                positions.append({
                    "name": name,
                    "isin": isin,
                    "quantity":            _parse_french_float(row.get("quantity", "0")),
                    "buying_price":        _parse_french_float(row.get("buyingPrice", "0")),
                    "last_price_csv":      _parse_french_float(row.get("lastPrice", "0")),
                    "total_var_pct_csv":   _parse_french_float(row.get("variation", "0")),
                    "last_movement_date":  row.get("lastMovementDate", "").strip(),
                })
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping CSV row %s: %s", isin, exc)
    return positions


def import_csv_to_redis(user_code: str) -> int:
    """
    Find the most recent CSV in TRADE_DATA_DIR and upsert positions into Redis.
    Skips silently if the file hasn't changed since the last import (mtime check).
    Preserves all Jarvis-managed fields that already exist in Redis.
    Returns number of positions imported (0 = nothing new).
    """
    path = _find_latest_csv()
    if not path:
        return 0

    r = _get_redis()
    last_ts = float(r.get(_import_ts_key(user_code)) or 0)
    if os.path.getmtime(path) <= last_ts:
        return 0

    positions = parse_boursorama_csv(path)
    if not positions:
        return 0

    idx_key = _idx_key(user_code)
    r.delete(idx_key)

    # Jarvis-managed fields that must never be overwritten by a CSV import
    _JARVIS_FIELDS = (
        "yahoo_ticker", "dividend_eur", "dividend_date",
        "threshold_high", "threshold_low",
        "last_alert_at", "last_alert_reason", "notes",
        "last_price", "intraday_var_pct", "price_updated_at",
    )

    for pos in positions:
        isin = pos["isin"]
        key = _pos_key(user_code, isin)

        # Preserve existing Jarvis-managed values
        existing = r.hgetall(key)
        jarvis_values = {k: existing[k] for k in _JARVIS_FIELDS if k in existing}

        r.hset(key, mapping={
            "name":                pos["name"],
            "isin":                isin,
            "quantity":            pos["quantity"],
            "buying_price":        pos["buying_price"],
            "last_price_csv":      pos["last_price_csv"],
            "total_var_pct_csv":   pos["total_var_pct_csv"],
            "last_movement_date":  pos["last_movement_date"],
            "imported_at":         datetime.now(timezone.utc).isoformat(),
        })
        if jarvis_values:
            r.hset(key, mapping=jarvis_values)

        r.sadd(idx_key, isin)

    r.set(_import_ts_key(user_code), os.path.getmtime(path))
    logger.info("Imported %d positions for %s from %s", len(positions), user_code, os.path.basename(path))
    return len(positions)


# ── Yahoo Finance price fetch ──────────────────────────────────────────────

def _resolve_ticker(isin: str, r: redis_lib.Redis, pos_key: str) -> str | None:
    """
    Resolve ISIN → Yahoo Finance ticker.
    Priority: Redis cache → static map → yfinance search (and cache result).
    """
    cached = r.hget(pos_key, "yahoo_ticker")
    if cached:
        return cached

    ticker = _ISIN_TICKER_MAP.get(isin)
    if not ticker:
        try:
            import yfinance as yf
            results = yf.Search(isin, max_results=1)
            quotes = results.quotes
            if quotes:
                ticker = quotes[0].get("symbol")
        except Exception as exc:
            logger.warning("yfinance ISIN lookup failed for %s: %s", isin, exc)

    if ticker:
        r.hset(pos_key, "yahoo_ticker", ticker)
        logger.debug("Resolved %s → %s", isin, ticker)
    return ticker


def fetch_live_prices(user_code: str) -> dict[str, dict]:
    """
    Synchronous: fetch current price + intraday % for every position via yfinance.
    Uses a per-ISIN Redis cache (55 min TTL) to avoid hammering Yahoo.
    Returns { isin: {price, intraday_var_pct} }.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot fetch prices")
        return {}

    r = _get_redis()
    isins = list(r.smembers(_idx_key(user_code)))
    if not isins:
        return {}

    results: dict[str, dict] = {}
    to_fetch: list[tuple[str, str]] = []  # (isin, ticker)

    for isin in isins:
        cached = r.get(_price_cache_key(isin))
        if cached:
            try:
                results[isin] = json.loads(cached)
                continue
            except json.JSONDecodeError:
                pass
        ticker = _resolve_ticker(isin, r, _pos_key(user_code, isin))
        if ticker:
            to_fetch.append((isin, ticker))

    for isin, ticker in to_fetch:
        try:
            obj = yf.Ticker(ticker)
            fi = obj.fast_info
            price = fi.last_price or fi.previous_close or 0.0
            change_pct = round(float(getattr(fi, "regular_market_change_percent", 0.0) or 0.0), 2)
            entry = {"price": round(float(price), 4), "intraday_var_pct": change_pct}
            results[isin] = entry
            r.setex(_price_cache_key(isin), PRICE_CACHE_TTL, json.dumps(entry))
            logger.debug("Price %s: %.4f (%+.2f%%)", ticker, price, change_pct)
        except Exception as exc:
            logger.warning("Price fetch failed for %s (%s): %s", ticker, isin, exc)

    return results


def update_prices_in_redis(user_code: str, prices: dict[str, dict]) -> None:
    """Write freshly fetched prices into each position hash."""
    r = _get_redis()
    now = datetime.now(timezone.utc).isoformat()
    for isin, data in prices.items():
        key = _pos_key(user_code, isin)
        if r.exists(key):
            r.hset(key, mapping={
                "last_price":       data["price"],
                "intraday_var_pct": data["intraday_var_pct"],
                "price_updated_at": now,
            })


# ── Portfolio read ─────────────────────────────────────────────────────────

def get_portfolio(user_code: str) -> list[dict]:
    """Return all positions enriched with live P&L calculations."""
    r = _get_redis()
    isins = sorted(r.smembers(_idx_key(user_code)))
    portfolio = []
    for isin in isins:
        raw = r.hgetall(_pos_key(user_code, isin))
        if not raw:
            continue
        try:
            qty  = float(raw.get("quantity", 0))
            buy  = float(raw.get("buying_price", 0))
            live = float(raw.get("last_price") or raw.get("last_price_csv") or 0)
            cost  = round(qty * buy,  2)
            value = round(qty * live, 2)
            portfolio.append({
                **raw,
                "quantity":            qty,
                "buying_price":        buy,
                "last_price":          live,
                "cost_basis_eur":      cost,
                "current_value_eur":   value,
                "unrealized_pnl_eur":  round(value - cost, 2),
                "unrealized_pnl_pct":  round((live - buy) / buy * 100, 2) if buy else 0.0,
            })
        except (ValueError, TypeError):
            portfolio.append(raw)
    return portfolio


def get_portfolio_summary_text(user_code: str) -> str:
    """
    Compact text block for LLM prompt injection.
    Returns empty string if no positions found.
    """
    positions = get_portfolio(user_code)
    if not positions:
        return ""

    lines = ["=== PORTEFEUILLE BOURSIER ==="]
    total_cost = total_value = 0.0

    for p in positions:
        try:
            cost  = float(p.get("cost_basis_eur", 0))
            value = float(p.get("current_value_eur", 0))
            pnl_pct = float(p.get("unrealized_pnl_pct", 0))
            intra   = float(p.get("intraday_var_pct", 0))
            total_cost  += cost
            total_value += value

            extras = ""
            if p.get("threshold_high") or p.get("threshold_low"):
                th = p.get("threshold_high", "—")
                tl = p.get("threshold_low", "—")
                extras += f" [seuils >{th}€ / <{tl}€]"
            if p.get("dividend_eur") and p.get("dividend_date"):
                extras += f" [div {p['dividend_eur']}€ le {p['dividend_date']}]"
            if p.get("notes"):
                extras += f" [{p['notes']}]"

            updated = (p.get("price_updated_at") or "")[:16]
            lines.append(
                f"• {p['name']} — qté:{p['quantity']:.0f} | achat:{p['buying_price']}€ | "
                f"cours:{p['last_price']}€ ({intra:+.2f}% J) | PV:{pnl_pct:+.2f}%"
                f"{extras}"
                + (f" [màj {updated}]" if updated else "")
            )
        except Exception:
            lines.append(f"• {p.get('name', p.get('isin', '?'))}")

    if total_cost > 0:
        total_pnl     = round(total_value - total_cost, 2)
        total_pnl_pct = round((total_value - total_cost) / total_cost * 100, 2)
        lines.append(
            f"\nTotal: investi {total_cost:.0f}€ | valeur {total_value:.0f}€ | "
            f"PV {total_pnl:+.0f}€ ({total_pnl_pct:+.2f}%)"
        )

    return "\n".join(lines)


# ── Alert evaluation ───────────────────────────────────────────────────────

_ALERT_SYSTEM = """\
Tu es Jarvis, l'assistant personnel de l'utilisateur. Tu surveilles son portefeuille boursier.
Analyse les positions et décide si une alerte mérite d'être envoyée maintenant.
Sois conservateur — n'alerte pas pour du bruit de marché normal.
N'alerte QUE si au moins une des conditions suivantes est vraie :
  - Un cours a franchi threshold_high ou threshold_low défini par l'utilisateur
  - Variation intraday > 3 % sur une position individuelle
  - Perte journalière totale du portefeuille > 2 %
  - Dividende prévu dans moins de 5 jours calendaires"""

_ALERT_USER = """\
Positions actuelles (intraday calculé par rapport à la veille) :
{portfolio}

Réponds en JSON uniquement : {{"alert": true/false, "message": "..."}}
Si alert=false, message doit être une chaîne vide."""


async def evaluate_alerts(user_code: str) -> tuple[bool, str]:
    """
    Ask PRIMARY_MODEL whether an alert should be fired.
    Rate-limits per position: same position won't alert twice within 4 h.
    Returns (should_alert, message).
    """
    r = _get_redis()
    positions = get_portfolio(user_code)
    if not positions:
        return False, ""

    now = datetime.now(timezone.utc)
    eligible = []
    for p in positions:
        last_alert = p.get("last_alert_at", "")
        if last_alert:
            try:
                if (now - datetime.fromisoformat(last_alert)).total_seconds() < _ALERT_MIN_INTERVAL_HOURS * 3600:
                    continue
            except ValueError:
                pass
        eligible.append(p)

    if not eligible:
        return False, ""

    lines = []
    for p in eligible:
        try:
            intra   = float(p.get("intraday_var_pct", 0))
            pnl_pct = float(p.get("unrealized_pnl_pct", 0))
            live    = float(p.get("last_price", 0))
            line    = f"{p['name']}: cours={live}€ (intraday={intra:+.2f}%, PV={pnl_pct:+.2f}%)"
            if p.get("threshold_high"):
                line += f" seuil_haut={p['threshold_high']}€"
            if p.get("threshold_low"):
                line += f" seuil_bas={p['threshold_low']}€"
            if p.get("dividend_date"):
                line += f" dividende_prévu={p['dividend_date']}"
            lines.append(line)
        except Exception:
            pass

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [
                        {"role": "system", "content": _ALERT_SYSTEM + no_think_suffix(PRIMARY_MODEL)},
                        {"role": "user",   "content": _ALERT_USER.format(portfolio="\n".join(lines))},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 200,
                    "temperature": 0,
                },
            )
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        should_alert = bool(result.get("alert", False))
        message      = result.get("message", "")

        if should_alert and message:
            now_iso = now.isoformat()
            for p in eligible:
                if p.get("name", "").upper() in message.upper():
                    r.hset(_pos_key(user_code, p["isin"]), mapping={
                        "last_alert_at":     now_iso,
                        "last_alert_reason": message[:200],
                    })

        return should_alert, message

    except Exception as exc:
        logger.error("Alert evaluation failed for %s: %s", user_code, exc)
        return False, ""


# ── Pending alert queue ─────────────────────────────────────────────────────

def push_pending_alert(user_code: str, message: str) -> None:
    r = _get_redis()
    key = _alert_queue_key(user_code)
    existing = json.loads(r.get(key) or "[]")
    existing.append({"message": message, "at": datetime.now(timezone.utc).isoformat()})
    r.setex(key, _ALERT_QUEUE_TTL, json.dumps(existing, ensure_ascii=False))


def pop_pending_alerts(user_code: str) -> list[dict]:
    """Return and clear all queued alerts for a user."""
    r = _get_redis()
    key = _alert_queue_key(user_code)
    raw = r.get(key)
    if not raw:
        return []
    r.delete(key)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


# ── Market hours helper ────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    """True if current Paris time is Mon–Fri 09:00–17:35."""
    tz  = pytz.timezone(BRIEFING_TIMEZONE or "Europe/Paris")
    now = datetime.now(tz)
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    open_  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    close_ = now.replace(hour=17, minute=35, second=0, microsecond=0)
    return open_ <= now <= close_


# ── Threshold auto-population ──────────────────────────────────────────────

def auto_set_thresholds(user_code: str) -> int:
    """
    Auto-populate threshold_high / threshold_low for positions that have neither.
    Uses yfinance 52-week high/low as defaults:
      threshold_high = 52-week high  (alert near historical resistance)
      threshold_low  = max(buying_price × 0.90, 52-week low)  (stop-loss vs historical support)
    Only positions missing BOTH thresholds are touched.
    Returns the number of positions updated.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot auto-set thresholds")
        return 0

    r = _get_redis()
    isins = list(r.smembers(_idx_key(user_code)))
    updated = 0

    for isin in isins:
        key = _pos_key(user_code, isin)
        pos = r.hgetall(key)

        # Skip if at least one threshold is already set (user or LLM configured)
        if pos.get("threshold_high") or pos.get("threshold_low"):
            continue

        ticker_sym = _resolve_ticker(isin, r, key)
        if not ticker_sym:
            continue

        try:
            info = yf.Ticker(ticker_sym).info
            high_52w = info.get("fiftyTwoWeekHigh")
            low_52w  = info.get("fiftyTwoWeekLow")

            if not high_52w or not low_52w:
                logger.warning("No 52-week data for %s (%s), skipping", ticker_sym, isin)
                continue

            buying_price = float(pos.get("buying_price") or 0)
            stop_loss    = round(buying_price * 0.90, 2) if buying_price else 0.0
            tl           = round(max(stop_loss, low_52w), 2)
            th           = round(high_52w, 2)

            r.hset(key, mapping={"threshold_high": str(th), "threshold_low": str(tl)})
            logger.info(
                "Auto-thresholds %s (%s): high=%.2f low=%.2f (52w: %.2f/%.2f)",
                pos.get("name", isin), isin, th, tl, high_52w, low_52w,
            )
            updated += 1

        except Exception as exc:
            logger.warning("Auto-threshold failed for %s (%s): %s", ticker_sym, isin, exc)

    return updated


async def suggest_thresholds_llm(user_code: str) -> dict:
    """
    Ask PRIMARY_MODEL to suggest threshold_high / threshold_low for every position.
    Writes suggestions directly into Redis, overwriting any existing values.
    Returns {isin: {threshold_high, threshold_low, rationale}}.
    """
    positions = get_portfolio(user_code)
    if not positions:
        return {}

    lines = []
    for p in positions:
        try:
            live = float(p.get("last_price") or 0)
            buy  = float(p.get("buying_price") or 0)
            pnl  = float(p.get("unrealized_pnl_pct") or 0)
            line = (
                f"- {p['name']} (ISIN: {p['isin']}): "
                f"achat={buy}€  cours={live}€  PV={pnl:+.1f}%"
            )
            if p.get("threshold_high"):
                line += f"  seuil_haut_actuel={p['threshold_high']}€"
            if p.get("threshold_low"):
                line += f"  seuil_bas_actuel={p['threshold_low']}€"
            lines.append(line)
        except Exception:
            pass

    prompt = (
        "Tu es un assistant de gestion de portefeuille boursier.\n"
        "Pour chaque position ci-dessous, suggère un seuil d'alerte haut (threshold_high) "
        "et bas (threshold_low) en euros.\n\n"
        "Critères :\n"
        "- threshold_high : niveau de prix justifiant une attention (prise de bénéfice possible, "
        "résistance technique, objectif de cours).\n"
        "- threshold_low : niveau de stop-loss ou d'alerte en cas de baisse significative "
        "(support technique, perte maximale acceptable).\n"
        "- Adapte les seuils à la volatilité typique de chaque titre et au contexte P&L.\n"
        "- Sois conservateur : évite les seuils trop proches du cours actuel.\n\n"
        f"Positions :\n" + "\n".join(lines) + "\n\n"
        "Réponds en JSON uniquement, format exact :\n"
        '{"positions": [{"isin": "...", "threshold_high": 0.0, "threshold_low": 0.0, "rationale": "..."}]}'
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{PRIMARY_API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {PRIMARY_API_KEY}"},
                json={
                    "model": PRIMARY_MODEL,
                    "messages": [{"role": "user", "content": prompt + no_think_suffix(PRIMARY_MODEL)}],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 1000,
                    "temperature": 0.2,
                },
            )
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        logger.error("LLM threshold suggestion failed for %s: %s", user_code, exc)
        return {}

    r = _get_redis()
    suggestions: dict = {}

    for item in result.get("positions", []):
        isin = item.get("isin", "").strip()
        th   = item.get("threshold_high")
        tl   = item.get("threshold_low")
        if not isin or th is None or tl is None:
            continue
        key = _pos_key(user_code, isin)
        if not r.exists(key):
            continue
        th = round(float(th), 2)
        tl = round(float(tl), 2)
        r.hset(key, mapping={"threshold_high": str(th), "threshold_low": str(tl)})
        suggestions[isin] = {
            "threshold_high": th,
            "threshold_low":  tl,
            "rationale":      item.get("rationale", ""),
        }
        logger.info("LLM threshold set for %s: high=%.2f low=%.2f", isin, th, tl)

    return suggestions


# ── Scheduled job ──────────────────────────────────────────────────────────

async def run_trade_check(user_codes: list[str]) -> None:
    """
    Hourly APScheduler job.
    - Always checks for a new CSV (cheap mtime check).
    - Only fetches prices + evaluates alerts during market hours.
    """
    for user_code in user_codes:
        try:
            imported = import_csv_to_redis(user_code)
            if imported:
                logger.info("Trade: imported %d positions for %s", imported, user_code)
                # Auto-populate thresholds for any new positions that have none
                filled = await asyncio.to_thread(auto_set_thresholds, user_code)
                if filled:
                    logger.info("Trade: auto-set thresholds for %d positions (%s)", filled, user_code)

            if not _get_redis().smembers(_idx_key(user_code)):
                continue  # No positions yet — nothing to do

            if _is_market_hours():
                prices = await asyncio.to_thread(fetch_live_prices, user_code)
                if prices:
                    update_prices_in_redis(user_code, prices)
                    logger.info("Trade: updated %d prices for %s", len(prices), user_code)
                    should_alert, msg = await evaluate_alerts(user_code)
                    if should_alert and msg:
                        push_pending_alert(user_code, msg)
                        logger.info("Trade alert queued for %s: %s", user_code, msg[:80])
            else:
                logger.debug("Trade: market closed, skipping price fetch for %s", user_code)

        except Exception as exc:
            logger.error("Trade check error for %s: %s", user_code, exc)
