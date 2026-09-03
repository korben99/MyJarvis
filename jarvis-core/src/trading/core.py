"""
Jarvis Trading Module
======================
Monitors a Boursorama stock portfolio from CSV exports.

Data flow:
  1. User drops a Boursorama CSV export in TRADE_DATA_DIR/<USER_CODE>/ — un dossier par
     utilisateur, l'export ne portant pas le nom de son détenteur (cf. _find_latest_csv).
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
import os
from datetime import datetime, timezone

import pytz
import redis as redis_lib
import yfinance as yf
from config import (
    BRIEFING_TIMEZONE,
    DEFAULT_TEMP,
    MAX_TOKENS_MEDIUM,
    MAX_TOKENS_THINK_MEDIUM,
    THINKING_BUDGET_MEDIUM,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
    llm_timeout,
)
from helpers import call_llm_async, extract_llm_json, get_logger, get_redis
from .keys import alert_queue_key, idx_key, import_ts_key, pos_key, price_cache_key

logger = get_logger("jarvis-trading")

TRADE_DATA_DIR = os.getenv("TRADE_DATA_DIR", "/opt/jarvis/RAGData/Trade")
PRICE_CACHE_TTL = 55 * 60  # 55 minutes
_ALERT_MIN_INTERVAL_HOURS = 8  # Don't re-alert same position within 8 h
_ALERT_QUEUE_TTL = 86400  # Pending alerts expire after 24 h


# ── CSV parsing ────────────────────────────────────────────────────────────


def _parse_french_float(val: str) -> float:
    """Convert French-format number to float.  '7 944,56' → 7944.56"""
    return float(
        val.strip()
        .strip('"')
        .replace("\u202f", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", ".")
    )


def _find_latest_csv(user_code: str) -> str | None:
    """CSV le plus récent DU dossier de cet utilisateur : TRADE_DATA_DIR/<USER_CODE>/*.csv.

    Un dossier par utilisateur, et aucun repli sur la racine. Un export Boursorama ne porte
    pas le nom de son propriétaire, donc un répertoire commun ne permet pas de savoir à qui
    appartient un fichier : tout utilisateur qui importe reçoit le dernier CSV déposé, quel
    qu'en soit l'auteur. Le dossier est la seule marque de propriété disponible.

    Sans dossier, on rend None — l'import est un no-op. C'est le bon défaut : mieux vaut
    n'importer aucune position qu'importer celles de quelqu'un d'autre.
    """
    files = glob.glob(os.path.join(TRADE_DATA_DIR, user_code, "*.csv"))
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
                positions.append(
                    {
                        "name": name,
                        "isin": isin,
                        "quantity": _parse_french_float(row.get("quantity", "0")),
                        "buying_price": _parse_french_float(
                            row.get("buyingPrice", "0")
                        ),
                        "last_price_csv": _parse_french_float(
                            row.get("lastPrice", "0")
                        ),
                        "total_var_pct_csv": _parse_french_float(
                            row.get("variation", "0")
                        ),
                        "last_movement_date": row.get("lastMovementDate", "").strip(),
                    }
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping CSV row %s: %s", isin, exc)
    return positions


def import_csv_to_redis(user_code: str) -> int:
    """
    Find the most recent CSV in TRADE_DATA_DIR/<user_code>/ and upsert positions into Redis.
    Skips silently if the file hasn't changed since the last import (mtime check).
    Preserves all Jarvis-managed fields that already exist in Redis.
    Returns number of positions imported (0 = nothing new, or no folder for this user).
    """
    path = _find_latest_csv(user_code)
    if not path:
        return 0

    r = get_redis()
    last_ts = float(r.get(import_ts_key(user_code)) or 0)
    if os.path.getmtime(path) <= last_ts:
        return 0

    positions = parse_boursorama_csv(path)
    if not positions:
        return 0

    idx_key_cur = idx_key(user_code)
    r.delete(idx_key_cur)

    # Jarvis-managed fields that must never be overwritten by a CSV import
    _JARVIS_FIELDS = (
        "yahoo_ticker",
        "dividend_eur",
        "dividend_date",
        "threshold_high",
        "threshold_low",
        "last_alert_at",
        "last_alert_reason",
        "notes",
        "last_price",
        "intraday_var_pct",
        "price_updated_at",
    )

    for pos in positions:
        isin = pos["isin"]
        key = pos_key(user_code, isin)

        # Preserve existing Jarvis-managed values
        existing = r.hgetall(key)
        jarvis_values = {k: existing[k] for k in _JARVIS_FIELDS if k in existing}

        r.hset(
            key,
            mapping={
                "name": pos["name"],
                "isin": isin,
                "quantity": pos["quantity"],
                "buying_price": pos["buying_price"],
                "last_price_csv": pos["last_price_csv"],
                "total_var_pct_csv": pos["total_var_pct_csv"],
                "last_movement_date": pos["last_movement_date"],
                "imported_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if jarvis_values:
            r.hset(key, mapping=jarvis_values)

        r.sadd(idx_key_cur, isin)

    r.set(import_ts_key(user_code), os.path.getmtime(path))
    logger.info(
        "Imported %d positions for %s from %s",
        len(positions),
        user_code,
        os.path.basename(path),
    )
    return len(positions)


# ── Yahoo Finance price fetch ──────────────────────────────────────────────


def _resolve_ticker(
    isin: str, r: redis_lib.Redis, pos_key: str, name: str = ""
) -> str | None:
    """
    Resolve ISIN → Yahoo Finance ticker.
    Priority:
      1. Redis cache
      2. yfinance Search(isin)  — premier candidat qui rend une série
      3. yfinance Search(name)  — idem
    Result is cached in Redis. Cache is cleared by fetch_live_prices on failure.

    **Pas de repli génératif.** Résoudre un ISIN est une consultation d'annuaire : soit une
    source trouve un titre réel, soit la réponse est « je ne sais pas ». Un modèle de langage
    interrogé sur un symbole en produit toujours un de forme plausible, qu'il existe ou non,
    et un contrôle de forme ne distingue pas les deux. Écrit dans `yahoo_ticker`, un symbole
    inventé est suivi comme une vraie ligne : cours, plus-value, seuils d'alerte, résumé de
    portefeuille. Le cas bénin est celui qui n'existe pas — Yahoo répond 404 et ça se voit.
    Le cas grave est celui qui existe et désigne un AUTRE titre : tout fonctionne en
    apparence et la ligne rapporte les chiffres d'une autre société, sans rien dans les
    journaux. Aucun contrôle automatique ne rattrape ce second cas : l'identité par ISIN
    rendue par yfinance n'est pas fiable (cf. le commentaire de `fetch_live_prices`).
    Non résolu → `UNKNOWN` et correction manuelle via PUT /portfolio/position.

    Les candidats sont VÉRIFIÉS avant d'être retenus : une recherche peut rendre une ligne
    d'un autre marché ou un instrument sans historique exploitable. La vérification porte
    sur l'existence (« Yahoo rend-il une série ? »), jamais sur la plausibilité d'un cours —
    un contrôle de plausibilité se déclenche sur une vraie variation, cf. `fetch_live_prices`.
    """
    cached = r.hget(pos_key, "yahoo_ticker")
    if cached:
        return None if cached == "UNKNOWN" else cached

    def _candidats(requete: str, via: str) -> list[tuple[str, str]]:
        try:
            quotes = yf.Search(requete, max_results=5).quotes
        except Exception as exc:
            logger.warning("yfinance %s lookup failed for '%s': %s", via, requete, exc)
            return []
        return [(q["symbol"], via) for q in quotes if q.get("symbol")]

    candidats = _candidats(isin, "ISIN")
    if name:
        candidats += _candidats(name, "name")

    vus: set[str] = set()
    ticker = None
    for symbole, via in candidats:
        if symbole in vus:
            continue
        vus.add(symbole)
        if _ticker_repond(symbole):
            ticker = symbole
            logger.info("Resolved ticker %s → %s (via %s search)", isin, ticker, via)
            break
        logger.debug("Candidat %s écarté pour %s : aucune série", symbole, isin)

    if ticker:
        r.hset(pos_key, "yahoo_ticker", ticker)
    else:
        logger.error(
            "TICKER UNRESOLVABLE: ISIN=%s name='%s' (%d candidat(s) examiné(s), aucun ne "
            "rend de série) — position will be skipped until manually set via "
            "PUT /portfolio/position",
            isin, name, len(vus),
        )
        # Cache the failure sentinel so we don't retry every hourly run
        r.hset(pos_key, "yahoo_ticker", "UNKNOWN")
    return ticker


def _ticker_repond(symbole: str) -> bool:
    """Vrai si Yahoo rend une série pour ce symbole. Appelé UNIQUEMENT à la résolution.

    Jamais sur un ticker déjà en cache : re-valider un ticker établi, c'est la boucle qui a
    fait retirer le garde-fou de `fetch_live_prices` — invalidation, re-résolution rendant
    le même symbole, re-déclenchement. Ici le résultat sert à CHOISIR entre candidats, une
    seule fois, et jamais à défaire un choix déjà inscrit.
    """
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return not yf.Ticker(symbole).history(period="5d", interval="1d").empty
    except Exception as exc:
        logger.debug("Vérification de %s impossible (%s)", symbole, type(exc).__name__)
        return False


def fetch_live_prices(user_code: str) -> dict[str, dict]:
    """
    Synchronous: fetch current price + intraday % for every position via yfinance.
    Uses a per-ISIN Redis cache (55 min TTL) to avoid hammering Yahoo.
    Returns { isin: {price, intraday_var_pct} }.
    """
    r = get_redis()
    isins = list(r.smembers(idx_key(user_code)))
    if not isins:
        return {}

    results: dict[str, dict] = {}
    to_fetch: list[tuple[str, str]] = []  # (isin, ticker)

    for isin in isins:
        cached = r.get(price_cache_key(isin))
        if cached:
            try:
                results[isin] = json.loads(cached)
                continue
            except json.JSONDecodeError:
                pass
        pos_key_u = pos_key(user_code, isin)
        name = r.hget(pos_key_u, "name") or ""
        ticker = _resolve_ticker(isin, r, pos_key_u, name)
        if ticker:
            to_fetch.append((isin, ticker))

    for isin, ticker in to_fetch:
        try:
            obj = yf.Ticker(ticker)
            fi = obj.fast_info
            price = fi.last_price or fi.previous_close or 0.0
            change_pct = round(
                float(getattr(fi, "regular_market_change_percent", 0.0) or 0.0), 2
            )

            # Aucune validation du prix : on fait confiance à Yahoo.
            #
            # Il y avait ici un garde-fou qui comparait le cours live au dernier prix du CSV
            # Boursorama et invalidait le ticker au-delà de 30 % d'écart. Retiré :
            # la référence CSV est un instantané figé dont l'âge n'est pas borné, donc toute
            # tendance réelle finissait par la franchir. Valneva (FR0004056851 → VLA.PA) est
            # passée de 2,17 le 5 août à 3,08 le 17 : +40 %, hausse authentique, prise pour une
            # erreur de résolution. Pire, le garde-fou ne pouvait pas se rétablir — il vidait le
            # ticker, la résolution suivante rendait le MÊME ticker (la preuve qu'il était bon),
            # et le contrôle re-déclenchait : 10 fois le 17/08. Le cours est resté gelé à 2,455
            # pendant 12 jours, sous-évaluant la ligne de 186 € et neutralisant ses seuils.
            #
            # L'identité par ISIN ne remplace pas ce contrôle : yfinance rend "-" pour VLA.PA et
            # 2CRSI.PA, et FR001400UEY9 au lieu de FR0000120578 pour Sanofi. Une valeur qui
            # dérape se voit à l'œil sur le portefeuille.

            entry = {"price": round(float(price), 4), "intraday_var_pct": change_pct}
            results[isin] = entry
            r.setex(price_cache_key(isin), PRICE_CACHE_TTL, json.dumps(entry))
            logger.debug("Price %s: %.4f (%+.2f%%)", ticker, price, change_pct)
        except Exception as exc:
            logger.warning("Price fetch failed for %s (%s): %s", ticker, isin, exc)
            # Clear cached ticker so next run retries resolution
            r.hdel(pos_key(user_code, isin), "yahoo_ticker")

    return results


def update_prices_in_redis(user_code: str, prices: dict[str, dict]) -> None:
    """Write freshly fetched prices into each position hash."""
    r = get_redis()
    now = datetime.now(timezone.utc).isoformat()
    for isin, data in prices.items():
        key = pos_key(user_code, isin)
        if r.exists(key):
            r.hset(
                key,
                mapping={
                    "last_price": data["price"],
                    "intraday_var_pct": data["intraday_var_pct"],
                    "price_updated_at": now,
                },
            )


# ── Portfolio read ─────────────────────────────────────────────────────────


def get_portfolio(user_code: str) -> list[dict]:
    """Return all positions enriched with live P&L calculations."""
    r = get_redis()
    isins = sorted(r.smembers(idx_key(user_code)))
    portfolio = []
    for isin in isins:
        raw = r.hgetall(pos_key(user_code, isin))
        if not raw:
            continue
        try:
            qty = float(raw.get("quantity", 0))
            buy = float(raw.get("buying_price", 0))
            live = float(raw.get("last_price") or raw.get("last_price_csv") or 0)
            cost = round(qty * buy, 2)
            value = round(qty * live, 2)
            portfolio.append(
                {
                    **raw,
                    "quantity": qty,
                    "buying_price": buy,
                    "last_price": live,
                    "cost_basis_eur": cost,
                    "current_value_eur": value,
                    "unrealized_pnl_eur": round(value - cost, 2),
                    "unrealized_pnl_pct": round((live - buy) / buy * 100, 2)
                    if buy
                    else 0.0,
                }
            )
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

    lines = ["<portfolio>"]
    total_cost = total_value = 0.0

    for p in positions:
        try:
            cost = float(p.get("cost_basis_eur", 0))
            value = float(p.get("current_value_eur", 0))
            pnl_pct = float(p.get("unrealized_pnl_pct", 0))
            intra = float(p.get("intraday_var_pct", 0))
            total_cost += cost
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

            lines.append(
                f"• {p['name']} : {p['quantity']:.0f} titres, "
                f"achat {p['buying_price']}€, cours {p['last_price']}€ ({intra:+.2f}% J), "
                f"PV {pnl_pct:+.2f}%"
                f"{extras}"
            )
        except Exception:
            lines.append(f"• {p.get('name', p.get('isin', '?'))}")

    if total_cost > 0:
        total_pnl = round(total_value - total_cost, 2)
        total_pnl_pct = round((total_value - total_cost) / total_cost * 100, 2)
        lines.append(
            f"\nTotal: investi {total_cost:.0f}€ | valeur {total_value:.0f}€ | "
            f"PV {total_pnl:+.0f}€ ({total_pnl_pct:+.2f}%)"
        )

    lines.append("</portfolio>")
    return "\n".join(lines)


# ── Alert evaluation ───────────────────────────────────────────────────────

_ALERT_SYSTEM = """\
Tu es Jarvis, l'assistant personnel de l'utilisateur. Tu surveilles son portefeuille boursier.
Analyse les positions et décide si une alerte mérite d'être envoyée maintenant.
Sois conservateur — n'alerte pas pour du bruit de marché normal.
N'alerte QUE si au moins une des conditions suivantes est vraie :
  - Un cours a franchi threshold_high ou threshold_low défini par l'utilisateur
  - Variation intraday ANORMALE : |J| dépasse le seuil_anormal indiqué sur la ligne.
    Ce seuil vaut deux fois la variation quotidienne habituelle de CETTE valeur — un
    mouvement plus petit est du bruit propre à la valeur, même s'il paraît grand dans
    l'absolu. Quand la ligne ne porte pas de seuil_anormal, applique 3 %.
  - Perte journalière totale du portefeuille > 2 %
  - Dividende ou résultats prévus dans moins de 5 jours calendaires"""

_ALERT_USER = """\
Variation journalière totale du portefeuille : {daily_pnl_pct:+.2f}%

Positions (J = variation intraday depuis la veille · PV_total = plus-value depuis l'achat) :
{portfolio}

Réponds UNIQUEMENT avec ce JSON, sans texte autour :
{{"alert": true, "message": "raison courte"}}  ← si alerte
{{"alert": false, "message": ""}}               ← si pas d'alerte
"""


def _stats_cache(ticker: str | None) -> dict:
    """Statistiques de tendance en cache, sans réseau. Dict vide si rien n'est en cache.

    Import tardif et isolé : `market` tire yfinance et lit `get_portfolio` d'ici même —
    l'importer paresseusement évite tout cycle et garde la boucle horaire insensible à une
    panne du module de marché."""
    if not ticker:
        return {}
    try:
        from .market import stats_ligne

        return stats_ligne(ticker, telecharger=False)
    except Exception as exc:
        logger.debug("trading: stats de %s indisponibles (%s)", ticker, type(exc).__name__)
        return {}


def _dates_cache(ticker: str | None) -> dict:
    """Échéances (détachement, résultats) en cache, sans réseau. Dict vide sinon."""
    if not ticker:
        return {}
    try:
        from .market import dates_a_venir

        return dates_a_venir(ticker, telecharger=False)
    except Exception as exc:
        logger.debug("trading: échéances de %s indisponibles (%s)", ticker, type(exc).__name__)
        return {}


async def evaluate_alerts(user_code: str) -> tuple[bool, str]:
    """
    Ask PRIMARY_MODEL whether an alert should be fired.
    Rate-limits per position: same position won't alert twice within 8 h.
    Returns (should_alert, message).
    """
    r = get_redis()
    positions = get_portfolio(user_code)
    if not positions:
        return False, ""

    now = datetime.now(timezone.utc)
    eligible = []
    for p in positions:
        last_alert = p.get("last_alert_at", "")
        if last_alert:
            try:
                if (
                    now - datetime.fromisoformat(last_alert)
                ).total_seconds() < _ALERT_MIN_INTERVAL_HOURS * 3600:
                    continue
            except ValueError:
                pass
        eligible.append(p)

    if not eligible:
        return False, ""

    # Compute total portfolio daily P&L explicitly so the LLM has a clear figure
    total_value = sum(float(p.get("current_value_eur", 0) or 0) for p in eligible)
    daily_pnl_eur = sum(
        float(p.get("current_value_eur", 0) or 0)
        * float(p.get("intraday_var_pct", 0) or 0)
        / 100
        for p in eligible
    )
    daily_pnl_pct = (
        round(daily_pnl_eur / total_value * 100, 2) if total_value > 0 else 0.0
    )

    lines = []
    for p in eligible:
        try:
            intra = float(p.get("intraday_var_pct", 0) or 0)
            pnl_pct = float(p.get("unrealized_pnl_pct", 0) or 0)
            live = float(p.get("last_price", 0) or 0)
            # Label clearly: J = intraday (daily), PV_total = since purchase
            line = f"{p['name']}: cours={live}€ (J={intra:+.2f}%, PV_total={pnl_pct:+.2f}%)"
            if p.get("threshold_high"):
                line += f" seuil_haut={p['threshold_high']}€"
            if p.get("threshold_low"):
                line += f" seuil_bas={p['threshold_low']}€"

            # Seuil d'anormalité PROPRE À LA VALEUR, à deux écarts-types de sa variation
            # quotidienne habituelle. Le seuil absolu de 3 % traitait à l'identique un ETF
            # World (±0,69 %/jour) et 2CRSI (±6,5 %/jour) : il criait sans arrêt sur la
            # seconde et restait muet sur la première, alors qu'un +2 % sur un ETF World est
            # un événement bien plus rare qu'un +5 % sur 2CRSI.
            #
            # Lecture du cache SEULE (`telecharger=False`) : on est dans la boucle horaire,
            # une quinzaine de téléchargements Yahoo n'y a pas sa place. Cache froid — avant
            # le premier briefing du jour — et la ligne ne porte pas de seuil : le modèle
            # retombe alors sur les 3 % historiques, explicitement prévus dans _ALERT_SYSTEM.
            infos = _stats_cache(p.get("yahoo_ticker"))
            if (vq := infos.get("volatilite_quotidienne_pct")):
                line += f" seuil_anormal={vq * 2:.1f}%"

            # Échéances : Yahoo d'abord, saisie manuelle en repli. Le champ dividend_date
            # n'existait que si l'utilisateur l'avait renseigné à la main, donc la règle
            # « dividende dans moins de 5 jours » ne se déclenchait quasiment jamais.
            dates = _dates_cache(p.get("yahoo_ticker"))
            if (d := dates.get("detachement_dividende") or p.get("dividend_date")):
                line += f" dividende_prévu={d}"
            if (d := dates.get("resultats")):
                line += f" résultats_prévus={d}"
            lines.append(line)
        except Exception:
            pass

    try:
        content = await call_llm_async(
            [
                {"role": "system", "content": _ALERT_SYSTEM},
                {
                    "role": "user",
                    "content": _ALERT_USER.format(
                        portfolio="\n".join(lines),
                        daily_pnl_pct=daily_pnl_pct,
                    ),
                },
            ],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_MEDIUM,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_MEDIUM),
        )
        if not content or not content.strip():
            logger.warning("Alert evaluation: empty LLM response, skipping")
            return False, ""
        result = extract_llm_json(content)
        should_alert = bool(result.get("alert", False))
        message = result.get("message", "")

        if should_alert and message:
            now_iso = now.isoformat()
            # Mark ALL eligible positions — not just name-matched ones.
            # A portfolio-level alert ("perte journalière > 2%") contains no position
            # name, so the old name-match logic never set last_alert_at, causing the
            # same alert to refire every hour until the cooldown actually engaged.
            for p in eligible:
                r.hset(
                    pos_key(user_code, p["isin"]),
                    mapping={
                        "last_alert_at": now_iso,
                        "last_alert_reason": message[:200],
                    },
                )

        return should_alert, message

    except Exception as exc:
        logger.error("Alert evaluation failed for %s: %s", user_code, exc)
        return False, ""


# ── Pending alert queue ─────────────────────────────────────────────────────


def push_pending_alert(user_code: str, message: str) -> None:
    r = get_redis()
    key = alert_queue_key(user_code)
    existing = json.loads(r.get(key) or "[]")
    existing.append({"message": message, "at": datetime.now(timezone.utc).isoformat()})
    r.setex(key, _ALERT_QUEUE_TTL, json.dumps(existing, ensure_ascii=False))


def pop_pending_alerts(user_code: str) -> list[dict]:
    """Return and clear all queued alerts for a user."""
    r = get_redis()
    key = alert_queue_key(user_code)
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
    tz = pytz.timezone(BRIEFING_TIMEZONE or "Europe/Paris")
    now = datetime.now(tz)
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    open_ = now.replace(hour=9, minute=0, second=0, microsecond=0)
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
    r = get_redis()
    isins = list(r.smembers(idx_key(user_code)))
    updated = 0

    for isin in isins:
        key = pos_key(user_code, isin)
        pos = r.hgetall(key)

        # Skip if at least one threshold is already set (user or LLM configured)
        if pos.get("threshold_high") or pos.get("threshold_low"):
            continue

        ticker_sym = _resolve_ticker(isin, r, key, pos.get("name", ""))
        if not ticker_sym:
            continue

        try:
            info = yf.Ticker(ticker_sym).info
            high_52w = info.get("fiftyTwoWeekHigh")
            low_52w = info.get("fiftyTwoWeekLow")

            if not high_52w or not low_52w:
                logger.warning(
                    "No 52-week data for %s (%s), skipping", ticker_sym, isin
                )
                continue

            buying_price = float(pos.get("buying_price") or 0)
            stop_loss = round(buying_price * 0.90, 2) if buying_price else 0.0
            tl = round(max(stop_loss, low_52w), 2)
            th = round(high_52w, 2)

            r.hset(key, mapping={"threshold_high": str(th), "threshold_low": str(tl)})
            logger.info(
                "Auto-thresholds %s (%s): high=%.2f low=%.2f (52w: %.2f/%.2f)",
                pos.get("name", isin),
                isin,
                th,
                tl,
                high_52w,
                low_52w,
            )
            updated += 1

        except Exception as exc:
            logger.warning(
                "Auto-threshold failed for %s (%s): %s", ticker_sym, isin, exc
            )

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
            buy = float(p.get("buying_price") or 0)
            pnl = float(p.get("unrealized_pnl_pct") or 0)
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
        content = await call_llm_async(
            [
                {
                    "role": "system",
                    "content": "Tu es un assistant de gestion de portefeuille boursier.",
                },
                {"role": "user", "content": prompt},
            ],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_THINK_MEDIUM,
            json_response=True,
            no_think=False,
            thinking_budget=THINKING_BUDGET_MEDIUM,
            timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
        )
        result = extract_llm_json(content)
    except Exception as exc:
        logger.error("LLM threshold suggestion failed for %s: %s", user_code, exc)
        return {}

    r = get_redis()
    suggestions: dict = {}

    for item in result.get("positions", []):
        isin = item.get("isin", "").strip()
        th = item.get("threshold_high")
        tl = item.get("threshold_low")
        if not isin or th is None or tl is None:
            continue
        key = pos_key(user_code, isin)
        if not r.exists(key):
            continue
        th = round(float(th), 2)
        tl = round(float(tl), 2)
        r.hset(key, mapping={"threshold_high": str(th), "threshold_low": str(tl)})
        suggestions[isin] = {
            "threshold_high": th,
            "threshold_low": tl,
            "rationale": item.get("rationale", ""),
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
                    logger.info(
                        "Trade: auto-set thresholds for %d positions (%s)",
                        filled,
                        user_code,
                    )

            if not get_redis().smembers(idx_key(user_code)):
                continue  # No positions yet — nothing to do

            if _is_market_hours():
                prices = await asyncio.to_thread(fetch_live_prices, user_code)
                if prices:
                    update_prices_in_redis(user_code, prices)
                    logger.info(
                        "Trade: updated %d prices for %s", len(prices), user_code
                    )
                    should_alert, msg = await evaluate_alerts(user_code)
                    if should_alert and msg:
                        push_pending_alert(user_code, msg)
                        logger.info(
                            "Trade alert queued for %s: %s", user_code, msg[:80]
                        )
            else:
                logger.debug(
                    "Trade: market closed, skipping price fetch for %s", user_code
                )

        except Exception as exc:
            logger.error("Trade check error for %s: %s", user_code, exc)
