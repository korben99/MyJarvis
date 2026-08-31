"""
Contexte de marché et statistiques de tendance.

Pendant *historique* de `core.py` : celui-ci ne connaît qu'un instantané — `last_price` et
`intraday_var_pct`, écrasés à chaque passage horaire —, celui-là porte la série. Sans série,
aucune tendance n'est calculable : c'est pourquoi le briefing ne savait dire qu'une seule
chose, « mouvement notable > 1 % intraday ». Un point n'est pas une trajectoire.

Trois productions, toutes issues de la MÊME série téléchargée une fois par jour :

  • statistiques par ligne — perf 5j/1m/6m, position vs MM50/MM200, extrêmes 52 semaines,
    volatilité, perte maximale ;
  • contexte de marché — indices, volatilité implicite, EUR/USD, matières premières, taux ;
  • dates à venir — détachement de dividende et résultats, lus chez Yahoo au lieu d'être
    saisis à la main.

**Pourquoi la volatilité change tout.** Elle ne décore pas, elle donne l'unité de mesure d'un
mouvement. Mesuré sur le portefeuille réel : +3 % sur WPEA (vol. 11 %/an) vaut plus de quatre
écarts-types quotidiens — un événement ; +3 % sur 2CRSI (vol. 103 %/an) vaut un demi
écart-type — un mardi ordinaire. Le seuil d'alerte absolu de `core.py` traitait les deux à
l'identique, donc criait sans cesse sur l'une et jamais sur l'autre.

Contraintes de conception :
  • **Un appel par ligne et par jour, jamais dans un tour.** `history()` est lent et Yahoo
    n'aime pas les rafales. Tout passe par un cache Redis de 20 h ; le briefing du matin est
    le seul déclencheur naturel, et il tourne déjà hors boucle de requête.
  • **Ne lève jamais.** Une ligne dont l'historique est indisponible est simplement absente
    du rendu. Un champ manquant est préférable à un champ faux : le modèle n'a aucun moyen
    de détecter une valeur inventée.
  • **Des faits, pas des ordres.** Ce module ne produit ni recommandation ni score. Il rend
    des mesures ; c'est le prompt du briefing qui les met en perspective.
"""

import time
from datetime import date, datetime

from helpers import get_logger, redis_get_json, redis_set_json

logger = get_logger("jarvis-market")

_CACHE_TTL = 20 * 3600  # 20 h — une séance par jour, on ne retélécharge pas dans la journée
_HIST_KEY = "jarvis:market:hist:"
_DATES_KEY = "jarvis:market:dates:"

_SEANCES_AN = 252  # séances de bourse par an, pour annualiser un écart-type quotidien

# Le contexte de marché, volontairement court. Chaque ligne doit changer la LECTURE du
# portefeuille : les grands indices pour situer une performance, le VIX pour dire si le
# marché est calme ou nerveux, l'EUR/USD parce qu'il pèse sur tout actif libellé en dollar
# (les ETF World en tête), l'or et le Brent comme repères macro, le 10 ans US parce que les
# taux commandent la valorisation des actions de croissance.
INDICES = [
    ("^FCHI", "CAC 40"),
    ("^STOXX50E", "EuroStoxx 50"),
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("^VIX", "VIX (volatilité implicite)"),
    ("EURUSD=X", "EUR/USD"),
    ("GC=F", "Or"),
    ("BZ=F", "Brent"),
    ("^TNX", "Taux US 10 ans"),
]


def _closes(ticker: str, telecharger: bool = True) -> list[float] | None:
    """Série des clôtures d'un an, en cache Redis 20 h. None si indisponible.

    On ne conserve QUE les clôtures, pas l'OHLC complet : tout ce que ce module calcule s'en
    déduit, et une liste de flottants se relit en JSON sans dépendre de pandas — le cache
    reste exploitable par n'importe quel processus.

    `telecharger=False` interdit tout accès réseau et se contente du cache. C'est le mode
    qu'emploie la boucle HORAIRE d'alerte : elle ne doit jamais payer une quinzaine de
    téléchargements Yahoo. Cache froid → None → l'appelant retombe sur son ancien
    comportement, jamais sur une attente.
    """
    cache = redis_get_json(_HIST_KEY + ticker, None)
    if isinstance(cache, dict) and isinstance(cache.get("c"), list) and cache["c"]:
        return cache["c"]
    if not telecharger:
        return None
    try:
        import warnings

        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if h.empty:
            logger.debug("market: historique vide pour %s", ticker)
            return None
        # `x == x` écarte les NaN sans importer numpy : un NaN n'est jamais égal à lui-même.
        closes = [round(float(x), 4) for x in h["Close"].tolist() if x == x]
    except Exception as exc:
        logger.warning("market: historique %s indisponible (%s)", ticker, type(exc).__name__)
        return None
    if len(closes) < 30:
        logger.debug("market: historique %s trop court (%d séances)", ticker, len(closes))
        return None
    redis_set_json(_HIST_KEY + ticker, {"c": closes, "at": time.time()}, ttl=_CACHE_TTL)
    return closes


def _perf(closes: list[float], seances: int) -> float | None:
    """Variation en % sur N séances glissantes, None si la série est trop courte."""
    if len(closes) <= seances or not closes[-seances - 1]:
        return None
    return round((closes[-1] / closes[-seances - 1] - 1) * 100, 1)


def _moyenne_mobile(closes: list[float], n: int) -> float | None:
    return round(sum(closes[-n:]) / n, 4) if len(closes) >= n else None


def _volatilite_quotidienne(closes: list[float]) -> float | None:
    """Écart-type des rendements quotidiens, en %. C'est l'unité de mesure d'un mouvement :
    savoir qu'une ligne bouge de 0,7 % par jour en moyenne rend un +3 % lisible."""
    rend = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]
    if len(rend) < 20:
        return None
    moy = sum(rend) / len(rend)
    var = sum((x - moy) ** 2 for x in rend) / (len(rend) - 1)
    return round(var ** 0.5 * 100, 3)


def _perte_max(closes: list[float]) -> float | None:
    """Pire repli depuis un sommet sur la période, en %. Dit ce que la ligne a déjà fait
    subir — une information que la performance seule masque."""
    sommet = closes[0]
    pire = 0.0
    for c in closes:
        sommet = max(sommet, c)
        if sommet:
            pire = min(pire, c / sommet - 1)
    return round(pire * 100, 1)


def _tendance(s: dict) -> str:
    """Qualification en un mot, à partir des deux moyennes mobiles.

    Définition explicite et vérifiable plutôt que laissée au modèle : au-dessus des deux
    moyennes = haussière, sous les deux = baissière, discordance = transition. C'est ce
    dernier cas qui porte l'information utile — Valneva mesurée à +28 % sur un mois DANS une
    baisse de 40 % sur six mois est un rebond dans une tendance baissière, pas un
    retournement, et seule la discordance MM50/MM200 le dit.
    """
    if s.get("vs_mm50") is None or s.get("vs_mm200") is None:
        return "indéterminée"
    haut50 = s["vs_mm50"] == "au-dessus"
    haut200 = s["vs_mm200"] == "au-dessus"
    if haut50 and haut200:
        return "haussière"
    if not haut50 and not haut200:
        return "baissière"
    return "transition (rebond dans une baisse)" if haut50 else "transition (repli dans une hausse)"


def stats_ligne(ticker: str, telecharger: bool = True) -> dict:
    """Toutes les mesures de tendance d'une valeur. Dict vide si l'historique manque.

    `telecharger=False` : lecture du cache seule, sans réseau — voir `_closes`."""
    closes = _closes(ticker, telecharger)
    if not closes:
        return {}
    dernier = closes[-1]
    mm50 = _moyenne_mobile(closes, 50)
    mm200 = _moyenne_mobile(closes, 200)
    vol_j = _volatilite_quotidienne(closes)
    s: dict = {
        "dernier": round(dernier, 4),
        "perf_5j": _perf(closes, 5),
        "perf_1m": _perf(closes, 22),
        "perf_6m": _perf(closes, 127),
        "plus_haut_52s": round(max(closes), 4),
        "plus_bas_52s": round(min(closes), 4),
        "perte_max_52s_pct": _perte_max(closes),
        "seances": len(closes),
    }
    if mm50:
        s["vs_mm50"] = "au-dessus" if dernier > mm50 else "en-dessous"
    if mm200:
        s["vs_mm200"] = "au-dessus" if dernier > mm200 else "en-dessous"
    if vol_j:
        s["volatilite_quotidienne_pct"] = vol_j
        s["volatilite_annuelle_pct"] = round(vol_j * (_SEANCES_AN ** 0.5), 1)
    s["tendance"] = _tendance(s)
    return {k: v for k, v in s.items() if v is not None}


def dates_a_venir(ticker: str, telecharger: bool = True) -> dict:
    """Détachement de dividende et résultats **à venir**, lus chez Yahoo.

    `telecharger=False` : cache seul, pour la boucle horaire d'alerte — voir `_closes`.

    Le filtre sur les dates futures n'est pas cosmétique : mesuré le 31/08/2026, Yahoo rend
    pour 2CRSI un détachement du 05/12/2023 et pour Valneva des résultats du 13/08/2026 déjà
    passés. Sans filtre, le briefing annoncerait comme « à venir » un événement vieux de
    trois ans. Les ETF rendent un calendrier vide : c'est normal, pas une panne.
    """
    cache = redis_get_json(_DATES_KEY + ticker, None)
    if isinstance(cache, dict):
        return {k: v for k, v in cache.items() if k != "at"}
    if not telecharger:
        return {}
    try:
        import warnings

        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = yf.Ticker(ticker).calendar or {}
    except Exception as exc:
        logger.debug("market: calendrier %s indisponible (%s)", ticker, type(exc).__name__)
        return {}

    aujourdhui = date.today()

    def _futur(v):
        """Rend une date ISO si elle est à venir, sinon None. Yahoo mélange les formes : une
        date nue pour le détachement, une LISTE pour les résultats."""
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, datetime):
            v = v.date()
        return v.isoformat() if isinstance(v, date) and v >= aujourdhui else None

    res: dict = {}
    if (d := _futur(cal.get("Ex-Dividend Date"))):
        res["detachement_dividende"] = d
    if (d := _futur(cal.get("Earnings Date"))):
        res["resultats"] = d
    redis_set_json(_DATES_KEY + ticker, {**res, "at": time.time()}, ttl=_CACHE_TTL)
    return res


def contexte_marche() -> list[dict]:
    """Indices, volatilité implicite, devise, matières premières et taux. Liste vide si
    aucune série n'aboutit — on n'invente jamais un repère de marché."""
    out = []
    for ticker, nom in INDICES:
        if (s := stats_ligne(ticker)):
            out.append({"nom": nom, **s})
    return out


# ── Rendu pour le briefing ────────────────────────────────────────────────

def _pct(v) -> str:
    return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/d"


def render_briefing_block(user_code: str) -> str:
    """Bloc <perspectives_marche> du briefing, ou chaîne vide si rien n'aboutit.

    Le portefeuille garde son bloc à lui (positions, prix, plus-values) : ici on n'ajoute QUE
    ce qu'un instantané ne peut pas dire — trajectoire, norme de variation, échéances. Aucun
    doublon de prix.

    On dit « dernière séance connue » et jamais « aujourd'hui » : mesuré le 31/08, Yahoo
    servait encore la clôture du 27/08. Annoncer ces chiffres comme ceux du jour serait faux
    trois jours sur sept.
    """
    lignes: list[str] = []

    if (marche := contexte_marche()):
        lignes.append("Marché (dernière séance connue) :")
        for m in marche:
            lignes.append(
                f"  - {m['nom']} : {m['dernier']:.2f} | 5j {_pct(m.get('perf_5j'))}"
                f" · 1m {_pct(m.get('perf_1m'))} · 6m {_pct(m.get('perf_6m'))}"
                f" | tendance {m.get('tendance', 'indéterminée')}"
            )

    try:
        from .core import get_portfolio

        positions = get_portfolio(user_code)
    except Exception as exc:
        logger.debug("market: portefeuille illisible (%s)", exc)
        positions = []

    lignes_pos: list[str] = []
    echeances: list[tuple[str, str]] = []
    for p in positions:
        ticker = p.get("yahoo_ticker")
        if not ticker:
            continue
        nom = p.get("name", p.get("isin", "?"))
        if (s := stats_ligne(ticker)):
            vol = (f" | variation quotidienne normale ±{s['volatilite_quotidienne_pct']}%"
                   if s.get("volatilite_quotidienne_pct") else "")
            # Moyennes mobiles écrites en toutes lettres : abrégées en « MM50 / MM200 », le
            # modèle les dépliait de travers — « sous sa moyenne mobile à 200 ANS » dans un
            # briefing de test. Trois mots de plus valent mieux qu'une unité fausse.
            lignes_pos.append(
                f"  - {nom} : tendance {s['tendance']} | 5j {_pct(s.get('perf_5j'))}"
                f" · 1m {_pct(s.get('perf_1m'))} · 6m {_pct(s.get('perf_6m'))}"
                f" | {s.get('vs_mm50', 'n/d')} de sa moyenne mobile 50 jours,"
                f" {s.get('vs_mm200', 'n/d')} de sa moyenne mobile 200 jours{vol}"
                f" | plage 52 semaines {s['plus_bas_52s']}–{s['plus_haut_52s']}"
            )
        d = dates_a_venir(ticker)
        if d.get("detachement_dividende"):
            echeances.append((d["detachement_dividende"],
                              f"  - {nom} : détachement du dividende le {d['detachement_dividende']}"))
        if d.get("resultats"):
            echeances.append((d["resultats"],
                              f"  - {nom} : publication des résultats le {d['resultats']}"))

    if lignes_pos:
        lignes.append("\nTendance de tes lignes :")
        lignes.extend(lignes_pos)
    if echeances:
        lignes.append("\nÉchéances à venir :")
        # Tri sur la date ISO portée à part, et non sur le libellé : découper la phrase pour
        # y repêcher la date remettrait une chaîne de présentation au cœur d'un tri.
        lignes.extend(texte for _, texte in sorted(echeances))

    return "\n".join(lignes)
