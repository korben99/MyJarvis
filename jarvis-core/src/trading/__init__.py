"""
Jarvis — Trading (portefeuille, cours, alertes, tendances de marché)
====================================================================

Ce paquet remplace l'ancien couple de monofichiers `trading.py` / `trade_keys.py`. L'API
publique est **identique** : tous les `from trading import …` existants continuent de
fonctionner, ce `__init__` ré-exportant les noms depuis les sous-modules.

    keys    schéma des clés Redis (aucune dépendance — la fondation du paquet)
    core    import CSV Boursorama, cours yfinance, lecture du portefeuille, alertes
    market  séries historiques : tendances, statistiques, contexte de marché, échéances

Graphe de dépendances (acyclique) :
    core → keys ;  market → core

Séparation voulue entre `core` et `market` : le premier vit dans la boucle HORAIRE et ne
manipule qu'un instantané ; le second télécharge un an d'historique et ne tourne qu'une fois
par jour. Les mélanger ferait entrer un appel lent dans un chemin fréquent.

Seul `trading.keys` est importé directement de l'extérieur (`self/actions.py`), parce qu'il
n'a besoin que du schéma de clés et surtout pas de tirer yfinance avec lui.
"""

from .core import (
    auto_set_thresholds,
    evaluate_alerts,
    fetch_live_prices,
    get_portfolio,
    get_portfolio_summary_text,
    import_csv_to_redis,
    parse_boursorama_csv,
    pop_pending_alerts,
    push_pending_alert,
    run_trade_check,
    suggest_thresholds_llm,
    update_prices_in_redis,
)
from .market import (
    contexte_marche,
    dates_a_venir,
    render_briefing_block,
    stats_ligne,
)

__all__ = [
    # core
    "auto_set_thresholds",
    "evaluate_alerts",
    "fetch_live_prices",
    "get_portfolio",
    "get_portfolio_summary_text",
    "import_csv_to_redis",
    "parse_boursorama_csv",
    "pop_pending_alerts",
    "push_pending_alert",
    "run_trade_check",
    "suggest_thresholds_llm",
    "update_prices_in_redis",
    # market
    "contexte_marche",
    "dates_a_venir",
    "render_briefing_block",
    "stats_ligne",
]
