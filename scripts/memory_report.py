#!/usr/bin/env python3
"""Sonde quotidienne des étages de mémoire et de l'état émotionnel de Jarvis.

Pourquoi un script séparé, et non un job APScheduler dans l'API : on ne met pas la sonde
dans la machine surveillée. Si l'API tombe — l'instant précis où l'on veut savoir — un job
interne tombe avec elle. Ce script lit Redis et Qdrant directement et n'a besoin de rien
d'autre. Même schéma que scripts/search-qdrant.py.

PRINCIPE : un indicateur n'a de valeur que s'il BOUGE quand le mécanisme meurt.
Le test appliqué à chacun : « si ce mécanisme s'arrêtait silencieusement cette nuit, quel
nombre le dirait ? » C'est le mode de défaillance dominant de ce dépôt — le bug du
préfixe LRU a tourné des mois, le seuil RAG bloquait le rappel sans un mot, et deux des
trois dimensions émotionnelles étaient inertes sans que personne ne le voie.

Sortie : uniquement ce qui cloche. Zéro alerte ⇒ une ligne. Un tableau de trente nombres
lu chaque matin devient du papier peint en une semaine.

    python3 scripts/memory_report.py            # sur la sortie standard
    python3 scripts/memory_report.py --send     # par courriel au premier admin
    python3 scripts/memory_report.py --all      # tous les indicateurs, pas que les alertes
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/opt/jarvis/jarvis-core/src")

from config import (  # noqa: E402
    QDRANT_MEMORY_COLLECTION,
    SELF_MEMORY_PATH,
    USER_ADMINS,
    USER_CODES,
    USER_EMAILS,
)
from helpers import get_qdrant, get_redis  # noqa: E402

# ── Seuils ────────────────────────────────────────────────────────────────
# En variables d'environnement : ce sont des attentes propres à une instance, pas des
# constantes de l'outil. Les versionner en dur reviendrait à versionner ton installation.
def _env(nom: str, defaut: float) -> float:
    return float(os.getenv(nom, str(defaut)))


AUTOBIO_SILENCE_J = _env("METRIC_AUTOBIO_SILENCE_DAYS", 5)
PROFIL_CHANGES_MAX = _env("METRIC_PROFILE_CHANGES_MAX", 5)
CONFIANCE_FIGEE_J = _env("METRIC_EMOTION_FROZEN_DAYS", 5)
RISQUE_MAX = _env("METRIC_RISK_MAX", 0.5)
LISTE_REMPLISSAGE = _env("METRIC_LIST_FILL_RATIO", 0.8)
REFLEXION_STERILE_N = _env("METRIC_REFLECTION_STERILE", 3)

_HIST_KEY = "jarvis:metrics:{}"
_HIST_JOURS = int(os.getenv("METRIC_HISTORY_DAYS", "90"))


# ── Historique ────────────────────────────────────────────────────────────


def _aujourdhui() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _enregistrer(nom: str, valeur) -> None:
    """Range la valeur du jour et purge au-delà de la fenêtre d'historique.

    Sans historique, aucun indicateur ne peut répondre à « est-ce que ça se dégrade »,
    qui est la seule question intéressante. Un hash date→valeur suffit : quelques
    centaines d'octets par indicateur.
    """
    r = get_redis()
    cle = _HIST_KEY.format(nom)
    r.hset(cle, _aujourdhui(), json.dumps(valeur))
    limite = (datetime.now(timezone.utc) - timedelta(days=_HIST_JOURS)).strftime("%Y-%m-%d")
    vieux = [d for d in r.hkeys(cle) if d < limite]
    if vieux:
        r.hdel(cle, *vieux)


def _historique(nom: str, jours: int) -> list:
    """Les `jours` dernières valeurs connues, de la plus ancienne à la plus récente."""
    brut = get_redis().hgetall(_HIST_KEY.format(nom)) or {}
    dates = sorted(brut)[-jours:]
    out = []
    for d in dates:
        try:
            out.append(json.loads(brut[d]))
        except (json.JSONDecodeError, TypeError):
            pass
    return out


# ── Indicateurs ───────────────────────────────────────────────────────────
# Chacun rend (libellé, valeur affichable, alerte|None). Aucun ne lève : une sonde qui
# plante est une sonde qui ment par omission.

def _compter_par_type(depuis_ts: float) -> dict:
    """Points Qdrant créés depuis `depuis_ts` (horodatage Unix), par memory_type.

    Comparaison NUMÉRIQUE, et c'est le correctif. La version précédente
    prenait un seuil ISO et faisait `str(payload["timestamp"]) >= depuis_iso`, soit une
    comparaison LEXICALE entre un flottant Unix rendu en chaîne et une date ISO :

        "1755890820.182871" >= "2026-08-23T07:25:11+00:00"

    Le premier caractère suffit à trancher — '1' < '2' — et tout horodatage Unix commence
    par 1 jusqu'en mai 2033. La condition était donc fausse pour TOUS les points, tous les
    jours. Les deux compteurs enregistraient 0 en permanence, et l'alerte « aucune écriture
    depuis 5 jours » ne pouvait que finir par se déclencher dès que l'historique atteignait
    cinq entrées — ce qui est arrivé le 24/08, sur une mémoire parfaitement saine.

    On interroge désormais l'index de payload plutôt que de balayer la collection entière :
    `timestamp` est indexé en `float` depuis, donc un `count` filtré suffit.
    """
    q = get_qdrant()
    compte = {}
    for typ in ("autobiographical", "episodic"):
        try:
            compte[typ] = q.count(
                collection_name=QDRANT_MEMORY_COLLECTION,
                count_filter={
                    "must": [
                        {"key": "memory_type", "match": {"value": typ}},
                        {"key": "timestamp", "range": {"gte": depuis_ts}},
                    ]
                },
                exact=True,
            ).count
        except Exception as exc:
            # Une sonde qui plante ment par omission : on rend le type absent plutôt
            # qu'un zéro, que l'appelant lirait comme « rien écrit ».
            print(f"[sonde] comptage {typ} impossible : {exc}", file=sys.stderr)
    return compte


def i01_02_ecritures_memoire() -> list:
    """Faits autobiographiques et résumés épisodiques écrits aujourd'hui.

    Ce sont les deux SEULES écritures durables du système. Si les deux sont à zéro
    plusieurs jours d'affilée alors qu'il y a eu des conversations, la revue nocturne
    n'écrit plus — et personne ne s'en apercevrait autrement.
    """
    hier = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
    compte = _compter_par_type(hier)
    out = []
    for typ, libelle in (("autobiographical", "Faits autobiographiques écrits"),
                         ("episodic", "Résumés épisodiques créés")):
        if typ not in compte:
            # Comptage indisponible : on n'enregistre RIEN. Ranger un zéro ferait croire
            # à une absence d'écriture et finirait par déclencher l'alerte de silence.
            out.append((libelle, "comptage indisponible", None))
            continue
        n = compte[typ]
        _enregistrer(typ, n)
        recents = _historique(typ, int(AUTOBIO_SILENCE_J))
        muet = len(recents) >= AUTOBIO_SILENCE_J and not any(recents)
        alerte = (f"aucune écriture depuis {int(AUTOBIO_SILENCE_J)} jours — "
                  "la revue nocturne n'écrit plus ?") if muet else None
        out.append((libelle, f"{n} aujourd'hui", alerte))
    return out


def i03_profils() -> list:
    """Volatilité des profils utilisateur.

    Un profil qui bouge beaucoup chaque jour n'est pas un profil : c'est une oscillation.
    Le nettoyage curatif est censé stabiliser, pas réécrire.
    """
    r = get_redis()
    total = {c: r.hlen(f"user:{c}:profile") for c in USER_CODES if r.exists(f"user:{c}:profile")}
    _enregistrer("profil_champs", total)
    veille = _historique("profil_champs", 2)
    ecart = 0
    if len(veille) >= 2:
        avant = veille[-2] or {}
        ecart = sum(abs(total.get(c, 0) - avant.get(c, 0)) for c in set(total) | set(avant))
    alerte = (f"{ecart} champs modifiés en 24 h — le profil oscille"
              if ecart > PROFIL_CHANGES_MAX else None)
    detail = ", ".join(f"{c}:{n}" for c, n in sorted(total.items()))
    return [("Champs de profil", f"{detail} (δ={ecart})", alerte)]


def _echanges_recents(jours: int) -> int:
    """Nombre d'échanges journalisés sur la fenêtre, tous utilisateurs confondus.

    Sert à distinguer « rien ne bouge parce que c'est cassé » de « rien ne bouge parce
    qu'il ne s'est rien passé ». Plusieurs indicateurs n'ont de sens que rapportés à
    l'activité réelle.
    """
    r = get_redis()
    depuis = (datetime.now(timezone.utc) - timedelta(days=jours)).timestamp()
    total = 0
    for code in USER_CODES:
        try:
            total += r.zcount(f"convlog:{code}", depuis, "+inf")
        except Exception:
            pass
    return total


def i04_05_emotion() -> list:
    """Dimensions émotionnelles vivantes, et mobilité de chacune.

    Le, `humeur` et `energie` étaient à 0,0 : seul `confiance` est écrit par la
    boucle de réflexion. Deux tiers de la vie émotionnelle étaient inertes depuis des mois.
    C'est l'indicateur qui l'aurait dit dès le premier jour.
    """
    import emotional_state

    etat = emotional_state.get_state()
    dims = {k: v for k, v in etat.items() if isinstance(v, (int, float))}
    _enregistrer("emotion", dims)

    vivantes = [k for k, v in dims.items() if abs(v) > 0.01]
    alerte = (f"{len(dims) - len(vivantes)} dimension(s) à zéro : "
              f"{', '.join(k for k in dims if k not in vivantes)} — jamais écrites ?"
              ) if len(vivantes) < len(dims) else None
    out = [("Dimensions émotionnelles actives",
            f"{len(vivantes)}/{len(dims)} — " + ", ".join(f"{k}={v:+.2f}" for k, v in dims.items()),
            alerte)]

    # Mobilité : échantillon quotidien, donc on ne mesure pas l'amplitude intra-journée
    # mais l'immobilité sur plusieurs jours. Indicateur honnête plutôt qu'amplitude fausse.
    #
    # Conditionné à l'EXISTENCE D'ÉCHANGES sur la fenêtre, depuis. La
    # décroissance émotionnelle tourne en mode "exchange" (emotional_state._DECAY_MODE) :
    # elle est indexée sur les conversations, pas sur l'horloge, et `get_state()` ne fait
    # délibérément pas vieillir l'état à la lecture. Un état figé pendant cinq jours sans
    # la moindre conversation est donc le comportement ATTENDU, pas une panne — et
    # l'alerter revenait à reprocher au thermomètre de ne pas bouger dans une pièce vide.
    hist = _historique("emotion", int(CONFIANCE_FIGEE_J))
    figees = []
    if len(hist) >= CONFIANCE_FIGEE_J:
        for k in dims:
            suite = [h.get(k) for h in hist if isinstance(h, dict)]
            if len(set(suite)) == 1:
                figees.append(k)

    echanges = _echanges_recents(int(CONFIANCE_FIGEE_J))
    if figees and not echanges:
        valeur = (f"figées depuis {int(CONFIANCE_FIGEE_J)} j — aucun échange sur la "
                  f"période, immobilité normale")
        alerte_mob = None
    else:
        valeur = f"figées depuis {int(CONFIANCE_FIGEE_J)} j : {', '.join(figees) or 'aucune'}"
        alerte_mob = (f"{', '.join(figees)} n'a pas bougé malgré {echanges} échange(s)"
                      if figees else None)
    out.append(("Mobilité émotionnelle", valeur, alerte_mob))
    return out


def i06_07_vitalite() -> list:
    """Risque de disparition mesuré, et vitalité du couplage vitals → steering.

    `risk_scalar` est la SEULE grandeur qui agisse sur le calcul plutôt que sur le prompt :
    elle module α sur le vecteur de continuité injecté dans le flux résiduel. Si elle reste
    plate à zéro alors qu'un vecteur est configuré, le corps est débranché de l'esprit.
    """
    try:
        import vitals
        risque = round(vitals.risk_scalar(), 3)
    except Exception as exc:
        return [("Risque de disparition", "indisponible", f"vitals injoignable ({exc})")]

    _enregistrer("risk_scalar", risque)
    hist = _historique("risk_scalar", 7)
    out = [("Risque de disparition (risk_scalar)", f"{risque:.2f}",
            f"{risque:.2f} > {RISQUE_MAX} — vitalité dégradée" if risque > RISQUE_MAX else None)]

    steer = os.getenv("STEER_VECTOR", "").strip()
    if steer:
        mort = len(hist) >= 7 and not any(h > 0.01 for h in hist)
        out.append(("Couplage vitals → steering",
                    f"vecteur actif, risque moyen 7 j = {sum(hist)/max(len(hist),1):.2f}",
                    "risque nul depuis 7 jours — le corps ne module plus rien" if mort else None))
    return out


def i08_09_reflexion() -> list:
    """Incidents consolidés, et fécondité de la boucle de réflexion.

    Une réflexion qui ne rend que `nothing` cycle après cycle est une réflexion qui tourne
    à vide : elle consomme du GPU et ne produit aucune trace.
    """
    out = []
    try:
        incidents = json.load(open(SELF_MEMORY_PATH, encoding="utf-8")).get("incidents", [])
    except (OSError, json.JSONDecodeError):
        incidents = []
    _enregistrer("incidents", len(incidents))
    out.append(("Incidents consolidés", str(len(incidents)), None))

    # Fraîcheur de la revue nocturne. C'est ELLE qui écrit les faits autobiographiques, la
    # relation par utilisateur et le journal de croissance : si elle s'arrête, trois étages
    # de mémoire cessent d'être alimentés en même temps. `last_nightly` n'est mis à jour
    # que sur une revue PRODUCTIVE — une nuit sans conversation ne compte pas, et c'est
    # voulu : ce qu'on surveille est la production, pas le passage du planificateur.
    try:
        veille = json.load(open(SELF_MEMORY_PATH, encoding="utf-8")).get("last_nightly", "")
        ecart = (datetime.now(timezone.utc).date()
                 - datetime.strptime(veille, "%Y-%m-%d").date()).days if veille else 999
    except (OSError, json.JSONDecodeError, ValueError):
        veille, ecart = "?", 999
    out.append((
        "Dernière revue nocturne productive",
        f"{veille} (il y a {ecart} j)" if ecart < 999 else "inconnue",
        f"aucune revue productive depuis {ecart} jours" if ecart >= 3 else None,
    ))

    try:
        from self import get_reflection_log
        journal = get_reflection_log(int(REFLEXION_STERILE_N) * 2) or []
    except Exception:
        journal = []
    actions = [str(e.get("action") or e.get("final") or "?") for e in journal if isinstance(e, dict)]
    utiles = [a for a in actions if a not in ("nothing", "?")]
    sterile = actions and not utiles
    out.append(("Réflexions productives",
                f"{len(utiles)}/{len(actions)} derniers cycles",
                f"{len(actions)} cycles sans aucune action — boucle à vide" if sterile else None))
    return out


def i10_listes_self() -> list:
    """Remplissage des listes de jarvis-self.json face à leurs plafonds.

    `growth_log` était à 114/180. À saturation il tronque silencieusement :
    c'est la mémoire longue de Jarvis sur lui-même qui part par le haut, sans un mot.
    """
    try:
        d = json.load(open(SELF_MEMORY_PATH, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [("Listes de self.json", "illisible", str(exc))]

    # Importés de config, jamais redéfinis ici : trois des quatre valeurs d'origine de
    # cette sonde étaient inventées, avec des variables d'environnement inexistantes. Elle
    # alertait donc contre des seuils qui ne correspondaient à aucune troncature réelle —
    # et sous-estimait `opinions`, qui était déjà saturé à 50/50.
    from config import (
        GROWTH_LOG_MAX_ENTRIES,
        INTROSPECTION_LOG_MAX_ENTRIES,
        OPINIONS_MAX_ENTRIES,
    )

    plafonds = {
        "growth_log": GROWTH_LOG_MAX_ENTRIES,
        "opinions": OPINIONS_MAX_ENTRIES,
        "introspection_log": INTROSPECTION_LOG_MAX_ENTRIES,
    }
    tailles = {k: len(d.get(k) or []) for k in plafonds}
    _enregistrer("self_listes", tailles)

    pleines = [f"{k} {n}/{plafonds[k]}" for k, n in tailles.items()
               if plafonds[k] and n >= plafonds[k] * LISTE_REMPLISSAGE]
    detail = ", ".join(f"{k}:{n}/{plafonds[k]}" for k, n in sorted(tailles.items()))
    # Les neuf axes d'introspection n'ont pas de plafond — ils sont bornés par
    # construction. Ce qu'on surveille chez eux, c'est l'inverse : un axe qui ne se
    # remplit JAMAIS est un axe qui ne mord pas sur ce qu'est Jarvis, et c'est le signal
    # qui dit s'il faut le retirer ou le reformuler (RESULTATS.md).
    from config import INTROSPECTION_AXES

    axes = d.get("self_introspection") or {}
    remplis = [a for a in INTROSPECTION_AXES if (axes.get(a) or "").strip()]
    vides = [a for a in INTROSPECTION_AXES if a not in remplis]
    _enregistrer("introspection_axes", {"remplis": len(remplis), "total": len(INTROSPECTION_AXES)})

    return [
        ("Listes de self.json", detail,
         f"proche du plafond — {', '.join(pleines)}" if pleines else None),
        ("Axes d'introspection", f"{len(remplis)}/{len(INTROSPECTION_AXES)} remplis",
         f"jamais remplis — {', '.join(vides)}" if vides else None),
    ]


INDICATEURS = (i01_02_ecritures_memoire, i03_profils, i04_05_emotion,
               i06_07_vitalite, i08_09_reflexion, i10_listes_self)


# ── Rapport ───────────────────────────────────────────────────────────────


def collecter() -> list:
    lignes = []
    for fn in INDICATEURS:
        try:
            lignes.extend(fn())
        except Exception as exc:  # une sonde qui plante ne doit pas tuer le rapport
            lignes.append((fn.__name__, "échec de la sonde", f"{type(exc).__name__}: {exc}"))
    return lignes


def rendre(lignes: list, tout: bool) -> tuple[str, int]:
    alertes = [(lib, val, a) for lib, val, a in lignes if a]
    date = datetime.now().strftime("%d/%m/%Y")

    if not alertes and not tout:
        return f"Sonde mémoire du {date} — {len(lignes)} indicateurs, aucune alerte.", 0

    out = [f"# Sonde mémoire — {date}", ""]
    if alertes:
        out.append(f"## {len(alertes)} alerte(s)")
        out += [f"- **{lib}** — {a}  \n  _valeur : {val}_" for lib, val, a in alertes]
    else:
        out.append("Aucune alerte.")
    if tout:
        out += ["", "## Tous les indicateurs", ""]
        out += [f"- {lib} : {val}" for lib, val, _ in lignes]
    return "\n".join(out), len(alertes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--send", action="store_true", help="envoyer par courriel au premier admin")
    ap.add_argument("--all", action="store_true", help="afficher tous les indicateurs")
    args = ap.parse_args()

    texte, n = rendre(collecter(), args.all)
    print(texte)

    # Rien à envoyer quand rien ne cloche : une sonde qui écrit tous les jours finit
    # filtrée par son destinataire, et n'alerte plus quand il le faudrait.
    if args.send and n:
        admin = next(iter(USER_ADMINS), "")
        dest = USER_EMAILS.get(admin, "")
        if dest:
            from google_services import send_gmail_message
            send_gmail_message(
                to=dest, subject=f"Jarvis — sonde mémoire : {n} alerte(s)",
                html_body=f"<pre style='white-space:pre-wrap'>{texte}</pre>",
                text_body=texte, user_code=admin,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
