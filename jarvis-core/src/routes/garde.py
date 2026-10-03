"""Garde d'accès partagée par les routeurs.

Une seule définition pour trois routeurs. Recopiée, une garde diverge : celui qui ajoute
un jeton de service d'un côté ne le voit pas manquer de l'autre, et l'écart ne se
remarque que le jour où quelqu'un s'en sert.

Ce que la garde protège n'est pas la lecture mais la DÉPENSE et la DESTRUCTION : un
cycle de réflexion mobilise le GPU plusieurs minutes, une fenêtre de maintenance éteint
le signal d'incident, et une purge mémoire est irréversible. Les routes de simple
consultation restent ouvertes — c'est la posture assumée du réseau de confiance, et le
tableau de bord les interroge sans jeton.
"""

from config import USER_ADMINS
from fastapi import Header, HTTPException


def exige_admin(authorization: str = Header(default=None)) -> str:
    """Code administrateur porté par l'en-tête, ou lève.

    En-tête `Authorization: Bearer <code>`, comme les routes de portefeuille : le code est
    un secret, et un secret n'a pas à voyager dans une chaîne de requête, que les journaux
    et l'historique du navigateur conservent.
    """
    code = (
        authorization[7:].strip()
        if authorization and authorization.startswith("Bearer ")
        else ""
    )
    if code not in USER_ADMINS:
        raise HTTPException(403, "réservé aux administrateurs")
    return code
