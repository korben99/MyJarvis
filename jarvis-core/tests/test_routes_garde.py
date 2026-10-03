"""Garde d'accès des routeurs — qui dépense, qui détruit, qui reste ouvert.

Le partage utile n'est pas « cette route est protégée » mais la LIGNE : les routes de
consultation restent ouvertes (le tableau de bord les interroge sans jeton, et c'est la
posture assumée du réseau de confiance), celles qui dépensent le GPU, détruisent de la
mémoire ou éteignent un signal exigent un administrateur.

Le test de câblage vaut plus que le test de la fonction : la garde elle-même est courte et
évidente, alors que l'oubli porte toujours sur UNE route ajoutée plus tard à un routeur
qui, lui, n'en a pas.
"""

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from config import USER_ADMINS, USER_CODES
from routes import memory_routes, self_routes
from routes.garde import exige_admin


def _gardes(router, chemin: str) -> set[str]:
    for r in router.routes:
        if isinstance(r, APIRoute) and r.path == chemin:
            return {d.call.__name__ for d in r.dependant.dependencies if d.call}
    raise AssertionError(f"route absente du routeur : {chemin}")


class TestGarde:

    def test_un_administrateur_passe(self):
        admin = sorted(USER_ADMINS)[0]
        assert exige_admin(f"Bearer {admin}") == admin

    @pytest.mark.parametrize("entete", [None, "", "Bearer", "Bearer ", "n'importe quoi"])
    def test_sans_jeton_utilisable_refus(self, entete):
        with pytest.raises(HTTPException) as e:
            exige_admin(entete)
        assert e.value.status_code == 403

    def test_un_utilisateur_non_admin_est_refuse(self):
        """Le code d'un utilisateur ordinaire ouvre le chat, pas la purge mémoire."""
        ordinaires = sorted(set(USER_CODES) - set(USER_ADMINS))
        if not ordinaires:
            pytest.skip("aucun utilisateur non-admin dans le jeu de test")
        with pytest.raises(HTTPException) as e:
            exige_admin(f"Bearer {ordinaires[0]}")
        assert e.value.status_code == 403

    def test_le_jeton_ne_voyage_pas_en_clair_dans_lurl(self):
        """La garde ne lit QUE l'en-tête : un code passé en requête serait conservé par
        les journaux et l'historique du navigateur."""
        with pytest.raises(HTTPException):
            exige_admin(sorted(USER_ADMINS)[0])  # sans le préfixe Bearer


class TestCablage:

    @pytest.mark.parametrize("router,chemin", [
        (memory_routes.router, "/memory/reset"),    # détruit la mémoire de tout le foyer
        (self_routes.router, "/self/reflect"),      # mobilise le GPU plusieurs minutes
        (self_routes.router, "/self/maintenance"),  # éteint la sévérité des incidents
    ])
    def test_les_routes_couteuses_exigent_un_admin(self, router, chemin):
        assert "exige_admin" in _gardes(router, chemin)

    @pytest.mark.parametrize("router,chemin", [
        (self_routes.router, "/self/state"),
        (self_routes.router, "/self/log"),
        (memory_routes.router, "/memory/emotional-state"),
    ])
    def test_les_lectures_restent_ouvertes(self, router, chemin):
        """Les fermer casserait jarvis-status.sh, qui les interroge sans jeton."""
        assert _gardes(router, chemin) == set()

    @pytest.mark.parametrize("router", [memory_routes.router, self_routes.router])
    def test_toute_route_destructive_est_gardee(self, router):
        """Règle de forme plutôt que liste à tenir : un DELETE ajouté plus tard sans garde
        échoue ici, sans que personne ait pensé à l'inscrire dans le test au-dessus."""
        for r in router.routes:
            if isinstance(r, APIRoute) and "DELETE" in r.methods:
                noms = {d.call.__name__ for d in r.dependant.dependencies if d.call}
                assert "exige_admin" in noms, f"{r.path} détruit sans garde"
