"""Récurrence des tâches d'agent.

La récurrence est une PROPRIÉTÉ d'une tâche, pas un type de tâche : le mécanisme ne sait
rien de ce qu'il relance, et les mêmes règles valent pour une revue de presse, une veille
ou un point hebdomadaire sur un projet.
"""

from unittest.mock import patch

import pytest

from agent.recurrent import (
    CADENCE_HEBDO,
    _domaine,
    dernier_tour_par_serie,
    nouvelle_serie,
    rappel_du_tour_precedent,
    relancer_les_series,
)

_BASE = {
    "id": "t", "user_code": "K", "objective": "o", "workspace": "/tmp/absent",
    "deliverables": [], "sources_seen": [], "origin": "human",
    "max_steps": 0, "timeout_minutes": 0, "status": "done",
}


def _meta(serie, cadence=CADENCE_HEBDO, **extra):
    return {"cadence": cadence, "serie": serie, **extra}


def _relancer(taches, arrets=frozenset()):
    """Rend les appels à create_task qu'aurait provoqués une ronde."""
    creees = []
    with patch("agent.recurrent.list_tasks", lambda n: list(taches)), \
         patch("agent.recurrent.create_task",
               lambda **kw: creees.append(kw) or {"id": "neuve", **kw}), \
         patch("agent.recurrent.series_arretees", lambda: set(arrets)):
        relancer_les_series()
    return creees


class TestSelectionDesSeries:
    def test_une_serie_terminee_est_relancee(self):
        assert len(_relancer([{**_BASE, "meta": _meta("s1")}])) == 1

    @pytest.mark.parametrize("statut", ["queued", "running"])
    def test_une_serie_encore_en_cours_est_sautee(self, statut):
        """Un tour qui déborde sur le suivant, ou un service redémarré au mauvais moment,
        empilerait des tâches qui se dupliqueraient le travail ET le courriel."""
        assert _relancer([{**_BASE, "status": statut, "meta": _meta("s1")}]) == []

    def test_une_serie_en_erreur_est_relancee(self):
        """Un échec n'interrompt pas la série : la semaine suivante est une occasion neuve,
        et l'alternative serait une récurrence qui s'éteint en silence."""
        assert len(_relancer([{**_BASE, "status": "error", "meta": _meta("s1")}])) == 1

    def test_une_tache_ponctuelle_nest_jamais_relancee(self):
        assert _relancer([{**_BASE, "meta": {}}]) == []

    def test_une_serie_arretee_nest_plus_relancee(self):
        assert _relancer([{**_BASE, "meta": _meta("s1")}], arrets={"s1"}) == []

    def test_deux_series_avancent_independamment(self):
        creees = _relancer([{**_BASE, "meta": _meta("s1")}, {**_BASE, "meta": _meta("s2")}])
        assert len(creees) == 2

    def test_une_serie_ne_produit_quun_tour_par_ronde(self):
        """L'index porte tous les tours passés ; les compter tous ferait d'une ronde autant
        de relances que la série a d'ancêtres."""
        creees = _relancer([
            {**_BASE, "id": "recent", "meta": _meta("s1")},
            {**_BASE, "id": "vieux", "meta": _meta("s1")},
        ])
        assert len(creees) == 1

    def test_le_dernier_tour_est_le_plus_recent(self):
        """`list_tasks` rend les plus récentes d'abord : c'est de CE tour que le successeur
        hérite, pas du premier venu."""
        taches = [{**_BASE, "id": "recent", "meta": _meta("s1")},
                  {**_BASE, "id": "vieux", "meta": _meta("s1")}]
        with patch("agent.recurrent.list_tasks", lambda n: taches), \
             patch("agent.recurrent.series_arretees", set):
            assert dernier_tour_par_serie()["s1"]["id"] == "recent"


class TestHeritageDeLObjectif:
    def test_lobjectif_source_est_conserve_tel_quel(self):
        creees = _relancer([{**_BASE, "meta": _meta("s1", objectif_source="veille stockage")}])
        assert creees[0]["objective"].startswith("veille stockage")
        assert creees[0]["meta"]["objectif_source"] == "veille stockage"

    def test_la_serie_et_les_budgets_sont_transmis(self):
        creees = _relancer([{**_BASE, "max_steps": 25, "timeout_minutes": 60,
                             "meta": _meta("s1")}])
        assert creees[0]["meta"]["serie"] == "s1"
        assert creees[0]["max_steps"] == 25 and creees[0]["timeout_minutes"] == 60

    def test_lobjectif_nenfle_pas_de_tour_en_tour(self):
        """LE défaut à ne pas laisser passer : si le successeur héritait de l'objectif
        AUGMENTÉ au lieu de la source, chaque édition traînerait toutes les précédentes et
        l'objectif grossirait sans fin."""
        tour1 = {**_BASE, "sources_seen": ["https://exemple.org/a"],
                 "meta": _meta("s1", objectif_source="veille stockage")}
        tour2 = _relancer([tour1])[0]
        suivant = {**_BASE, "objective": tour2["objective"], "sources_seen": tour1["sources_seen"],
                   "meta": tour2["meta"]}
        tour3 = _relancer([suivant])[0]
        assert len(tour3["objective"]) == len(tour2["objective"])

    def test_sans_antecedent_lobjectif_reste_nu(self):
        """Premier tour d'une série, ou prédécesseur sans trace : un rappel vide serait une
        consigne qui n'interdit rien tout en occupant le contexte."""
        assert rappel_du_tour_precedent(None) == ""
        assert rappel_du_tour_precedent({**_BASE}) == ""


class TestRappelDuTourPrecedent:
    def test_les_domaines_remplacent_les_urls(self):
        """Une même source paginée rend autant d'adresses que de pages ; les compter
        séparément remplirait le rappel de doublons qui n'en sont pas."""
        tache = {**_BASE, "sources_seen": [
            "https://www.exemple.org/revue-2", "https://exemple.org/revue-3",
            "https://autre.fr/x",
        ]}
        rappel = rappel_du_tour_precedent(tache)
        assert "exemple.org, autre.fr" in rappel

    def test_le_rappel_est_borne(self, monkeypatch):
        import agent.recurrent as R

        monkeypatch.setattr(R, "AGENT_RECURRENT_MAX_DOMAINES", 3)
        tache = {**_BASE, "sources_seen": [f"https://d{i}.fr/x" for i in range(50)]}
        assert rappel_du_tour_precedent(tache).count(".fr") == 3

    def test_www_est_retire(self):
        assert _domaine("https://www.exemple.org/a") == "exemple.org"

    def test_une_url_illisible_ne_fait_pas_tomber_le_rappel(self):
        assert _domaine("pas une url") == ""

    def test_un_livrable_absent_ne_prive_pas_des_sources(self, tmp_path):
        """L'enregistrement vit en Redis sous TTL, le fichier sur disque : perdre l'un ne
        doit pas coûter l'autre."""
        tache = {**_BASE, "workspace": str(tmp_path), "deliverables": ["disparu.md"],
                 "sources_seen": ["https://exemple.org/a"]}
        assert "exemple.org" in rappel_du_tour_precedent(tache)

    def test_le_sommaire_du_livrable_est_repris(self, tmp_path):
        (tmp_path / "r.md").write_text("# Titre\n\ntexte\n\n## Section A\n\n### Détail\n")
        tache = {**_BASE, "workspace": str(tmp_path), "deliverables": ["r.md"]}
        rappel = rappel_du_tour_precedent(tache)
        assert "- Titre" in rappel and "- Section A" in rappel


class TestIdentifiantDeSerie:
    def test_deux_series_ne_se_confondent_pas(self):
        assert nouvelle_serie() != nouvelle_serie()

    def test_lidentifiant_tient_dans_le_motif_de_la_commande_darret(self):
        """La commande « agent: arrête <id> » ne reconnaît que [0-9a-f]{6,16} : un
        identifiant hors de ce format rendrait la série impossible à couper depuis le chat."""
        import re

        assert re.fullmatch(r"[0-9a-f]{6,16}", nouvelle_serie())


class TestCommandeDeChat:
    """« tâche agent hebdo: » — le seul endroit où une récurrence se déclare."""

    @pytest.fixture
    def banc(self, monkeypatch):
        import agent
        import routes.chat as C

        appels = {}

        def _faux_create(user_code, objective, origin="human", meta=None, **kw):
            appels.update({"objective": objective, "meta": meta, "user_code": user_code})
            return {"id": "abcd1234ef567890"}

        monkeypatch.setattr(agent, "create_task", _faux_create)
        monkeypatch.setattr(C, "_instant_reply", lambda req, uc, txt: {"txt": txt})
        monkeypatch.setattr(C, "USER_ADMINS", {"K"})
        monkeypatch.setattr(C, "AGENT_ENABLED", True)
        return C, appels

    def _envoyer(self, C, message, user="K"):
        class _Req:
            def __init__(self, m):
                self.message = m

        return C._handle_agent_task(_Req(message), user)

    @pytest.mark.parametrize("prefixe", [
        "tâche agent hebdo:", "tache agent hebdo :", "agent hebdo:", "agent weekly:",
    ])
    def test_les_variantes_declarent_une_cadence(self, banc, prefixe):
        C, appels = banc
        self._envoyer(C, f"{prefixe} veille concurrentielle sur le stockage")
        assert appels["meta"]["cadence"] == CADENCE_HEBDO
        assert appels["meta"]["objectif_source"] == "veille concurrentielle sur le stockage"

    def test_une_tache_ponctuelle_reste_sans_cadence(self, banc):
        """La régression qui compte : « agent: » ne doit pas se mettre à récidiver."""
        C, appels = banc
        self._envoyer(C, "tâche agent: un travail ponctuel bien assez long")
        assert appels["meta"] is None

    def test_le_prefixe_recurrent_nemporte_pas_le_mot_hebdo_dans_lobjectif(self, banc):
        C, appels = banc
        self._envoyer(C, "agent hebdo: revue de presse des innovations")
        assert appels["objective"] == "revue de presse des innovations"

    def test_un_message_ordinaire_nest_pas_une_commande(self, banc):
        C, _ = banc
        assert self._envoyer(C, "bonjour, comment vas-tu ?") is None

    def test_le_premier_tour_part_tout_de_suite(self, banc):
        """Attendre le prochain créneau ferait d'une commande une promesse, et personne ne
        saurait avant plusieurs jours si l'objectif était seulement tenable."""
        C, appels = banc
        reponse = self._envoyer(C, "agent hebdo: veille sur le stockage objet")
        assert appels["objective"]
        assert "arrière-plan" in reponse["txt"]

    def test_la_reponse_annonce_la_recurrence(self, banc):
        C, _ = banc
        assert "chaque semaine" in self._envoyer(C, "agent hebdo: veille stockage objet")["txt"]

    def test_un_non_administrateur_ne_peut_pas_lancer_de_serie(self, banc):
        C, appels = banc
        reponse = self._envoyer(C, "agent hebdo: veille stockage objet", user="AUTRE")
        assert not appels and "administrateurs" in reponse["txt"]

    def test_une_serie_sarrete_depuis_le_chat(self, banc, monkeypatch):
        """Sans cette commande, couper une récurrence demanderait d'éditer Redis à la main."""
        C, _ = banc
        import agent.recurrent as R

        arretees = []
        monkeypatch.setattr(R, "arreter_serie", arretees.append)
        reponse = self._envoyer(C, "agent: arrête 1a2b3c4d5e6f")
        assert arretees == ["1a2b3c4d5e6f"] and "arrêtée" in reponse["txt"]

    def test_un_non_administrateur_ne_peut_pas_couper_une_serie(self, banc, monkeypatch):
        C, _ = banc
        import agent.recurrent as R

        arretees = []
        monkeypatch.setattr(R, "arreter_serie", arretees.append)
        self._envoyer(C, "agent: arrête 1a2b3c4d5e6f", user="AUTRE")
        assert arretees == []


class TestPlanificationAuDemarrage:
    """`main` était inimportable sous les stubs de test — aucun travail planifié n'avait
    donc de couverture possible, celui-ci compris. La cause tenait au simulacre de
    `qdrant_client.models`, qui énumérait les symboles au lieu de répondre à tous.
    """

    def test_main_est_importable(self):
        import main

        assert main

    def test_la_relance_est_posee_sur_le_planificateur(self):
        """Une fonction qui existe mais que personne n'appelle ne relance rien.

        Lu dans la source : le planificateur est monté dans le `lifespan`, qu'on ne peut
        pas dérouler sans charger le modèle. Vérifier le texte vaut mieux que ne rien
        vérifier — ce test tombe si le travail est supprimé, renommé, ou décroché de ses
        constantes de configuration.
        """
        import inspect

        import main

        src = inspect.getsource(main.lifespan)
        assert 'id="agent_recurrent"' in src
        assert "relancer_les_series" in src
        assert "day_of_week=AGENT_RECURRENT_DAY" in src
        assert "hour=AGENT_RECURRENT_HOUR" in src
