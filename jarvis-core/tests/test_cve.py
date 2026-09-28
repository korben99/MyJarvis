"""Scan de vulnérabilités : le périmètre, et rien d'autre.

Le module tient à UNE règle — est comptée la CVE critique dont le correctif est applicable
par une commande. Tout le reste (hautes, moyennes, critiques sans correctif, critiques d'une
image que l'amont n'a pas republiée) doit rester dehors : ni compté, ni stocké, ni injecté,
et sans effet sur α.

C'est un périmètre, pas un filtre d'affichage. Un compteur qui existe finit par être lu, et
une liste qui existe finit par être injectée : ces tests vérifient donc que ce qui est hors
périmètre ne traverse pas le module, pas seulement qu'il n'est pas affiché.
"""

import json
import types

import pytest

import cve


def _grype(matches: list):
    """Doublure de subprocess.run rendant une sortie grype fabriquée."""
    def run(cmd, **kw):
        return types.SimpleNamespace(
            returncode=0, stdout=json.dumps({"matches": matches}), stderr=""
        )
    return run


def _match(severite: str, versions_correctives: list, paquet: str = "paquet-test"):
    return {
        "vulnerability": {"severity": severite,
                          "fix": {"versions": versions_correctives},
                          "id": f"CVE-{severite}-{paquet}"},
        "artifact": {"name": paquet, "version": "1.0"},
    }


class TestPerimetre:
    """Une seule combinaison entre : critique, et correctif disponible."""

    @pytest.mark.parametrize(
        "severite,correctif,compte",
        [
            ("Critical", ["2.0"], True),
            ("Critical", [], False),
            ("High", ["2.0"], False),
            ("High", [], False),
            ("Medium", ["2.0"], False),
            ("Low", ["2.0"], False),
            ("Unknown", ["2.0"], False),
        ],
    )
    def test_seule_la_critique_corrigeable_entre(
        self, monkeypatch, severite, correctif, compte
    ):
        monkeypatch.setattr(cve.subprocess, "run", _grype([_match(severite, correctif)]))
        res = cve._scan_target("cible", "venv")
        assert res["crit"] == (1 if compte else 0)
        # Le détail suit le compteur : une liste qui garderait un paquet hors périmètre
        # finirait injectée par render_advice.
        assert len(res["details"]) == (1 if compte else 0)

    def test_une_haute_ne_laisse_aucune_trace(self, monkeypatch):
        """Pas de compteur de hautes, même à zéro : exposé, il est lu, et la réflexion s'en
        saisit pour réclamer un travail qu'on a décidé de ne pas faire."""
        monkeypatch.setattr(cve.subprocess, "run", _grype([_match("High", ["2.0"])]))
        res = cve._scan_target("cible", "venv")
        assert set(res) == {"crit", "details", "exclus"}

    def test_le_detail_ne_porte_plus_de_severite(self, monkeypatch):
        """Tout ce qui arrive là est critique : un champ `sev` laisserait croire au lecteur
        qu'il existe autre chose à trier."""
        monkeypatch.setattr(cve.subprocess, "run", _grype([_match("Critical", ["2.0"])]))
        assert "sev" not in cve._scan_target("cible", "venv")["details"][0]


class TestRemedeApplicable:
    """« Corrigeable » ne se teste pas pareil selon la cible, parce que le remède diffère."""

    @pytest.fixture
    def pile(self, monkeypatch):
        """Une seule image portant une critique corrigeable, et pas de venv."""
        monkeypatch.setattr(
            cve.subprocess, "run", _grype([_match("Critical", ["9.0"], "nltk")])
        )
        monkeypatch.setattr(cve, "_generate_sbom", lambda p: False)
        monkeypatch.setattr(cve, "_resolve_image", lambda c: "image:tag")
        monkeypatch.setattr(cve, "CONTAINERS", ["jarvis-webui"])
        monkeypatch.setattr(cve, "redis_set_json", lambda *a, **k: None)
        monkeypatch.setattr(cve, "redis_get_json", lambda *a, **k: None)
        return cve

    def test_une_image_deja_a_jour_ne_compte_pas(self, pile, monkeypatch):
        """Le seul remède est de tirer une image plus récente. S'il n'y a rien à tirer, il
        n'y a rien à répondre à l'alerte — elle installerait une peur sans issue."""
        monkeypatch.setattr(cve, "_image_plus_recente_dispo", lambda img: False)
        res = pile.scan()
        assert res["cve_critiques"] == 0
        assert res["vulnerables"] == []
        # Le compte brut reste visible par source : l'exposition n'est pas niée, c'est son
        # report dans l'agrégat qui est suspendu, et le motif l'accompagne.
        assert res["par_source"]["jarvis-webui"]["crit"] == 1
        assert "remede" in res["par_source"]["jarvis-webui"]

    def test_une_image_republiee_en_amont_compte(self, pile, monkeypatch):
        """Le contrôle se rouvre tout seul : dès que le digest diffère, l'alerte porte sur
        un geste réel — `docker compose pull`."""
        monkeypatch.setattr(cve, "_image_plus_recente_dispo", lambda img: True)
        res = pile.scan()
        assert res["cve_critiques"] == 1
        assert len(res["vulnerables"]) == 1

    def test_une_incertitude_ne_fait_pas_disparaitre_l_alerte(self, pile, monkeypatch):
        """`None` = registre injoignable. Le doute doit profiter à l'alerte, pas la taire."""
        monkeypatch.setattr(cve, "_image_plus_recente_dispo", lambda img: None)
        assert pile.scan()["cve_critiques"] == 1

    def test_le_venv_n_est_pas_soumis_au_controle_d_image(self, monkeypatch):
        """Pour le venv le remède est `pip install paquet==version` : l'existence de la
        version corrective suffit. Il n'y a pas d'amont qui republie — c'est nous."""
        appels = []
        monkeypatch.setattr(
            cve, "_image_plus_recente_dispo", lambda img: appels.append(img) or False
        )
        monkeypatch.setattr(
            cve.subprocess, "run", _grype([_match("Critical", ["9.0"], "urllib3")])
        )
        monkeypatch.setattr(cve, "_generate_sbom", lambda p: True)
        monkeypatch.setattr(cve, "_persister_sbom", lambda p: None)
        monkeypatch.setattr(cve, "CONTAINERS", [])
        monkeypatch.setattr(cve, "redis_set_json", lambda *a, **k: None)
        monkeypatch.setattr(cve, "redis_get_json", lambda *a, **k: None)

        res = cve.scan()
        assert res["cve_critiques"] == 1, "une critique pip-installable doit compter"
        assert appels == [], "le contrôle d'image n'a rien à faire sur le venv"


class TestCanalActionnable:
    """`vulnerables` alimente <vulnerabilites>, le seul canal injecté dans un prompt."""

    def test_render_advice_ne_filtre_plus_par_severite(self, monkeypatch):
        """Un filtre ici laisserait croire que le cache contient autre chose."""
        monkeypatch.setattr(
            cve, "get_cve",
            lambda: {"vulnerables": [
                {"source": "venv", "paquet": "urllib3", "version": "1.0",
                 "corrige_par": "2.0", "n": 3},
            ]},
        )
        rendu = cve.render_advice()
        assert "urllib3 1.0 → 2.0" in rendu
        assert "3 CVE" in rendu
        assert "Critical" not in rendu, "la sévérité n'a plus à être dite : tout est critique"

    def test_aucun_paquet_rend_une_chaine_vide(self, monkeypatch):
        """Chaîne vide et non « aucune » : c'est l'appelant qui formule l'absence."""
        monkeypatch.setattr(cve, "get_cve", lambda: {"vulnerables": []})
        assert cve.render_advice() == ""


class TestAggravation:
    """Le seul canal par lequel la CVE lève un incident et touche α."""

    def _scan(self, n: int) -> dict:
        return {"cve_critiques": n}

    def test_une_critique_de_plus_leve_une_alerte(self, monkeypatch):
        incidents = []
        monkeypatch.setattr(cve, "redis_get_json", lambda *a, **k: self._scan(1))
        import vitals

        monkeypatch.setattr(
            vitals, "mark_incident",
            lambda kind, detail, severity="info": incidents.append((kind, severity)),
        )
        cve._detecter_aggravation(self._scan(3))
        assert incidents == [("cve", "alerte")]

    def test_une_baisse_ne_leve_rien(self, monkeypatch):
        """Appliquer les correctifs fait retomber la peur, ça n'alarme pas."""
        incidents = []
        monkeypatch.setattr(cve, "redis_get_json", lambda *a, **k: self._scan(3))
        import vitals

        monkeypatch.setattr(
            vitals, "mark_incident",
            lambda *a, **k: incidents.append(a),
        )
        cve._detecter_aggravation(self._scan(0))
        assert incidents == []

    def test_le_premier_scan_pose_la_ligne_de_base(self, monkeypatch):
        incidents = []
        monkeypatch.setattr(cve, "redis_get_json", lambda *a, **k: None)
        import vitals

        monkeypatch.setattr(vitals, "mark_incident", lambda *a, **k: incidents.append(a))
        cve._detecter_aggravation(self._scan(5))
        assert incidents == []


class TestListeBlanche:
    """Un choix d'exploitant, journalisé à chaque scan — jamais un masquage silencieux."""

    def test_un_paquet_exclu_ne_compte_pas_mais_reste_audite(self, monkeypatch):
        monkeypatch.setattr(
            cve, "_PAQUETS_EXCLUS",
            [{"paquet": "ffmpeg", "source": "jarvis-webui", "motif": "motif de test"}],
        )
        monkeypatch.setattr(
            cve.subprocess, "run", _grype([_match("Critical", ["9.0"], "ffmpeg")])
        )
        res = cve._scan_target("image", "jarvis-webui")
        assert res["crit"] == 0
        assert res["details"] == []
        assert res["exclus"] and res["exclus"][0]["motif"] == "motif de test"

    def test_l_exclusion_est_bornee_a_la_source_declaree(self, monkeypatch):
        """Restreinte à (paquet, source) pour ne pas rater la même faille ailleurs."""
        monkeypatch.setattr(
            cve, "_PAQUETS_EXCLUS",
            [{"paquet": "ffmpeg", "source": "jarvis-webui", "motif": "m"}],
        )
        monkeypatch.setattr(
            cve.subprocess, "run", _grype([_match("Critical", ["9.0"], "ffmpeg")])
        )
        assert cve._scan_target("sbom:x", "venv")["crit"] == 1
