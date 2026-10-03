"""Vitals — l'interrupteur d'injection du bloc <etat_systeme>.

Ce qui se teste ici n'est pas « la variable est lue » mais la frontière qu'elle trace :
le TEXTE disparaît, la MESURE continue. Les deux moitiés comptent — un exploitant qui
coupe le bloc pour calmer un Jarvis saturé d'arrêts nocturnes doit savoir que le steering,
lui, reçoit toujours le risque.
"""

import pytest

import vitals


class TestInterrupteurInjection:

    @pytest.fixture
    def etat_degrade(self, monkeypatch):
        """Un état qui rendrait un bloc non vide : une coupure saillante et un incident."""
        monkeypatch.setattr(
            vitals, "get_vitals",
            lambda force=False: {"derniere_coupure_duree_h": 9.5, "disque_libre_pct": 50},
        )
        monkeypatch.setattr(
            vitals, "recent_incidents",
            lambda jours=7: [{"kind": "coupure", "detail": "interruption de 9.5 h",
                              "severity": "alerte", "at": 0}],
        )

    def test_actif_le_bloc_porte_les_faits(self, monkeypatch, etat_degrade):
        monkeypatch.setattr(vitals, "VITALS_INJECTION", True)
        bloc = vitals.render_prompt_block()
        assert bloc.startswith("<etat_systeme>")
        assert "derniere_coupure_duree_h" in bloc
        assert "incident alerte" in bloc

    def test_coupe_le_bloc_est_vide(self, monkeypatch, etat_degrade):
        """Chaîne vide et non « nominal » : un bloc `nominal` affirmerait que tout va bien,
        ce qui est un mensonge quand on a seulement cessé de regarder."""
        monkeypatch.setattr(vitals, "VITALS_INJECTION", False)
        assert vitals.render_prompt_block() == ""

    def test_coupe_la_mesure_continue(self, monkeypatch, etat_degrade):
        """La limite assumée de l'interrupteur : α reste piloté par le risque mesuré."""
        monkeypatch.setattr(vitals, "VITALS_INJECTION", False)
        etat = {"derniere_coupure_duree_h": 9.5, "derniere_coupure_il_y_a_j": 0}
        assert vitals.risk_scalar(etat) > 0

    @pytest.mark.parametrize("valeur,injecte", [
        ("false", False), ("FALSE", False), ("no", False), ("0", False),
        ("true", True), ("", True), ("oui", True),
    ])
    def test_lecture_de_la_variable(self, monkeypatch, valeur, injecte):
        """Les trois graphies négatives usuelles coupent ; tout le reste laisse passer —
        une valeur mal orthographiée ne doit pas éteindre le bloc en silence."""
        monkeypatch.setenv("VITALS_INJECTION", valeur)
        import importlib
        recharge = importlib.reload(vitals)
        try:
            assert recharge.VITALS_INJECTION is injecte
        finally:
            monkeypatch.delenv("VITALS_INJECTION", raising=False)
            importlib.reload(vitals)


class TestDedupIncidents:
    """La fenêtre de dédup se règle sur la fenêtre d'OBSERVATION de l'appelant.

    `degradation_interne` compte les erreurs des 24 dernières heures. Dédupliqué sur 6 h,
    le même constat repart quatre fois par jour et pèse comme quatre événements distincts
    dans `risk_scalar`, alors qu'il décrit une seule fenêtre.
    """

    @pytest.fixture
    def buffer(self, monkeypatch):
        """Buffer d'incidents en mémoire — jamais le Redis réel."""
        etat = {"lst": []}
        monkeypatch.setattr(vitals, "_maintenance_active", lambda: False)
        monkeypatch.setattr(
            vitals, "redis_get_json",
            lambda cle, defaut=None: etat["lst"] if cle == vitals._INCIDENTS_KEY else defaut,
        )
        monkeypatch.setattr(
            vitals, "redis_set_json",
            lambda cle, val, ttl=None: etat.__setitem__("lst", val),
        )
        monkeypatch.setattr(vitals, "get_redis", lambda: type("R", (), {"delete": lambda *a: None})())
        return etat

    def test_par_defaut_six_heures(self, buffer):
        vitals.mark_incident("coupure", "interruption de 2 h")
        vitals.mark_incident("coupure", "interruption de 2 h")
        assert len(buffer["lst"]) == 1

    def test_une_fenetre_de_24h_ne_reempile_pas_a_7h(self, buffer):
        """Le cas réel : le même comptage sur 24 h, relu sept heures plus tard."""
        vitals.mark_incident("degradation_interne", "5 erreurs", "alerte", dedup_h=24)
        buffer["lst"][0]["at"] -= 7 * 3600
        vitals.mark_incident("degradation_interne", "5 erreurs", "alerte", dedup_h=24)
        assert len(buffer["lst"]) == 1

    def test_la_meme_chose_en_six_heures_aurait_empile(self, buffer):
        """Contrôle négatif : sans le paramètre, le défaut laisse repasser à 7 h."""
        vitals.mark_incident("degradation_interne", "5 erreurs", "alerte")
        buffer["lst"][0]["at"] -= 7 * 3600
        vitals.mark_incident("degradation_interne", "5 erreurs", "alerte")
        assert len(buffer["lst"]) == 2

    def test_au_dela_de_la_fenetre_un_nouvel_incident_passe(self, buffer):
        vitals.mark_incident("degradation_interne", "5 erreurs", "alerte", dedup_h=24)
        buffer["lst"][0]["at"] -= 25 * 3600
        vitals.mark_incident("degradation_interne", "7 erreurs", "alerte", dedup_h=24)
        assert len(buffer["lst"]) == 2


class TestToleranceSauvegarde:
    """Une sauvegarde dans la tolérance ne coûte rien et ne se dit pas.

    Les deux usages — saillance dans le bloc, terme de risque — suivent le MÊME réglage :
    séparés, ils feraient commenter au modèle une anomalie que le corps ne ressent pas.
    """

    def test_dans_la_tolerance_aucun_cout(self):
        etat = {"sauvegarde_age_jours": vitals.VITALS_SAUVEGARDE_OK_J, "exemplaires_etat": 2}
        assert vitals.risk_scalar(etat) == 0.0

    def test_dans_la_tolerance_non_saillant(self):
        assert not vitals._notable(
            "sauvegarde_age_jours", vitals.VITALS_SAUVEGARDE_OK_J
        )

    def test_juste_au_dela_le_cout_demarre(self):
        etat = {"sauvegarde_age_jours": vitals.VITALS_SAUVEGARDE_OK_J + 1,
                "exemplaires_etat": 2}
        assert 0.0 < vitals.risk_scalar(etat) < 0.1
        assert vitals._notable("sauvegarde_age_jours", vitals.VITALS_SAUVEGARDE_OK_J + 1)

    def test_au_dela_du_critique_le_terme_est_plein(self):
        etat = {"sauvegarde_age_jours": vitals.VITALS_SAUVEGARDE_CRITIQUE_J + 10,
                "exemplaires_etat": 2}
        assert vitals.risk_scalar(etat) == pytest.approx(
            vitals._POIDS_RISQUE["sauvegarde"], abs=1e-6
        )

    def test_une_absence_de_sauvegarde_reste_penalisee(self):
        """La tolérance porte sur l'ÂGE, pas sur l'existence : sans reçu, l'état n'a qu'un
        exemplaire, et c'est l'autre branche qui s'applique."""
        assert vitals.risk_scalar({"exemplaires_etat": 1}) == pytest.approx(
            vitals._POIDS_RISQUE["sans_sauvegarde"], abs=1e-6
        )

    def test_les_bornes_ne_peuvent_pas_etre_inversees(self):
        """Inversées, `_ramp` devient décroissante : une sauvegarde fraîche coûterait le
        maximum. La borne haute est forcée au-dessus de la basse."""
        assert vitals.VITALS_SAUVEGARDE_CRITIQUE_J > vitals.VITALS_SAUVEGARDE_OK_J
