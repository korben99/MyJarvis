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
