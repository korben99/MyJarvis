"""Prompt système : les blocs conditionnels par utilisateur.

Ce qui est gardé ici, c'est le CONDITIONNEMENT, pas le texte des consignes. Un bloc
réservé à une catégorie d'utilisateurs est un bloc dont personne ne voit l'absence :
servi à tous, il annonce une capacité ou une adaptation qui n'a pas lieu d'être ;
servi à personne, il disparaît en silence — c'est exactement ce qui est arrivé à
AGENT_CAPABILITY, appelé par le pipeline sans constante derrière.
"""

import pipeline
from prompts import get_prompt


class TestConsigneEnfant:
    """`"enfant": true` dans users_list.json → CHILD_GUIDANCE dans le prompt système."""

    def test_un_enfant_recoit_la_consigne(self):
        prompt = pipeline.build_system_prompt("CHLOE4")
        assert get_prompt("CHILD_GUIDANCE") in prompt

    def test_un_adulte_ne_la_recoit_pas(self):
        assert get_prompt("CHILD_GUIDANCE") not in pipeline.build_system_prompt("ALICE1")

    def test_un_code_inconnu_ne_la_recoit_pas(self):
        assert get_prompt("CHILD_GUIDANCE") not in pipeline.build_system_prompt("")

    def test_la_consigne_suit_le_profil(self):
        """La consigne demande de calibrer le registre sur l'année de naissance, qui n'est
        lisible que dans <profil_utilisateur>. Le bloc doit donc être servi avec elle."""
        prompt = pipeline.build_system_prompt("CHLOE4")
        assert "<profil_utilisateur>" in prompt
        assert prompt.index("<profil_utilisateur>") < prompt.index(
            get_prompt("CHILD_GUIDANCE")
        )
