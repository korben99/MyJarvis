"""Contrat de sortie de l'analyzer.

Le piège que ces tests gardent est documenté dans AGENTS.md et a déjà coûté une
fonctionnalité entière : `model_config = {"extra": "ignore"}` fait que **tout champ non
déclaré dans le modèle pydantic est supprimé en silence**. Ajouter un champ au prompt sans
l'ajouter au modèle produit donc une fonctionnalité qui ne marche jamais, sans la moindre
erreur, et que `logs/prompts.log` montre pourtant comme présente — puisque ce journal
affiche la sortie BRUTE, avant validation.

Un test de dérive vaut ici bien plus qu'un test de comportement : il échoue au moment où
l'écart est introduit, pas six semaines plus tard en production.
"""

import re

import pytest

import prompts
from analyzer import AnalysisResult, InterestWeight, ProjectEvent, UserFact


def _cles_du_prompt() -> set[str]:
    """Les clés JSON que ANALYSIS_PROMPT demande au modèle de produire."""
    return set(re.findall(r'"([a-z_]+)"\s*:', prompts.ANALYSIS_PROMPT))


def _champs_declares() -> set[str]:
    """Les champs acceptés par les modèles pydantic, tous niveaux confondus."""
    champs: set[str] = set()
    for modele in (AnalysisResult, UserFact, InterestWeight, ProjectEvent):
        champs |= set(modele.model_fields)
    return champs


class TestContratPromptModele:
    """Le prompt et les modèles pydantic doivent décrire la même structure."""

    def test_aucun_champ_du_prompt_nest_silencieusement_jete(self):
        orphelins = _cles_du_prompt() - _champs_declares()
        assert not orphelins, (
            f"ANALYSIS_PROMPT demande {sorted(orphelins)} au modèle, mais aucun modèle "
            "pydantic ne déclare ces champs : ils seront supprimés en silence par "
            "extra=ignore. Ajouter le champ au modèle, ou le retirer du prompt."
        )

    def test_les_modeles_ignorent_bien_les_champs_inconnus(self):
        """Vérifie le mécanisme lui-même, pour que le test ci-dessus ait un sens."""
        r = AnalysisResult.model_validate({"topics": ["a"], "champ_invente": "valeur"})
        assert not hasattr(r, "champ_invente")
        assert r.topics == ["a"]

    @pytest.mark.parametrize("modele", [AnalysisResult, UserFact, InterestWeight, ProjectEvent])
    def test_extra_ignore_est_bien_pose(self, modele):
        """Passer un modèle en extra=forbid casserait le pipeline sur toute sortie bavarde."""
        assert modele.model_config.get("extra") == "ignore"


class TestValeursParDefaut:
    """Une sortie LLM partielle ne doit jamais faire tomber le pipeline."""

    def test_objet_vide_est_valide(self):
        r = AnalysisResult.model_validate({})
        assert r.topics == []
        assert r.mood == "neutral"
        assert r.satisfaction == "unknown"
        assert r.memory_summary is None
        assert r.importance is None

    def test_listes_par_defaut_ne_sont_pas_partagees(self):
        """`default_factory` et non `default=[]` : sinon deux analyses partagent la liste."""
        a, b = AnalysisResult(), AnalysisResult()
        a.topics.append("pollution")
        assert b.topics == []

    def test_project_event_due_accepte_null(self):
        """`due` a déjà été mangé par le modèle une fois ; il doit rester nullable."""
        assert "due" in ProjectEvent.model_fields
        assert ProjectEvent.model_validate({"name": "p", "action": "update"}).due is None
        assert ProjectEvent.model_validate(
            {"name": "p", "action": "update", "due": "2026-09-01"}
        ).due == "2026-09-01"


class TestPromptAnalyse:
    """Garde-fous de forme sur ANALYSIS_PROMPT."""

    def test_le_prompt_demande_bien_du_json(self):
        assert "json" in prompts.ANALYSIS_PROMPT.lower()

    def test_les_huit_champs_de_premier_niveau_sont_cites(self):
        attendus = set(AnalysisResult.model_fields)
        cites = _cles_du_prompt()
        manquants = attendus - cites
        assert not manquants, (
            f"{sorted(manquants)} sont déclarés dans AnalysisResult mais jamais demandés "
            "au modèle : ils resteront à leur valeur par défaut pour toujours."
        )
