"""Parité entre les jeux de langue.

Trois modules portent la langue, et la distinction compte :

    prompts_XX.py   ce qu'on ENVOIE au modèle
    textes_XX.py    ce qu'on RESTITUE à l'utilisateur   (pas encore extrait)
    lexique_XX.py   ce qu'on RECONNAÎT dans son message

Ces tests vérifient qu'aucun jeu n'a pris de retard sur l'autre. Un nom manquant
produirait un `KeyError` au moment de servir une requête — c'est-à-dire au pire moment,
et sur une seule route à la fois, donc difficile à rattacher à sa cause.
"""

import importlib

import pytest

import prompts_fr


def _constantes(module) -> set[str]:
    return {n for n, v in vars(module).items() if n.isupper() and isinstance(v, str)}


class TestPariteDesPrompts:

    def test_le_jeu_francais_est_complet(self):
        """Sanité : c'est le jeu de référence."""
        assert len(_constantes(prompts_fr)) >= 40

    def test_le_jeu_anglais_couvre_le_francais(self):
        """Le test qui autorise à basculer JARVIS_LANG=en.

        Tant qu'il échoue, l'instance anglaise refuse de démarrer — c'est voulu, et la
        garde de `prompts.py` le dit en nommant les constantes manquantes.

        """
        try:
            prompts_en = importlib.import_module("prompts_en")
        except ImportError:
            pytest.skip("prompts_en.py absent — jeu anglais pas encore écrit")
        manquantes = sorted(_constantes(prompts_fr) - _constantes(prompts_en))
        assert not manquantes, (
            f"{len(manquantes)} constantes absentes de prompts_en.py : {manquantes}"
        )

    def test_aucune_constante_orpheline_en_anglais(self):
        """Une constante qui n'existe qu'en anglais ne serait jamais servie en français."""
        try:
            prompts_en = importlib.import_module("prompts_en")
        except ImportError:
            pytest.skip("prompts_en.py absent")
        orphelines = sorted(_constantes(prompts_en) - _constantes(prompts_fr))
        assert not orphelines, f"présentes seulement en anglais : {orphelines}"

    def test_les_champs_de_substitution_sont_identiques(self):
        """La parité des NOMS ne suffit pas : ce sont les `{champs}` qui plantent.

        Un prompt traduit en oubliant un champ lève un `KeyError` au moment de servir la
        requête ; un champ ajouté d'un seul côté est passé pour rien, donc silencieux
        jusqu'au jour où l'instance change de langue. Les deux se voient ici et nulle part
        ailleurs — la relecture humaine de deux fichiers de 1600 lignes ne les attrape pas.
        """
        import string

        try:
            prompts_en = importlib.import_module("prompts_en")
        except ImportError:
            pytest.skip("prompts_en.py absent")

        def champs(texte: str) -> set[str]:
            return {f for _, f, _, _ in string.Formatter().parse(texte) if f}

        ecarts = []
        for nom in sorted(_constantes(prompts_fr) & _constantes(prompts_en)):
            fr, en = champs(getattr(prompts_fr, nom)), champs(getattr(prompts_en, nom))
            if fr != en:
                ecarts.append(f"{nom} (FR seul : {sorted(fr - en)}, EN seul : {sorted(en - fr)})")
        assert not ecarts, "champs désynchronisés — " + " ; ".join(ecarts)


class TestPariteDuLexique:

    def test_le_lexique_anglais_couvre_le_francais(self):
        from llm import lexique_fr
        try:
            lexique_en = importlib.import_module("llm.lexique_en")
        except ImportError:
            pytest.skip("lexique_en.py absent — lexique anglais pas encore écrit")
        attendus = {n for n in vars(lexique_fr) if n.isupper() or n.startswith("_")}
        attendus -= {"__builtins__", "__doc__", "__file__", "__loader__",
                     "__name__", "__package__", "__spec__", "__cached__"}
        manquants = sorted(n for n in attendus if not hasattr(lexique_en, n))
        assert not manquants, f"absents de lexique_en.py : {manquants}"


class TestBalises:
    """Les balises XML sont des délimiteurs d'injection, pas de la prose."""

    def test_les_balises_restent_identiques_entre_langues(self):
        """Les renommer casserait tous les blocs de contexte : les sites d'injection les
        écrivent littéralement dans le code, pas via les prompts."""
        import re
        try:
            prompts_en = importlib.import_module("prompts_en")
        except ImportError:
            pytest.skip("prompts_en.py absent")

        def balises(module):
            tout = " ".join(v for n, v in vars(module).items()
                            if n.isupper() and isinstance(v, str))
            return set(re.findall(r"</?([a-z_]{4,})>", tout))

        communes = _constantes(prompts_fr) & _constantes(prompts_en)
        assert communes, "aucune constante commune à comparer"
        ecart = balises(prompts_en) - balises(prompts_fr)
        assert not ecart, f"balises inventées côté anglais : {sorted(ecart)}"
