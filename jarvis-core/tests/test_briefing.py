"""Briefing matinal : la reprise sur génération dégénérée, et le repli.

Le briefing est l'appel le plus exposé à la boucle : une sortie STRUCTURÉE dont les champs
portent de la PROSE longue. La famille répétition — le garde d'échantillonnage contre la
boucle — est écartée en mode JSON parce qu'elle pénalise les jetons que la grammaire
impose de répéter. Le garde est donc absent là où le contenu est le plus propice à boucler.

Ce qui est vérifié ici n'est pas que la boucle n'arrive plus (on ne sait pas la déclencher
à la demande), mais que ses conséquences sont bornées : une seconde tentative, puis un
repli qui ne sert jamais la sortie dégénérée.
"""

import asyncio
from unittest.mock import patch

import briefing


SECTIONS = {"calendar": [], "weather": "Beau temps", "gmail": [], "news": []}


def _appeler(reponses):
    """Joue `_assemble_with_llm` contre une suite de réponses LLM simulées.

    Une entrée `Exception` est levée, une chaîne est rendue. Retourne le couple
    (texte, html) et la liste des appels effectivement passés.
    """
    appels = []

    async def _faux_llm(messages, **kw):
        appels.append(kw)
        r = reponses[len(appels) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    with patch.object(briefing, "call_llm_async", _faux_llm), \
         patch.object(briefing, "now_user") as _nu:
        _nu.return_value.strftime.return_value = "lundi 05 octobre 2026"
        texte, html = asyncio.run(
            briefing._assemble_with_llm("Alice", "ALICE1", SECTIONS)
        )
    return texte, html, appels


class TestRepriseBriefing:

    def test_un_json_valide_du_premier_coup_nappelle_quune_fois(self):
        texte, html, appels = _appeler(['{"text": "Bonjour Alice", "html": "<p>ok</p>"}'])
        assert texte == "Bonjour Alice"
        assert html == "<p>ok</p>"
        assert len(appels) == 1

    def test_une_sortie_degeneree_declenche_une_seconde_tentative(self):
        """Le cas constaté : la boucle sature le plafond, le JSON reste inachevé."""
        degenere = "Bonjour Alice. " * 40
        texte, _, appels = _appeler(
            [degenere, '{"text": "vrai briefing", "html": "<p>x</p>"}']
        )
        assert len(appels) == 2, "la seconde tentative n'a pas eu lieu"
        assert texte == "vrai briefing"

    def test_deux_echecs_mènent_au_repli_jamais_a_la_sortie_degeneree(self):
        """Le garde qui compte : l'utilisateur reçoit agenda + météo, pas la boucle."""
        degenere = "Bonjour Alice. " * 40
        texte, html, appels = _appeler([degenere, degenere])
        assert len(appels) == 2
        assert "Bonjour Alice. Bonjour Alice." not in texte
        assert "Météo" in texte and "Beau temps" in texte
        assert html.startswith("<p>")

    def test_une_panne_de_transport_nest_pas_rejouee(self):
        """Un dépassement de délai se reproduirait et coûterait une génération entière :
        seul un JSON illisible mérite une seconde chance."""
        texte, _, appels = _appeler([TimeoutError("timeout")])
        assert len(appels) == 1
        assert "Météo" in texte

    def test_la_reprise_garde_les_memes_parametres_dappel(self):
        """La seconde tentative ne doit pas changer le contrat — c'est la graine qui
        change, pas les réglages."""
        degenere = "Bonjour Alice. " * 40
        _, _, appels = _appeler([degenere, '{"text": "t", "html": "h"}'])
        assert appels[0] == appels[1]
        assert appels[0]["json_response"] is True
        assert appels[0]["no_think"] is True
