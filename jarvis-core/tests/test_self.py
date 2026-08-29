"""Proto-self : catalogue d'actions, axes d'introspection, revue nocturne.

Le découpage à garder en tête : **la nuit APPREND, la réflexion AGIT**. Une action qui
écrit ce que Jarvis sait appartient à la revue nocturne ; une action qui fait quelque chose
vers l'extérieur appartient au cycle de réflexion. Plusieurs bugs sont nés de la confusion
des deux — trois écrivains concurrents pour l'autobio, des prompts réécrits sur des défauts
jamais constatés.

Les tests les plus utiles ici sont des tests de COHÉRENCE : entre ce qu'on annonce au
modèle et ce que le code sait exécuter, entre les axes déclarés et ceux que le prompt
demande. C'est exactement la classe d'écart qui se voit en production sous forme d'appel de
raisonnement payé pour rien, et jamais dans un journal.
"""

import pytest

import prompts
from self.actions import _ACTION_CATALOG
from self.engine import _SELF_ACTIONS, _SELF_REVIEW_REQUIRED, _USER_ACTIONS, _USER_SCOPED


# ── Cohérence du catalogue d'actions ─────────────────────────────────────────

class TestCatalogueActions:

    def test_toute_action_annoncee_a_un_handler(self):
        """Invariant déjà gardé au démarrage par engine.py — on le vérifie sans booter.

        Une action annoncée au modèle mais absente du catalogue retombe silencieusement sur
        « nothing » : un appel de raisonnement payé pour rien, visible seulement en relisant
        le journal.
        """
        orphelines = (_SELF_ACTIONS | _USER_ACTIONS) - set(_ACTION_CATALOG)
        assert not orphelines, f"annoncées au modèle mais sans handler : {sorted(orphelines)}"

    def test_nothing_est_disponible_des_deux_cotes(self):
        """C'est le repli de toute chaîne : son absence ferait échouer un cycle entier."""
        assert "nothing" in _SELF_ACTIONS
        assert "nothing" in _USER_ACTIONS

    def test_les_actions_utilisateur_sont_derivees_du_catalogue(self):
        """`_USER_SCOPED` a été réécrit à la main une fois, et il manquait
        `flag_project_stall` : l'action était morte par le chemin LLM."""
        assert _USER_SCOPED == _USER_ACTIONS - {"nothing"}

    def test_les_deux_phases_ne_partagent_que_nothing(self):
        """Agir sur soi et agir sur un utilisateur sont deux registres distincts."""
        assert (_SELF_ACTIONS & _USER_ACTIONS) == {"nothing"}

    def test_les_actions_contestees_existent(self):
        toutes = _SELF_ACTIONS | _USER_ACTIONS
        assert _SELF_REVIEW_REQUIRED <= toutes

    def test_alert_admin_passe_par_lauto_contestation(self):
        """Elle réveille quelqu'un : elle ne part jamais sans être contestée."""
        assert "alert_admin" in _SELF_REVIEW_REQUIRED

    def test_refine_prompt_ne_passe_pas_par_lauto_contestation(self):
        """Choix mesuré, pas un oubli : elle ne fait que PROPOSER, un humain tranche
        ensuite. La contester en plus avait coûté 19 vetos sur 19 en quatre jours."""
        assert "refine_prompt" not in _SELF_REVIEW_REQUIRED

    @pytest.mark.parametrize("partie", ["store_insight", "correct_profile", "check_health"])
    def test_les_actions_retirees_ne_reviennent_pas(self, partie):
        """La nuit est propriétaire de l'autobio et du profil ; `check_health` n'était pas
        une action, son résultat est déjà dans le contexte."""
        assert partie not in (_SELF_ACTIONS | _USER_ACTIONS)


# ── Axes d'introspection ─────────────────────────────────────────────────────

class TestAxesIntrospection:

    def test_il_y_a_bien_neuf_axes(self, cfg):
        """Le nombre est fixe par conception : la liste `learnings` qu'ils remplacent
        croissait sans borne utile."""
        assert len(cfg.INTROSPECTION_AXES) == 9

    def test_chaque_axe_porte_une_definition(self, cfg):
        for nom, definition in cfg.INTROSPECTION_AXES.items():
            assert definition.strip(), f"l'axe {nom} n'a pas de définition"

    def test_les_axes_sont_tous_cites_au_modele(self, cfg):
        """Un axe déclaré mais jamais demandé reste vide pour toujours."""
        manquants = [a for a in cfg.INTROSPECTION_AXES if a not in prompts.NIGHTLY_SELF_SYSTEM]
        assert not manquants, f"axes absents de NIGHTLY_SELF_SYSTEM : {manquants}"


# ── Prompt de revue nocturne ─────────────────────────────────────────────────

class TestPromptNocturne:

    @pytest.mark.parametrize(
        "champ", ["self_introspection", "jarvis_opinions", "knowledge_gaps"]
    )
    def test_les_sorties_attendues_sont_demandees(self, champ):
        assert champ in prompts.NIGHTLY_SELF_SYSTEM

    def test_les_lacunes_exigent_un_echec_observe(self):
        """Sans cette exigence, le modèle produit « lacune identifiée dans mes capacités
        d'assistance » — une phrase vague que le code rejette de toute façon."""
        p = prompts.NIGHTLY_SELF_SYSTEM.lower()
        assert "observé" in p or "concret" in p

    def test_lintrospection_est_une_revision_pas_une_accumulation(self):
        """Une ligne par axe, la dernière compte : c'est ce qui borne la croissance."""
        assert "axe" in prompts.NIGHTLY_SELF_SYSTEM.lower()

    def test_aucune_date_de_session_ne_traine_dans_le_prompt(self):
        """Une date de développement dans un prompt part au modèle à chaque appel et
        invalide le cache LRU sans rien lui apprendre."""
        import re
        for nom in ("NIGHTLY_SELF_SYSTEM", "ANALYSIS_PROMPT", "SYSTEM_BASE_FR", "IDENTITY_FR"):
            texte = getattr(prompts, nom)
            dates = re.findall(r"\d{2}/\d{2}/20\d{2}", texte)
            assert not dates, f"{nom} contient une date de session : {dates}"
