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

from datetime import date, datetime, time, timedelta

import pytest
import pytz

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
        for nom in ("NIGHTLY_SELF_SYSTEM", "ANALYSIS_PROMPT", "SYSTEM_BASE", "IDENTITY"):
            texte = getattr(prompts, nom)
            dates = re.findall(r"\d{2}/\d{2}/20\d{2}", texte)
            assert not dates, f"{nom} contient une date de session : {dates}"


# ── Autocoding : liste blanche et résolution des prompts ─────────────────────

class TestPromptsResolvent:
    """Trois défauts réels, trouvés en production, gardés ici.

    `get_prompt` rendait `""` sur un nom inconnu. Un prompt vide part au modèle sans lever
    la moindre erreur : le seul symptôme est une dégradation des réponses, constatée des
    jours plus tard et impossible à rattacher à sa cause.
    """

    def test_get_prompt_leve_sur_un_nom_inconnu(self):
        with pytest.raises(KeyError):
            prompts.get_prompt("CE_PROMPT_NEXISTE_PAS")

    def test_tous_les_noms_litteraux_du_code_resolvent(self):
        """C'est ce test qui a trouvé `AGENT_CAPABILITY`, appelé par pipeline.py sans
        jamais avoir été écrit — la capacité agent n'était donc annoncée à aucun admin."""
        import pathlib
        import re
        src = pathlib.Path(prompts.__file__).parent
        noms: set[str] = set()
        for f in src.rglob("*.py"):
            noms |= set(re.findall(r'get_prompt\(\s*"([A-Z_][A-Z0-9_]*)"', f.read_text()))
        noms.discard("NAME")  # exemple cité dans une docstring
        introuvables = sorted(n for n in noms if not isinstance(getattr(prompts, n, None), str))
        assert not introuvables, (
            f"get_prompt est appelé avec {introuvables}, qui n'existent pas. "
            "Le prompt correspondant partirait vide au modèle."
        )


class TestListeBlancheRefinePrompt:
    """La liste des prompts modifiables n'existait qu'en PROSE dans REFLECTION_PROMPT.

    Côté code, le seul garde était « la constante existe » : les 41 prompts du module
    étaient donc acceptés, dont ceux de la boucle agentique et REFINE_PROMPT_SYSTEM
    lui-même, que le modèle pouvait ainsi réécrire.
    """

    def test_tous_les_refinables_existent(self):
        fantomes = sorted(
            n for n in prompts.REFINABLE_PROMPTS
            if not isinstance(getattr(prompts, n, None), str)
        )
        assert not fantomes, f"REFINABLE_PROMPTS cite des constantes inexistantes : {fantomes}"

    def test_la_prose_et_le_code_disent_la_meme_chose(self):
        """Le garde qui manquait. Un nom annoncé au modèle mais refusé par le code fait
        payer un appel de raisonnement pour un refus — visible seulement dans le journal."""
        import re
        bloc = prompts.REFLECTION_PROMPT.split("Noms valides :")[1].split("• Routing")[0]
        annonces = set(re.findall(r"\b([A-Z][A-Z0-9_]{4,})\b", bloc))
        assert annonces == set(prompts.REFINABLE_PROMPTS), (
            f"annoncés au modèle mais refusés par le code : "
            f"{sorted(annonces - set(prompts.REFINABLE_PROMPTS))} · "
            f"acceptés par le code mais jamais annoncés : "
            f"{sorted(set(prompts.REFINABLE_PROMPTS) - annonces)}"
        )

    def test_les_prompts_de_lagent_ne_sont_pas_modifiables(self):
        """Ils se jugent sur un livrable, pas sur un échange : le cycle de réflexion n'a
        aucune observation qui les concerne."""
        agent = {n for n in vars(prompts) if n.startswith("AGENT_")}
        assert not (agent & prompts.REFINABLE_PROMPTS)

    def test_refine_prompt_ne_peut_pas_se_reecrire_lui_meme(self):
        for n in ("REFINE_PROMPT_SYSTEM", "REFINE_PROMPT_USER"):
            assert n not in prompts.REFINABLE_PROMPTS


# ── Fenêtre de la revue nocturne ─────────────────────────────────────────────

class TestFenetreVeille:
    """La journée relue se découpe dans le fuseau de l'utilisateur, pas en UTC.

    Des bornes UTC décalent la coupure de l'écart au méridien — cinq heures à New York —
    et rangent la soirée de l'un dans la journée de l'autre. Le défaut ne se voit nulle
    part : la revue tourne, lit un jour, et rend un rapport cohérent avec ce qu'elle a lu.
    """

    def _fenetre(self, tz, jour):
        """La fenêtre rendue pour `jour`, ancrée à midi le lendemain — midi existe dans
        tous les fuseaux, y compris ceux qui basculent l'heure à minuit."""
        from self.nightly import _fenetre_veille

        ancre = tz.localize(datetime.combine(jour + timedelta(days=1), time(12, 0)))
        return _fenetre_veille(tz, ancre)

    @pytest.mark.parametrize(
        "tz_name", ["Europe/Paris", "America/New_York", "America/Los_Angeles", "Asia/Tokyo"]
    )
    def test_les_bornes_tombent_a_minuit_local(self, tz_name):
        tz = pytz.timezone(tz_name)
        jour, debut, fin = self._fenetre(tz, date(2026, 10, 1))
        assert jour == "2026-10-01"
        assert datetime.fromtimestamp(debut, tz).strftime("%H:%M:%S") == "00:00:00"
        assert datetime.fromtimestamp(fin, tz).strftime("%H:%M:%S") == "23:59:59"

    @pytest.mark.parametrize(
        "tz_name,jour,heures",
        [
            ("Europe/Paris", date(2026, 3, 29), 23),
            ("Europe/Paris", date(2026, 10, 25), 25),
            ("America/New_York", date(2026, 3, 8), 23),
            ("America/New_York", date(2026, 11, 1), 25),
        ],
    )
    def test_un_jour_de_bascule_dure_23_ou_25_heures(self, tz_name, jour, heures):
        """`replace(hour=0)` + `normalize` donnait ici une fenêtre démarrée une heure
        avant ou après minuit : l'offset du moment du tir n'est pas celui de minuit."""
        _, debut, fin = self._fenetre(pytz.timezone(tz_name), jour)
        assert round((fin - debut + 0.000001) / 3600) == heures

    @pytest.mark.parametrize("tz_name", ["Europe/Paris", "America/Santiago"])
    def test_les_fenetres_pavent_lannee_sans_trou_ni_recouvrement(self, tz_name):
        """Un trou perdrait des échanges, un recouvrement les relirait deux fois — et la
        dédup autobio absorberait le second sans rien signaler."""
        tz = pytz.timezone(tz_name)
        jour = date(2026, 1, 1)
        _, _, fin_precedente = self._fenetre(tz, jour)
        for _ in range(364):
            jour += timedelta(days=1)
            rendu, debut, fin = self._fenetre(tz, jour)
            assert rendu == jour.isoformat()
            assert debut == pytest.approx(fin_precedente + 0.000001, abs=1e-6), jour
            fin_precedente = fin
