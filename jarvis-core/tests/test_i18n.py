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


class TestMotifsDeRaisonnement:
    """Le drapeau de raisonnement se lève sur deux familles, et pas sur une troisième.

    Les intents disent où CHERCHER ; ils ne disent pas quoi faire du résultat. Une
    recherche web peut être une restitution (un prix, une recette) ou une synthèse
    (comparer, trancher), et seule la formulation de la demande les sépare. C'est pourquoi
    la réflexion est pilotée par ces motifs et non par l'intent.
    """

    @pytest.fixture(params=["fr", "en"])
    def lexique(self, request):
        return importlib.import_module(f"llm.lexique_{request.param}"), request.param

    ORDRES = {
        "fr": ["raisonne bien là-dessus", "explique étape par étape", "analyse en profondeur"],
        "en": ["reason about this", "walk me through it step by step", "go in depth"],
    }
    SYNTHESES = {
        "fr": ["compare les deux offres", "la M4 par rapport à la M5",
               "dis-moi ce que tu en penses", "ton avis sur ce montage ?",
               "quels sont les avantages et inconvénients", "lequel choisir ?"],
        "en": ["compare the two offers", "the M4 compared to the M5",
               "what do you think of it", "your opinion on this setup?",
               "pros and cons please", "which one should i take?"],
    }
    RESTITUTIONS = {
        "fr": ["donne-moi une recette de gaufres", "les dernières infos sur le conflit",
               "quelle est la capitale du Pérou ?", "le cours de l'or aujourd'hui"],
        "en": ["give me a waffle recipe", "latest news on the conflict",
               "what is the capital of Peru?", "gold price today"],
    }

    def test_un_ordre_explicite_leve_le_drapeau(self, lexique):
        mod, langue = lexique
        for m in self.ORDRES[langue]:
            assert mod._REASON_REGEX.search(m), m

    def test_une_demande_de_synthese_leve_le_drapeau(self, lexique):
        """Sans ça, « compare ces deux offres » arrivant en `memory` ne réfléchit pas."""
        mod, langue = lexique
        for m in self.SYNTHESES[langue]:
            assert mod._REASON_REGEX.search(m), m

    def test_une_restitution_ne_leve_pas_le_drapeau(self, lexique):
        """Le garde qui compte : ces motifs s'appliquent à TOUT le trafic, donc un motif
        trop large paie de la réflexion sur des questions factuelles."""
        mod, langue = lexique
        for m in self.RESTITUTIONS[langue]:
            assert not mod._REASON_REGEX.search(m), m

    def test_les_deux_familles_existent_dans_les_deux_langues(self):
        """Parité d'intention, pas de formulation : un jeu qui n'aurait que les ordres
        explicites rendrait l'instance sourde aux demandes de synthèse."""
        from llm import lexique_en, lexique_fr

        for mod, langue in ((lexique_fr, "fr"), (lexique_en, "en")):
            assert mod._REASON_REGEX.search(self.ORDRES[langue][0])
            assert mod._REASON_REGEX.search(self.SYNTHESES[langue][0])


class TestEcritureCalendrier:
    """Reconnaître une DEMANDE d'écriture, pas une mention d'agenda.

    Ce motif est évalué sur chaque message avant tout routage, donc ses deux moitiés sont
    des garde-fous symétriques : sans le verbe d'ordre, une affirmation (« j'ai un
    rendez-vous demain ») partirait en création d'événement ; sans la tolérance aux mots
    intercalés, un pronom ou un titre entre guillemets suffit à faire manquer la demande,
    qui repart alors en conversation ordinaire.
    """

    DEMANDES = {
        "fr": ["ajoute un rendez-vous jeudi 14h",
               "mets moi un rendez-vous vendredi à 16h",
               'ajoute "week end Saint Raymond" a mon agenda du 15 mai',
               "planifie une réunion lundi matin",
               "bloque un créneau mardi après-midi"],
        "en": ["add an appointment on thursday at 2pm",
               "put me down for a meeting friday at 4",
               'add "family weekend" to my calendar on may 15th',
               "schedule a meeting monday morning",
               "block a slot tuesday afternoon"],
    }
    NON_DEMANDES = {
        "fr": ["j'ai un rendez-vous demain",
               "mon rendez-vous de jeudi est annulé",
               "regarde mon agenda de la semaine",
               "c'est quoi mon planning jusqu'à vendredi ?",
               "la réunion s'est bien passée"],
        "en": ["i have an appointment tomorrow",
               "my thursday meeting is cancelled",
               "check my calendar for the week",
               "what is my schedule until friday?",
               "the meeting went well"],
    }

    @pytest.fixture(params=["fr", "en"])
    def lexique(self, request):
        return importlib.import_module(f"llm.lexique_{request.param}"), request.param

    def test_une_demande_est_reconnue(self, lexique):
        mod, langue = lexique
        for m in self.DEMANDES[langue]:
            assert mod._CALENDAR_WRITE_RE.search(m), m

    def test_une_mention_nest_pas_une_demande(self, lexique):
        mod, langue = lexique
        for m in self.NON_DEMANDES[langue]:
            assert not mod._CALENDAR_WRITE_RE.search(m), m

    def test_les_mots_intercales_ne_rompent_pas_la_detection(self, lexique):
        """La cause des ratés constatés : le test de sous-chaîne contiguë."""
        mod, langue = lexique
        intercale = {"fr": "mets moi donc un rendez-vous demain",
                     "en": "put me down for an appointment tomorrow"}[langue]
        assert mod._CALENDAR_WRITE_RE.search(intercale)

    def test_la_detection_suit_la_langue_de_linstance(self):
        """`is_calendar_write` est appelée quelle que soit JARVIS_LANG : sans jeu anglais,
        l'écriture au calendrier est injoignable sur une instance EN."""
        from llm import lexique_en, lexique_fr

        assert lexique_fr._CALENDAR_WRITE_RE.search("ajoute un rendez-vous jeudi")
        assert lexique_en._CALENDAR_WRITE_RE.search("add an appointment on thursday")
