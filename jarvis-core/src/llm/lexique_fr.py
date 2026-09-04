"""
lexique_fr.py — Ce qui RECONNAÎT le français en entrée.
===============================================================================
Trois modules portent la langue, et la distinction compte :

    prompts_XX.py   ce qu'on ENVOIE au modèle
    textes_XX.py    ce qu'on RESTITUE à l'utilisateur
    lexique_XX.py   ce qu'on RECONNAÎT dans son message   ← ce fichier

Ici : les phrases-exemples du routeur sémantique, les acquiescements de small
talk, les déclencheurs de mode raisonnement, et les motifs d'extraction (ville,
commande RAG). Rien de tout cela n'est rendu à qui que ce soit — c'est de la
reconnaissance pure.

Le pendant anglais est `lexique_en.py`. Les deux doivent exposer les mêmes noms.

**Les exemples d'intent n'ont PAS besoin d'être traduits pour fonctionner.** Le
modèle d'embedding est multilingue : une requête anglaise contre son équivalent
français donne 0,82 à 0,98 de similarité, au-dessus du seuil de bascule de 0,74.
Une traduction améliore la marge, elle n'est pas un prérequis. Les régex et les
listes exactes, elles, ne survivent pas au changement de langue : ce sont elles
qu'il faut traduire en priorité.
"""

import re

# ── Phrases-exemples par intent (toutes en français) ─────────────────────────
#
# Règles d'édition :
#   • 8 à 15 phrases par intent — assez pour couvrir les formulations courantes
#   • Phrases courtes, naturelles, telles qu'un utilisateur les taperait
#   • Pas de doublons inter-intents (sinon l'ambiguïté monte)
#   • Pas de noms propres sauf si vraiment caractéristiques (ex: "PEA" pour portfolio)

INTENT_EXAMPLES: dict[str, list[str]] = {
    # ── Conversation / mémoire générale ──────────────────────────────────────
    "memory": [
        # questions / demandes d'aide — nécessitent le profil pour personnaliser
        "c'est quoi ton avis là-dessus ?",
        "aide-moi à rédiger un email",
        "explique-moi comment ça marche",
        "qu'est-ce que tu penses de ça ?",
        "aide-moi avec ce code",
        "qu'est-ce que tu sais sur moi ?",
        "résume ce texte",
        "donne-moi des idées pour",
        "c'est quoi la différence entre",
        # partage d'informations personnelles — profil à mettre à jour
        "je viens de décider de",
        "j'ai changé d'avis sur",
        "depuis quelque temps je",
        "j'adore ce sport",
        "je déteste travailler le soir",
        "j'ai toujours préféré",
        "je ne supporte pas le",
        # référence au passé / continuité de conversation
        "tu te souviens de",
        "comme je t'avais expliqué",
        "rappelle-toi ce projet",
        "on avait parlé de",
        "tu sais ce que j'ai fait hier",
    ],
    # ── Météo ────────────────────────────────────────────────────────────────
    "weather": [
        "quelle est la météo à",
        "quel temps fait-il ?",
        "météo à Paris demain",
        "température ce week-end",
        "météo de la semaine prochaine",
        "prévisions météo pour demain matin",
        "quel temps demain ?",
    ],
    # ── Emails / Gmail ───────────────────────────────────────────────────────
    "gmail": [
        "lit mon dernier mail",
        "lit mes mails non lus",
        "lit mes mails",
        "vérifie mes emails",
        "résume mes emails non lus",
        "recherche dans mes mails",
        "check mes mails",
        "regarde dans mes mails",
    ],
    # ── Agenda / Calendrier ──────────────────────────────────────────────────
    "calendar": [
        "mon agenda aujourd'hui",
        "qu'est-ce que j'ai de prévu ?",
        "planning cette semaine",
        "mes rendez-vous du jour",
        "agenda de la semaine",
        "qu'est-ce que j'ai prévu ce soir ?",
        "montre-moi mon calendrier",
        "regarde dans mon agenda",
    ],
    # ── Briefing matinal ─────────────────────────────────────────────────────
    "briefing": [
        "briefing du matin",
        "briefing matinal",
        "lance le briefing",
        "donne-moi le briefing",
        "le point du matin",
        "briefing s'il te plaît",
    ],
    # ── Recherche web / actualités ───────────────────────────────────────────
    "web": [
        "les news du jour",
        "cours actuel du pétrole",
        "qui a gagné le match hier",
        "cherche sur internet",
        "cherche sur le net",
        "dernières actualités",
        "recherche des infos",
        "recherche sur le net",
        "trouve-moi des informations",
        "quelles sont les news",
        "recherche en ligne",
        "le cours de l'action Engie",
        "analyse de l'action Total, sa valorisation et ses perspectives",
        "quelle est la valorisation de cette action ?",
        # Recherche d'un commerce ou d'un service DANS une ville. Sans ces formes, un
        # « à <Ville> » n'existait que dans les exemples météo : toute recherche
        # géolocalisée y était aspirée, d'autant plus fort que le corpus local contient
        # de vraies requêtes météo sur la ville de l'utilisateur.
        "trouve-moi un bon restaurant à Lyon",
        "un bon garagiste près de Bordeaux",
        "cherche une pizzeria ouverte ce soir",
        "recommande-moi un hôtel à Nantes",
    ],
    # ── Documents personnels / RAG ───────────────────────────────────────────
    # Phrases COMPLÈTES uniquement, jamais de fragments.
    #
    # Un exemple tronqué (« ma fiche sur », « dans mes documents », « RAG ») n'encode plus
    # que sa structure — déterminant + possessif — et non un sens. N'importe quelle phrase
    # courte de forme voisine s'y accroche alors : « par curiosité » et « donne moi mon
    # portefeuille » décrochaient 0.78 sur ce pool, au-dessus du seuil de routage, alors
    # qu'aucun des deux ne parle de documents. Compléter les exemples ramène ces deux-là
    # sous le seuil sans rien coûter aux vrais positifs.
    "rag": [
        "cherche dans mes documents",
        "j'ai un fichier sur ce sujet",
        "retrouve le document sur ce sujet",
        "cherche dans ma base documentaire",
        "regarde dans mes fichiers",
        "cherche ça dans mes documents",
        "lis ma fiche sur ce sujet",
        "lis un extrait de mon fichier",
        "base-toi sur mon fichier",
        "montre-moi ma fiche sur ce sujet",
        "extrait de mon document",
        "cherche dans mon RAG",
        "regarde dans le RAG",
        "consulte mon fichier",
    ],
    # ── Portefeuille boursier ────────────────────────────────────────────────
    "portfolio": [
        "mon portefeuille",
        "mon PEA",
        "mes actions",
        "performance de mes actions",
        "analyse mon portefeuille",
        "mes positions boursières",
        "comment va mon portefeuille ?",
        "j'hésite à acheter cette action, elle a sa place dans mon portefeuille ?",
        "intégrer une nouvelle action à mon portefeuille",
        "est-ce que je devrais acheter du Engie ?",
        # Formes portant un marqueur temporel. Sans elles, « aujourd'hui » ou « ce matin »
        # tirait la question vers `calendar`, dont les exemples en sont saturés
        # (« mon agenda aujourd'hui », « mes rendez-vous du jour »).
        "mes positions du jour",
        "où en est mon portefeuille ce matin",
        "l'état de mes actions aujourd'hui",
        "mon PEA a-t-il bougé aujourd'hui ?",
    ],
    # ── Questions sur l'état d'un projet ────────────────────────────────────
    # Couvre les requêtes de STATUS ("où en est", "comment avance").
    # Les mises à jour conversationnelles ("j'ai avancé sur X") routent vers
    # "memory" — la mémoire épisodique fournit le contexte, l'analyzer capture
    # l'update dans la timeline. L'injection de détail n'est pas nécessaire
    # quand l'utilisateur donne une information (il connaît son propre projet).
    "project": [
        "comment avance le projet",
        "où en est le projet",
        "état d'avancement du projet",
        "donne-moi l'avancement",
        "mets à jour le projet",
        "j'ai avancé sur le projet",
        "j'ai terminé la partie",
        "j'ai fini le projet",
        "on avance sur le projet",
        "prochaine étape du projet",
        "il reste encore à faire",
        "j'ai commencé un nouveau projet",
    ],
    # ── État interne de Jarvis ───────────────────────────────────────────────
    "self": [
        "comment vas-tu Jarvis ?",
        "Salut Jarvis, en forme ?",
        "qu'est-ce que tu fais en ce moment ?",
        "ton état interne",
        "tes dernières réflexions",
        "qu'est-ce que tu as appris récemment ?",
        "comment tu te sens ?",
        "donne-moi ton introspection",
        "tu as réfléchi à quoi récemment ?",
        "tes auto-réflexions",
        "tes réflexions",
        "montre les propositions de prompt",
        "montre les prompts en attente",
        "montre les prompts",
        "liste les propositions en attente",
        "accepte la proposition",
        "rejette la proposition",
        "montre la proposition",
        "approuve la proposition de prompt",
        "parle-moi de toi",
        "ton identité",
        # Sécurité de SA propre pile. Le possessif ("tes", "ton") est ce qui sépare ces
        # phrases d'une question de sécurité générale, qui doit rester en web/memory :
        # « tes CVE » = état interne, « la faille log4j » = information externe.
        "tu as des CVE critiques ?",
        "fais-moi un point sur tes CVE critiques",
        "tes vulnérabilités critiques",
        "où en est ton scan de vulnérabilités ?",
        "ta pile est à jour côté sécurité ?",
        "ton état vitals",
        "ton état système",
    ],
}

_REASON_EXACT = {
    "mode expert",
    "analyse approfondie",
    "analyse complète",
    "analyse détaillée",
    "réflexion approfondie",
    "réfléchis en profondeur",
    "réfléchis bien",
    "prends le temps de réfléchir",
    "prends le temps d'analyser",
    "debug complet",
}

_REASON_REGEX = re.compile(
    r"\braisonne\b|\bréfléchis\b|\bétape par étape\b|\bpas à pas\b|\ben profondeur\b",
    re.IGNORECASE,
)

# ── Small talk — acquiescements purs (≤ 50 chars, pas de ?, pas de contenu) ──
# Bypasse profil, mémoire et opinions : le LLM n'a besoin que de l'historique.
# WHITELIST conservative : uniquement mots qui n'apportent aucun fait nouveau.
# Critères bloquants : présence de "?" OU longueur > 50 chars → jamais small talk.
_SMALL_TALK_EXACT = {
    "merci",
    "merci !",
    "merci beaucoup",
    "super merci",
    "merci bien",
    "parfait",
    "c'est parfait",
    "top",
    "génial",
    "excellent",
    "nickel",
    "super",
    "très bien",
    "bien",
    "c'est bon",
    "c'est bien",
    "ok",
    "okay",
    "oki",
    "d'accord",
    "ok ok",
    "oui oui",
    "non non",
    "vas-y",
    "go",
    "continue",
    "allez",
    "fais-le",
    "fais",
    "bonne idée",
    "oui bonne idée",
    "oui c'est ça",
    "ah ok",
    "ah je vois",
    "ah d'accord",
    "ah oui",
    "je vois",
    "j'ai compris",
    "compris",
    "reçu",
    "haha",
    "lol",
    "😄",
    "👍",
    # Salutations pures (ajoutées)
    "bonjour",
    "salut",
    "salut jarvis",
    "hello",
    "hey",
    "yo",
    "coucou",
    "bonsoir",
    "hi",
    "hola",
    "re",
    "rebonjour",
}

# Briefing exact (avant l'embedding, ces formulations sont sans ambiguïté)
_BRIEFING_EXACT = {
    "briefing",
    "mon briefing",
    "briefing matinal",
    "briefing du matin",
    "lance le briefing",
    "le briefing",
    "fais le briefing",
}

# Liants minuscules dans les noms de villes françaises
_FR_CITY_LIANTS = r"(?:de|du|des|le|la|les|aux|en|sur|sous|sainte?|saint)"

# Capture une ville après préposition, en préservant les noms composés :
# La Rochelle · Aix-en-Provence · Saint-Germain-en-Laye · Boulogne-sur-Mer
_CITY_AFTER_PREP_RE = re.compile(
    r"\b(?:à|au|aux|pour|sur|vers|en)\s+"
    r"("
    r"[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][\wÀ-ÿ'\-]*"
    r"(?:[-\s]+(?:" + _FR_CITY_LIANTS + r"[-\s]+)?"
    r"[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]?[\wÀ-ÿ'\-]+){0,3}"
    r")"
)

# Faux positifs fréquents après préposition (jours, moments)
_TEMPORAL_WORDS = {
    "aujourd'hui",
    "demain",
    "hier",
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
    "matin",
    "soir",
    "midi",
}


# ── RAG query extraction ─────────────────────────────────────────────────────
# Phase 1: strip command/routing phrases (compound patterns first, then single verbs)
_RAG_CMD_RE = re.compile(
    r"(?:"
    r"base-toi sur (?:mon|ma)\s*|"
    r"lis (?:un extrait de )?(?:mon|ma)\s*|"
    r"montre(?:-moi)? (?:mon|ma)\s*|"
    r"extrait de (?:mon|ma)\s*|"
    r"consulte(?:r)? (?:mon|ma|mes)\s*|"
    r"depuis (?:le |la |mon |ma |mes )?RAG\b\s*|"
    r"dans (?:le |la |mon |ma |mes )?RAG\b\s*|"
    r"sur (?:le |la |mon |ma |mes )?RAG\b\s*|"
    r"qui est dans le RAG\b\s*|"
    r"dans mes (?:documents?|notes?|fichiers?|base documentaire)\s*|"
    r"dans (?:mon|ma) (?:document|fichier|fiche|note)\s*|"
    r"(?:mes|mon|ma) (?:documents?|notes?|fichiers?)\s*|"
    r"j'ai (?:un|une) (?:fichier|fiche|document|note) sur\s*|"
    r"retrouve(?:r)? (?:le |la |un |une )?(?:document|fichier|fiche) sur\s*|"
    r"\b(?:"
    r"cherche(?:r)?|retrouve(?:r)?|trouve(?:r)?|recherche(?:r)?|"
    r"regarde(?:r)?|consulte(?:r)?|montre(?:-moi)?|lis|lit|"
    r"extrais?|extrait"
    r")\b\s*|"
    r"\bRAG\b\s*"
    r")",
    re.IGNORECASE,
)

# Phase 2: strip leading articles/possessives left after command removal
_RAG_LEAD_NOISE_RE = re.compile(
    r"^(?:(?:mon|ma|mes|le|la|les|l'|un|une|du|de|des|d')\s+)+",
    re.IGNORECASE,
)


