"""Le bloc d'état que l'agent d'autocodage reçoit avant de relire son propre code.

Séparé de `build_dynamic_prefix` (pipeline de chat) à dessein : celui-ci assemble neuf
blocs dont plusieurs portent sur l'UTILISATEUR (profil, relation, projets), sans objet
quand Jarvis relit son code. On ne garde ici que ce qui le concerne, lui.

Quatre blocs, et rien d'autre :

    <etat_systeme>            son exposition mesurée (vitals)
    <souvenirs_jarvis>        ce que des échanges passés lui ont laissé
    <etat_emotionnel_jarvis>  son humeur du moment
    <introspection_jarvis>    ce qu'il sait de sa propre conduite

`IDENTITY` décrit déjà ces balises ; sans ce préfixe, l'agent lisait une notice à cadrans
vides. C'est ce que ce module remplit.
"""

from config import SELF_MEMORY_PERMANENT_N, SELF_MEMORY_SIMILAR_N
from helpers import get_logger

logger = get_logger("jarvis-autocode")


def build_autocode_prefix() -> str:
    """Les quatre blocs d'état de Jarvis, concaténés. Chaîne vide si aucun n'est rempli.

    Chaque source est isolée : une sonde indisponible retire son bloc, elle ne prive pas
    l'agent des trois autres. Aucune donnée utilisateur n'entre ici.
    """
    parts: list[str] = []

    try:
        from vitals import render_prompt_block

        if (bloc := render_prompt_block()):
            parts.append(bloc)
    except Exception as exc:
        logger.debug("autocode: <etat_systeme> non rendu (%s)", type(exc).__name__)

    try:
        from memory import recall_self_memories

        souvenirs = recall_self_memories(SELF_MEMORY_PERMANENT_N, SELF_MEMORY_SIMILAR_N)
        if souvenirs:
            parts.append(
                "<souvenirs_jarvis>\n"
                + "\n".join(f"- {s['text']}" for s in souvenirs)
                + "\n</souvenirs_jarvis>"
            )
    except Exception as exc:
        logger.debug("autocode: <souvenirs_jarvis> non rendu (%s)", type(exc).__name__)

    try:
        import emotional_state

        if (lignes := emotional_state.render_prompt_lines()):
            parts.append(
                "<etat_emotionnel_jarvis>\n"
                + "\n".join(f"- {l}" for l in lignes)
                + "\n</etat_emotionnel_jarvis>"
            )
    except Exception as exc:
        logger.debug("autocode: <etat_emotionnel_jarvis> non rendu (%s)", type(exc).__name__)

    try:
        from memory import get_self_memory

        introspection = [
            t for t in (get_self_memory().get("self_introspection") or {}).values() if t
        ]
        if introspection:
            parts.append(
                "<introspection_jarvis>\n"
                + "\n".join(f"- {t}" for t in introspection)
                + "\n</introspection_jarvis>"
            )
    except Exception as exc:
        logger.debug("autocode: <introspection_jarvis> non rendu (%s)", type(exc).__name__)

    return "\n\n".join(parts)
