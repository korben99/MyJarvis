"""
PROJECT JARVIS v10
Function calling — traduction entre le format OpenAI et le format natif de Qwen3.6
===================================================================================

Deux conversions, dans les deux sens :

  sortie modèle → OpenAI   parse_tool_calls()
  entrée OpenAI → template normalise_messages_for_template()

Le format natif est imposé par le template de chat (models/templates/qwen36_ninja.jinja,
lignes 50-58 pour la consigne, 110-133 pour le rendu des tours passés) :

    <tool_call>
    <function=read_file>
    <parameter=path>
    src/main.py
    </parameter>
    </function>
    </tool_call>

Ce n'est PAS du JSON. L'ancien proxy Anthropic (supprimé) échouait précisément là : il
décrivait les outils dans le prompt à la manière d'Anthropic et espérait que Qwen imite
du `<tool_use>` JSON, au lieu de passer `tools` au template et de laisser le modèle
utiliser le format sur lequel il a été entraîné.
"""

import json
import re
import uuid

from helpers import get_logger

logger = get_logger("jarvis-tools")

# Un bloc par appel. Le modèle peut en émettre plusieurs à la suite.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _param_types(tool_schema: dict) -> dict[str, str]:
    """Extrait {nom_paramètre: type JSON} du schéma d'un outil, au format OpenAI."""
    function = tool_schema.get("function") or tool_schema
    properties = (function.get("parameters") or {}).get("properties") or {}
    return {
        name: (spec or {}).get("type", "string")
        for name, spec in properties.items()
        if isinstance(spec, dict) or spec is None
    }


# Le modèle n'écrit pas toujours du JSON strict pour les booléens : "True" (majuscule,
# style Python) est fréquent, "yes"/"on" apparaissent aussi. Observé sur un
# paramètre boolean, où json.loads échouait et laissait passer la chaîne "True" — ce qui
# fait échouer la validation de schéma côté client.
_TRUE_LITERALS = {"true", "yes", "on", "1"}
_FALSE_LITERALS = {"false", "no", "off", "0"}


def _coerce(raw: str, json_type: str, function_name: str, param_name: str):
    """Rend à une valeur son type déclaré.

    Les <parameter> arrivent en texte brut. Le client (AI SDK côté OpenCode) valide les
    arguments contre le schéma JSON de l'outil : un "42" là où un entier est attendu fait
    échouer l'appel. Le template sérialise les non-chaînes avec `tojson` (jinja ligne 131),
    donc json.loads est la réciproque *théorique* — mais le modèle ne respecte pas toujours
    la syntaxe JSON, d'où les replis ci-dessous.

    En dernier recours on renvoie la chaîne : mieux vaut une validation client qui échoue
    avec un message clair qu'une valeur inventée ici.
    """
    value = raw.strip()
    if json_type == "string":
        return value

    if json_type == "boolean":
        if value.lower() in _TRUE_LITERALS:
            return True
        if value.lower() in _FALSE_LITERALS:
            return False

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    else:
        # Chaîne JSON là où un scalaire est attendu ('"42"' pour un integer) : on repart du
        # contenu déballé, sinon le repli numérique retenterait sur la version guillemetée.
        if isinstance(parsed, str) and json_type != "string":
            value = parsed.strip()
            if json_type == "boolean":
                if value.lower() in _TRUE_LITERALS:
                    return True
                if value.lower() in _FALSE_LITERALS:
                    return False
        # json.loads réussit parfois en donnant le mauvais type : "42" pour un integer
        # est du JSON valide, mais c'est une chaîne. On corrige avant de rendre.
        if json_type == "integer" and isinstance(parsed, bool):
            pass  # bool est un int en Python — ne pas le laisser passer pour un entier
        elif json_type in ("integer", "number") and isinstance(parsed, (int, float)):
            return int(parsed) if json_type == "integer" else float(parsed)
        elif json_type == "array" and isinstance(parsed, list):
            return parsed
        elif json_type == "object" and isinstance(parsed, dict):
            return parsed
        elif json_type == "boolean" and isinstance(parsed, bool):
            return parsed

    if json_type == "integer":
        try:
            return int(value)
        except ValueError:
            pass
    elif json_type == "number":
        try:
            return float(value)
        except ValueError:
            pass

    logger.warning(
        "tool_calls: %s.%s attendu en %s, reçu %r — laissé en chaîne",
        function_name, param_name, json_type, value[:80],
    )
    return value


def _resolve_name(name: str, declared: dict, by_lower: dict) -> str:
    """Ramène un nom d'outil à l'orthographe exacte déclarée par le client.

    Le modèle capitalise parfois : observé, `<function=Bash>` alors qu'OpenCode
    déclare `bash`. Le client valide le nom au caractère près et rejette l'appel — l'agent
    perd son tour pour une majuscule. On ne corrige QUE la casse : un nom réellement inventé
    doit rester visible plutôt que d'être rapproché de force du plus ressemblant.
    """
    if not declared or name in declared:
        return name
    corrected = by_lower.get(name.lower())
    if corrected:
        logger.info("tool_calls: '%s' → '%s' (casse corrigée)", name, corrected)
        return corrected
    return name


def parse_tool_calls(text: str, tools: list | None) -> tuple[str, list[dict]]:
    """Sépare le texte des appels d'outil dans la sortie brute du modèle.

    Retourne (texte_restant, tool_calls au format OpenAI). Le texte restant est ce que le
    modèle a écrit hors des blocs — le template autorise du raisonnement en langage
    naturel AVANT l'appel (jinja ligne 58), il faut donc le conserver.

    tool_calls vide → aucun appel détecté, le texte est inchangé.
    """
    if not text or "<tool_call>" not in text:
        return text, []

    types_by_function = {}
    for tool in tools or []:
        function = tool.get("function") or tool
        if function.get("name"):
            types_by_function[function["name"]] = _param_types(tool)
    # Index de repli sur la casse — voir _resolve_name.
    names_by_lower = {name.lower(): name for name in types_by_function}

    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(text):
        function_name = _resolve_name(match.group(1).strip(), types_by_function, names_by_lower)
        param_types = types_by_function.get(function_name, {})

        arguments = {}
        for param_match in _PARAMETER_RE.finditer(match.group(2)):
            param_name = param_match.group(1).strip()
            arguments[param_name] = _coerce(
                param_match.group(2),
                param_types.get(param_name, "string"),
                function_name,
                param_name,
            )

        if function_name not in types_by_function:
            # Outil non déclaré : on le remonte quand même, le client tranchera. Une
            # hallucination de nom d'outil doit être visible, pas silencieusement ignorée.
            logger.warning(
                "tool_calls: le modèle a appelé '%s', absent des outils déclarés",
                function_name,
            )

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    remaining_text = _TOOL_CALL_RE.sub("", text).strip()
    return remaining_text, tool_calls


def normalise_messages_for_template(messages: list[dict]) -> list[dict]:
    """Adapte des messages OpenAI à ce que le template sait rendre.

    Un seul écart, mais il est bloquant : OpenAI transporte `arguments` en CHAÎNE JSON,
    alors que le template itère `tool_call.arguments|items` (jinja ligne 130) et attend
    donc un dict. Sans cette conversion, le second tour d'un agent — celui qui renvoie
    l'appel précédent dans l'historique — produit un prompt corrompu.

    Le reste (role "tool", content null) est déjà géré par le template.
    """
    normalised = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            normalised.append(message)
            continue

        converted = []
        for call in tool_calls:
            call = dict(call)
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments) if arguments.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "tool_calls: arguments illisibles pour '%s' — dict vide",
                        function.get("name"),
                    )
                    function["arguments"] = {}
            call["function"] = function
            converted.append(call)

        message = dict(message)
        message["tool_calls"] = converted
        normalised.append(message)
    return normalised
