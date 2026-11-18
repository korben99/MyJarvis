"""Function calling : typage des paramètres lus dans le schéma des outils.

Le piège gardé ici : un paramètre facultatif est souvent déclaré sans `type` direct
(`anyOf` avec `null`, ou `type` en liste). Lu comme une chaîne, un tableau part en texte
JSON et la validation du client refuse l'appel (« expected array »), alors que le modèle
avait écrit une valeur correcte.
"""

import json

import pytest

from tool_calls import _param_types, parse_tool_calls


def _outil(proprietes: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "memory",
            "parameters": {"type": "object", "properties": proprietes},
        },
    }


def _arguments(sortie: str, outils: list) -> dict:
    _, appels = parse_tool_calls(sortie, outils)
    assert len(appels) == 1
    return json.loads(appels[0]["function"]["arguments"])


class TestTypeDeclare:

    @pytest.mark.parametrize(
        "spec, attendu",
        [
            ({"type": "integer"}, "integer"),
            ({"anyOf": [{"type": "array"}, {"type": "null"}]}, "array"),
            ({"anyOf": [{"type": "null"}, {"type": "object"}]}, "object"),
            ({"oneOf": [{"type": "boolean"}, {"type": "null"}]}, "boolean"),
            ({"type": ["number", "null"]}, "number"),
            ({"type": ["null", "integer"]}, "integer"),
            ({"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "null"}]}, "string"),
            ({"description": "sans type"}, "string"),
            (None, "string"),
        ],
    )
    def test_type_retenu(self, spec, attendu):
        assert _param_types(_outil({"p": spec}))["p"] == attendu


class TestParseToolCallsNullable:

    def test_tableau_nullable_rendu_en_tableau(self):
        outils = [_outil({
            "target": {"type": "string"},
            "operations": {"anyOf": [
                {"type": "array", "items": {"type": "object"}}, {"type": "null"},
            ]},
        })]
        sortie = (
            "<tool_call>\n<function=memory>\n"
            "<parameter=target>\ndevice\n</parameter>\n"
            '<parameter=operations>\n[{"action": "add", "content": "pare-feu du salon"}]\n'
            "</parameter>\n</function>\n</tool_call>"
        )
        arguments = _arguments(sortie, outils)
        assert arguments["target"] == "device"
        assert arguments["operations"] == [{"action": "add", "content": "pare-feu du salon"}]

    def test_entier_en_liste_de_types(self):
        outils = [_outil({"limit": {"type": ["integer", "null"]}})]
        sortie = (
            "<tool_call>\n<function=memory>\n"
            "<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>"
        )
        assert _arguments(sortie, outils)["limit"] == 50

    def test_chaine_nullable_reste_chaine(self):
        # Un identifiant numérique déclaré chaîne ne doit pas devenir un entier.
        outils = [_outil({"device": {"anyOf": [{"type": "string"}, {"type": "null"}]}})]
        sortie = (
            "<tool_call>\n<function=memory>\n"
            "<parameter=device>\n42\n</parameter>\n</function>\n</tool_call>"
        )
        assert _arguments(sortie, outils)["device"] == "42"
