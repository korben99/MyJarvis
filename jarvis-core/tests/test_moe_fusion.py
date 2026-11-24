"""Noyaux MoE fusionnés : activation et repli (llm/moe_fusion.py).

mlx et unsloth-zoo ne sont pas exercés ici : on vérifie que le module reste inerte sans
la variable, et qu'une erreur à l'installation laisse le chemin d'origine sans lever.
L'identité des sorties est vérifiée sur le vrai modèle, hors suite.
"""

import sys
import types

import llm.moe_fusion as moe


class _Modele:
    def __init__(self):
        self.eval_appele = False

    def eval(self):
        self.eval_appele = True

    def named_modules(self):
        return []


def test_desactive_sans_variable(monkeypatch):
    monkeypatch.setattr(moe, "MOE_FUSED", False)
    m = _Modele()
    assert moe.install(m, "m") is False
    assert m.eval_appele is False


def test_erreur_d_installation_ne_leve_pas(monkeypatch):
    monkeypatch.setattr(moe, "MOE_FUSED", True)
    monkeypatch.setattr(moe, "_scopes", {})
    faux = types.ModuleType("unsloth_zoo.mlx.inference")

    def casse(model):
        raise RuntimeError("noyau indisponible")

    faux.fused_moe_gate_up = faux.fused_moe_router = faux.fused_moe_routed_experts = casse
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx", types.ModuleType("unsloth_zoo.mlx"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.inference", faux)
    assert moe.install(_Modele(), "m") is False
    assert moe._scopes == {}


def test_aucun_bloc_eligible_referme_les_scopes(monkeypatch):
    from contextlib import contextmanager

    monkeypatch.setattr(moe, "MOE_FUSED", True)
    monkeypatch.setattr(moe, "_scopes", {})
    fermes = []

    @contextmanager
    def scope(model):
        yield
        fermes.append(1)

    faux = types.ModuleType("unsloth_zoo.mlx.inference")
    faux.fused_moe_gate_up = faux.fused_moe_router = faux.fused_moe_routed_experts = scope
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx", types.ModuleType("unsloth_zoo.mlx"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.inference", faux)
    assert moe.install(_Modele(), "m") is False
    assert len(fermes) == 3 and moe._scopes == {}
