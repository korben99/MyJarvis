"""Tête de sortie réduite : garde-fous d'activation (llm/vocab.py).

mlx est simulé par conftest : on exerce ici la validation du fichier de sous-vocabulaire,
qui décide seule si la tête complète reste en place. Le calcul des logits est vérifié sur
le vrai modèle, hors suite.
"""

import json
from types import SimpleNamespace

import llm.vocab as vocab


class _Tok:
    """Tokenizer minimal : longueur et tokens ajoutés, comme un tokenizer HF."""

    def __init__(self, n, ajoutes):
        self._n = n
        self.added_tokens_decoder = {i: object() for i in ajoutes}

    def __len__(self):
        return self._n


def _fichier(tmp_path, ids, n=100):
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"vocab": n, "ids": ids}))
    return str(f)


def test_desactive_sans_variable(monkeypatch):
    monkeypatch.setattr(vocab, "OUTPUT_VOCAB", "")
    assert vocab.install(object(), object(), "m") is False


def test_fichier_absent(tmp_path):
    assert vocab._ids_valides(str(tmp_path / "absent.json"), _Tok(100, []), 128) is None


def test_autre_tokenizer_refuse(tmp_path):
    chemin = _fichier(tmp_path, [0, 1, 2], n=100)
    assert vocab._ids_valides(chemin, _Tok(101, []), 128) is None


def test_ids_hors_tete_refuses(tmp_path):
    chemin = _fichier(tmp_path, [0, 1, 200])
    assert vocab._ids_valides(chemin, _Tok(100, []), 128) is None


def test_token_ajoute_manquant_refuse(tmp_path):
    # 99 joue le rôle de </think> : sans lui, la réflexion ne pourrait plus se fermer
    chemin = _fichier(tmp_path, [0, 1, 2])
    assert vocab._ids_valides(chemin, _Tok(100, [99]), 128) is None


def test_valide_trie_et_dedoublonne(tmp_path):
    chemin = _fichier(tmp_path, [5, 99, 1, 5])
    assert vocab._ids_valides(chemin, _Tok(100, [99]), 128) == [1, 5, 99]


def test_tokenizer_enveloppe(tmp_path):
    # TokenizerWrapper de mlx_lm expose le tokenizer HF sous _tokenizer
    chemin = _fichier(tmp_path, [1, 99])
    enveloppe = SimpleNamespace(_tokenizer=_Tok(100, [99]))
    assert vocab._ids_valides(chemin, enveloppe, 128) == [1, 99]


def test_marqueur_tete_reduite():
    # La greffe vision (llm/local.py) s'appuie sur ce marqueur pour traiter la tête à part
    class Tete:
        _tete_reduite = True

    assert vocab.est_tete_reduite(Tete()) is True
    assert vocab.est_tete_reduite(object()) is False
    assert vocab.est_tete_reduite(None) is False
