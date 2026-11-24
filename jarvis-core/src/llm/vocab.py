"""
Tête de sortie réduite à un sous-vocabulaire (FR, EN, langues latines, code, emoji).

Désactivée par défaut. S'active en posant dans `.env` :

    OUTPUT_VOCAB=/opt/jarvis/models/vocab/qwen36_fr_en.json

Le fichier liste les ids gardés ; il est produit par `RESEARCH/BENCH-27B/vocab_fr_en.py`
pour un tokenizer donné. Sur un MoE à ~3 Md de paramètres actifs, la tête de sortie
(vocabulaire × dimension cachée) pèse environ 17 % des octets lus par token : n'en garder
que 57 % des lignes accélère le décodage d'environ 4 %.

Les logits sont recalculés sur le sous-vocabulaire puis replacés dans un vecteur de taille
complète, rempli de −∞ ailleurs. Sampler, processeurs de logits, jetons d'arrêt et cache
de prompts voient donc une sortie de forme inchangée. Un texte hors sous-vocabulaire reste
écrivable : les 256 tokens d'octets et les fragments multi-octets sont toujours gardés,
seul le découpage s'allonge.

L'installation ne lève jamais : un fichier absent, construit pour un autre tokenizer ou
privé d'un token ajouté (`</think>`, fin de tour, appel d'outil) laisse la tête complète
en place et le journalise.
"""

import json
import os

from helpers import get_logger

logger = get_logger("jarvis-vocab")

OUTPUT_VOCAB = os.getenv("OUTPUT_VOCAB", "").strip()


def est_tete_reduite(module) -> bool:
    """Vrai pour une tête posée par install(). Ses paramètres n'ont pas la forme de la tête
    d'origine : toute greffe de poids nom à nom doit la traiter à part."""
    return bool(getattr(module, "_tete_reduite", False))


def _ids_valides(chemin: str, tokenizer, vocab_tete: int):
    """Ids du fichier, ou None si le fichier ne correspond pas au modèle chargé."""
    if not os.path.isfile(chemin):
        logger.warning("Tête réduite : fichier introuvable (%s) — tête complète", chemin)
        return None
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    ids = sorted({int(i) for i in data.get("ids", [])})
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    # Un même nombre de tokens ne garantit pas le même tokenizer, mais une différence
    # garantit un autre : les ids pointeraient alors vers d'autres tokens.
    if data.get("vocab") != len(hf):
        logger.warning("Tête réduite : %s construit pour un vocabulaire de %s tokens, "
                       "le tokenizer en a %d — tête complète", os.path.basename(chemin),
                       data.get("vocab"), len(hf))
        return None
    if not ids or ids[-1] >= vocab_tete:
        logger.warning("Tête réduite : ids hors de la tête (%d lignes) — tête complète", vocab_tete)
        return None
    # Les tokens ajoutés portent la structure (fin de tour, réflexion, appels d'outil) :
    # en retirer un rendrait un arrêt ou une balise impossible à émettre.
    ajoutes = {int(i) for i in getattr(hf, "added_tokens_decoder", {})}
    manquants = ajoutes - set(ids)
    if manquants:
        logger.warning("Tête réduite : %d tokens ajoutés absents de %s — tête complète",
                       len(manquants), os.path.basename(chemin))
        return None
    return ids


def install(model, tokenizer, model_path: str) -> bool:
    """Remplace la tête de sortie par sa version réduite. Retourne True si installée."""
    if not OUTPUT_VOCAB:
        return False
    try:
        import mlx.core as mx
        import mlx.nn as nn

        lm = getattr(model, "language_model", model)
        tete = getattr(lm, "lm_head", None)
        # Une tête liée à l'embedding n'existe pas comme module séparé : la réduire
        # imposerait de toucher l'embedding, qui doit rester complet en entrée.
        if not isinstance(tete, nn.QuantizedLinear):
            logger.warning("Tête réduite : pas de tête quantifiée séparée sur %s — ignoré",
                           model_path.split("/")[-1])
            return False
        ids = _ids_valides(OUTPUT_VOCAB, tokenizer, tete.weight.shape[0])
        if ids is None:
            return False

        class TeteReduite(nn.Module):
            _tete_reduite = True

            def __init__(self):
                super().__init__()
                idx = mx.array(ids, dtype=mx.uint32)
                self.weight = tete.weight[idx]
                self.scales = tete.scales[idx]
                self.biases = tete.biases[idx]
                self.idx = idx.astype(mx.int32)
                self.vocab = tete.weight.shape[0]
                self.group_size, self.bits = tete.group_size, tete.bits
                self.mode = getattr(tete, "mode", "affine")
                self.freeze()

            def __call__(self, x):
                petit = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                            transpose=True, group_size=self.group_size,
                                            bits=self.bits, mode=self.mode)
                plein = mx.full((*petit.shape[:-1], self.vocab), -mx.inf, dtype=petit.dtype)
                idx = mx.broadcast_to(self.idx, petit.shape)
                return mx.put_along_axis(plein, idx, petit, axis=-1)

        reduite = TeteReduite()
        mx.eval(reduite.parameters())
        lm.lm_head = reduite
        logger.info("Tête de sortie réduite sur %s : %d/%d tokens (%s)",
                    model_path.split("/")[-1], len(ids), tete.weight.shape[0],
                    os.path.basename(OUTPUT_VOCAB))
        return True
    except Exception as exc:
        logger.warning("Tête réduite indisponible (%s) — tête complète", exc)
        return False
