#!/usr/bin/env python3
"""ARCHIVÉ — banc bâti sur un modèle causal faux, conservé pour mémoire seulement.

Ce banc fait varier les pénalités d'échantillonnage pour reproduire les boucles. Ce n'était
pas la cause : jusqu'à MLX 0.32.0, une fonction décorée
`@mx.compile(inputs=mx.random.state, outputs=mx.random.state)` ne propage pas l'état du
générateur hors du thread principal. Les samplers de mlx_lm sont tous décorés ainsi et
toute génération tourne dans un thread de génération, donc le tirage rendait le même
quantile à chaque token — un greedy déguisé, qui boucle sur les sorties longues. Les
pénalités ne faisaient que relever la barrière d'entrée ; le banc mesurait l'épaisseur du
pansement. Correctif : plancher `mlx>=0.32.2` plus une graine par génération
(`llm/local.py::_setup_gen`).

L'en-tête ci-dessous affirme qu'un rejeu du prompt exact d'une boucle avérée n'en a pas
reproduit. C'est faux, et c'est ce qui a égaré l'enquête : le rejeu donne au contraire la
MÊME sortie au caractère près, puisque la génération était déterministe. C'est d'ailleurs
ce rejeu qui a fini par désigner la cause.

Banc de reproduction des boucles de génération, et évaluation des correctifs.

**Jarvis doit être arrêté** (`jarvis-stop`) : le banc charge le modèle dans son propre
processus et lui parle en direct, sans passer par l'API. C'est ce qui permet de faire varier
les pénalités, que `/v1/raw` n'expose pas — et ce qui évite de monopoliser le GPU de
production, ce qui avait fini par bloquer le serveur le 09/09.

Deux cas réels, rejoués depuis logs/prompts.log :
  A  vente M4        — 2 messages, motif court « Q4–Q1 prochain observable » ×45
  B  élastiques      — 4 messages, motif énuméré « Apparence : Une grande culotte… » ×12

Chaque cas est rejoué sous plusieurs bras. Le bras `0-base` est la production telle qu'elle
est configurée ; les autres modifient UNE chose, prompt ou pénalités, jamais les deux.

Usage :
  jarvis-stop
  ./venv/bin/python scripts/bench_boucles.py                       # tous les bras, 3 essais
  ./venv/bin/python scripts/bench_boucles.py --bras 0-base 2a-freq64 --essais 5
  ./venv/bin/python scripts/bench_boucles.py --cas A --essais 1    # rodage rapide
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace

sys.path.insert(0, "/opt/jarvis/jarvis-core/src")

JOURNAL = "/opt/jarvis/logs/prompts.log"


def _jarvis_tourne() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8000/docs", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


if _jarvis_tourne():
    sys.exit("Jarvis tourne — le banc a besoin du GPU seul. Lancer `jarvis-stop` d'abord.")

os.environ.setdefault("LLM_DEBUG_PROMPTS", "")  # le banc journalise lui-même

import config as C  # noqa: E402
import llm.local as L  # noqa: E402

# ── Les deux cas de référence ────────────────────────────────────────────────

CAS = {
    "A": {"horo": "2026-09-09 14:56:42", "titre": "vente M4 — motif court (7 tok)"},
    "B": {"horo": "2026-09-02 13:51:42", "titre": "élastiques — motif énuméré"},
}


def charger_cas(cle: str) -> list[dict]:
    """Rejoue le prompt rendu en messages : le template le re-rend à l'identique."""
    txt = open(JOURNAL, encoding="utf-8", errors="replace").read()
    blocs = [x for x in txt.split("=" * 80) if CAS[cle]["horo"] in x]
    if not blocs:
        sys.exit(f"cas {cle} introuvable dans {JOURNAL} ({CAS[cle]['horo']})")
    rendu = blocs[0].split("--- PROMPT ---", 1)[1].split("--- RESPONSE (raw) ---", 1)[0]
    msgs = [
        {"role": r, "content": c.strip()}
        for r, c in re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", rendu, re.S)
    ]
    if not msgs or msgs[0]["role"] != "system":
        sys.exit(f"cas {cle} : messages illisibles")
    return msgs


# ── Les bras ─────────────────────────────────────────────────────────────────
# `systeme`  : transformation du message système (cas 1 — ajustement du prompt)
# `profil`   : champs du _ModelProfile à remplacer (cas 2 — pénalités)
# Un bras ne touche qu'à un seul des deux, pour que l'écart soit attribuable.

CLAUSE_MD = "Réponds en français, sans markdown — sauf si JSON ou code explicitement demandé. "
BRIEVETE = (
    "Réponds en français. Va au fait : quelques phrases suffisent, et tu t'arrêtes quand tu "
    "as répondu. Rien ne t'oblige à remplir. "
)

BRAS: dict[str, dict] = {
    # Les conditions EXACTES de l'incident du 09/09 14:56, seul bras qui puisse servir de
    # référence : le serveur tournait alors avec la fenêtre de fréquence partagée à 64 et
    # un plafond de 10000. C'est celui-ci qu'il faut savoir faire boucler avant d'évaluer
    # quoi que ce soit d'autre.
    "0-incident": {"profil": {"frequency_context_size": 64}, "plafond": 10000},

    # la production d'aujourd'hui : fenêtre 256, plafond dérivé du budget de réflexion
    "0-base": {},

    # ── Cas 1 — ajustement du prompt ─────────────────────────────────────────
    "1a-sans-markdown": {"systeme": lambda s: s.replace(CLAUSE_MD, "Réponds en français. ")},
    "1b-brievete": {"systeme": lambda s: s.replace(CLAUSE_MD, BRIEVETE)},

    # ── Cas 2 — valeurs de pénalité ──────────────────────────────────────────
    # repetition_penalty reste à 1.1, plafond recommandé par Qwen : on ne touche qu'aux
    # deux pénalités additives, qui ne viennent pas de ses préconisations.
    "2a-freq64": {"profil": {"frequency_context_size": 64}},      # avant correctif
    "2b-freq-fort": {"profil": {"frequency_penalty": 0.30}},
    "2c-presence-fort": {"profil": {"presence_penalty": 2.5}},
}


def profil_du_bras(bras: str):
    """Remplace _model_profile par une version patchée, le temps d'un essai."""
    champs = BRAS[bras].get("profil")
    base = L._model_profile(C.PRIMARY_MODEL)
    return base if not champs else replace(base, **champs)


# ── Mesure ───────────────────────────────────────────────────────────────────

def periode_max(txt: str, pas: int = 40) -> int:
    seg = [txt[i : i + pas] for i in range(0, max(0, len(txt) - pas), pas)]
    return max((seg.count(s) for s in set(seg)), default=0)


def motif_dominant(txt: str, pas: int = 40) -> str:
    seg = [txt[i : i + pas] for i in range(0, max(0, len(txt) - pas), pas)]
    return max(set(seg), key=seg.count) if seg else ""


async def _streamer(msgs: list[dict], plafond: int) -> str:
    """Passe par stream_local, le chemin réel du chat — pas par _generate_sync.

    Les deux ne sont pas interchangeables : `_generate_sync` remplace toute la réponse par
    un repli quand la réflexion est tronquée, ce que le streaming ne fait pas.
    """
    morceaux: list[str] = []
    async for c in L.stream_local(
        msgs, C.PRIMARY_MODEL,
        temperature=None,
        max_tokens=plafond,
        no_think=False,
        thinking_budget=C.THINKING_BUDGET_MEDIUM,
        skip_debug_log=True,
    ):
        morceaux.append(c)
    return "".join(morceaux)


def essai(messages: list[dict], bras: str, tok) -> dict:
    msgs = [dict(m) for m in messages]
    if "systeme" in BRAS[bras]:
        msgs[0]["content"] = BRAS[bras]["systeme"](msgs[0]["content"])

    plafond = BRAS[bras].get("plafond", PLAFOND)
    profil = profil_du_bras(bras)
    vrai_profil = L._model_profile
    L._model_profile = lambda _p: profil        # patché le temps de la génération
    try:
        t0 = time.monotonic()
        brut = asyncio.run(_streamer(msgs, plafond))
        dt = time.monotonic() - t0
    finally:
        L._model_profile = vrai_profil

    # stream_local rend les morceaux BRUTS : le bloc <think> est filtré par chat.py, pas
    # ici. On le sépare pour mesurer séparément là où la dérive s'installe.
    i = brut.find("</think>")
    pensee, reponse = (brut[:i], brut[i + 8 :]) if i >= 0 else ("", brut)
    n_p, n_r = len(tok.encode(pensee)) if pensee else 0, len(tok.encode(reponse))
    rep_p, rep_r = periode_max(pensee), periode_max(reponse)
    return {
        "think": n_p,
        "tokens": n_r,
        "repet": max(rep_p, rep_r),
        "repet_think": rep_p,
        "boucle": max(rep_p, rep_r) >= 8,
        "ferme": i >= 0,
        "sature": (n_p + n_r) >= plafond * 0.95,
        "dt": dt,
        "motif": motif_dominant(reponse if rep_r >= rep_p else pensee)[:46],
        "fin": reponse[-70:].replace("\n", " "),
    }


PLAFOND = C.MAX_TOKENS_THINK_MEDIUM  # même borne qu'en production après correctif


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cas", nargs="*", default=list(CAS), choices=list(CAS))
    ap.add_argument("--bras", nargs="*", default=list(BRAS), choices=list(BRAS))
    ap.add_argument("--essais", type=int, default=3)
    ap.add_argument("--sortie", default="/opt/jarvis/logs/bench_boucles.jsonl")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.PRIMARY_MODEL)

    print(f"plafond {PLAFOND} tok (réflexion {C.THINKING_BUDGET_MEDIUM} + marge)")
    print(f"{len(a.cas)} cas × {len(a.bras)} bras × {a.essais} essais\n")

    sortie = open(a.sortie, "a", encoding="utf-8")
    resultats: dict[tuple[str, str], list[dict]] = {}

    # Bras entrelacés : grouper laisserait une dérive d'état passer pour un effet de bras.
    for n_essai in range(1, a.essais + 1):
        for cle in a.cas:
            messages = charger_cas(cle)
            for bras in a.bras:
                r = essai(messages, bras, tok)
                resultats.setdefault((cle, bras), []).append(r)
                marque = "BOUCLE" if r["boucle"] else ("saturé" if r["sature"] else "ok")
                if not r["ferme"]:
                    marque += " /think-non-fermé"
                print(
                    f"  {n_essai} | cas {cle} | {bras:18s} think {r['think']:5d} "
                    f"rép {r['tokens']:5d} | ×{r['repet']:2d} | {r['dt']:5.1f}s | {marque}"
                )
                if r["boucle"]:
                    print(f"        motif : {r['motif']!r}")
                sortie.write(json.dumps({"cas": cle, "bras": bras, **r}, ensure_ascii=False) + "\n")
                sortie.flush()

    print("\n" + "=" * 78)
    print(f"{'cas':4s} {'bras':18s} {'boucles':>8s} {'tok médian':>11s} {'répét. max':>11s}")
    for (cle, bras), v in sorted(resultats.items()):
        print(
            f"{cle:4s} {bras:18s} {sum(x['boucle'] for x in v):3d}/{len(v):<4d} "
            f"{int(statistics.median(x['tokens'] for x in v)):11d} "
            f"{max(x['repet'] for x in v):11d}"
        )
    print(f"\ndétail : {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
