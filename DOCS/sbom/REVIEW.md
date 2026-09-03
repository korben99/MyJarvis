# SBOM & dependency review — 2026-07-29

> **Ce document est l'archive d'un audit ponctuel.** Il n'est pas tenu à jour.
> La SBOM **vivante** est `sbom-venv.json`, réécrite à chaque scan quotidien par
> `cve.py::_persister_sbom` : c'est elle qui décrit le venv réel, et un diff dessus signale
> qu'une dépendance a bougé. C'est désormais le seul fichier de ce dossier qui vaille comme
> état courant. La surveillance passe par `grype` + le cache Redis `jarvis:cve` ; les
> commandes manuelles sont dans le cheatsheet, section « Sécurité — vérifier les CVE à la
> main ».

**Statut : appliqué.** Les 3 groupes ci-dessous ont été installés dans
`/opt/jarvis/venv` et figés dans `requirements.txt` le 2026-07-29 :
mlx/mlx-metal → 0.32.0, mlx-vlm → 0.6.8, et les 5 CVE (pillow, torch,
cryptography, python-multipart, setuptools). `torchvision` a dû être aligné
à 0.28.0 (conflit de version avec torch 2.13.0, `pip check` propre après).
`pip-audit` : 0 vulnérabilité restante. Imports de `main.py`/`llm_local.py`
et pipeline `describe_images_local()` (mlx-vlm 0.6.8) testés OK en isolation.
**Le service `jarvis-api` tourne encore sur l'ancien code chargé en mémoire —
redémarrage nécessaire pour que le processus live utilise ces versions.**

## Génération

```
pip install cyclonedx-bom pip-audit   # dans venv
cyclonedx-py environment --output-format JSON --output-file DOCS/sbom/sbom-python.json /opt/jarvis/venv
pip-audit -r requirements.txt -f json > DOCS/sbom/pip-audit.json
```

- `sbom-python.json` : SBOM CycloneDX du venv Python (156 composants).
- `pip-audit.json` : scan de vulnérabilités connues (OSV/PyPA).
- Périmètre : uniquement le venv Python (`jarvis-core`). Le SBOM ne couvre pas
  JarvisApp (Swift/Xcode) ni les images Docker (qdrant, redis, open-webui) —
  celles-ci sont des binaires tiers non buildés par le projet, pas de fichier
  de deps à auditer côté SBOM Python.

## Vulnérabilités connues (pip-audit)

28 CVE/advisories sur 5 paquets :

| Paquet | Version | Corrigé dans | Gravité notable |
|---|---|---|---|
| `pillow` | 12.2.0 | 12.3.0 | 16 advisories (dont plusieurs heap-overflow sur decoders d'images) |
| `torch` | 2.11.0 | 2.13.0 | PYSEC-2025-194 |
| `cryptography` | 46.0.7 | 48.0.1 | GHSA-537c-gmf6-5ccf |
| `python-multipart` | 0.0.26 | 0.0.27 → 0.0.31 | 4 advisories (parsing de formulaires multipart) |
| `setuptools` | 81.0.0 | 83.0.0 | PYSEC-2026-3447 |

`pillow` traite des images uploadées par l'utilisateur (vision pipeline) —
c'est le paquet le plus exposé de la liste, à prioriser.

## mlx / mlx-lm / mlx-vlm

| Paquet | Installé | Dernière | Écart |
|---|---|---|---|
| `mlx-lm` | 0.31.3 | 0.31.3 | déjà à jour |
| `mlx` / `mlx-metal` | 0.31.2 | 0.32.0 | 1 minor |
| `mlx-vlm` | 0.4.4 | 0.6.8 | ~2 ans de retard |

**mlx 0.32.0** : le changelog (`v0.31.2...v0.32.0`) contient plusieurs correctifs
de correction numérique sur les kernels quantifiés Metal utilisés en génération
token-par-token (le chemin exact emprunté par le briefing/self-reflection) :
- `Fix qvm_split_k incorrect batch stride calculation`
- `Fix fp quantized matvec for output dim < 8`
- `Add small-batch quantized matvec kernel (qmv_wide)`
- `Fix rope single token multiple sequences`

C'est potentiellement lié (sans certitude) au bug d'échappement JSON du
16-07-29 (`\ ` au lieu de `\n\n` généré par Qwen3.6-35B RotorQuant) — un logit
corrompu par un bug de qmv/qmm expliquerait un token aberrant rare et
dépendant du contenu. `mlx-lm` 0.31.3 n'impose aucune borne haute sur `mlx`
(`mlx>=0.31.2`), donc la mise à jour est sans risque de conflit de version.

**mlx-vlm 0.4.4 → 0.6.8** : saut beaucoup plus large, à traiter séparément.
`mlx-vlm==0.6.8` exige `mlx>=0.32.0` et `mlx-lm>=0.31.3` (satisfaits une fois
mlx bumpé). Changelog entre les deux couvre des mois de correctifs
(Gemma4, TurboQuant Metal — non utilisé ici, cf. mémoire projet — auth API
serveur, etc.). `llm_local.py::_load_vlm` / `_describe_images_sync` appellent
directement l'API `mlx_vlm` : à valider par un test réel (`describe_images`)
avant bascule en prod, pas un simple bump de version.

## Autres paquets notables obsolètes (hors sécurité)

`fastapi` 0.136→0.140, `huggingface_hub` 1.11→1.25, `transformers` 5.5→5.14,
`torch` 2.11→2.13 (cf. CVE ci-dessus), `uvicorn` 0.46→0.51, `qdrant-client`
1.17→1.18. Rien d'urgent en dehors du volet sécurité déjà listé.
