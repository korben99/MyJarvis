# AGENTS.md — Jarvis

Assistant personnel : API FastAPI (`jarvis-core/src/`), LLM local MLX sur Mac Mini M4 Pro,
Redis + Qdrant en conteneurs, clients Open WebUI et iOS.

## Commandes

```bash
./venv/bin/python -m pytest jarvis-core/tests/test_quality.py -m "not integration"  # unitaires, <1s
./venv/bin/python -m pytest jarvis-core/tests/test_quality.py                       # + intégration : serveur sur :8000 requis
./venv/bin/python -m py_compile jarvis-core/src/<fichier>.py                        # vérif syntaxe rapide
./jarvis-status.sh                                                                  # santé de la stack
jarvis-restart                                                                      # recharge le code (PAS de --reload)
```

**Le serveur ne recharge pas à chaud.** Toute modification sous `jarvis-core/src/` exige un
`jarvis-restart`, suivi de ~2 min de chargement du modèle. Vérifier qu'un correctif est
réellement actif : comparer l'heure de démarrage du process au `mtime` du fichier.

## Règles de travail

- **Correctifs chirurgicaux, pas de refonte.** Plusieurs mécanismes ont des invariants
  subtils et coûteux à rétablir : cache LRU de prompts, strip des `<think>` en streaming,
  résumé de session, fenêtre de back-fill du convlog. Combler un cas manquant ou ajuster un
  seuil ; ne pas restructurer. Préserver les branches existantes à l'identique.
- **Ne jamais supposer qu'un correctif marche : l'exercer.** Ce dépôt a déjà connu des
  correctifs validés sur les journaux alors que rien n'arrivait en base. Vérifier l'effet
  côté Redis/Qdrant, pas dans `logs/prompts.log`.
- **`logs/prompts.log` montre la sortie LLM BRUTE, avant validation pydantic.** Un champ
  visible là n'est pas un champ arrivé en base.
- Commentaires en français, comme le code existant. Expliquer le *pourquoi* — surtout un
  choix contre-intuitif —, pas le *quoi*.
- Ne pas commiter sauf demande explicite.

## Architecture — ce qu'il faut savoir avant de toucher

**Deux routes de chat, à ne pas confondre :**

| | `/v1/chat/completions` | `/v1/raw/chat/completions` |
|---|---|---|
| usage | Open WebUI, iOS | agents de code |
| messages | dernier `user` seulement | tous |
| `system` du client | écrasé | respecté |
| contexte injecté | profil, mémoire, RAG, Gmail, Calendar | aucun |
| écrit en mémoire | oui | non |

**Pipeline mémoire.** `post_analysis` (chaud, ~5 ms) n'écrit qu'une entrée convlog.
L'extraction de faits, le scoring et la vectorisation Qdrant sont faits par le job planifié
`analyse_recent_conversations` (`analyzer.py`). Un effet n'est donc pas immédiat : forcer
une passe avec `POST /memory/analyze/{user_code}`.

**Sortie LLM structurée.** Elle est validée par pydantic (`analyzer.py`) avant d'être
consommée. **Tout champ non déclaré dans le modèle est supprimé en silence**
(`model_config = {"extra": "ignore"}`). Ajouter un champ au prompt sans l'ajouter au modèle
donne une fonctionnalité qui ne marche jamais, sans la moindre erreur.

**Écriture des projets utilisateur.** `update_user_projects` (`memory.py`) reconstruit
chaque entrée champ par champ : tout champ absent de la liste blanche est effacé à la
fusion nocturne. Ajouter un champ = l'ajouter aussi à cette liste.

**Templates de chat.** Qwen3.6 utilise un template maison
(`models/templates/qwen36_ninja.jinja`), pas celui du tokenizer. C'est lui qui impose le
format des appels d'outil (`<tool_call><function=…>`, pas du JSON) et le rendu du mode
sans réflexion. Le modifier change tous les prompts, donc le cache LRU.

## Pièges de test

- Sessions Redis persistantes 90 j → **horodater les `session_id` de test**, sinon un
  contexte pollué fait rejeter les faits par l'analyzer.
- L'utilisateur `TEST` est un clone du profil réel : ses prompts sont indiscernables du
  vrai trafic dans les journaux.
- L'analyzer 35B rejette les jetons artificiels (`valeur_qa_42`) : formuler les faits de
  test comme l'exemple canonique d'`ANALYSIS_PROMPT`.
- `json.dumps` sans `ensure_ascii=False` échappe les accents et rend muettes les
  assertions de sous-chaîne.

## Ne pas faire

- Ne pas lancer `scripts/uploadrag.py` sans `--dry-run` : il écrit dans le RAG de production.
- Ne pas élargir la fenêtre de back-fill du convlog (`session["msgs"]`) : elle réécrirait
  des entrées déjà renseignées, et `satisfaction` s'y écrase sans garde.
- Ne pas exposer `/v1/raw` hors du réseau local : aucune authentification.
