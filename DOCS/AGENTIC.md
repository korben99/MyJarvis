# Boucle agentique — ROADMAP

> Donner une tâche à Jarvis, et le laisser itérer seul jusqu'à l'avoir menée à bien :
> surfer, lire, écrire des documents, exécuter, coder — en arrière-plan, sans jamais
> passer devant le chat.

Document de travail du chantier `agent/`. Complète `ROADMAP.md`, qui reste la référence
pour le proto-self et l'autocoding.

---

## Position dans l'architecture

Deux régimes autonomes coexistent, et il faut les garder **distincts** :

| | `self/` — proto-self | `agent/` — boucle agentique |
|---|---|---|
| Déclenchement | tout seul, toutes les `REFLECTION_INTERVAL_HOURS` | **jamais** tout seul — un humain poste une tâche |
| Rapport au monde | observe, réfléchit, **propose** | **agit** : surfe, lit, écrit des fichiers |
| Portée d'une itération | une action du catalogue | un appel d'outil, jusqu'à `AGENT_MAX_STEPS` |
| Garde-fous | cooldowns, self-review, approbation humaine | sandbox de chemins, budgets, annulation |

Les mélanger reviendrait à réviser d'un coup toutes les garanties du proto-self. Le
principe immuable de `ROADMAP.md` reste entier : **le cycle de réflexion ne crée jamais de
tâche agentique**, et l'agent n'écrit jamais dans le code source de Jarvis.

---

## Décision d'architecture — pourquoi pas une lib

Étudié le 19/08/2026, avant d'écrire une ligne.

| Option | Verdict | Raison |
|---|---|---|
| Claude Agent SDK | ❌ | Exige un endpoint au format Anthropic. Le proxy correspondant a été supprimé le 08/08/2026, précisément parce qu'il échouait sur les appels d'outil. |
| LangGraph | ❌ | Machine à états + persistance + tracing qui doublonnent Redis, Qdrant et `reflection_log`. |
| openai-agents / pydantic-ai | ⚠️ | Forcent un aller-retour HTTP de Jarvis vers lui-même : re-sérialisation, et perte du lock de priorité GPU (ils passeraient par `/v1/raw`). |
| smolagents | ⚠️ | Le plus léger, mais tire LiteLLM et réécrit ce que `tool_calls.py` fait déjà pour le template maison. |
| OpenCode headless comme moteur | ⚠️ | Excellent sur le code, muet sur le reste (pas de web, pas de mémoire Jarvis, traces hors du système). Retenu comme **outil délégué** en Phase 3. |
| **Boucle maison `agent/`** | ✅ | ~600 lignes. La partie difficile — parsing des appels d'outil au format natif Qwen — existait déjà, en production, validée par OpenCode. |

Ce qui a fait pencher la balance : la valeur ajoutée d'un framework (parsing d'outils,
retries, tracing, mémoire) est exactement la partie déjà présente, et en meilleure
adéquation avec MLX et le template `qwen36_ninja.jinja`.

---

## Ordonnancement GPU — la contrainte structurante

Un seul GPU, sérialisé par `_infer_lock` (`llm/local.py`). Le chat est prioritaire, non
négociable. Trois mécanismes, du plus grossier au plus fin :

1. **`stream_local(priority="bg")`** — un pas d'agent ne prend le lock que si aucun appel
   de chat n'attend. C'est le seul chemin d'inférence qui accepte `tools` ; avant la
   Phase 0, il était câblé en priorité chat, donc aucun agent outillé ne pouvait
   s'exécuter en arrière-plan.
2. **`AGENT_STEP_MAX_TOKENS`** (1200) — le lock ne cède qu'**entre** deux générations. Ce
   plafond est donc le pire cas d'attente imposé à un tour de chat. Mesuré ~50 tok/s sur
   Qwen3.6-35B-A3B-5bit ⇒ ≈ 24 s. **Ne pas monter sans mesurer.**
3. **`AGENT_QUIET_SECONDS`** (45) — le lock ignore le tour de chat *à venir*. Sans fenêtre
   de calme, l'agent prend le GPU pendant que l'utilisateur lit la réponse précédente.

> **Préemption en cours de génération** — envisagée, **non implémentée**. Elle consisterait
> à poser `stop_flag` dans le worker de `stream_local` dès qu'un `_chat_waiter` apparaît,
> et à rejouer le pas (le cache LRU rend la re-prefill peu coûteuse). À n'écrire que si les
> trois leviers ci-dessus se révèlent insuffisants **à la mesure** : ça touche une mécanique
> délicate pour un gain qui reste à démontrer.

Ordre de grandeur d'une tâche : 15 pas × ~25 s ≈ **7 min de GPU**, étalés en arrière-plan.

---

## Phase 0 — Priorité GPU — TERMINÉ (2026-08-19)

- [x] `_acquire_infer_lock_bg()` extrait de `call_llm_local_async_bg` — boucle d'attente
      background partagée, comportement inchangé pour les appelants existants
- [x] `stream_local(priority="chat"|"bg")` — défaut `"chat"`, aucun changement par défaut
- [x] `seconds_since_chat()` — horodatage de la dernière demande d'inférence en priorité
      chat, alimente la fenêtre de calme
- [x] `/v1/raw` bascule en **priorité background par défaut** (`RAW_PRIORITY=bg`) :
      l'endpoint est par définition celui des agents, l'info est portée par la route et non
      par un réglage client. Override par requête via le champ `priority`.
- [x] Attente non bornée assumée en cas de chat soutenu — l'agent patiente, pas de repli

**Gain immédiat, indépendant du reste** : une session OpenCode ne dégrade plus le TTFT de
l'assistant sur iPhone.

---

## Phase 1 — Squelette de boucle, outils sans exécution — TERMINÉ (2026-08-19)

### Modules

```
agent/store.py     enregistrement Redis + file d'attente + contexte sur disque
agent/sandbox.py   confinement des chemins (realpath + racines autorisées)
agent/tools.py     7 schémas d'outils + implémentations
agent/loop.py      objectif → outil → observation → finish
agent/worker.py    consommation de la file, une tâche à la fois, notification finale
routes/agent_routes.py   POST/GET/cancel/transcript
```

### Outils

| Outil | Effet | Réutilise |
|---|---|---|
| `web_search` | recherche web | `web_search.search_web` (Tavily → DDG) |
| `fetch_url` | contenu texte d'une page | `web_search._fetch_page_text` |
| `search_docs` | base documentaire perso | `rag.search_documents` |
| `threat_intel` | groupes d'attaquants, sites de fuite darknet | `agent/cti.py` |
| `list_dir` | listing | — |
| `read_file` | lecture numérotée, paginée | — |
| `write_file` | écriture **workspace uniquement** | — |
| `finish` | sortie normale + livrables | — |

Huit outils, pas plus : leurs schémas sont rendus en tête de prompt **à chaque pas**, et
chaque outil supplémentaire est une occasion de se tromper de choix.

### Renseignement darknet (`agent/cti.py`)

Le web de surface ne dit presque rien d'utile sur un groupe de ransomware : les sources
primaires sont les sites de fuite en `.onion` que ces groupes opèrent eux-mêmes. On passe
par des agrégateurs qui les moissonnent et republient en clearnet — lecture seule, aucune
infrastructure, aucune exposition.

| Source | Contenu | Accès |
|---|---|---|
| **RansomLook** | 613 groupes suivis, publications récentes | libre, sans clé |
| **ransomwatch** | métadonnées des groupes (`.onion`, alias, état en ligne), historique des victimes | libre, sans clé |
| ~~ransomware.live~~ | meilleure couverture des trois | **passée sous clé API** (19/08/2026) — toutes ses routes rendent du HTML. À reprendre si une clé est obtenue. |

Deux pièges de volumétrie, rencontrés à la mise au point et traités dans le code :
`RansomLook /api/group/<nom>` rend jusqu'à **161 Mo** pour un groupe actif (jamais appelé) ;
`ransomwatch posts.json` fait 2,3 Mo (téléchargé une fois, cache mémoire de 6 h).

Vérifié en conditions réelles le 19/08/2026 : `lockbit3` → 57 `.onion` connus avec leur
état, 2178 victimes publiées ; flux d'actualité daté du jour.

> **Accès Tor direct** — non implémenté, décision en attente. Voir « Phase 2 bis ».

### Garde-fous

- **Chemins** : tout chemin est résolu par `realpath` (chemin **entier**, dernier composant
  compris — ne résoudre que le parent laissait passer un lien symbolique fichier) puis
  comparé aux racines autorisées. Écriture : workspace de la tâche. Lecture : workspace +
  `AGENT_READONLY_ROOTS`. `/opt/jarvis/.env` est hors des racines, donc illisible.
- **Budgets indépendants** : `AGENT_MAX_STEPS` (dérive), `AGENT_TASK_TIMEOUT_MINUTES`
  (temps réel), détection de boucle serrée (3 appels identiques d'affilée).
- **Un appel d'outil par tour** : les appels surnuméraires d'un même tour porteraient sur
  des résultats que le modèle n'a pas encore vus. Le premier est exécuté, les autres
  ignorés — et le modèle en est informé.
- **Aucune exception ne remonte d'un outil** : un échec doit donner de quoi corriger au pas
  suivant, pas tuer la tâche.
- **Réservé aux `USER_ADMINS`.**
- **Désactivé par défaut** (`AGENT_ENABLED=false`) : aucun worker, routes en 503.

### Reprise après redémarrage

Le contexte est écrit dans `messages.json` (écriture atomique) après **chaque** pas, et
`transcript.jsonl` garde la trace lisible. Au démarrage, `requeue_interrupted()` remet en
file les tâches restées `running` : elles reprennent où elles en étaient. À l'arrêt du
service, `CancelledError` sauvegarde le contexte et laisse le statut à `running`.

### Contexte long

Au-delà de ~100 000 caractères, les plus anciens résultats d'outil sont élidés. Ni le
system, ni l'objectif, ni les tours de l'assistant ne sont touchés : ce sont eux qui
portent le fil. Ce qui devait être retenu d'un résultat a normalement été écrit sur disque.

### Continuité du raisonnement — la panne qui a tout expliqué (19/08/2026)

Un tour d'agent ne laisse presque rien de visible : sa sortie se réduit à l'appel d'outil,
tout le « pourquoi » vit dans le `<think>`. En le supprimant, on obtenait un contexte fait
d'un objectif suivi d'un tas de résultats bruts — **32 messages, 17 000 tokens, tous les
tours assistant à 0 caractère de contenu**. Le modèle redécouvrait le problème à chaque
tour et rejouait six fois la même recherche.

Le raisonnement du **dernier** tour est désormais réinjecté (un seul : au-delà, la
croissance redevient quadratique). Effet mesuré sur la même tâche : **6 pas au lieu de 16,
et un livrable au lieu de rien**.

Corollaire : le découpage `<think>` doit précéder `parse_tool_calls`. Le modèle ébauche
souvent un appel *dans* sa réflexion pour l'écarter ensuite — parser le brut l'exécuterait.

### Dimensionnement du contexte — mesuré, pas supposé

| Grandeur | Valeur | Note |
|---|---|---|
| Contexte d'une tâche complète | **7 784 tokens** | 24 % du plafond pratique (~32 k) |
| `AGENT_MAX_TOOL_OUTPUT` | 6000 → **15000** | à 6000, une recherche de 9984 car perdait 40 % |
| `AGENT_PAGE_MAX_CHARS` | **14000** | distinct de `_PAGE_MAX_CHARS` (6000), calibré pour le chat |
| `AGENT_STEP_MAX_TOKENS` / `AGENT_THINKING_BUDGET` | **2200 / 1000** | `think` mesuré à 3500-3750 car sur les pas difficiles |

On affamait l'agent résultat par résultat en laissant les trois quarts du contexte vides.
Le bon endroit pour arbitrer est global (`_CONTEXT_SOFT_CAP`), pas par appel.

### Discipline de sourçage

Le premier livrable produit était structurellement correct et **factuellement faux** :
dates de 2024/2025 sur des faits de 2026, volumes inventés, sources sans URL. Le modèle
avait cherché, mais jamais **lu** une source — il rédigeait depuis les extraits de
`web_search`, complétés par sa mémoire d'entraînement.

Trois règles dans `AGENT_SYSTEM`, plus un garde-fou mécanique :
1. chercher ne suffit pas — au moins un `fetch_url` complet avant toute rédaction ;
2. aucune date, aucun chiffre, aucune citation qui ne vienne d'une source lue ;
3. toute source se cite avec son URL.

Le garde-fou : `finish` est **refusé une fois** si aucun livrable ne contient d'URL
(`_has_sources`). Une seule objection, jamais deux — sinon un livrable qui n'a
légitimement pas de source (un script) devient impossible à rendre.

### API

```bash
curl -X POST localhost:8000/agent/tasks \
  -H 'Content-Type: application/json' \
  -d '{"user_code":"<CODE_ADMIN>","objective":"Compare X et Y, écris-moi une note de synthèse"}'

curl localhost:8000/agent/tasks                      # liste
curl localhost:8000/agent/tasks/{id}                 # état + résultat + livrables
curl localhost:8000/agent/tasks/{id}/transcript      # ce que l'agent a réellement fait
curl -X POST localhost:8000/agent/tasks/{id}/cancel  # pris en compte entre deux pas
```

Livrables dans `/opt/jarvis/agent_workspace/{task_id}/` (gitignoré). Notification iOS à la
fin, via le même chemin de livraison que les push du proto-self.

---

## Phase 2 — Shell — À FAIRE

Le point sensible : Jarvis tourne sous le compte `korben`, avec ses droits pleins.

- [ ] Outil `shell(cmd, timeout)` — `cwd` forcé sur le workspace
- [ ] `sandbox-exec` (seatbelt macOS) : profil autorisant l'écriture au seul workspace
      + `/tmp`, lecture globale
- [ ] Denylist (`sudo`, `launchctl`, `docker`, `rm -rf /`, `curl | sh`, `git push`),
      timeout 120 s par commande, quota de commandes par tâche, sortie tronquée
- [ ] Outil `ask_human` — suspend la tâche, push, reprise à la réponse
- [ ] Durcissement optionnel : conteneur `jarvis-agent` dans `docker-compose.yml`,
      workspace en volume, `/opt/jarvis` monté en lecture seule

---

## Phase 2 bis — Accès Tor direct — DÉCISION EN ATTENTE

Les agrégateurs republient les *posts* des sites de fuite, mais pas leurs pièces jointes,
pas les forums, et avec un décalage. Pour aller à la source :

- [ ] Conteneur `tor` dans `docker-compose.yml` (SOCKS5 sur 9050) — la stack est déjà là
- [ ] `fetch_onion(url)` : `httpx` via `socks5h://`, extraction texte, mêmes plafonds que
      `fetch_url`
- [ ] **Liste blanche de domaines `.onion`** alimentée depuis `ransomwatch groups.json` :
      l'agent ne visite que des sites de fuite déjà identifiés comme tels. Sans elle, un
      modèle qui suit un lien se retrouve sur une place de marché.
- [ ] Lecture seule, stricte : aucune soumission de formulaire, aucun identifiant, aucune
      interaction. Journalisation de chaque URL visitée.

Coût : un conteneur, ~80 lignes. Le point à trancher n'est pas technique — c'est le
périmètre : quels sites, et jusqu'où.

---

## Phase 3 — Code — À FAIRE

- [ ] `delegate_coding(brief)` → `opencode run` headless dans le workspace, en priorité
      background (`/v1/raw` l'est désormais par défaut)
- [ ] Modification du code source de Jarvis : **diff uniquement**, versé dans le mécanisme
      de proposals existant → approbation humaine, conformément à `ROADMAP.md`

---

## Phase 4 — Intégration Jarvis — À FAIRE

- [ ] Intent `agent` dans `INTENT_EXAMPLES` + branche dans `chat.py` — déclenchement en
      langage naturel depuis l'iPhone, réponse immédiate, notification à la fin
- [ ] Résultat versé en mémoire épisodique ; échecs versés en incidents `vitals`
- [ ] Les tâches en cours visibles depuis le contexte de réflexion

---

## Risques

| Risque | Gravité | Parade |
|---|---|---|
| **Qwen3.6-35B-A3B 5-bit ne tient pas 15 pas** — dérive, boucle, oubli de l'objectif | 🔴 | Peu d'outils ; compteur de pas dans chaque résultat ; rappel de budget bas ; détection de boucle ; sortie forcée en synthèse. **À mesurer sur 3 tâches réelles avant la Phase 2.** |
| Un pas long fige le chat | 🟠 | Les trois leviers d'ordonnancement ci-dessus |
| Cache LRU mis sous pression (`LRU_KV_SIZE=4`, une entrée en croissance par agent) | 🟡 | Concurrence 1. Passer à 6 si des re-prefills apparaissent dans les logs |
| Contexte qui explose | 🟡 | Troncature par outil + élision des vieux résultats |
| Tâche zombie | 🟡 | Timeout global + annulation + reprise explicite au boot |

---

## Mise en service

```bash
echo "AGENT_ENABLED=true" >> /opt/jarvis/.env
launchctl kickstart -k gui/$(id -u)/com.jarvis.api
```

Vérifier au démarrage : `agent: worker démarré` dans `logs/jarvis-api.log`.
