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
- **Trois détecteurs de blocage**, chacun né d'une panne observée : 3 générations vides
  d'affilée (prompt cassé) · 3 appels d'outil identiques (méthode en échec) · 3 tours de
  prose identiques sans appel d'outil (modèle figé). Les trois sont nécessaires — chacun
  laisse passer ce que les deux autres attrapent.
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

**Deux tentatives de rapatrier le raisonnement ont échoué**, chacune en figeant la boucle,
et il faut les garder en mémoire — elles disent quelque chose de général sur ce qu'on peut
remettre dans un contexte :

| Placement | Panne |
|---|---|
| Fusionné dans le `content` du tour assistant | Le modèle en déduit « contenu assistant = mon raisonnement » : il émet sa réflexion, ferme `</think>`, puis **réécrit le même texte en sortie visible** (`think=3556 car, visible=3556 car`) et se fige 6 pas. |
| Message `user` en fin de contexte | Le plan périmé devient le signal **le plus récent** : il relit « ton plan était de lire l'article RTL » juste après avoir reçu cet article, et le refetche. 3 fois. |

La leçon : du raisonnement réinjecté est soit pris pour un modèle de sortie, soit pour un
ordre frais. **La solution n'est pas de mieux le replacer, c'est de ne pas en mettre.**

### L'outil `plan` — l'agent suit son propre plan

Un plan est une **donnée**, pas du raisonnement : il ne peut être confondu ni avec une
sortie attendue, ni avec une instruction fraîche.

- premier appel obligatoire : `plan(steps=[…])`, 3 à 6 étapes courtes ;
- le plan est **réaffiché sous chaque résultat d'outil**, avec l'étape courante fléchée ;
- `plan(done=N)` coche une étape ; de nouveaux `steps` remplacent le plan quand la réalité
  le dément ;
- **`plan` est le seul outil autorisé en plus d'une action dans le même tour.** L'exception
  est de principe : marquer une étape ne dépend d'aucun résultat. L'exiger dans un tour
  séparé ferait payer un pas par étape — un quart du budget en comptabilité.

Le modèle écrit en plus **une phrase en clair** avant chaque appel (`AGENT_SYSTEM`) : elle
atterrit dans `content`, se persiste d'elle-même et occupe la bonne place chronologique.
C'est tout ce qu'il relit de son propre cheminement, et ça ne coûte aucune mécanique.

### Où le compteur de tours est injecté

Le pied de chaque résultat d'outil porte `[pas N/max]` — mais il n'arrive qu'**à partir du
tour 2**. Au tour 1, le modèle ne voyait que l'objectif : ni sa position, ni ce qu'on
attendait de lui en premier. `AGENT_OBJECTIVE` porte donc lui aussi le compteur et la
consigne de planification, sur le seul message dont il dispose à cet instant.

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

#### Le garde-fou de sources : de bloquant à informatif (même journée)

Première version : `finish` refusé si aucun livrable ne citait de source. Bilan sur une
journée d'essais — **zéro fabrication rattrapée**, deux rustines nécessaires pour éviter
les faux positifs (documentation de code, base documentaire), et un rapport de 11 ko
**détruit** parce que l'objection a poussé le modèle à réécrire son fichier au lieu d'y
ajouter une section.

Beaucoup de livrables n'ont légitimement rien à citer : un script, un fichier de
configuration, une synthèse des données de l'utilisateur lui-même. Et ce qui corrige
réellement les inventions, ce sont les règles de sourçage d'`AGENT_SYSTEM` — l'article
ZeroBytes est sorti avec dix citations sans que le garde-fou n'intervienne.

Il est donc devenu un **signalement joint au compte rendu** : *« ce livrable ne cite
aucune source consultée »*. L'information reste, la coercition disparaît, l'humain juge.

#### Écrasement d'un livrable — garde `write_file`

`append` par défaut à false : une écriture sans ce drapeau REMPLACE le fichier. Un rapport
bâti en cinq pas est ainsi passé de 11 ko à 726 octets. `write_file` refuse désormais un
remplacement qui raccourcit un fichier existant de plus de 30 %, et indique les deux
sorties : `append=true` pour ajouter, `overwrite=true` pour remplacer délibérément.

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

Livrables dans `/opt/jarvis/agent_workspace/{task_id}/` (gitignoré).

### Restitution : le push annonce, le courriel transporte

À la fin d'une tâche réussie, **deux canaux**, et la division du travail entre eux est le
point important :

| Canal | Porte | Limite |
|---|---|---|
| Push iOS | l'annonce et le résumé | 500 caractères, écran verrouillé |
| Courriel | le **livrable entier** | 120 000 caractères, puis renvoi vers le workspace |

Le courriel part depuis le compte Google du demandeur, **vers lui-même** — jamais vers un
tiers. `send_gmail_message` n'accepte pas de pièce jointe (multipart/alternative texte +
HTML) : le document part donc dans le corps, ce qui a l'avantage d'être lisible sans rien
ouvrir. Le contenu est échappé avant rendu HTML.

L'envoi précède le push, ce qui permet à la notification de dire « envoyé par mail » —
et donc de savoir sans ouvrir sa boîte s'il y a quelque chose à lire.

Silencieux et non bloquant dans tous les cas de bord : aucun livrable, fichier illisible
ou vide, utilisateur sans courriel configuré, Gmail indisponible. Le livrable reste de
toute façon sur disque. `AGENT_EMAIL_REPORT=false` pour couper.

> **Injection dans l'historique** — le push est aussi ajouté à la conversation iOS
> (`iphone-main`), donc Jarvis y fait référence quand on lui parle **depuis l'iPhone**.
> Depuis Open WebUI, chaque conversation est une session distincte : les comptes rendus
> n'y apparaissent pas.

---

### Validation en conditions réelles (19/08/2026)

Tâche : *« note de synthèse sur le groupe Zero Bytes, actif contre des cibles françaises,
sources citées »* — sujet d'actualité, absent des données d'entraînement du modèle.

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Issue | 20 pas, aucun livrable | figé, aucun livrable | **8 pas, livrable** |
| Dates | inventées (2024/25 sur des faits de 2026) | — | **exactes et sourcées** |
| Sources | 2, sans URL | — | **10 citations numérotées, URL réelles** |
| Réflexion par tour | ~3500 car | 3556 (figé) | **72-335 car** |
| Durée par pas | 30-80 s | — | **5-17 s** |

Enchaînement du run 3 : `plan` → `web_search` → 2 × `fetch_url` (lectures complètes) →
replanification → 2 × `write_file` → `finish`. La reprise sur troncature s'est déclenchée
une fois et a fonctionné.

L'effondrement de la réflexion (3500 → 300 caractères) est l'effet direct du plan : le
modèle ne redérive plus le problème entier à chaque tour, il lit son plan et agit.

Le livrable a été recoupé avec des recherches indépendantes : 678 000 comptes DGFiP, juin
2026, VPN compromis, Intermarché Drive — tout tient. Le modèle a même produit de lui-même
une section « Limites de l'analyse » signalant les revendications non confirmées, ce qui
est la pratique CTI correcte.

Deux défauts résiduels, tous deux corrigés depuis : une bibliographie par morceau écrit
(consigne ajoutée) et un `append` collé au contenu précédent sans saut de ligne (jointure
posée mécaniquement dans `write_file`).

---

## Phase 2 — Shell — LIVRÉ (2026-08-19)

Jarvis tourne sous le compte `korben`, avec ses droits pleins. Le confinement n'est donc
pas une option de configuration, c'est la condition d'existence de l'outil.

- [x] Outil `shell(cmd, timeout)` — `cwd` forcé sur le workspace, `HOME` redirigé dessus
- [x] `sandbox-exec` (seatbelt) : écriture limitée au workspace + `/tmp`, **réseau coupé**,
      secrets (`.env`, `keys/`, `.ssh`, Keychains) illisibles
- [x] Liste noire (`sudo`, `launchctl`, `docker`, `rm -rf /`, `curl | sh`, `git push`,
      `diskutil`, arrêt machine…) — garde-fou contre l'erreur franche, pas une barrière
- [x] Environnement réduit : aucune clé d'API de Jarvis à portée d'un `env`
- [x] Délai par commande (60 s), quota par tâche (25), sortie tronquée
- [x] **Désactivé par défaut** (`AGENT_SHELL_ENABLED=false`), et le schéma n'est même pas
      déclaré au modèle quand la capacité est éteinte
- [ ] Outil `ask_human` — suspend la tâche, push, reprise à la réponse
- [ ] Durcissement optionnel : conteneur `jarvis-agent`, `/opt/jarvis` en lecture seule

### Validation en conditions réelles

Tâche : *« analyse `jarvis-api.log` et fais un rapport des erreurs de la journée »* — 2,7 Mo,
donc hors de portée de `read_file` : le shell est le seul chemin. Quatre exécutions, dont
trois échecs qui ont chacun désigné un défaut précis.

| Run | Issue | Défaut révélé |
|---|---|---|
| 1 | 20 pas, aucun fichier | shell trouvé au pas 8 ; relance de fin de budget trop douce |
| 2 | figé sur 4 `plan` | tours « plan seul » sans pied de page — aucun signal de budget |
| 3 | rapport **détruit** | `write_file` sans `append` écrase ; garde-fou sources coercitif |
| 4 | **réussite** | 20 pas, 8 commandes, rapport de 5,8 ko |

Contrôle du livrable du run 4 contre le journal réel :

| Affirmation | Réel |
|---|---|
| 113 entrées ERROR/WARNING le 19/08 | 112 — à une unité près |
| Boucle `item pairs from a mapping`, 10h03–10h38 | 40 occurrences, **toutes** dans cette fenêtre |
| Outils absents des déclarés, 13h44–13h47 | 3, noms cités correctement |
| Google en échec pour deux utilisateurs | 124 occurrences |

Le rapport a diagnostiqué **les bugs de la journée dans ce dépôt**, dont
`normalise_messages_for_template` (*« un dict mal sérialisé entre le LLM et le parser
interne — inspection de jarvis-core/src/llm/ nécessaire »*) et les noms d'outils hallucinés.

Seul défaut : « 40 » devient « ~68 » dans le résumé de notification, alors que le document
est exact. Un chiffre approximé au moment de résumer, pas au moment d'analyser.

### Auto-correction observée

Le modèle a écrit `grep -oP` (GNU), lu l'échec BSD, et réécrit
`grep -o '^[0-9]\{4\}-…'` sans aide. Il a aussi filtré ses propres commandes du journal
avant analyse — *« pour éviter les biais autocentrés »* —, ce que personne ne lui avait dit.

### Ce que quatre runs ont appris sur les garde-fous

Sur douze pannes de la journée, **la moitié venait de garde-fous ajoutés en réaction à la
panne précédente** : le refus de `finish` sans source a détruit un rapport, la relance de
marquage a provoqué quatre `plan` stériles, le plancher RAG à 0.60 a rendu la base
documentaire inaccessible. La règle qui en sort, et qui vaut pour la suite :

> **Mesurer avant de durcir. Préférer signaler à interdire.**

C'est le motif classique du risque introduit par la mesure qui contre un risque.

### Pourquoi `(allow default)` et non `(deny default)`

Un profil deny-default casse la moitié des outils Unix sur macOS (mach-lookup, sysctl,
dyld) et aurait produit un shell inutilisable. On ferme les deux voies par lesquelles une
erreur sort de la machine — **écriture hors zone et réseau** — et on garde le reste ouvert.
La lecture reste large à dessein : l'agent doit pouvoir inspecter le système pour être
utile, et tout ce qu'il lit finit dans un contexte que l'utilisateur relit.

Le réseau est coupé alors même que l'agent a `web_search` et `fetch_url` : ces outils
passent par du code journalisé et borné. Un `curl` libre est le chemin d'exfiltration le
plus court qui soit.

### Vérification du confinement (macOS 26.6, avant mise en service)

| Tentative | Résultat |
|---|---|
| Écriture dans le workspace | autorisée |
| Écriture dans `/opt/jarvis` | `Operation not permitted` |
| `touch /Users/korben/PWNED` | `Operation not permitted` |
| Écriture via `python3 open(…,'w')` | `PermissionError` |
| `cat /opt/jarvis/.env` | refusé (liste noire) |
| `cat /opt/jarvis/.e''nv` — **contournement de la regex** | `Operation not permitted` (noyau) |
| `curl https://example.com` | exit 6 (pas de réseau) |
| `socket.create_connection()` en Python | `PermissionError` |
| `env \| grep -c 'api\|key\|token'` | `0` |
| `sleep 5` avec délai 2 s | interrompu |

Le dernier point est le plus important : **la liste noire est contournable, le bac à sable
ne l'est pas.** C'est la seule couche sur laquelle reposer.

---

### Second registre validé — lecture de code (19/08/2026)

Tâche : *« lis `vitals.py` et documente son fonctionnement, dont où le score de risque est
consommé ailleurs dans le code »*. Registre volontairement éloigné du web.

**Résultat : 12 pas, livrable exact.** L'agent a lu `vitals.py`, cherché le consommateur,
ouvert `steering.py` de lui-même, et compris le lien. Recoupé contre le code : les 5
familles de disparition, `risk_scalar`, `_ramp`, `steering.set_risk`, et **les 12
identifiants de sondes cités existent tous**.

Trois défauts, dont deux corrigés :

| Défaut | Traitement |
|---|---|
| 4 appels `plan` identiques sans rien changer (pas 8-11), soit un tiers du budget | `plan` répond désormais « inchangé, rien n'a bougé » et rappelle d'utiliser `done` ; les tours sans action comptent dans la fenêtre de boucle |
| 2 noms d'outils inventés (`list_directory`, `search_files`), 2 pas perdus | Récupération déjà correcte — le message d'outil inconnu liste les outils réels |
| 2 caractères CJK (`部分`) dans 4738 caractères de français | Artefact de décodage de Qwen quantifié. Consigne d'alphabet ajoutée au prompt — atténuation partielle, la cause est au niveau du décodage |

Enseignement transversal des sept runs : **presque toutes les pannes venaient de la
plomberie, pas du modèle** — prompt enseignant le mauvais format, budget exprimé en lignes
au lieu de caractères, compteur de boucle mesurant les répétitions consécutives au lieu
d'une fenêtre. Le 35B choisit correctement quand on lui présente les choses correctement.

---

### Cache LRU et prompt d'agent (19/08/2026)

Le prompt d'agent respecte la doctrine du chat, en plus strict : **tête constante, queue
strictement croissante par ajout**.

```
[bloc système : outils + AGENT_SYSTEM]   constant pour toute la tâche
[user : OBJECTIF]                        constant
[assistant → tool → assistant → tool …]  croît, jamais réécrit
```

Rien de dynamique ne s'insère en tête — ni date, ni mémoire, ni état émotionnel,
contrairement à `build_dynamic_prefix` du chat.

#### Le bug qui annulait tout : préfixe système calculé sans les outils

`_system_prefix_text` construisait son candidat de préfixe **sans passer `tools`**, et
vérifiait son invariant contre un rendu lui aussi sans outils — la vérification passait
donc toujours. Or le template ouvre le bloc système par `# Tools` et les schémas, et ne
place le contenu système qu'ensuite (`qwen36_ninja.jinja:50-59`).

```
AVANT : préfixe annoncé  499 tok | réellement communs    3 tok | valide=False
APRÈS : préfixe annoncé 1853 tok | réellement communs 1853 tok | valide=True
```

`_lru_get_cache` tranchait le prompt réel à 499 tokens et faisait reprendre le modèle **au
milieu du JSON des outils**, avec un cache portant autre chose. Aucune exception — la
docstring annonçait le symptôme : *« garbled answers, no exception raised »*. Concernait
chaque génération outillée : la boucle agentique **et** OpenCode via `/v1/raw`.

Réexplique plusieurs pannes d'abord imputées au modèle : noms d'outils inventés
(`list_directory`, `search_files`), appels écrits en prose, consignes de résultat ignorées.

#### Ce que le mode thinking coûte encore

La queue reste non réutilisable en mode thinking : la clé LRU contient le `<think>` brut,
absent du prompt suivant, et `ArraysCache` (Qwen3.6 hybride) n'est pas trimmable — donc
`fetch_nearest_cache` saute l'entrée. Coût mesuré sur le run de lecture de code :

```
pas 2-3   :  6-8 s
pas 7-12  : 34-68 s     ← re-prefill d'un contexte qui grossit
```

En `no_think`, `_build_prompt` passe `preserve_thinking=True` et la clé redevient un vrai
préfixe. Vérifié **pour notre forme de messages**, `tool_calls` compris (leur mesure
d'origine portait sur une conversation de chat sans outils) : clé de 5490 caractères,
préfixe du prompt suivant à 100 %.

#### Résultat de l'A/B no_think — TRANCHÉ, on reste en thinking (19/08/2026)

Même tâche (documenter `vitals.py`), `AGENT_THINKING_BUDGET=0`, `AGENT_MAX_STEPS=30`.

| | thinking | no_think |
|---|---|---|
| Pas | 12 | 21 |
| Durée moyenne / pas | ~28 s | **27 s** — aucun gain |
| Exactitude du livrable | 12 identifiants vérifiés réels | **fabrication complète** |

Le document produit en no_think ne cite **aucune** des 5 familles réelles et invente
`_check_redis()`, `_check_qdrant()`, `_check_disk_space()`, `_boot_checks()`, un
`/proc/statfs` (Linux, sur une machine macOS), Qdrant, OpenAI, des inodes et des tableaux
de seuils. Le fichier avait pourtant été lu sept fois : le modèle avait le contenu réel
sous les yeux et a rédigé un module de monitoring générique tiré de ses a priori.

**Le raisonnement est ce qui intègre ce qui vient d'être lu.** Le supprimer ne rend pas
l'agent plus rapide, il le rend confiant et faux — le pire des deux mondes pour un
livrable destiné à être utilisé.

Réserve méthodologique : le correctif du préfixe LRU a été livré dans la même fournée,
donc l'A/B est confondu. Le gain LRU, lui, est isolé et net : **1 seul prefill de 1874
tokens pour 21 pas**, contre 12 prefills de 1140 auparavant.

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
