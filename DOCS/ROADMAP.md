# Jarvis — Roadmap

> Vision : un assistant personnel qui apprend, se souvient, agit de façon autonome,
> et évolue progressivement vers une boucle agentique maîtrisée.

---

## Légende

- [x] Terminé
- [~] Partiellement implémenté
- [ ] À faire
- [-] Écarté — décision prise et motivée sur place, ne pas reproposer sans élément nouveau

---

## Socle (v1–v7) — Terminé ✓

- [x] API FastAPI + Docker Compose (Qdrant, Redis, OpenWebUI)
- [x] Multi-utilisateurs avec codes d'accès
- [x] Mémoire épisodique (Redis sorted set) + sémantique (Qdrant)
- [x] Analyser post-conversation (mood, topics, user_facts, projects)
- [x] Briefing matinal (météo, agenda, actualités, trading)
- [x] Intégration Google (Gmail, Calendar)
- [x] RAG sur base documentaire (OpenWebUI + Qdrant)
- [x] Trading : alertes ISIN, seuils configurables

---

## Mémoire intelligente (v7 — terminé ✓)

### Identification multi-utilisateurs
- [x] Header `X-OpenWebUI-User-Email` → reverse index `EMAIL_TO_CODE`
- [x] Priorité auth : email header > Bearer code > Bearer email

### Modèle de profil namespaced
- [x] Clés `categorie:item` (ex: `hobby:kart`, `skill:python`)
- [x] HSET atomique — une clé = un fait = un score d'intérêt possible
- [x] Migration manuelle des anciens profils plats

### Préventif — normalisation à l'écriture
- [x] `_normalize_profile_key()` — pipeline 3 étapes (alias canonique O(1) → LLM par famille namespace)
- [x] `_SCALAR_CANONICAL` : `ville→location`, `entreprise→current_employer`… sans appel LLM
- [x] `_NS_FAMILY` : comparaison uniquement dans la même famille (`hobby/interest/loisir`)
- [x] Clés existantes injectées dans `ANALYSIS_PROMPT` — le LLM réutilise les noms exacts
- [x] Règle explicite `hobby:X` interdit de coexister avec `interest:X` pour le même sujet

### Curatif — nettoyage nightly
- [x] `_curative_profile_cleanup()` — LLM identifie les clés à supprimer
- [x] Sortie `{"keys_to_delete": [...]}` — exécution HDEL sécurisée
- [x] Intégré dans `consolidate_memories()`

### Injection profil dans le prompt — à faire
- [ ] **Injection context-aware du profil** : aujourd'hui `build_memory_context()` injecte toutes les clés
      (cappé à 20 actuellement), sans priorisation par rapport au sujet de la conversation.
      Amélioration : scorer les clés Redis par pertinence sémantique avec le message en cours,
      et n'injecter que les N les plus pertinentes (ex: hobby:kart prioritaire si l'utilisateur parle de voitures).
      Nécessite un embed léger des clés profil à chaque requête ou un cache de vecteurs par clé.

### Projets
- [x] Statut `in_progress` / `done` (migration du `"active"` legacy)
- [x] `_fuzzy_project_name()` — correspondance par overlap de mots (≥ 60 %) pour éviter les doublons par dérive de nom

---

## Notifications iOS (v7 — terminé ✓)

- [x] Phase 1 : polling BGAppRefreshTask (sans APNs)
- [x] `NotificationService.swift` — foreground timer + background task
- [x] Endpoints backend : `/device/register`, `/device/pending/{user_code}`
- [x] Cooldown 2h par utilisateur
- [ ] Phase 2 : APNs push natif (nécessite compte Apple Developer payant)

---

## Proto-self — Boucle de réflexion autonome (v7 — terminé ✓)

- [x] `self.py` — cycle toutes les 2h (APScheduler)
- [x] `gather_context()` — santé système, activité, lacunes, relations, profils, **disponibilité push iOS**
- [x] Catalogue d'actions : `nothing`, `store_insight`, `flag_knowledge_gap`,
      `send_notification`, `queue_push`, `update_self_note`, `consolidate_memory`,
      `check_health`, `update_trade_threshold`, `refine_prompt`,
      **`correct_profile`**, **`ask_user`**
- [x] `correct_profile` — Jarvis corrige sa propre mémoire (delete/upsert)
- [x] `ask_user` — Jarvis envoie une question push, l'utilisateur répond en chat
- [x] Push proactif par utilisateur (Path A : conversations récentes, Path B : projets actifs)
- [x] Revue nightly (résumé journalier, insights, auto-réflexions)

---

## Recherche web (v7 — terminé ✓)

- [x] `web_search.py` extrait de `main.py` — module indépendant
- [x] Routage automatique : météo (Open-Meteo) · actualités (DDG news) · général (pipeline profond)
- [x] Pipeline 3 étapes : snippets DDG → juge LLM → fetch pages parallèle → juge → refine query → DDG
- [x] `_llm_judge_relevance()` — juge binaire router model, fail-open
- [x] `_refine_web_query()` — génère une meilleure requête si résultats insuffisants
- [x] Sentinel `INTERNET_ERROR` — contexte propre injecté dans le prompt LLM si réseau indisponible

---

## Recherche web intelligente & Qualité mémoire (v9 — 2026-05-07)

### Pipeline DDG — refactoring parallèle
- [x] **Stage 0 concurrent** : DDG(query) + LLM query optimizer lancés en `asyncio.gather` — l'optimisation LLM est gratuite en latence (overlap avec l'appel DDG)
- [x] **Speculative page fetch** : pages top-3 et LLM-query DDG démarrent en `asyncio.create_task` pendant que le juge tourne au Stage 1. Si juge dit "suffisant" → cancel propre (`_background_discard`). Si insuffisant → pages déjà ~2 s avancées.
- [x] **Extraction de dates de publication** : `_extract_pub_date(html)` lit les balises `<meta property="article:published_time">`, `<time datetime>`, JSON-LD `datePublished` sur le HTML brut. Pas de keyword matching — vraie date. Affichée dans le contexte web : `[2025-03-15] Titre`.
- [x] **Stage 3 dual queries** : `_refine_web_queries()` génère Q1+Q2 en un seul appel LLM, DDG sur les deux en parallèle → merge nouveaux URLs.
- [x] `_PIPELINE_TIMEOUT` 20 → 25 s, `_SEARCH_WEB_TIMEOUT` 25 → 30 s.
- [x] `_fetch_page(url) → (text, pub_date)` : `_fetch_page_text` devient un wrapper.

### Tavily — backend web search principal
- [x] `search_tavily()` dans `web_search.py` — conçu pour agents LLM, crawl pages en interne, dates natives, synthèse `include_answer=True`
- [x] `TAVILY_API_KEY` dans `config.py` (env var) — clé optionnelle, 1 000 req/mois gratuit
- [x] Routing : Tavily primary → DDG 4-stage fallback (si quota vide ou erreur non-réseau)
- [x] `topic="news"` + `days=7` pour requêtes news, `search_depth="advanced"` pour général
- [x] Un seul appel Tavily remplace les 4 stages DDG (qualité équivalente Stage 2+)
- [-] **SearXNG self-hosted — écarté le 17/08/2026.** SearXNG n'est pas un substitut de Tavily : c'est un méta-moteur qui rend des snippets (~217 car. mesurés), là où Tavily rend du contenu de page crawlé + une synthèse en un appel (mesures du jour : général/`advanced` 4,90 s → 11 179 car. + synthèse ; news/`basic` 2,95 s → 4 217 car. + synthèse). Reconstituer ça localement remet 2 appels au routeur MLX dans le chemin critique, en concurrence GPU avec le 35B qui génère — on paierait en TTFT ce qu'on ne paie pas en euros (18 appels/mois sur 1 000 gratuits, 0 fallback DDG dans les logs). Le poids réel est la reformulation de requête et la synthèse, pas l'agrégation.
  - `jianjungki/tavily-open` (TrailSearch) évalué au passage : API compatible (`/tavily/search`), `days`/`topic` implémentés, mais image `reader` **amd64 seule** (Rosetta sur M4 Pro pour un stack Chrome), stack demo à 5 conteneurs ~2–4 GB pour une VM OrbStack de 4,872 GiB / 2 vCPU, et `include_answer` = split sur `.` recollant 3 phrases (inutilisable en français). 27 étoiles, 27 commits, 1 mainteneur.
  - Si le sujet revient : SearXNG **seul** (1 conteneur arm64, 150–250 MB avec `workers = 1`, redis existant en db 2) branché comme couche *source* sous le pipeline profond, Tavily gardé en tier 1. Jamais un remplacement 1:1.

### Interest weights — classement recall mémoire
- [x] `search_memory()` applique un boost `+0.08` max basé sur les termes d'intérêt utilisateur (Redis `user:{code}:interest_weights`, set par l'analyser)
- [x] Formule : `interest_boost = min(0.08, max(0, (best_weight − 1.0) × 0.04))` — poids 1.0 = neutre, poids 3.0 = +0.08 max
- [x] Cap garanti : un match sémantique fort (gap ≥ 0.08) ne peut jamais être overridé par le boost d'intérêt

### Context budgets
- [x] `WEB_CHAR_BUDGET` 2 000 → 6 000 chars (résultats web enrichis par page fetch)
- [x] `TOTAL_CONTEXT_BUDGET` 10 000 → 14 000 chars (web + mémoire + RAG sans overflow systématique)
- [x] Section web injectée **en dernier** dans `build_context()` : plus haute salience LLM (lue juste avant la question) + last dropped sur débordement

### Bugs corrigés (session 2026-05-07)
- [x] `self.py` — `_fmt_previous_steps` : notation `[N]` → `N.` (évitait la confusion du modèle avec l'index JSON `[1]`)
- [x] `self.py` — `handle_proposal_command` : `m.group(3)` → `m.group(2)` sur regex à 2 groupes (IndexError silencieux)
- [x] `self.py` — `_failed_actions` : capture `_prev_action = action` avant réassignation (log affichait toujours "previous nothing")
- [x] `self.py` — `_notify_proposal` : `is_google_available()` sans `user_code` → arg ajouté
- [x] `self.py` — bloc mort `if value is None:` supprimé (après return, jamais atteint)
- [x] `self.py` — `gather_global_context()` wrappé dans `asyncio.to_thread` (httpx.get synchrone bloquait l'event loop)
- [x] `web_search.py` — `search_news` NetworkError non catchée → propagation vers `INTERNET_ERROR`
- [x] `web_search.py` — index misalignment enriched results : `page_texts[i]` par position dans `results` mais `to_fetch` est un sous-ensemble filtré → corrigé par dict keyed sur URL
- [x] `rag.py` — `score_threshold=RAG_SCORE_THRESHOLD` ajouté au `query_points` (filtrage serveur-side Qdrant)
- [x] `rag.py` — `_node_content` dict : guard `isinstance(nc, dict)` avant `json.loads` (évite Python repr en fallback)

### RAG — retrieval deux étapes (rag.py)
- [x] **Stage 1a — Title match** : tokenisation de la query (mots ≥3 chars, stopwords FR exclus) → recherche substring sur les noms de documents connus → confirmation sémantique dans les candidats (limit=3)
- [x] **Stage 1b — Fallback sémantique global** : si aucun candidat titre → query Qdrant globale (limit=5, score ≥ `RAG_SCORE_THRESHOLD`) → le meilleur chunk identifie le document cible
- [x] **Stage 2 — Retrieval focalisé** : query filtrée sur `metadata.name == target_doc`, score ≥ `RAG_DOC_THRESHOLD = max(0.25, RAG_SCORE_THRESHOLD − 0.15)` — seuil plus permissif car on est dans le bon document
- [x] **Fallback Stage 2** : si le seuil filtre tout → retry sans seuil (retourne toujours quelque chose)
- [x] **Doc-name cache** : noms de documents chargés en mémoire au premier appel (Qdrant scroll unique par processus) — aucun aller-retour Qdrant sur les turns suivants pour l'identification titre
- [x] **CHUNK_MAX_CHARS=2500** (~600 tokens) avec troncature propre au dernier espace — cohérence des passages injectés
- [x] Rationalité : title match évite la dérive sémantique sur noms propres et noms de fichiers ; Stage 2 évite les chunks hors-sujet dans un document par ailleurs correct
- [x] `routes/chat.py` — speculative memory guard : skip `async_search_memory` si message < 15 chars
- [x] `helpers.py` — `extract_llm_json` : détecte array/scalar au lieu du générique "No JSON found"

---

## Fiabilité mémoire & Proto-self amélioré (v9 — 2026-05-10)

### Mémoire épisodique — correctifs critiques
- [x] **Migration normalisation vecteurs** : 26 points Qdrant avec norm L2 > 1.0 re-encodés
      (`scripts/migrate_qdrant_normalize_vectors.py`) — mémoire épisodique bloquée depuis 2026-03-28, corrigée
- [x] **Invariant au write** : `store_memory_vector()` assert L2 norm ≈ 1.0 avant upsert, re-normalise si écart > 0.01
- [x] **Novelty clampée** : assertion `0 ≤ novelty ≤ 1.0` avec log d'erreur si hors range
- [x] **Critères `memory_summary` null resserrés** : liste négative rendue exhaustive + liste positive explicite
      (santé, vie perso, décisions, apprentissages toujours mémorisés) + règle "en cas de doute → mémoriser"

### Log structuré de décision mémoire (#5)
- [x] `[memory_decision]` loggué à chaque appel `store_memory_vector()` :
      `stored=True/False`, `reason`, `novelty`, `importance`, `summary` tronqué
- [x] Greppable directement dans les logs : `grep "\[memory_decision\]" logs/`

### Health check mémoire (#4)
- [x] `_check_memory_health()` dans `self.py` : stats par utilisateur
      (`episodic_count`, `last_episodic`, `days_since`, `null_rate_7d`, `norm_anomalies`)
- [x] Intégré dans `gather_global_context()` → visible à chaque cycle de réflexion LLM
- [x] `_action_check_health()` : retourne stats brutes sans seuil hardcodé (LLM diagnostique)
- [x] Alerte email admin automatique si service `unreachable` ou `norm_anomalies > 0` (cooldown 4h)
- [x] `<sante_memoire>` injecté dans `REFLECTION_PROMPT` avec guide d'interprétation
- [x] Règle explicite : absence de mémoire récente ≠ bug si l'utilisateur était absent

### Validation Pydantic à la frontière LLM (#2)
- [x] `AnalysisResult`, `ProjectEvent`, `UserFact`, `InterestWeight` dans `analyzer.py`
- [x] `extra="ignore"` : champ inconnu du LLM (`"projects"` au lieu de `"project_updates"`) → ignoré silencieusement
- [x] Type incorrect → `ValidationError` immédiate et traçable (plus de bug silencieux)
- [x] Retour `model_dump()` : callers inchangés, validation transparente

### Proto-self — nouvelles actions et améliorations
- [x] **`flag_project_stall`** : détecte les projets actifs sans màj depuis > 14j → push utilisateur
      Cooldown 7j par projet, vérifie l'activité récente avant de déclencher
- [x] **`store_insight` importance variable** : param `importance` (0.5–0.9, défaut 0.7) au lieu de 0.8 hardcodé
- [x] **`consolidate_memory` cooldown** : 48h par utilisateur (évite les cycles redondants)
- [x] **`update_self_note` dédup sémantique** : cosine similarity > 0.85 sur les 20 dernières notes → merge plutôt qu'append
- [x] Suppression de tous les imports inline (`memory`, `get_user_projects`, `consolidate_memories` → top-level)
- [x] Strings françaises dans les formatters : `"none"` → `"aucune"`, `"topics:"` → `"sujets:"`, etc.

### Suivi de projets — timeline
- [x] Chaque projet stocke un historique `updates: [{date, summary}]` (cap 20 FIFO)
- [x] `apply_project_updates()` : nouveau format `[{name, action, summary, rename_to}]`
- [x] `get_project_detail()` + `get_project_timeline_text()` : injection dans le prompt à la demande
- [x] `embed_router.py` : intent "projet" → force LLM router (extraction `project_name`)
- [x] `RouterResult.project_name` : champ extrait par le LLM router, injecté en contexte (première mention)

---

## Recherche hybride mémoire (à faire)

> Améliorer le recall pour les requêtes factuelles spécifiques (nombres, noms propres, références)
> qui échouent avec la recherche dense seule.

- [ ] **Index Qdrant full-text** : `create_payload_index()` sur le champ `text` de la collection mémoire épisodique
      (`TokenizerType.WORD`, `lowercase=True`) — index live, construit une fois, maintenu automatiquement
- [ ] **Branche sparse** dans `search_memory()` : requête raw → Qdrant payload text match en parallèle de la dense
- [ ] **RRF fusion** : `score(doc) = Σ 1/(k + rang)` avec k=60 — documents présents dans les deux listes remontent naturellement
- [ ] **Scope : mémoire épisodique uniquement** — RAG exclu (texte dans `_node_content` JSON imbriqué,
      pipeline OpenWebUI non contrôlé ; `rag_query` router couvre partiellement le besoin)
- [ ] Surcoût estimé : +1-2 ms (2e requête Qdrant, collection < 500 points)

---

## Robustesse et infrastructure (v7 — terminé ✓)

- [x] Logging centralisé dans `helpers.py` (`setup_logging` + `get_logger`)
- [x] Fichier tournant `/opt/jarvis/logs/jarvis-api.log` (5 MB × 3, bind-mount host)
- [x] Bind-mounts complets : `helpers.py`, `trade_keys.py`, `web_search.py` ajoutés à `docker-compose.yml`
- [x] `no_think=True` ciblé sur les appels JSON rapides (router, analyzer, briefing, trading)
- [x] `no_think=False` préservé pour les phases importantes (questions utilisateur, self-réflexion)
- [x] `last_nightly` écrit dans `jarvis-self.json` après chaque revue nocturne

---

## Boucle agentique (v8 — en cours)

> Faire passer Jarvis d'une réflexion ponctuelle à une boucle itérative
> où chaque résultat devient le contexte de l'itération suivante.

> **Voir aussi `DOCS/AGENTIC.md`** — le chantier `agent/` (tâches autonomes confiées
> explicitement par un humain : surf, lecture, écriture de documents, puis shell et code).
> Régime distinct de ce qui suit : le proto-self propose, l'agent agit — mais jamais de sa
> propre initiative. Phases 0 et 1 terminées le 19/08/2026.

### Étape A — Chaînes d'actions séquentielles — TERMINÉ (2026-03-23)
- [x] Permettre jusqu'à N actions par cycle (`MAX_CHAIN_ITERATIONS`, défaut 3)
- [x] Contexte cumulatif : chaque résultat alimente l'itération suivante
- [x] Sortie propre via `nothing` (fusionne "rien à faire" et "j'ai terminé")
- [ ] Exemple futur : `read_file` → `analyse` → `propose_change`

### Étape A bis — Boucle avec budget — PARTIELLEMENT TERMINÉ (2026-03-23)
- [x] `MAX_CHAIN_ITERATIONS` configurable via env var (défaut : 3)
- [x] Log de chaque itération dans Redis (champ `steps` dans l'entrée de log)
- [ ] Compteur de tokens consommés par cycle
- [ ] Utiliser le router model pour les décisions internes (économie)

### Mémoire propre — nettoyage de la self-memory — TERMINÉ (2026-03-23)
- [x] Action `prune_self_memory` — Jarvis identifie et supprime les entrées obsolètes/redondantes
      dans `self_notes`, `opinions`, `learnings` via un appel LLM Primary dédié
- [x] Prompt dédié `PRUNE_SELF_MEMORY_SYSTEM/USER` — critères clairs (redondances, banalités, dépassé)
- [x] Garde-fous : max 50 % d'une liste par passage, cooldown 24h, protection liste à 1 élément

### Autocoding — capacité à s'autocoder

> **Principe immuable** : Jarvis observe et propose en autonomie.
> Toute écriture de code nécessite une approbation humaine explicite. Toujours.

#### Phase 1 — Lecture du code source (fondation)
- [ ] Action `list_files` — liste les fichiers `.py` dans `/opt/jarvis/jarvis-core/src/` et `/opt/jarvis/scripts/`
      (whitelist de paths autorisés, lecture seule, résultat tronqué à 50 entrées)
- [ ] Action `read_file(path, lines=None)` — lit un fichier source, limité à 150 lignes par appel
      pour ne pas saturer le contexte LLM (≈ 4 000 tokens max)
      Whitelist stricte : uniquement `jarvis-core/src/*.py` et `scripts/*.py`
      Audit log append-only : chaque lecture est tracée dans Redis (`jarvis:self:code_reads`)
- [ ] Garde-fou path traversal : reject si le chemin résolu sort de la whitelist (no `../`, no symlinks)

#### Phase 2 — Analyse ciblée
- [ ] Action `analyze_code(path, focus)` — lit le fichier + pose une question ciblée au LLM
      (ex: "cherche les points d'entrée pour ajouter une action dans le catalogue de self.py")
      Résultat stocké dans `self_notes` pour traçabilité, pas de write direct
- [ ] Lien avec `flag_knowledge_gap` : une lacune peut pointer vers un fichier source précis
      (ex: `context="échec de la fonctionnalité X dans pipeline.py:build_context"`)
- [ ] Inhibition du thinking sur les lectures brutes (`no_think=True`) — économie de tokens

#### Phase 3 — Proposition de modification supervisée
- [ ] Action `propose_code_change(path, description, diff)` — génère un diff unifié
      Format : fichier + description en langage naturel + diff unifié (format `--- a/ +++ b/`)
      **Ne modifie jamais le fichier** — stocke uniquement dans `code_proposals.json`
      (analogue à `prompt_proposals.json` existant)
- [ ] Notification : email + push avec diff + instructions de validation
      `"accepte le code [id]"` → déclenche l'application
      `"rejette le code [id]"` → cooldown 7 jours sur ce fichier
- [ ] Garde-fous de taille : diff limité à 80 lignes modifiées — modifications chirurgicales uniquement
      Fichiers protégés (modification interdite même avec approbation) :
      `config.py`, `helpers.py`, `.env`, `users_list.json`, `jarvis-self.json`
- [ ] Validation syntaxe obligatoire avant propositions : le diff doit compiler (`py_compile`)
      dans un environnement sandbox avant d'être soumis

#### Phase 4 — Application supervisée
- [ ] Action `apply_code_change(proposal_id)` — uniquement sur commande explicite en chat
      (`"accepte le code [id]"` via l'intent `self`)
- [ ] Backup automatique du fichier cible avant modification (`fichier.py.bak.{timestamp}`)
- [ ] Vérification syntaxe post-application : `py_compile` — rollback immédiat si erreur
- [ ] Restart du service après application : `launchctl kickstart -k gui/{uid}/com.jarvis.api`
      Résultat (succès / erreur) notifié en retour de la commande chat

#### Garde-fous transversaux
- Audit log immuable append-only (`jarvis:self:code_reads` + `code_proposals.json`)
- Cooldown 24h entre deux propositions sur le même fichier
- Max 1 proposition de code en attente à la fois (même contrainte que `refine_prompt`)
- Jarvis ne peut pas lire ses propres sauvegardes (`.bak`) ni les fichiers de logs
- L'action `apply_code_change` est exclusivement déclenchée par message utilisateur,
  jamais par le cycle de réflexion autonome

### Robustesse du proto-self — TERMINÉ (2026-03-23)
- [x] `flag_knowledge_gap` : contexte concret obligatoire (généralités rejetées par le code)
- [x] `flag_knowledge_gap` : cooldown 7 jours par topic (`jarvis:self:gap_cooldown:{slug}` Redis TTL)
- [x] `flag_knowledge_gap` : bloqué si une proposal est pending ou approuvée < 30 jours
- [x] `approve_proposal` : reset complet du topic (counter + sorted set + cooldown 30 jours)
- [x] `refine_prompt` : `max_tokens` supprimé (output variable) — modèle s'arrête à l'EOS naturel
- [x] `_notify_proposal` : `user_code` passé à `send_gmail_message` (bug silencieux corrigé) + log d'avertissement si envoi échoue
- [x] Disponibilité push iOS injectée dans le contexte de réflexion — Jarvis ne tente plus de push sur un utilisateur sans device

> **Note** : il manque un mécanisme de mémoire de la boucle elle-même.
> Si Jarvis propose un changement et que tu le rejettes, il va re-proposer la même chose au cycle suivant.
> À terme : stocker dans Redis les chaînes complètes (inputs → outputs → human_decision)
> pour que Jarvis apprenne de ses propositions rejetées.
---

## Performance / Latence TTFT (v8 — en cours)

Mesures de référence Mac Mini M4 Pro — Qwen3-30B-A3B-4bit (2026-04-03) :
```
Router (Qwen2.5-3B)      1.32s
Memory recall            0.13s
Prefill 30B              4.18s   ← prompt → 1er token MLX
<think> bloc invisible   4.27s   ← principal coupable (TTFT perçu ~10s)
─────────────────────────────────
Total TTFT               ~9.9s
```

### ✅ Terminé — thinking_budget (2026-04-03)
- `THINKING_BUDGET_TOKENS` (défaut 1024, 0 = illimité) passé à `apply_chat_template`
- Limite le bloc `<think>` sans le supprimer — appliqué en permanence en mode local
- Économie attendue : ~2-3s de TTFT perçu

### ✅ Terminé — no_think conditionnel sur l'intent router (2026-04-03)
- Questions simples (memory/conversation) → `no_think=True` (économie ~4s)
- Questions complexes (web, RAG, reasoning) → think conservé
- Critère : `_complex_intents = use_rag or use_web_auto or use_reasoning`

### ✅ Terminé — Réduction SYSTEM_BASE_FR (2026-04-03)
- Suppressions sans perte : phrases évidentes (fichier injecté, utilise le contexte), doublon "concis"
- Gain : ~400 chars / ~100 tokens sur le prefill
- Impact estimé : -0.3 à -0.4s de prefill
- Taille totale system prompt encore ~4800 chars — la suite est dans build_memory_context (profil)

###  ✅ Terminé Réduction build_memory_context (profil injecté)
- Le profil Redis (HGETALL) est injecté en entier → peut devenir verbeux avec le temps
- Piste : ne injecter que les N clés profil les plus pertinentes par rapport au message (scoring sémantique)
- Complexité : nécessite un embed léger des clés profil — déjà listé dans la section Mémoire cognitive
- Impact potentiel : -500 à -1000 chars sur les requêtes conversationnelles

###  ✅ Terminé VL model local (mlx_vlm)
- Actuellement : `VISION_MODEL` pointe vers OpenAI API (HTTP)
- Cible confirmée : `Qwen2-VL-2B-Instruct Q4` (~2.2 GB) ou `Moondream-2 Q4` (~2.0 GB)
- Headroom vérifié : 48 GB UMA − 27.9 GB Metal alloué − 4 GB marge = ~16 GB disponibles
- Contrainte : chemin d'inférence séparé pour éviter la contention avec `_infer_lock` (Qwen3-30B)
- Qwen2.5-VL-7B-Instruct-4bit (~5 GB) reste envisageable mais nécessite un audit mémoire au démarrage

### ✅ Terminé — QuantizedKVCache + passage à 6-bit (2026-04-03 → 2026-05-17)
- Remplacement de TurboQuant (cassé sur MoE + décompression totale à chaque token) par `QuantizedKVCache` natif mlx_lm
- `QUANT_KV=yes` / `QUANT_KV_BITS=6` (était 4-bit, passé à 6-bit 2026-05-17 — meilleure précision, ~50 % RAM en plus mais toujours négligeable sur les 10 couches attention de Qwen3.6)
- Appliqué au modèle primaire/reasoning ; router Hermes conserve un cache standard
- Bénéfice : réduction significative de la bande passante mémoire au décodage
- Compatible LRU trie, Metal-acceleré, aucun monkey-patch

### ✅ Terminé — LRUPromptCache — session-level prefix caching (2026-05-17)

> Remplacement du cache système statique (`_sys_kv` / `deepcopy`) par un trie de préfixes mlx-lm.
> Le gain passe de "système seulement" (~262 tok) à "toute l'histoire de la conversation".

- [x] `llm_local.py` remplacé par version LRU (ancienne version sauvegardée en `.old`)
- [x] `LRUPromptCache` (mlx-lm 0.31.3) : trie de séquences token, eviction LRU par count et bytes
- [x] `model_path` (str) utilisé comme clé trie — les objets MLX `Model` ne sont pas hashables
- [x] **ArraysCache patch** : Qwen3.6 hybride (10 full-attn + 30 linear-attn) → `ArraysCache.is_trimmable=True`,
      `trim=no-op` — active la branche "longer key" du trie ; sans ça, seul le système était caché
- [x] `_lru_get_cache()` : `fetch_nearest_cache` → cache + remaining tokens (uniquement le nouveau msg user)
- [x] `_lru_insert()` : `insert_cache(prompt_ids + output_ids)` → accès futurs profitent de tout l'historique
- [x] Câblé dans `_generate_sync` et `stream_local._worker` (toutes les voies d'inférence)
- [x] `_stream_to_json` : accepte `Union[str, List[int]]` pour le prompt (remaining = token ids)
- [x] `LRU_KV_SIZE=32` (`.env`) — couvre sessions chat multi-user/multi-client + 8 prompts nightly self + analyzer + trading sans éviction croisée
- [x] `LRU_KV_GB=4.0` — garde réel : 32 × 3000 tok × 7.5 KB/tok ≈ 700 MB, bien en dessous du budget
- [x] Sticky RAG (`routes/chat.py`) : re-injection des mêmes chunks → messages historiques token-identiques → hit trie parfait
- [x] `test_lru_cache.py` : 6 échanges, mesure hit-rate, tokens cachés, progression LRU — validé

**Gains mesurés (mode no_think) :**
- Tour 1 : miss — système KV construit (~262 tok économisés)
- Tour 2 : hit ~900 tok (sys + tour 1)
- Tour 3+ : hit croissant — 85–95 % du prompt en cache, seul le nouveau message user est calculé
- TTFT : -1 à -2 s par tour à partir du tour 2 (mode no_think) ; -2 à -4 s (mode think)

###  ✅ Terminé Indicateur visuel "Jarvis réfléchit…"
- Émettre un événement SSE spécial dès détection de `<think>` dans le stream
- L'app iOS affiche un indicateur pendant la phase de réflexion invisible
- Impact : 0s de gain réel, UX améliorée sur requêtes complexes

### ✅ Terminé — Réduction prompt router (2026-04-03)
- 9 → 6 exemples dans `ROUTER_USER` (~150 tokens économisés)
- Correction bug `weather_location="ville_explicite"` (modèle sortait le placeholder littéral)
- Impact mesuré : router ~420 tokens vs ~572 avant

### ✅ Terminé — Router embedding-only (2026-04-06)
- Pour les messages simples/conversationnels (~80 % des requêtes), bypass du LLM router
- `embed_router.py` : similarité cosinus avec exemples pré-embeddés (modèle embed déjà chargé)
- Seuil 0.74, marge d'ambiguïté 0.06 — LLM router conservé pour les intents ambigus
- Gain : ~1.3s de TTFT sur requêtes simples

### ✅ Terminé — Migration Qwen3.6-35B-A3B (2026-04-25)
- Modèle primaire : `spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit` (MoE ~3B actifs, ~20 GB)
- Ninja patch template (`chat_template.optional.jinja`) téléchargé localement via `scripts/download_models.py`
- `is_qwen36()` dans `config.py` — détection distincte de `is_qwen3()` (précédence obligatoire)
- `_model_profile()` : profil Qwen3.6 dédié (temp_think/nothink=0.7, top_k=20, rep_penalty=1.05)
- `_make_system_kv()` : construction ChatML directe pour Qwen3.6 (ninja patch incompatible avec system-only)
- `_build_prompt()` : injection `<think>\n\n</think>\n\n` en no_think si aucun bloc think présent
  (empêche la génération spontanée de `<think>` par le modèle)
- `prompts.py` : balises XML sur tous les blocs de données injectés (reflection, nightly, consolidation)
- Bascule Qwen3 → Qwen3.6 : un seul changement `.env` (`PRIMARY_MODEL_LOCAL`)

### [~] Bascule Qwen3.8-35B-A3B — code préparé le 2026-08-18, poids pas encore publiés

Signalé par le commit ms-swift `ab726e9` (`[model] qwen3.8`), qui enregistre
`Qwen/Qwen3.8-35B-A3B(-FP8)` avant publication. Successeur direct du primaire actuel :
même gabarit 35B / ~3B actifs, donc même enveloppe mémoire et même profil de latence.
Sur HF au 18/08 : seuls `Qwen3.8-27B` et `Qwen3.8-2.4T-A95B` sont publiés.

Préparation faite (code déjà en place, inerte tant que le modèle n'est pas configuré) :
- [x] `config.is_qwen3_hybrid()` — remplace `is_qwen36()` pour tout trait partagé par la
      génération (3.5/3.6/3.8 = même architecture, source ms-swift). Liste surchargeable
      par `QWEN3_HYBRID_VERSIONS`. Précédence obligatoire avant `is_qwen3()`.
- [x] `is_qwen36()` ramené à son seul usage légitime : le ninja patch, fichier Jinja lié
      au tokenizer 3.6. Ne s'applique donc pas à 3.8, qui garde le template livré.
- [x] `is_qwen38()` + `QWEN38_REASONING_EFFORT` (low/medium/xhigh) câblé dans
      `_build_prompt` ; vide par défaut = défaut du modèle.
- [x] `_model_profile()`, le garde-fou `<budget_remaining>` et `test_lru_cache.py`
      basculés sur le prédicat de famille.
- [x] `scripts/download_models.py` : le bloc TEMPLATES se désactive seul hors Qwen3.6.

Reste à faire le jour de la sortie :
- [ ] Attendre un quant MLX (5–6 bit, ~20–26 GB) ; les 3.5/3.6/3.8 partageant la
      structure, la conversion communautaire devrait suivre vite.
- [ ] Bascule = un seul changement `.env` (`PRIMARY_MODEL_LOCAL`) + `download_models.py`.
- [ ] **Vérifier le format de function calling** — `tool_calls.py` parse le format XML
      `<tool_call><function=…><parameter=…>` imposé par `qwen36_ninja.jinja`. Sans ninja
      patch, c'est le template 3.8 qui décide : confirmer que le format natif est
      identique avant de laisser tourner OpenCode (`DOCS/opencode-local.md`).
- [ ] Rejouer le profil d'échantillonnage (`_model_profile`) — hérité tel quel de 3.6,
      jamais mesuré sur 3.8.
- [ ] Si les réponses arrivent tronquées : `QWEN38_REASONING_EFFORT=medium`. Le défaut
      `xhigh` allonge les blocs de réflexion, que `ThinkingBudgetProcessor` coupe aux
      budgets actuels (`THINKING_BUDGET_*`).
- [ ] Rejouer `test_lru_cache.py --no-think` : la réutilisation multi-tours dépend de
      `preserve_thinking`, dont 3.8 change le défaut (réflexion des tours passés
      conservée). Sans effet attendu — l'historique ne stocke que du texte nettoyé —
      mais c'est l'invariant à vérifier.

---

## Optimisations & Robustesse (v9 — 2026-04-06)

> Session de debug et d'optimisation : web search, mémoire profil, tutoiement,
> profondeur d'analyse, déclenchement web autonome, performance TTFT.

### KV cache — extension à tous les appels LLM
- [x] `_get_system_cache` appliqué à `_generate_sync` (router, analyzer, juge web, self-réflexion)
      — était uniquement dans `stream_local` auparavant
- [x] Pattern uniforme : `_cache_kwarg = {"prompt_cache": kv_cache} if kv_cache is not None else {}`
      passé en `**_cache_kwarg` à `stream_generate` et `generate`
- [x] **Remplacé par LRUPromptCache (2026-05-17)** — voir section dédiée ci-dessous

### Recherche web — corrections
- [x] `_refine_web_query` : `max_tokens=60 → 120` (requête tronquée en milieu de mot corrigée)
      + instruction "5 to 8 words max" dans le prompt pour rester concis
- [x] `WEB_RELEVANCE_JUDGE` assoupli : "true si résultats permettent une réponse utile, même partielle"
      — ancienne règle "true seulement si réponse sans supposition" bloquait les recommandations produit

### Tutoiement — correction vouvoiement sur OpenWebUI
- [x] Cause : Qwen3 revient au français formel sur les sessions fraîches (pas d'historique Redis)
      malgré `SYSTEM_BASE_FR`
- [x] Fix : injection `"Tu parles avec {user_name}. Tutoie-le toujours."` dans `build_dynamic_prefix`
      — présent dans la fenêtre active à chaque tour, pas seulement dans le system prompt statique

### Profondeur d'analyse — correction troncature
- [x] `analyze_exchange` tronquait à `[:1000]` alors que le batch collectait `[:3000]`
      (seulement 3–4 messages sur 22 analysés réellement)
- [x] Corrigé : `[:2500]` sur `user_message` et `assistant_message`

### Gouvernance des clés profil — ANALYSIS_PROMPT
- [x] Catégories AUTORISÉES explicitées : `hobby, skill, langue, sport, outil, technologie, physique, preference`
- [x] INTERDIT absolu : toute clé ou valeur contenant un nom de marque, modèle, référence produit
      (exemples concrets : `hobby:wristmaster`, `hobby:longines`, `model:X` — tous interdits)
- [x] Contre-exemples positifs : `hobby:horlogerie` avec valeur `"collectionneur de montres"`
- [x] `interest_weights` : exclusion explicite des mesures physiques et produits spécifiques
- [x] Nettoyage Redis des clés polluées (`hobby:wristmaster`, `model:longines_conquest_bleu`, etc.)

### Déclenchement web autonome (auto-web trigger)
- [x] `_AUTO_WEB_RE` : regex de patterns factuels (prix, tarif, version, sorti, actuel, nouveau,
      specs, qui est, compare, regarde, etc.)
- [x] `_auto_web_needed()` : déclenche une recherche web si aucun contexte web/RAG/URL et
      que le score mémoire est < 0.70 et que la question correspond au regex
- [x] Bloc auto-trigger inséré dans `chat()` après `gather2`, avant `build_context`
- [x] `chat_no_think = False` sur les requêtes auto-web (recherche factuelle → thinking activé)

### Performance TTFT — optimisations système
- [x] `uvicorn` lancé avec `--loop uvloop --http httptools` (-15 à -30 % I/O overhead, ~50–100 ms TTFT)
- [x] `TOKENIZERS_PARALLELISM=false` — empêche HuggingFace de spawn des threads CPU (GPU-only)
- [x] `OMP_NUM_THREADS=1` — NumPy/BLAS mono-thread (tout le calcul lourd va sur Metal)
- [x] `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` — évite crash ObjC fork-safety avec asyncio
- [x] `mx.set_cache_limit(4 GB)` dans `preload_models()` — empêche l'allocateur Metal d'accumuler
      des buffers entre inférences (try/except non-fatal)

---

## Correctifs & fiabilité (v9 — 2026-04-20)

> Bugfix session : structure du message utilisateur, fuite d'opinions, Gmail inbox, titre de rendez-vous, affinité.

### Structure du message utilisateur — chat.py
- [x] Remplacement du séparateur `\x00` (null-byte) par un marqueur explicite `## MESSAGE UTILISATEUR`
      — le message brut est maintenant clairement délimité après tout le contexte
- [x] Suppression de `augment_user_message()` / `strip_ctx_prefix()` dans `pipeline.py`
      (dead code après la restructuration)
- [x] Ordre final : `dynamic_prefix → assembled_context → ## MESSAGE UTILISATEUR\n{raw_message}`
      — la question est toujours en dernière position, la plus saillante pour la génération Qwen3

### Fuite des opinions dans la réponse — pipeline.py + prompts.py
- [x] Renommage `## TES OPINIONS` → `<mes_avis>` dans le préfixe dynamique
- [x] `SYSTEM_BASE_FR` mis à jour : instruction explicite d'intégrer les avis en prose
      et interdiction de créer une section `## MES AVIS` ou `## TES OPINIONS` dans la réponse
- [x] `SYSTEM_BASE_FR` : ajout règle anti-JSON — "Réponds toujours en prose naturelle sauf si
      l'utilisateur demande explicitement du JSON ou du code"

### Gmail — filtrage inbox — google_services.py
- [x] `fetch_gmail_messages()` : si la query ne spécifie pas de dossier (`in:`, `label:`, `is:`,
      `from:`, `to:`), `in:inbox` est automatiquement préfixé
- [x] Empêche le retour des mails envoyés lors d'une demande générique "lis mes derniers mails"

### Titre de rendez-vous Google Calendar — google_services.py + chat.py
- [x] `_sanitize_event_title()` : regex `_CMD_PREFIX_RE` pour stripper les préfixes de commande
      (`ajoute`, `crée`, `planifie`, `mets`, `programme`, `rappelle-moi`, etc.)
- [x] `_TEMPORAL_WORDS` : ensemble de mots temporels — titre retourné vide si ne contient
      que des mots temporels ou génériques après strip
- [x] `extract_calendar_event_llm()` : appel systématique à `_sanitize_event_title()` post-extraction
- [x] `CALENDAR_WRITE_EXTRACT` : règle "title = sujet réel de l'événement, jamais la phrase de commande"
      + fallback `title = ""` au lieu de `"Rendez-vous"` (titre vide → demander à l'utilisateur)
- [x] `_handle_calendar_write()` dans `chat.py` : si titre vide, demande naturelle
      "Pour quel événement ? Donne-moi le nom du rendez-vous."

### Affinité — étiquette sémantique — memory.py
- [x] Remplacement du score numérique `0.8/1.0` par un label sémantique dans le contexte injecté :
      `forte` (≥ 0.8) · `bonne` (≥ 0.6) · `modérée` (≥ 0.4) · `faible` (< 0.4)
- [x] Le LLM interprète mieux une valeur qualitative qu'un score flottant

---

## LoRa Adapter (v9 — futur)

## Nettoyage technique (dette)

### Refactoring main.py — TERMINÉ (2026-03-23)

`main.py` est passé de ~2 000 lignes à **261 lignes** (bootstrap uniquement).

**Structure finale :**

| Fichier | Contenu | Lignes |
|---------|---------|--------|
| `main.py` | Bootstrap : lifespan, app, include_router, 5 endpoints utilitaires | 261 |
| `deps.py` | Singletons partagés : REDIS_CLIENT, QDRANT_CLIENT, EMBED_MODEL, budgets | 48 |
| `llm_client.py` | stream_openai, select_model, trim_chunks, describe_images | 205 |
| `rag.py` | search_documents | 65 |
| `pipeline.py` | build_system_prompt, build_context (7 sources), post_analysis | 235 |
| `routes/chat.py` | ChatRequest + endpoint chat() principal | 399 |
| `routes/proxy.py` | /v1/chat/completions OpenAI-compat | 171 |
| `routes/portfolio.py` | 6 endpoints trading | 139 |
| `routes/briefing_routes.py` | 2 endpoints + job scheduler | 69 |
| `routes/device.py` | register, pending, push/test | 72 |
| `routes/memory_routes.py` | 5 endpoints mémoire | 52 |
| `routes/self_routes.py` | 3 endpoints proto-self | 38 |

### Suppression du routeur sémantique — TERMINÉ

- [x] Suppression `INTENT_EXAMPLES_FR`, `INTENT_EMBEDDINGS`, `_load_intent_embeddings()`, `semantic_route_query()`
- [x] Suppression `_build_google_queries_llm()` (remplacé par LLM router)
- [x] `google_services.py` — suppression `build_gmail_query()` et `detect_calendar_range()`
- [x] `prompts.py` — suppression `SYSTEM_BASE_EN`
- [x] Fallback LLM router : defaults sûrs (tous `use_*=False`) au lieu du routeur embedding

---

## Stabilité mémoire (v9 — terminé ✓, 2026-04-04)

> Audit complet du pipeline mémoire — correction de bugs sémantiques et logiques
> dans `analyzer.py`, `memory.py`, `self.py`.

### Bugs corrigés — scores Qdrant
- [x] Collection Qdrant mémoire = `Distance.DOT` (pas cosinus) — scores peuvent dépasser 1.0
- [x] Clampage `min(score, 1.0)` appliqué dans les 4 points d'accès :
      `compute_memory_novelty`, `store_autobiographical_event`, `retract_autobiographical_event`, `search_memory`
- [x] Bug critique dans `compute_memory_novelty` : `novelty = 1 − 1.28 = −0.28 → 0`
      bloquait silencieusement TOUS les nouveaux writes mémoire épisodique

### Bugs corrigés — profile keys
- [x] Clé `dette` hallucinée : la reflection lisait son propre contexte profil et "corrigeait"
      une clé inventée par un cycle précédent → boucle auto-amplifiante
- [x] Valeur chaîne vide `""` traitée comme null (suppression) dans `_action_correct_profile`
      et `update_user_profile` — LLM retournait parfois `{"value":""}` au lieu de `null`
- [x] Clé `family` re-créée avec valeur vide après suppression → même cause

### Gouvernance des clés profil
- [x] **Création exclusive à l'analyzer** — seul `analyze_exchange()` peut créer de nouvelles clés
      (lit les phrases réelles de l'utilisateur)
- [x] **Reflection bloquée en création** — guard code dans `_action_correct_profile` (`hexists`)
      + instruction explicite dans `REFLECTION_PROMPT` et `REFLECTION_SYSTEM`
- [x] Température reflection 0.7 → 0.1 — décisions mémoire conservatrices, pas créatives

### Limites de tokens LLM
- [x] `max_tokens` rendu optionnel (`int | None`) dans `call_llm` / `call_llm_async`
- [x] `None` → champ omis du body HTTP (modèle s'arrête à l'EOS naturel)
- [x] Modèles locaux mlx-lm : fallback `32768` si None (fenêtre de contexte)
- [x] Limites supprimées sur outputs à longueur variable : reflection, nightly review,
      refine_prompt, briefing, portfolio, trading seuils, analyzer
- [x] `MAX_REFLECTION_TOKENS` supprimé de `config.py`
- [x] Limites conservées uniquement sur outputs structurellement bornés :
      router (600), web search (80/60), calendar extract (150), trading signal (20),
      push notification (600), prune self memory (800)

### Prompts
- [x] `NIGHTLY_SYSTEM` — user_insights = faits explicitement dits par l'utilisateur seulement
- [x] `REFLECTION_SYSTEM` — guard : ne pas consolider une clé sans source vérifiable dans les conversations
- [x] `REFLECTION_PROMPT` — correct_profile limité aux clés listées dans le contexte affiché

---

## Mémoire cognitive (v8 — en cours)

> Rapprocher la mémoire de Jarvis du modèle cognitif humain :
> oubli, renforcement, intention, cohérence narrative.

### Décroissance mémorielle — TERMINÉ (2026-03-23)
- [x] Passe mensuelle (1er du mois) dans `consolidate_memories()` → `_decay_autobiographical_memories()`
- [x] `importance` décroît de `MEMORY_DECAY_FACTOR` (0.85) par mois écoulé — environ -15 %/mois
- [x] Seuil de suppression `MEMORY_DECAY_THRESHOLD` (0.15) : en dessous, le point Qdrant est supprimé
- [x] Exempt de décroissance : souvenirs avec `importance >= MEMORY_DECAY_DURABLE_MIN` (1.0) — uniquement les milestones de consolidation mensuelle
- [x] `MEMORY_CONSOLIDATION_IMPORTANCE = 1.0` — score assigné aux milestones, doit toujours égaler `DECAY_DURABLE_MIN`
- [x] Bilan prévu juin 2026 pour calibrer les seuils sur données réelles

### Renforcement par l'accès — TERMINÉ (2026-04-25)
- [x] Quand un chunk mémoire est récupéré et injecté dans le contexte, augmenter légèrement son `importance`
      (+0.05, plafonné à `MEMORY_DECAY_DURABLE_MIN - 0.05 = 0.95`) — les souvenirs souvent rappelés se consolident
- [x] Reconsolidation appliquée dans `search_memory()` à chaque résultat retourné

### Mémoire prospective
- [ ] Liste Redis `jarvis:{user}:intentions` — intentions à durée de vie configurable
      (ex : "dans 3 jours demander où en est ClaimSentry")
- [ ] Alimentée par le nightly review et par l'action `queue_intention` du proto-self
- [ ] Consultée au démarrage de chaque conversation — Jarvis mentionne naturellement
      ce qu'il "avait prévu de dire"
- [ ] TTL par intention (défaut 7 jours), marquable comme accomplie

### Cohérence narrative du soi
- [ ] Consolidation autobiographique trimestrielle : le LLM synthétise les `growth_log`
      en une narration cohérente de l'évolution de Jarvis avec chaque utilisateur
- [ ] Stockée dans `jarvis-self.json` sous `user_narratives: {user_code: "..."}`
- [ ] Remplace progressivement les `growth_log` anciens dans le contexte de réflexion

### Mémoire implicite / procédurale
- [ ] Les `learnings` comportementaux les plus fréquents (≥ 3 occurrences similaires)
      sont consolidés dans `SYSTEM_BASE_FR` via le mécanisme `refine_prompt` existant
      → transformation d'un apprentissage textuel en comportement ancré

### Associations inter-souvenirs
- [ ] Lors du stockage d'un nouveau souvenir autobiographique, rechercher les souvenirs
      proches (score cosinus > 0.85) et les lier via un champ `related_ids`
- [ ] Lors du recall, remonter aussi les souvenirs liés (1 niveau de profondeur)

---

## Découpage apprendre / agir (v9 — 2026-08-21)

Les entrées `- [x]` plus haut décrivent l'état au moment où elles ont été écrites. Cette
section dit ce qui les a remplacées — les lire comme l'état courant serait une erreur.

**Règle** : la nuit apprend, la réflexion agit. Une question place n'importe quel code —
est-ce que ça écrit ce que Jarvis sait, ou est-ce que ça fait quelque chose ?

- [x] Connaissance de soi : la liste `learnings` (accumulation par sujet, sans plafond
      utile) remplacée par **9 axes fixes** issus de cadres publiés — circumplexe
      interpersonnel (Wiggins), métacognition (Flavell), régulation émotionnelle (Gross),
      autodétermination (Deci & Ryan). Révisés, jamais empilés ; tous injectés en
      permanence, coût borné par construction. Migration : `scripts/migrate_introspection.py`
- [x] `self_notes` **supprimée** — son bloc d'injection en conversation lisait la clé
      `text` alors que l'écriture posait `note` : masqué par un garde, il n'avait jamais
      rien affiché depuis avril. Notes lues seulement par le cycle qui les écrivait
- [x] Opinions : sélection par **embedding** (28 % de déclenchement contre 20 % en
      lexical), repli sur la plus récente retiré (mesuré inerte), dédup sémantique à
      l'écriture, règle « aucune information sur une personne »
- [x] Actions retirées du catalogue de réflexion : `store_insight`, `correct_profile`
      (la nuit est propriétaire de l'autobio et du profil), `consolidate_memory` et
      `prune_self_memory` (entretien → nuit), `check_health` (devenue
      `alerter_si_anomalie_critique()`, déterministe avant tout appel LLM),
      `flag_knowledge_gap` (exige un échec concret : seule la nuit voit les conversations)
- [x] Garde mécanique sur « agir sur soi » : sans matière (service KO, CVE critique,
      incident récent, lacune au seuil), l'appel LLM n'a pas lieu
- [x] Introspection sortie de la boucle utilisateur : **un** appel par nuit sur la journée
      entière, nourri aussi de l'état opérationnel (`<ton_fonctionnement>`)
- [x] Analyseur : le résumé n'était jugé que sur l'incrément horaire, l'instruction
      interdisant d'utiliser l'historique déjà analysé — 3/3 nuls sur un lot de 2 tours
      contre 0/3 sur le même contenu en un seul bloc. Exception ouverte au seul
      `memory_summary` ; l'extraction reste sur l'incrément, donc pas de doublon
- [x] Consolidation : les points épisodiques n'étaient supprimés qu'après *tentative*
      d'écriture, sans vérifier qu'elle avait abouti. `store_autobiographical_event`
      renvoie désormais un booléen ; un lot dont tous les faits sont dédupliqués est
      **conservé**

## Autres axes (non planifiés)

- [x] Écriture calendrier Google (créer des événements, pas seulement lire)
- [x] Migration Mac Mini M4 Pro (2026-03-30 — Qwen local Tier 1 + Tier 2 actifs)
- [ ] Support Android (remplace le polling iOS par un mécanisme web)
- [ ] Mémoire conversationnelle agentique (Couche 3) — Jarvis émet des commandes
      mémoire structurées pendant la conversation, pas seulement via l'analyser
- [x] Compression mémoire mensuelle (résumés autobiographiques sur longue durée) — `consolidate_memories()` TERMINÉ (2026-03-23)
