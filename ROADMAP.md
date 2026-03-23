# Jarvis — Roadmap

> Vision : un assistant personnel qui apprend, se souvient, agit de façon autonome,
> et évolue progressivement vers une boucle agentique maîtrisée.

---

## Légende

- [x] Terminé
- [~] Partiellement implémenté
- [ ] À faire

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

### Étape D — Accès lecture au code source
- [ ] Action `read_file` — lire ses propres `.py` (lecture seule)
- [ ] Action `search_web` — recherche web sanitisée (lecture seule)
- [ ] Garde-fou : sanitisation du contenu web avant injection dans le contexte LLM
  (protection contre l'injection de prompt via des pages malveillantes)

### Étape C — Auto-modification supervisée
- [ ] Action `propose_code_change` — génère un diff, ne l'applique pas
- [ ] Le diff est envoyé par push + email pour validation humaine
- [ ] Action `write_file` — uniquement après confirmation explicite de l'utilisateur
- [ ] Rollback automatique si les tests échouent après application

> **Principe immuable** : Jarvis pense seul, agit seul sur les actions réversibles.
> Toute écriture de code nécessite une approbation humaine. Toujours.

### Robustesse du proto-self — TERMINÉ (2026-03-23)
- [x] `flag_knowledge_gap` : contexte concret obligatoire (généralités rejetées par le code)
- [x] `flag_knowledge_gap` : cooldown 7 jours par topic (`jarvis:self:gap_cooldown:{slug}` Redis TTL)
- [x] `flag_knowledge_gap` : bloqué si une proposal est pending ou approuvée < 30 jours
- [x] `approve_proposal` : reset complet du topic (counter + sorted set + cooldown 30 jours)
- [x] `refine_prompt` : `max_tokens` porté à 4000 + `current_text` cappé à 6000 chars — évite les propositions tronquées
- [x] `_notify_proposal` : `user_code` passé à `send_gmail_message` (bug silencieux corrigé) + log d'avertissement si envoi échoue
- [x] Disponibilité push iOS injectée dans le contexte de réflexion — Jarvis ne tente plus de push sur un utilisateur sans device

> **Note** : il manque un mécanisme de mémoire de la boucle elle-même.
> Si Jarvis propose un changement et que tu le rejettes, il va re-proposer la même chose au cycle suivant.
> À terme : stocker dans Redis les chaînes complètes (inputs → outputs → human_decision)
> pour que Jarvis apprenne de ses propositions rejetées.
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

### Renforcement par l'accès
- [ ] Quand un chunk mémoire est récupéré et injecté dans le contexte, augmenter légèrement son `importance`
      (ex: +0.05, plafonné à 1.0) — les souvenirs souvent rappelés se consolident
- [ ] Logguer les accès mémoire pour identifier les souvenirs les plus "vivants"

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

## Autres axes (non planifiés)

- [x] Écriture calendrier Google (créer des événements, pas seulement lire)
- [ ] Migration Mac Mini M4 Pro (livraison fin avril 2026 — Qwen local Tier 1 + Tier 2)
- [ ] Support Android (remplace le polling iOS par un mécanisme web)
- [ ] Mémoire conversationnelle agentique (Couche 3) — Jarvis émet des commandes
      mémoire structurées pendant la conversation, pas seulement via l'analyser
- [x] Compression mémoire mensuelle (résumés autobiographiques sur longue durée) — `consolidate_memories()` TERMINÉ (2026-03-23)
