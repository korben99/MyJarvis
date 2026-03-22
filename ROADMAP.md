# Jarvis — Roadmap

> Vision : un assistant personnel qui apprend, se souvient, agit de façon autonome,
> et évolue progressivement vers une boucle agentique maîtrisée.

---

## Légende

- [x] Terminé
- [~] Partiellement implémenté
- [ ] À faire

---

## Socle (v1–v7) — Terminé

- [x] API FastAPI + Docker Compose (Qdrant, Redis, OpenWebUI)
- [x] Multi-utilisateurs avec codes d'accès
- [x] Mémoire épisodique (Redis sorted set) + sémantique (Qdrant)
- [x] Analyser post-conversation (mood, topics, user_facts, projects)
- [x] Briefing matinal (météo, agenda, actualités, trading)
- [x] Intégration Google (Gmail, Calendar)
- [x] RAG sur base documentaire (OpenWebUI + Qdrant)
- [x] Trading : alertes ISIN, seuils configurables

---

## Mémoire intelligente (v7 — en cours)

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

## Notifications iOS (v7 — en cours)

- [x] Phase 1 : polling BGAppRefreshTask (sans APNs)
- [x] `NotificationService.swift` — foreground timer + background task
- [x] Endpoints backend : `/device/register`, `/device/pending/{user_code}`
- [x] Cooldown 2h par utilisateur
- [ ] Phase 2 : APNs push natif (nécessite compte Apple Developer payant)

---

## Proto-self — Boucle de réflexion autonome (v7)

- [x] `self.py` — cycle toutes les 2h (APScheduler)
- [x] `gather_context()` — santé système, activité, lacunes, relations, **profils**
- [x] Catalogue d'actions : `nothing`, `store_insight`, `flag_knowledge_gap`,
      `send_notification`, `queue_push`, `update_self_note`, `consolidate_memory`,
      `check_health`, `update_trade_threshold`, `refine_prompt`,
      **`correct_profile`**, **`ask_user`**
- [x] `correct_profile` — Jarvis corrige sa propre mémoire (delete/upsert)
- [x] `ask_user` — Jarvis envoie une question push, l'utilisateur répond en chat
- [x] Push proactif par utilisateur (Path A : conversations récentes, Path B : projets actifs)
- [x] Revue nightly (résumé journalier, insights, auto-réflexions)

---

## Recherche web (v7 — terminé)

- [x] `web_search.py` extrait de `main.py` — module indépendant
- [x] Routage automatique : météo (Open-Meteo) · actualités (DDG news) · général (pipeline profond)
- [x] Pipeline 3 étapes : snippets DDG → juge LLM → fetch pages parallèle → juge → refine query → DDG
- [x] `_llm_judge_relevance()` — juge binaire router model, fail-open
- [x] `_refine_web_query()` — génère une meilleure requête si résultats insuffisants
- [x] Sentinel `INTERNET_ERROR` — contexte propre injecté dans le prompt LLM si réseau indisponible

---

## Robustesse et infrastructure (v7 — terminé)

- [x] Logging centralisé dans `helpers.py` (`setup_logging` + `get_logger`)
- [x] Fichier tournant `/opt/jarvis/logs/jarvis-api.log` (5 MB × 3, bind-mount host)
- [x] Bind-mounts complets : `helpers.py`, `trade_keys.py`, `web_search.py` ajoutés à `docker-compose.yml`
- [x] `no_think=True` ciblé sur les appels JSON rapides (router, analyzer, briefing, trading)
- [x] `no_think=False` préservé pour les phases importantes (questions utilisateur, self-réflexion)
- [x] `last_nightly` écrit dans `jarvis-self.json` après chaque revue nocturne

---

## Boucle agentique (v8 — futur)

> Faire passer Jarvis d'une réflexion ponctuelle à une boucle itérative
> où chaque résultat devient le contexte de l'itération suivante.

### Étape A — Chaînes d'actions séquentielles
- [ ] Permettre jusqu'à 3 actions par cycle de réflexion (au lieu d'une seule)
- [ ] Contexte cumulatif : chaque résultat alimente l'itération suivante
- [ ] Action `done` pour que le LLM sorte proprement de la chaîne
- [ ] Exemple : `read_file` → `analyse` → `propose_change`

### Étape A bis — Boucle avec budget
- [ ] `max_iterations` configurable (défaut : 5)
- [ ] Compteur de tokens consommés par cycle
- [ ] Utiliser le router model pour les décisions internes (économie)
- [ ] Log de chaque itération dans Redis

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
Ce qui manque dans la roadmap
Un mécanisme de mémoire de la boucle elle-même. Si Jarvis exécute une chaîne read_file → analyse → propose_change et que tu rejettes la proposition, il va re-proposer la même chose au cycle suivant. Il n'y a pas de persistance du résultat des chaînes passées — seulement du résultat des actions individuelles. Il faudrait stocker dans Redis les chaînes complètes (inputs → outputs → human_decision) pour que Jarvis apprenne de ses propositions rejetées.
---
## LoRa Adapter (v9 — futur)

## Nettoyage technique (dette)

### Suppression du routeur sémantique (fallback embedding)
> Prérequis : LLM router considéré stable et toujours disponible.

- [ ] `main.py` — supprimer `INTENT_EXAMPLES_FR` (~105 lignes), `INTENT_EMBEDDINGS`, `_load_intent_embeddings()`, `semantic_route_query()`, 8 constantes `ROUTER_*_THRESHOLD`, appel lifespan, bloc `else` dans `chat()` → 3 lignes de defaults
- [ ] `main.py` — supprimer `_build_google_queries_llm()` et ses TODO (fallback Google query builder, remplacé par le LLM router)
- [ ] `main.py` — supprimer 5 lignes de code commenté (`# hist.append(...)` × 4, `# hist = conversation_history...`)
- [ ] `google_services.py` — supprimer `build_gmail_query()` et `detect_calendar_range()` (section "DEPRECATED", non appelées)
- [ ] `prompts.py` — supprimer `SYSTEM_BASE_EN` (constante anglaise non utilisée, seul `SYSTEM_BASE_FR` est actif)
- [ ] `llm_router.py` — supprimer commentaire `# CHECK IF 250 IS ENOUGH` (question résolue)

---

## Autres axes (non planifiés)

- [x] Écriture calendrier Google (créer des événements, pas seulement lire)
- [ ] Migration Mac Mini M4 Pro (livraison fin avril 2026 — Qwen local Tier 1 + Tier 2)
- [ ] Support Android (remplace le polling iOS par un mécanisme web)
- [ ] Mémoire conversationnelle agentique (Couche 3) — Jarvis émet des commandes
      mémoire structurées pendant la conversation, pas seulement via l'analyser
- [ ] Compression mémoire mensuelle (résumés autobiographiques sur longue durée)
