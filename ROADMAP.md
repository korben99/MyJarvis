# Jarvis — Roadmap

> Vision : un assistant personnel qui apprend, se souvient, agit de façon autonome,
> et évolue progressivement vers une boucle agentique maîtrisée.

---

## Légende

- [x] Terminé
- [~] Partiellement implémenté
- [ ] À faire

---

## Socle (v1–v6) — Terminé

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
- [x] `_normalize_profile_key()` — router LLM détecte les doublons sémantiques
- [x] Exemples : `hobby:kart` ≠ `hobby:tennis`, `hobby:kart` == `loisir:kart`
- [x] Prompt analyser mis à jour (règles de nommage, `interest_weights` explicites)

### Curatif — nettoyage nightly
- [x] `_curative_profile_cleanup()` — LLM identifie les clés à supprimer
- [x] Sortie `{"keys_to_delete": [...]}` — exécution HDEL sécurisée
- [x] Intégré dans `consolidate_memories()`

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

## Boucle agentique (v8 — futur)

> Faire passer Jarvis d'une réflexion ponctuelle à une boucle itérative
> où chaque résultat devient le contexte de l'itération suivante.

### Étape A — Chaînes d'actions séquentielles
- [ ] Permettre jusqu'à 3 actions par cycle de réflexion (au lieu d'une seule)
- [ ] Contexte cumulatif : chaque résultat alimente l'itération suivante
- [ ] Action `done` pour que le LLM sorte proprement de la chaîne
- [ ] Exemple : `read_file` → `analyse` → `propose_change`

### Étape B — Boucle avec budget
- [ ] `max_iterations` configurable (défaut : 5)
- [ ] Compteur de tokens consommés par cycle
- [ ] Utiliser le router model pour les décisions internes (économie)
- [ ] Log de chaque itération dans Redis

### Étape C — Accès lecture au code source
- [ ] Action `read_file` — lire ses propres `.py` (lecture seule)
- [ ] Action `search_web` — recherche web sanitisée (lecture seule)
- [ ] Garde-fou : sanitisation du contenu web avant injection dans le contexte LLM
  (protection contre l'injection de prompt via des pages malveillantes)

### Étape D — Auto-modification supervisée
- [ ] Action `propose_code_change` — génère un diff, ne l'applique pas
- [ ] Le diff est envoyé par push + email pour validation humaine
- [ ] Action `write_file` — uniquement après confirmation explicite de l'utilisateur
- [ ] Rollback automatique si les tests échouent après application

> **Principe immuable** : Jarvis pense seul, agit seul sur les actions réversibles.
> Toute écriture de code nécessite une approbation humaine. Toujours.

---

## Autres axes (non planifiés)

- [ ] Écriture calendrier Google (créer des événements, pas seulement lire)
- [ ] Migration Mac Mini M4 Pro (infrastructure)
- [ ] Support Android (remplace le polling iOS par un mécanisme web)
- [ ] Mémoire conversationnelle agentique (Couche 3) — Jarvis émet des commandes
      mémoire structurées pendant la conversation, pas seulement via l'analyser
- [ ] Compression mémoire mensuelle (résumés autobiographiques sur longue durée)
