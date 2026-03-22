# Redis — Guide opérationnel Jarvis

Connexion au container Redis :
```bash
docker exec -it jarvis-redis redis-cli
```

---

## Carte des clés

| Pattern | Type | Contenu |
|---|---|---|
| `user:{code}:profile` | hash | Profil utilisateur (faits appris) |
| `chat:{code}:{session}` | list | Historique de conversation |
| `episodic:{code}:conversations` | zset | Mémoire épisodique (score = timestamp) |
| `jarvis:self:reflection_log` | zset | Journal des réflexions autonomes |
| `jarvis:self:knowledge_gaps` | zset | Lacunes de connaissance détectées |
| `jarvis:self:gap_counts` | hash | Compteurs par lacune (slug → count) |
| `jarvis:self:notif:{code}:{date}` | string | Garde anti-doublon email (TTL 24h) |
| `jarvis:push:pending:{code}` | list | Notifications iOS en attente |
| `jarvis:push:cooldown:{code}` | string | Cooldown push (TTL 2h) |
| `jarvis:device:token:{code}` | string | Token APNs de l'app iOS |
| `jarvis:{code}:pending_calendar_action` | string | Action calendrier en attente (TTL 10min) |
| `jarvis:{code}:tomorrow_suggestions` | string | Suggestions du lendemain (TTL 24h) |
| `trade:{code}:index` | set | Index ISINs du portefeuille |
| `trade:{code}:pos:{isin}` | hash | Données d'une position (prix, seuils…) |
| `trade:{code}:last_import_ts` | string | Timestamp dernier import CSV |
| `trade:price_cache:{isin}` | string | Cache prix yfinance |
| `trade:{code}:pending_alerts` | list | Alertes trading en attente |

`{code}` = code utilisateur (ex: `KORBEN99`)

---

## Lister des clés

```bash
# Toutes les clés (dangereux sur gros volume — préférer SCAN)
KEYS *

# Par namespace
KEYS user:*
KEYS chat:KORBEN99:*
KEYS jarvis:self:*
KEYS trade:KORBEN99:*

# Scan paginé (safe en production)
SCAN 0 MATCH "jarvis:*" COUNT 50
```

---

## Inspecter

```bash
# Type d'une clé
TYPE user:KORBEN99:profile

# TTL (-1 = permanent, -2 = n'existe pas)
TTL jarvis:self:notif:KORBEN99:2026-03-22

# Profil utilisateur complet
HGETALL user:KORBEN99:profile
-> GET pour un string

# Nombre de champs du profil
HLEN user:KORBEN99:profile

# Une clé du profil
HGET user:KORBEN99:profile hobby:kart

# Dernières réflexions (les 5 plus récentes)
ZREVRANGE jarvis:self:reflection_log 0 4 WITHSCORES

# Lacunes connaissance avec scores
ZREVRANGE jarvis:self:knowledge_gaps 0 -1 WITHSCORES

# Compteurs de lacunes
HGETALL jarvis:self:gap_counts

# Historique conversation (50 derniers messages)
LRANGE chat:KORBEN99:iphone-main -50 -1

# Mémoire épisodique récente (dernières 24h)
ZRANGEBYSCORE episodic:KORBEN99:conversations -inf +inf LIMIT 0 10

# Notifications iOS en attente
LRANGE jarvis:push:pending:KORBEN99 0 -1

# Position trading
HGETALL trade:KORBEN99:pos:FR0000131104
```

---

## Nettoyer

### Profil utilisateur
```bash
# Supprimer une clé du profil
HDEL user:KORBEN99:profile hobby:tennis

# Corriger une valeur
HSET user:KORBEN99:profile name Sébastien

# Vider tout le profil (irréversible)
DEL user:KORBEN99:profile
```

### Lacunes de connaissance
```bash
# Vider toutes les lacunes (après acceptation d'une proposition)
DEL jarvis:self:knowledge_gaps
DEL jarvis:self:gap_counts

# Réinitialiser un compteur spécifique
HDEL jarvis:self:gap_counts test_validation
```

### Notifications et cooldowns
```bash
# Vider la file de push d'un utilisateur
DEL jarvis:push:pending:KORBEN99

# Lever le cooldown push (forcer un push immédiat possible)
DEL jarvis:push:cooldown:KORBEN99

# Supprimer le token device (désactive les push)
DEL jarvis:device:token:KORBEN99
```

### Conversations
```bash
# Vider une session (aussi accessible via DELETE /conversations/{session_id})
DEL chat:KORBEN99:iphone-main

# Supprimer toutes les sessions d'un utilisateur
# (depuis le shell, pas depuis redis-cli)
docker exec jarvis-redis redis-cli --scan --pattern "chat:KORBEN99:*" | xargs docker exec jarvis-redis redis-cli DEL
```

### Mémoire épisodique
```bash
# Taille de la mémoire épisodique
ZCARD episodic:KORBEN99:conversations

# Supprimer les entrées les plus anciennes (garder les 200 dernières)
ZREMRANGEBYRANK episodic:KORBEN99:conversations 0 -201

# Vider complètement (irréversible — Qdrant non affecté)
DEL episodic:KORBEN99:conversations
```

### Trading
```bash
# Supprimer une position
SREM trade:KORBEN99:index FR0000131104
DEL trade:KORBEN99:pos:FR0000131104

# Vider le cache prix
docker exec jarvis-redis redis-cli --scan --pattern "trade:price_cache:*" | xargs docker exec jarvis-redis redis-cli DEL

# Forcer un réimport CSV au prochain cycle
DEL trade:KORBEN99:last_import_ts
```

### Réflexion autonome
```bash
# Vider le journal de réflexion (repart de zéro)
DEL jarvis:self:reflection_log
```

---

## Divers

```bash
# Taille totale de la base (nombre de clés)
DBSIZE

# Mémoire utilisée
INFO memory

# Monitorer les commandes en temps réel (ctrl+c pour stopper)
MONITOR

# Quitter
EXIT
```
