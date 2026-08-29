# Redis — operations guide

Connect to the Redis container:

```bash
docker exec -it jarvis-redis redis-cli
```

Throughout, `{code}` is a user code (e.g. `ALICE1`).

---

## Key map

### Per user

| Pattern | Type | Contents |
|---|---|---|
| `user:{code}:profile` | hash | User profile — learned stable facts |
| `user:{code}:profile:ts` | hash | Per-field timestamps for the profile |
| `user:{code}:profile_narrative` | string | Narrative rendering of the profile |
| `user:{code}:preferences` | hash | Explicit user preferences |
| `user:{code}:interest_weights` | hash | Interest weights used for briefing selection |
| `user:{code}:projects` | string | User projects and their state |
| `chat:{code}:{session}` | list | Conversation history for one session |
| `episodic:{code}:conversations` | zset | Episodic memory (score = timestamp) |

### Self / reflection

| Pattern | Type | Contents |
|---|---|---|
| `jarvis:self:reflection_log` | zset | Autonomous reflection journal |
| `jarvis:self:knowledge_gaps` | zset | Detected knowledge gaps |
| `jarvis:self:gap_cooldown:{slug}` | string | Per-topic cooldown after a gap is handled |
| `jarvis:self:refine_cooldown` | string | Rate limit on `refine_prompt` proposals |
| `jarvis:self:matiere_empreinte` | string | Fingerprint of persistent signals (TTL 7 d) |
| `jarvis:self:stall` | — | Stalled-project tracking |
| `jarvis:self:health_alert` | string | Internal-health alert guard |
| `jarvis:self:last_consolidate` | string | Last memory consolidation pass |
| `jarvis:self:last_prune` | string | Last memory prune pass |
| `jarvis:self:notif:{code}:{date}` | string | Email dedup guard (TTL 24 h) |
| `jarvis:emotional_state` | string | Three-dimensional emotional state |

### Vitals, incidents, maintenance

| Pattern | Type | Contents |
|---|---|---|
| `jarvis:vitals` | string | Vitals snapshot (TTL 15 min) |
| `jarvis:incidents` | — | Consolidated incident buffer |
| `jarvis:cve` | — | Latest CVE scan result |
| `jarvis:maintenance` | string | Active maintenance window |
| `jarvis:boot_at` | string | Last boot timestamp |
| `jarvis:last_shutdown` | string | Last clean shutdown |
| `jarvis:usage_counters` | hash | Usage counters |
| `jarvis:metrics:{name}` | — | Runtime metrics |
| `jarvis:admin_alert:cooldown` | string | `alert_admin` cooldown |

### Push notifications

| Pattern | Type | Contents |
|---|---|---|
| `jarvis:push:pending:{code}` | list | Pending iOS notifications |
| `jarvis:push:cooldown:{code}` | string | Push cooldown (TTL 2 h) |
| `jarvis:device:token:{code}` | string | APNs token of the iOS app |

### Agent

| Pattern | Type | Contents |
|---|---|---|
| `jarvis:agent:queue` | list | Task queue |
| `jarvis:agent:index` | — | Task index |
| `jarvis:agent:task:{id}` | — | One task's state |
| `jarvis:agent:cancel:{id}` | string | Cancellation request |
| `jarvis:agent:push:{id}` | — | Completion notification for a task |

### Session-scoped

| Pattern | Type | Contents |
|---|---|---|
| `jarvis:sticky_rag:{code}:{session}` | string | RAG chunks pinned across turns of a session |
| `jarvis:{code}:pending_calendar_action` | string | Pending calendar action (TTL 10 min) |
| `jarvis:{code}:tomorrow_suggestions` | string | Next-day suggestions (TTL 24 h) |
| `jarvis:{code}:nightly_review:{date}` | string | Nightly review guard |
| `jarvis:{code}:nightly_maint:{date}` | string | Nightly maintenance guard |

### Trading

| Pattern | Type | Contents |
|---|---|---|
| `trade:{code}:index` | set | Portfolio ISIN index |
| `trade:{code}:pos:{isin}` | hash | One position (price, thresholds…) |
| `trade:{code}:last_import_ts` | string | Last CSV import timestamp |
| `trade:{code}:pending_alerts` | list | Pending trading alerts |
| `trade:price_cache:{isin}` | string | yfinance price cache |

---

## Listing keys

```bash
# Everything (dangerous on a large database — prefer SCAN)
KEYS *

# By namespace
KEYS user:*
KEYS chat:ALICE1:*
KEYS jarvis:self:*
KEYS trade:ALICE1:*

# Paginated scan (safe in production)
SCAN 0 MATCH "jarvis:*" COUNT 50
```

---

## Inspecting

```bash
# Type of a key
TYPE user:ALICE1:profile

# TTL (-1 = permanent, -2 = does not exist)
TTL jarvis:self:notif:ALICE1:2026-03-22

# Full user profile
HGETALL user:ALICE1:profile

# Number of profile fields
HLEN user:ALICE1:profile

# One profile field
HGET user:ALICE1:profile hobby:kart

# Five most recent reflections
ZREVRANGE jarvis:self:reflection_log 0 4 WITHSCORES

# Knowledge gaps with scores
ZREVRANGE jarvis:self:knowledge_gaps 0 -1 WITHSCORES

# Conversation history (last 50 messages)
LRANGE chat:ALICE1:iphone-main -50 -1

# Recent episodic memory
ZRANGEBYSCORE episodic:ALICE1:conversations -inf +inf LIMIT 0 10

# Pending iOS notifications
LRANGE jarvis:push:pending:ALICE1 0 -1

# Vitals snapshot
GET jarvis:vitals

# A trading position
HGETALL trade:ALICE1:pos:FR0000131104
```

---

## Cleaning up

### User profile

```bash
# Remove one profile field
HDEL user:ALICE1:profile hobby:tennis

# Correct a value
HSET user:ALICE1:profile name Alice

# Wipe the whole profile (irreversible)
DEL user:ALICE1:profile
```

### Knowledge gaps

```bash
# Clear all gaps
DEL jarvis:self:knowledge_gaps

# Lift the cooldown on one topic so it can be proposed again
DEL jarvis:self:gap_cooldown:<slug>
```

### Notifications and cooldowns

```bash
# Empty a user's push queue
DEL jarvis:push:pending:ALICE1

# Lift the push cooldown (allows an immediate push)
DEL jarvis:push:cooldown:ALICE1

# Remove the device token (disables push)
DEL jarvis:device:token:ALICE1
```

### Conversations

```bash
# Clear one session (also available via DELETE /conversations/{session_id})
DEL chat:ALICE1:iphone-main

# Remove every session of a user (from the shell, not from redis-cli)
docker exec jarvis-redis redis-cli --scan --pattern "chat:ALICE1:*" \
  | xargs docker exec jarvis-redis redis-cli DEL
```

### Episodic memory

```bash
# Size of episodic memory
ZCARD episodic:ALICE1:conversations

# Drop the oldest entries, keeping the last 200
ZREMRANGEBYRANK episodic:ALICE1:conversations 0 -201

# Wipe entirely (irreversible — Qdrant is not affected)
DEL episodic:ALICE1:conversations
```

### Trading

```bash
# Remove a position
SREM trade:ALICE1:index FR0000131104
DEL trade:ALICE1:pos:FR0000131104

# Flush the price cache
docker exec jarvis-redis redis-cli --scan --pattern "trade:price_cache:*" \
  | xargs docker exec jarvis-redis redis-cli DEL

# Force a CSV re-import on the next cycle
DEL trade:ALICE1:last_import_ts
```

### Autonomous reflection

```bash
# Clear the reflection journal (starts from scratch)
DEL jarvis:self:reflection_log
```

---

## Miscellaneous

```bash
# Total number of keys
DBSIZE

# Memory usage
INFO memory

# Watch commands live (ctrl+c to stop)
MONITOR

# Quit
EXIT
```
