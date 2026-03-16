#!/bin/bash
# ══════════════════════════════════════════════════════════
#  Jarvis Production Test Suite
#  Usage: ./scripts/jarvis-test.sh [user_code]
#  Default user: KORBEN99
# ══════════════════════════════════════════════════════════

API="http://localhost:8000"
USER="${1:-KORBEN99}"
PASS=0
FAIL=0
WARN=0

# ── Colours ──────────────────────────────────────────────
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
BOLD="\033[1m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✓${RESET} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${RESET} $1"; ((FAIL++)); }
warn() { echo -e "  ${YELLOW}⚠${RESET} $1"; ((WARN++)); }
section() { echo -e "\n${BOLD}${BLUE}▶ $1${RESET}"; }

check_json_field() {
    local json="$1" field="$2"
    echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d$field)" 2>/dev/null
}

echo -e "\n${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║     Jarvis Test Suite — $(date '+%Y-%m-%d %H:%M')     ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo -e "  User: ${BOLD}$USER${RESET}  |  API: ${BOLD}$API${RESET}"

# ══════════════════════════════════════════════════════════
section "1. Infrastructure — Docker Containers"
# ══════════════════════════════════════════════════════════

for svc in jarvis-api jarvis-redis jarvis-qdrant; do
    state=$(docker inspect --format='{{.State.Status}}' "$svc" 2>/dev/null)
    if [ "$state" = "running" ]; then
        ok "$svc is running"
    else
        fail "$svc is not running (state: ${state:-not found})"
    fi
done

# ══════════════════════════════════════════════════════════
section "2. Infrastructure — Redis"
# ══════════════════════════════════════════════════════════

redis_ping=$(docker exec jarvis-redis redis-cli ping 2>/dev/null)
if [ "$redis_ping" = "PONG" ]; then
    redis_mem=$(docker exec jarvis-redis redis-cli info memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
    redis_keys=$(docker exec jarvis-redis redis-cli dbsize 2>/dev/null)
    ok "Redis responding (memory: ${redis_mem}, keys: ${redis_keys})"
else
    fail "Redis not responding"
fi

# Check key Redis data structures
for key_pattern in "episodic:*" "briefing:*" "jarvis:self:*"; do
    count=$(docker exec jarvis-redis redis-cli --scan --pattern "$key_pattern" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        ok "Redis pattern '$key_pattern': $count keys"
    else
        warn "Redis pattern '$key_pattern': no keys (may be normal on fresh install)"
    fi
done

# ══════════════════════════════════════════════════════════
section "3. Infrastructure — Qdrant"
# ══════════════════════════════════════════════════════════

qdrant_resp=$(curl -s --max-time 5 "http://localhost:6333/collections" 2>/dev/null)
if echo "$qdrant_resp" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
    collections=$(echo "$qdrant_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
cols=d.get('result',{}).get('collections',[])
for c in cols:
    print(c['name'])
" 2>/dev/null)
    ok "Qdrant responding, collections: $(echo "$collections" | tr '\n' ' ')"

    for col in open-webui_knowledge jarvis_memory; do
        col_info=$(curl -s --max-time 5 "http://localhost:6333/collections/$col" 2>/dev/null)
        pts=$(echo "$col_info" | python3 -c "import json,sys; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null)
        if [ -n "$pts" ]; then
            ok "  Collection '$col': $pts vectors"
        else
            warn "  Collection '$col': not found or empty"
        fi
    done
else
    fail "Qdrant not responding"
fi

# ══════════════════════════════════════════════════════════
section "4. API — Health & Status"
# ══════════════════════════════════════════════════════════

status_resp=$(curl -s --max-time 10 "$API/status" 2>/dev/null)
if [ -z "$status_resp" ]; then
    fail "API /status: no response"
else
    api_status=$(echo "$status_resp" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    if [ "$api_status" = "online" ]; then
        ok "API /status: online"
    else
        fail "API /status: unexpected response ($api_status)"
    fi

    google_status=$(echo "$status_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
g=d.get('services',{}).get('google',{})
print(g.get('status','unknown'))
" 2>/dev/null)
    if [ "$google_status" = "configured" ]; then
        ok "Google services: configured (Gmail + Calendar)"
    else
        warn "Google services: $google_status"
    fi

    emotion=$(echo "$status_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d.get('services',{}).get('memory',{}).get('emotional_state','unknown'))
" 2>/dev/null)
    ok "Emotional state: $emotion"

    # ── LLM tiers (added with two-tier architecture) ──
    router_status=$(echo "$status_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('services',{}).get('router',{})
print(r.get('status','missing'), r.get('model','?'))
" 2>/dev/null)
    if echo "$router_status" | grep -q "missing"; then
        fail "Router tier: not present in /status response"
    else
        ok "Router tier: $router_status"
    fi

    reasoning_status=$(echo "$status_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('services',{}).get('reasoning',{})
print(r.get('status','missing'), r.get('model','?'))
" 2>/dev/null)
    if echo "$reasoning_status" | grep -q "missing"; then
        fail "Reasoning tier: not present in /status response"
    else
        ok "Reasoning tier: $reasoning_status"
    fi
fi

# ══════════════════════════════════════════════════════════
section "5. LLM tiers — endpoint reachability"
# ══════════════════════════════════════════════════════════

# Validate model config from /status (already fetched above — no new call needed).
# Then do a lightweight GET /models against each tier's API URL to confirm
# the endpoint is reachable without triggering any inference.

reasoning_url=$(echo "$status_resp" | python3 -c "
import json,sys
print(json.load(sys.stdin).get('services',{}).get('reasoning',{}).get('url',''))
" 2>/dev/null)
router_url=$(echo "$status_resp" | python3 -c "
import json,sys
print(json.load(sys.stdin).get('services',{}).get('router',{}).get('url',''))
" 2>/dev/null)

for tier_name in "reasoning:$reasoning_url" "router:$router_url"; do
    tname="${tier_name%%:*}"
    turl="${tier_name#*:}"
    if [ -z "$turl" ] || [ "$turl" = "None" ]; then
        warn "LLM $tname: URL not reported by /status"
        continue
    fi
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
        -H "Authorization: Bearer dummy" "$turl/models" 2>/dev/null)
    # 200 = OK, 401 = reachable but auth required (also fine — proves endpoint up)
    if [ "$http_code" = "200" ] || [ "$http_code" = "401" ]; then
        ok "LLM $tname endpoint reachable ($turl) — HTTP $http_code"
    elif [ "$http_code" = "000" ]; then
        fail "LLM $tname endpoint unreachable ($turl) — connection refused or timeout"
    else
        warn "LLM $tname endpoint returned HTTP $http_code ($turl)"
    fi
done

# ══════════════════════════════════════════════════════════
section "6. User memory state"
# ══════════════════════════════════════════════════════════

# Read existing memory entries from Redis — no writes, no LLM calls.
episodic_count=$(docker exec jarvis-redis redis-cli zcard "episodic:$USER:conversations" 2>/dev/null)
semantic_count=$(docker exec jarvis-redis redis-cli hlen "semantic:$USER" 2>/dev/null)
working_count=$(docker exec jarvis-redis redis-cli hlen "working:$USER" 2>/dev/null)

if [ "${episodic_count:-0}" -gt 0 ] 2>/dev/null; then
    ok "Episodic memory ($USER): $episodic_count conversations"
else
    warn "Episodic memory ($USER): empty (no past conversations recorded)"
fi
ok "Semantic memory ($USER): ${semantic_count:-0} entries"
ok "Working memory ($USER): ${working_count:-0} entries"

# Show most recent episodic entry timestamp (score = unix timestamp)
last_ts=$(docker exec jarvis-redis redis-cli zrevrange "episodic:$USER:conversations" 0 0 WITHSCORES 2>/dev/null \
    | tail -1)
if [ -n "$last_ts" ]; then
    last_dt=$(python3 -c "import datetime; print(datetime.datetime.fromtimestamp(float('$last_ts')).strftime('%Y-%m-%d %H:%M'))" 2>/dev/null)
    ok "  Last episodic entry: $last_dt"
fi

# ══════════════════════════════════════════════════════════
section "7. External data — weather feed"
# ══════════════════════════════════════════════════════════

# Check wttr.in is reachable and returns weather data — read-only HTTP GET.
weather_resp=$(curl -s --max-time 8 "https://wttr.in/Paris?format=j1" 2>/dev/null)
temp=$(echo "$weather_resp" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d['current_condition'][0]['temp_C'] + '°C')
except:
    print('')
" 2>/dev/null)
if [ -n "$temp" ]; then
    ok "Weather feed (wttr.in): Paris currently $temp"
else
    warn "Weather feed (wttr.in): no response or unexpected format"
fi

# ══════════════════════════════════════════════════════════
section "8. External services — Google"
# ══════════════════════════════════════════════════════════

# Google auth status already fetched via /status — no new calls.
if [ "$google_status" = "configured" ]; then
    ok "Gmail: credentials configured (client_id + refresh_token present)"
    ok "Calendar: credentials configured (same OAuth flow)"
elif [ "$google_status" = "unconfigured" ]; then
    warn "Google services: not configured — Gmail and Calendar routes inactive"
else
    warn "Google services: status=$google_status"
fi

# ══════════════════════════════════════════════════════════
section "9. Conversation history endpoint"
# ══════════════════════════════════════════════════════════

# GET-only check of the history endpoint — no writes.
history_resp=$(curl -s --max-time 5 "$API/history/$USER" 2>/dev/null)
history_count=$(echo "$history_resp" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    # endpoint returns list or {sessions:[...]} depending on version
    if isinstance(d, list): print(len(d))
    else: print(len(d.get('sessions', d.get('history', []))))
except:
    print('')
" 2>/dev/null)
if [ -n "$history_count" ]; then
    ok "History endpoint ($USER): $history_count session(s) available"
else
    warn "History endpoint: no data or unexpected format"
fi

# ══════════════════════════════════════════════════════════
section "10. Routing config — embedding model"
# ══════════════════════════════════════════════════════════

# Check embedding model loaded successfully from container logs — read-only.
jarvis_logs_head=$(docker logs jarvis-api --tail=500 2>&1)
if echo "$jarvis_logs_head" | grep -qi "embed\|sentence.transform\|paraphrase-multilingual"; then
    ok "Embedding model: referenced in startup logs"
else
    warn "Embedding model: no mention in recent logs (may load lazily on first use)"
fi

router_mode=$(echo "$status_resp" | python3 -c "
import json,sys
r=json.load(sys.stdin).get('services',{}).get('router',{})
print(r.get('status','unknown'))
" 2>/dev/null)
if [ "$router_mode" = "llm" ]; then
    ok "Active router: LLM tier (fast intent classification)"
elif [ "$router_mode" = "embedding_fallback" ]; then
    warn "Active router: embedding fallback (LLM router disabled or unreachable)"
else
    warn "Active router: status=$router_mode"
fi

# ══════════════════════════════════════════════════════════
section "11. Briefing cache"
# ══════════════════════════════════════════════════════════

# Read-only: check whether today's briefing is cached in Redis.
# Never trigger generation — that is an LLM call with side effects.
briefing_cached=$(curl -s --max-time 5 "$API/briefing/$USER" \
    -H "Authorization: Bearer $USER" 2>/dev/null)
cached_ts=$(echo "$briefing_cached" | python3 -c "
import json,sys; print(json.load(sys.stdin).get('generated_at','')[:16])
" 2>/dev/null)
if [ -n "$cached_ts" ]; then
    ok "Briefing cached (generated: $cached_ts)"
    preview=$(echo "$briefing_cached" | python3 -c "
import json,sys; print(json.load(sys.stdin).get('text','')[:100])
" 2>/dev/null)
    ok "  Preview: ${preview}…"
else
    warn "No briefing cached for $USER today — will be generated at scheduled time"
fi

# ══════════════════════════════════════════════════════════
section "12. API — Proto-self"
# ══════════════════════════════════════════════════════════

self_state=$(curl -s --max-time 10 "$API/self/state" 2>/dev/null)
focus=$(echo "$self_state" | python3 -c "import json,sys; print(json.load(sys.stdin).get('current_focus','')[:80])" 2>/dev/null)
refl_count=$(echo "$self_state" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reflection_count',0))" 2>/dev/null)
last_refl=$(echo "$self_state" | python3 -c "import json,sys; print(json.load(sys.stdin).get('last_reflection','')[:16])" 2>/dev/null)
goals=$(echo "$self_state" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(len(d.get('goals',[])))
" 2>/dev/null)

if [ -n "$focus" ]; then
    ok "Self state: $goals goals, $refl_count reflections, last: $last_refl"
    ok "  Focus: $focus"
else
    fail "Self state endpoint failed or current_focus empty"
fi

self_log=$(curl -s --max-time 5 "$API/self/log?n=1" 2>/dev/null)
last_action=$(echo "$self_log" | python3 -c "
import json,sys
log=json.load(sys.stdin).get('log',[])
if log: print(f\"{log[0].get('action','?')} — {log[0].get('outcome','?')[:60]}\")
else: print('empty')
" 2>/dev/null)
ok "  Last action: $last_action"

# ── Validate last reflection entry structure (read-only — no new cycle triggered) ──
# The scheduler already runs a reflection at startup + every REFLECTION_INTERVAL_HOURS.
# We validate the state left by those cycles without firing a new one.
last_entry=$(docker exec jarvis-redis redis-cli zrevrange jarvis:self:reflection_log 0 0 2>/dev/null)
if [ -z "$last_entry" ]; then
    warn "Redis jarvis:self:reflection_log is empty — scheduler reflection may not have completed yet"
else
    reflect_ok=$(echo "$last_entry" | python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.read().strip())
    required=['focus','action','reason','params','outcome','health']
    missing=[k for k in required if k not in d]
    if missing:
        print('missing:' + ','.join(missing))
    elif not d.get('focus','').strip():
        print('empty_focus')
    elif d.get('action','') not in ['nothing','store_insight','flag_knowledge_gap','send_notification','update_self_note','consolidate_memory','check_health']:
        print('invalid_action:' + d.get('action',''))
    else:
        print('ok:' + d['action'])
except Exception as e:
    print('parse_error:' + str(e))
" 2>/dev/null)

    if echo "$reflect_ok" | grep -q "^ok:"; then
        action_taken=$(echo "$reflect_ok" | cut -d: -f2)
        ok "Last reflection entry: valid structure (action=$action_taken)"
    else
        fail "Last reflection entry: invalid — $reflect_ok"
    fi

    # Validate health sub-object in last reflection
    reflect_health=$(echo "$last_entry" | python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.read().strip())
    h=d.get('health',{})
    bad=[s for s,v in h.items() if v != 'ok']
    print(f\"services={list(h.keys())} unhealthy={bad}\")
except:
    print('parse_error')
" 2>/dev/null)
    ok "  Health snapshot: $reflect_health"
fi

# Redis key counts (read-only)
gaps_count=$(docker exec jarvis-redis redis-cli zcard jarvis:self:knowledge_gaps 2>/dev/null)
log_count=$(docker exec jarvis-redis redis-cli zcard jarvis:self:reflection_log 2>/dev/null)
if [ "${log_count:-0}" -gt 0 ] 2>/dev/null; then
    ok "Redis: reflection_log=$log_count entries, knowledge_gaps=${gaps_count:-0} entries"
else
    warn "Redis: jarvis:self:reflection_log is empty"
fi

# ══════════════════════════════════════════════════════════
section "13. API — Memory endpoints"
# ══════════════════════════════════════════════════════════

profile_resp=$(curl -s --max-time 5 "$API/memory/profile/$USER" 2>/dev/null)
profile_keys=$(echo "$profile_resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(len(d.get('profile',{})))
" 2>/dev/null)
if [ -n "$profile_keys" ] && [ "$profile_keys" -gt 0 ] 2>/dev/null; then
    ok "Memory profile: $profile_keys keys"
else
    warn "Memory profile: empty or unavailable"
fi

recent_resp=$(curl -s --max-time 5 "$API/memory/recent/$USER" 2>/dev/null)
recent_count=$(echo "$recent_resp" | python3 -c "
import json,sys
print(len(json.load(sys.stdin).get('conversations',[])))
" 2>/dev/null)
ok "Recent conversations (24h): ${recent_count:-0}"

# ══════════════════════════════════════════════════════════
section "14. Jarvis-self.json integrity"
# ══════════════════════════════════════════════════════════

self_json=$(docker exec jarvis-api cat /app/data/jarvis-self.json 2>/dev/null)
if [ -z "$self_json" ]; then
    fail "jarvis-self.json: not readable"
else
    # Top-level counts
    goals_count=$(echo "$self_json" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('goals',[])))" 2>/dev/null)
    learnings_count=$(echo "$self_json" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('learnings',[])))" 2>/dev/null)
    growth_count=$(echo "$self_json" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('growth_log',[])))" 2>/dev/null)
    version=$(echo "$self_json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('identity',{}).get('version','?'))" 2>/dev/null)
    ok "jarvis-self.json v$version: $goals_count goals, $learnings_count learnings, $growth_count growth entries"

    # current_focus must be non-empty (set by each reflection cycle)
    current_focus=$(echo "$self_json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('current_focus','').strip())" 2>/dev/null)
    if [ -n "$current_focus" ]; then
        ok "  current_focus: ${current_focus:0:80}"
    else
        warn "  current_focus is empty — reflection may not have run yet"
    fi

    # self_notes array must exist (even if empty is acceptable on first boot)
    self_notes_count=$(echo "$self_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('self_notes',[])))" 2>/dev/null)
    ok "  self_notes: ${self_notes_count:-0} entries"

    # Each goal must have 'label' and 'description' keys (required by _fmt_goals in self.py)
    goals_integrity=$(echo "$self_json" | python3 -c "
import json,sys
goals=json.load(sys.stdin).get('goals',[])
bad=[i for i,g in enumerate(goals) if 'label' not in g or 'description' not in g]
if bad:
    print('missing_fields:goals[' + ','.join(str(i) for i in bad) + ']')
else:
    print(f'ok ({len(goals)} goals with label+description)')
" 2>/dev/null)
    if echo "$goals_integrity" | grep -q "^ok"; then
        ok "  Goals structure: $goals_integrity"
    else
        fail "  Goals structure: $goals_integrity"
    fi

    # reflection_count must be a number and > 0 after startup reflection
    refl_count_json=$(echo "$self_json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reflection_count',0))" 2>/dev/null)
    if [ "${refl_count_json:-0}" -gt 0 ] 2>/dev/null; then
        ok "  reflection_count: $refl_count_json"
    else
        warn "  reflection_count is 0 or missing — reflection may not have completed"
    fi
fi

# ══════════════════════════════════════════════════════════
section "15. Scheduled jobs check"
# ══════════════════════════════════════════════════════════

# APScheduler logs natural-language messages, not job IDs in quotes.
# Patterns matched below correspond to logger.info() calls in lifespan():
#   "Morning briefing scheduled at ..."  → id="morning_briefing"
#   "Self reflection scheduled every ..."→ id="self_reflection"
#   "Nightly review scheduled at 23:00"  → id="nightly_review"
jarvis_logs=$(docker logs jarvis-api --tail=300 2>&1)

if echo "$jarvis_logs" | grep -q "Morning briefing scheduled"; then
    ok "Scheduler: morning_briefing registered"
else
    warn "Scheduler: morning_briefing not found in logs (check BRIEFING_ENABLED)"
fi

if echo "$jarvis_logs" | grep -q "Self reflection scheduled"; then
    ok "Scheduler: self_reflection registered"
else
    fail "Scheduler: self_reflection not found in logs (APScheduler may have failed to start)"
fi

if echo "$jarvis_logs" | grep -q "Nightly review scheduled"; then
    ok "Scheduler: nightly_review registered"
else
    fail "Scheduler: nightly_review not found in logs (APScheduler may have failed to start)"
fi

# Confirm APScheduler itself started (not just job registrations)
if echo "$jarvis_logs" | grep -qE "Scheduler started|AsyncIOScheduler started"; then
    ok "Scheduler: APScheduler started successfully"
elif echo "$jarvis_logs" | grep -q "Scheduler failed to start"; then
    fail "Scheduler: APScheduler failed to start — check logs"
else
    warn "Scheduler: no explicit 'started' log line found (may still be running)"
fi

# Check for a completed reflection in the log (startup reflection runs immediately)
if echo "$jarvis_logs" | grep -q "Reflection complete"; then
    ok "Scheduler: startup reflection completed at least once"
else
    warn "Scheduler: no completed reflection found in recent logs"
fi

# ══════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════

TOTAL=$((PASS + FAIL + WARN))
echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Results: ${GREEN}${PASS} passed${RESET}  ${RED}${FAIL} failed${RESET}  ${YELLOW}${WARN} warnings${RESET}  (${TOTAL} total)"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
