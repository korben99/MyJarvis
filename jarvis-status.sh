#!/bin/bash
echo "=== JARVIS STATUS ==="
echo ""

source /opt/jarvis/.env

echo "── Local Services ──"
for svc in jarvis-qdrant jarvis-redis jarvis-webui jarvis-api; do
    if docker ps --format '{{.Names}}' | grep -q "$svc"; then
        echo "  ✅ $svc"
    else
        echo "  ⬜ $svc (not running)"
    fi
done

echo ""
echo "── LLM Providers ──"

# OpenAI
if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "sk-your-openai-key-here" ]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  ✅ OpenAI API — connected"
    else
        echo "  ❌ OpenAI API — error (HTTP $HTTP_CODE)"
    fi
else
    echo "  ⬜ OpenAI API — no key configured"
fi

# Claude
if [ -n "$ANTHROPIC_API_KEY" ] && [ "$ANTHROPIC_API_KEY" != "sk-ant-your-claude-key-here" ]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
        -H "x-api-key: $ANTHROPIC_API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        https://api.anthropic.com/v1/models)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  ✅ Claude API — connected"
    else
        echo "  ❌ Claude API — error (HTTP $HTTP_CODE)"
    fi
else
    echo "  ⬜ Claude API — no key configured"
fi

# Ollama
if curl -s --connect-timeout 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✅ Ollama (local) — online"
else
    echo "  ⬜ Ollama (local) — not running"
fi

echo ""
echo "── External APIs ──"

# Weather (Open-Meteo — no key required)
WEATHER_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
    "https://api.open-meteo.com/v1/forecast?latitude=48.85&longitude=2.35&current_weather=true")
if [ "$WEATHER_CODE" = "200" ]; then
    echo "  ✅ Open-Meteo (weather) — reachable"
else
    echo "  ❌ Open-Meteo (weather) — unreachable (HTTP $WEATHER_CODE)"
fi

# DuckDuckGo
DDG_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "https://duckduckgo.com/")
if [ "$DDG_CODE" = "200" ] || [ "$DDG_CODE" = "301" ] || [ "$DDG_CODE" = "302" ]; then
    echo "  ✅ DuckDuckGo (web search) — reachable"
else
    echo "  ❌ DuckDuckGo (web search) — unreachable (HTTP $DDG_CODE)"
fi

echo ""
echo "── RAG Knowledge Base ──"
QDRANT_INFO=$(curl -s http://localhost:6333/collections/open-webui_knowledge 2>/dev/null)
if echo "$QDRANT_INFO" | python3 -c "import sys,json; info=json.load(sys.stdin)['result']; print(f'  ✅ {info[\"points_count\"]} vectors in Qdrant')" 2>/dev/null; then
    :
else
    echo "  ❌ Qdrant collection not found"
fi

echo ""
echo "── Memory ──"
FIRST_USER=$(python3 -c "
import json, os
path = os.getenv('USERS_LIST', '/opt/jarvis/jarvis-core/JarvisData/users_list.json')
try:
    users = json.load(open(path))
    print(users[0]['code'] if users else 'KORBEN99')
except Exception:
    print('KORBEN99')
" 2>/dev/null)
FIRST_USER=${FIRST_USER:-KORBEN99}

MEM_STATUS=$(curl -s "http://localhost:8000/memory/profile/${FIRST_USER}" 2>/dev/null)
if echo "$MEM_STATUS" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    python3 -c "
import json, sys
d = json.load(sys.stdin)
keys = len(d.get('profile', {}))
proj = len(d.get('projects', []))
print(f'  ✅ User profile (${FIRST_USER}): {keys} facts, {proj} projects')
" <<< "$MEM_STATUS" 2>/dev/null
else
    echo "  ⬜ Memory not available (jarvis-api not running?)"
fi

EMOTION=$(curl -s http://localhost:8000/memory/emotional-state 2>/dev/null)
MOOD=$(echo "$EMOTION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mood','?'))" 2>/dev/null)
echo "  ✅ Current mood: ${MOOD:-unknown}"

echo ""
echo "── Proto-Self ──"
SELF=$(curl -s http://localhost:8000/self/state 2>/dev/null)
if echo "$SELF" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    python3 -c "
import json, sys
d = json.load(sys.stdin)
focus  = (d.get('current_focus') or '(not set)')[:70]
count  = d.get('reflection_count', 0)
action = d.get('last_action') or {}
last   = action.get('action', '?') if action else '?'
rels   = d.get('user_relations', {})
print(f'  ✅ Reflection cycles: {count}')
print(f'  ✅ Focus: {focus}')
print(f'  ✅ Last autonomous action: {last}')
if rels:
    for code, rel in rels.items():
        print(f'  ✅ Relation {code}: affinity={rel[\"affinity\"]} | style={rel[\"interaction_style\"]} | mood={rel[\"average_interaction_mood\"]}')
else:
    print('  ⬜ No user relations yet (first nightly review pending)')
" <<< "$SELF" 2>/dev/null
else
    echo "  ⬜ Proto-Self not available"
fi

echo ""
echo "── Anthropic Proxy (Claude Code local) ──"
PROXY_PID=$(launchctl list | awk '/com.jarvis.anthropic-proxy/ {print $1}')
if [ -n "$PROXY_PID" ] && [ "$PROXY_PID" != "-" ]; then
    PROXY_HEALTH=$(curl -s --connect-timeout 2 http://localhost:8090/health 2>/dev/null)
    if [ -n "$PROXY_HEALTH" ]; then
        echo "  ✅ anthropic-proxy — PID $PROXY_PID, écoute :8090"
        # Vérifie que l'endpoint raw de Jarvis répond
        RAW_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 \
            -X POST http://localhost:8000/v1/raw/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"messages":[{"role":"user","content":"ping"}],"stream":false}')
        if [ "$RAW_CODE" = "200" ]; then
            echo "  ✅ /v1/raw/chat/completions — OK"
        else
            echo "  ❌ /v1/raw/chat/completions — HTTP $RAW_CODE (Jarvis redémarré ?)"
        fi
    else
        echo "  ❌ anthropic-proxy — process $PROXY_PID mais ne répond pas sur :8090"
    fi
else
    echo "  ⬜ anthropic-proxy — non démarré"
    echo "     launchctl start com.jarvis.anthropic-proxy"
fi

echo ""
echo "── Access ──"
IP=$(hostname -I | awk '{print $1}')
echo "  Open WebUI:  http://${IP}:3000"
echo "  Jarvis API:  http://${IP}:8000"
echo "  Proxy CC:    http://${IP}:8090  (ANTHROPIC_BASE_URL)"
echo "  Qdrant:      http://${IP}:6333/dashboard"
