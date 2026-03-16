#!/bin/bash
echo "=== JARVIS STATUS ==="
echo ""

source /opt/jarvis/.env

echo "── Local Services ──"
for svc in jarvis-qdrant jarvis-redis jarvis-webui jarvis-api jarvis-claude-proxy; do
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
    HTTP_CODE=$(curl -s -o /tmp/oai_resp.json -w "%{http_code}" --connect-timeout 5 \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  ✅ OpenAI API — connected"
    else
        echo "  ❌ OpenAI API — error (HTTP $HTTP_CODE)"
    fi
    rm -f /tmp/oai_resp.json
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

# RunPod / Local Ollama (for future Mac Studio)
if [ -n "$RUNPOD_OLLAMA_URL" ] && [ "$RUNPOD_OLLAMA_URL" != "http://localhost:11434" ]; then
    HTTP_CODE=$(curl -s -o /tmp/ollama_resp.json -w "%{http_code}" --connect-timeout 5 "$RUNPOD_OLLAMA_URL/v1/models")
    if [ "$HTTP_CODE" = "200" ] && python3 -c "import json; json.load(open('/tmp/ollama_resp.json'))" 2>/dev/null; then
        echo "  ✅ Ollama (RunPod) — online"
        python3 -c "
import json
data = json.load(open('/tmp/ollama_resp.json'))
for m in data.get('data', []):
    print(f\"     • {m['id']}\")" 2>/dev/null
    else
        echo "  ❌ Ollama (RunPod) — offline"
    fi
    rm -f /tmp/ollama_resp.json
else
    # Check local Ollama (future Mac Studio)
    if curl -s --connect-timeout 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✅ Ollama (local) — online"
    else
        echo "  ⬜ Ollama (local) — not running"
    fi
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
source /opt/jarvis/.env
MEM_STATUS=$(curl -s http://localhost:8000/memory/profile 2>/dev/null)
if [ -n "$MEM_STATUS" ]; then
    PROFILE_KEYS=$(echo "$MEM_STATUS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('profile',{})))" 2>/dev/null)
    PROJECTS=$(echo "$MEM_STATUS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('projects',[])))" 2>/dev/null)
    echo "  ✅ User profile: ${PROFILE_KEYS:-0} facts"
    echo "  ✅ Projects tracked: ${PROJECTS:-0}"

    EMOTION=$(curl -s http://localhost:8000/memory/emotional-state 2>/dev/null)
    MOOD=$(echo "$EMOTION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mood','?'))" 2>/dev/null)
    echo "  ✅ Jarvis mood: ${MOOD:-unknown}"
else
    echo "  ⬜ Memory not available (jarvis-api not running)"
fi

# Check reflections
REFL_COUNT=$(ls /opt/jarvis/data/reflections/*.json 2>/dev/null | wc -l)
echo "  ✅ Reflections: ${REFL_COUNT} days"


echo ""
echo "── Access ──"
IP=$(hostname -I | awk '{print $1}')
echo "  Open WebUI:  http://${IP}:3000"
echo "  Qdrant:      http://${IP}:6333/dashboard"
