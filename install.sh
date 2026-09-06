#!/usr/bin/env bash
# Jarvis — one-shot installer.
#
# Idempotent: safe to re-run after a git pull. Never overwrites .env or
# users_list.json once they exist. Gets you to the point where the only
# things left to do are: pick your models / fill in API keys in .env,
# edit users_list.json, and (if running local models) download them.
set -euo pipefail

JARVIS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$JARVIS_HOME"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  [ok] %s\n' "$1"; }
info() { printf '  [..] %s\n' "$1"; }
warn() { printf '  [!!] %s\n' "$1"; }

# ── Interactive helpers ───────────────────────────────────────────────────
# L'interactif ne se déclenche que sur un vrai terminal, et JARVIS_NONINTERACTIVE=1 le
# désarme : le script doit rester rejouable après un git pull, y compris sans humain.
INTERACTIVE=0
if [[ -t 0 && "${JARVIS_NONINTERACTIVE:-0}" != "1" ]]; then INTERACTIVE=1; fi

# ask <prompt> <default> — rend la saisie, ou le défaut si la ligne est vide.
ask() {
    local _v
    read -r -p "     $1 [$2]: " _v </dev/tty || true
    printf '%s' "${_v:-$2}"
}

# ask_yn <prompt> <y|n> — rend "true" ou "false", pour injection directe en JSON.
ask_yn() {
    local _v
    read -r -p "     $1 [$2/$([[ $2 == y ]] && echo n || echo y)]: " _v </dev/tty || true
    [[ "${_v:-$2}" =~ ^[yYoO] ]] && printf 'true' || printf 'false'
}

# set_env <clé> <valeur> — pose la clé dans .env, qu'elle y soit active, commentée ou
# absente. Ne fait rien sur une valeur vide : une clé vide n'est pas la même chose qu'une
# clé absente, `config.py` applique son défaut sur l'absence, pas sur la chaîne vide.
set_env() {
    local k="$1" v="$2"
    [[ -z "$v" ]] && return 0
    if grep -qE "^${k}=" .env; then
        sed -i '' "s|^${k}=.*|${k}=${v}|" .env
    elif grep -qE "^# *${k}=" .env; then
        sed -i '' "s|^# *${k}=.*|${k}=${v}|" .env
    else
        printf '%s=%s\n' "$k" "$v" >> .env
    fi
}

bold "Jarvis install — $JARVIS_HOME"

# ── 1. Preflight ─────────────────────────────────────────────────────────
bold "1/6 Checking prerequisites"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    warn "Jarvis requires macOS on Apple Silicon (mlx is a hard dependency, see requirements.txt)."
    exit 1
fi
ok "macOS on Apple Silicon"

if ! command -v python3.13 >/dev/null 2>&1; then
    warn "python3.13 not found. Install it with: brew install python@3.13"
    exit 1
fi
ok "python3.13 found"

if ! command -v docker >/dev/null 2>&1; then
    warn "docker CLI not found. Install Docker Desktop or OrbStack (recommended: https://orbstack.dev), then re-run."
    exit 1
fi
ok "docker CLI found"

if ! docker info >/dev/null 2>&1; then
    info "Docker daemon not running — attempting to start it"
    if [[ -d "/Applications/OrbStack.app" ]]; then
        open -a OrbStack
    elif [[ -d "/Applications/Docker.app" ]]; then
        open -a Docker
    fi
    for _ in $(seq 1 15); do
        docker info >/dev/null 2>&1 && break
        sleep 2
    done
fi
if docker info >/dev/null 2>&1; then
    ok "Docker daemon running"
else
    warn "Docker daemon still not up — start it manually before running jarvis-start"
fi

# ── 2. Python venv ───────────────────────────────────────────────────────
bold "2/6 Python environment"

if [[ ! -d venv ]]; then
    info "Creating venv"
    python3.13 -m venv venv
else
    ok "venv already exists"
fi

source venv/bin/activate
pip install --quiet --upgrade pip
info "Installing requirements.txt (can take a while — mlx/torch are large)"
pip install --quiet -r requirements.txt
ok "Python dependencies installed"

# ── 3. Runtime directories (all gitignored, none shipped in the repo) ───
bold "3/6 Creating data directories"

mkdir -p \
    logs \
    models \
    keys \
    RouterData \
    TradeData \
    RAGData/personal RAGData/work RAGData/documents RAGData/company RAGData/reflexions \
    jarvis-core/JarvisData
ok "Directories ready"

# ── 4. Config files ───────────────────────────────────────────────────────
bold "4/6 Config files"

if [[ ! -f .env ]]; then
    cp .env.example .env
    ok "Created .env from .env.example (LLM_LOCAL=yes by default)"
else
    ok ".env already exists — left untouched"
fi

# Secret de session Open WebUI. docker-compose.yml le rend obligatoire : en dur dans le
# compose, il devenait la même valeur pour tous les clones du dépôt. On en génère un par
# installation, et on ne touche pas à celui qui existe déjà — le changer déconnecterait
# toutes les sessions Open WebUI en cours.
if grep -qE '^WEBUI_SECRET_KEY=.+' .env 2>/dev/null; then
    ok "WEBUI_SECRET_KEY already set — left untouched"
else
    _secret="$(openssl rand -hex 32)"
    if grep -qE '^WEBUI_SECRET_KEY=' .env 2>/dev/null; then
        # BSD sed (macOS) : -i exige un suffixe de sauvegarde, ici vide.
        sed -i '' "s|^WEBUI_SECRET_KEY=.*|WEBUI_SECRET_KEY=$_secret|" .env
    else
        printf '\nWEBUI_SECRET_KEY=%s\n' "$_secret" >> .env
    fi
    unset _secret
    ok "Generated a WEBUI_SECRET_KEY in .env"
fi

USERS_LIST=jarvis-core/JarvisData/users_list.json
if [[ ! -f "$USERS_LIST" ]]; then
    cp DOCS/examples/users_list.example.json "$USERS_LIST"
    ok "Created $USERS_LIST from template"
    FRESH_USERS=1
else
    ok "$USERS_LIST already exists — left untouched"
    FRESH_USERS=0
fi

# ── 4b. Minimum viable configuration ──────────────────────────────────────
# Ne tourne QUE sur un fichier fraîchement créé : re-questionner à chaque git pull
# écraserait une configuration en service. Tout est skippable — Entrée prend le défaut.
if [[ "$FRESH_USERS" == "1" && "$INTERACTIVE" == "1" ]]; then
    echo
    bold "4b/6 Minimum setup — press Enter to accept each default"
    echo

    info "First user (administrator)"
    U_FIRST=$(ask "First name" "Alice")
    # Le code EST le secret d'API de cet utilisateur : pas de défaut, et on refuse celui
    # du gabarit. Un « changeme1 » laissé en place ouvre le compte admin à quiconque.
    while :; do
        U_CODE=$(ask "Access code (their API secret — long and random)" "")
        [[ -n "$U_CODE" && "$U_CODE" != "changeme1" ]] && break
        warn "The access code cannot be empty or 'changeme1' — it is the admin's password."
    done
    U_MAIL=$(ask "Email (leave empty if no Gmail/Calendar)" "")
    U_CITY=$(ask "City (for weather and briefing)" "Paris")
    U_TZ=$(ask "Timezone" "Europe/Paris")
    U_GOOGLE=false
    [[ -n "$U_MAIL" ]] && U_GOOGLE=$(ask_yn "Connect Gmail and Google Calendar for them?" n)

    python3 - "$USERS_LIST" "$U_FIRST" "$U_CODE" "$U_MAIL" "$U_CITY" "$U_TZ" "$U_GOOGLE" <<'PY'
import json, sys
path, first, code, mail, city, tz, google = sys.argv[1:8]
json.dump([{
    "id": 1, "firstname": first, "name": "", "code": code, "admin": True,
    "mail": mail, "city": city, "timezone": tz,
    "briefing_enabled": True, "trading": False,
    "google": google == "true", "profile": {},
}], open(path, "w"), ensure_ascii=False, indent=2)
PY
    ok "Wrote $USERS_LIST"

    echo
    info "Language — one per instance; prompts, lexicon and replies follow"
    set_env JARVIS_LANG "$(ask "Language (fr/en)" "en")"

    echo
    info "Local models — defaults are public on Hugging Face, no token needed"
    set_env PRIMARY_MODEL_LOCAL  "$(ask "Primary (chat, analysis, reflection)" "spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit")"
    set_env ROUTER_MODEL_LOCAL   "$(ask "Router (fast intent classifier)"      "mlx-community/Qwen2.5-1.5B-Instruct-4bit")"
    set_env VISION_MODEL_LOCAL   "$(ask "Vision (image description)"           "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit")"
    ok "Models set — reasoning tier reuses the primary unless you set REASONING_MODEL_LOCAL"

    echo
    info "Hugging Face token — only for gated models; the defaults above are not"
    set_env HF_TOKEN "$(ask "HF_TOKEN (leave empty to skip)" "")"

    echo
    ok "Minimum configuration written. Everything else has a working default in .env."
elif [[ "$FRESH_USERS" == "1" ]]; then
    warn "Non-interactive run — edit $USERS_LIST and .env by hand before starting"
fi

# ── 5. launchd service ────────────────────────────────────────────────────
bold "5/6 launchd service"

"$JARVIS_HOME/scripts/jarvis-launchd.sh" install
ok "launchd service installed (start it with: jarvis-start)"

ALIASES_LINE="source $JARVIS_HOME/DOCS/examples/jarvis-aliases.sh"
SHELL_RC="$HOME/.zshrc"
# Le test ne porte QUE sur la ligne source. Il matchait aussi "alias jarvis-start=", donc
# toute installation antérieure — celle qui définit les vieux alias launchctl bruts, non
# idempotents — se voyait déclarée à jour et n'était jamais migrée.
if [[ -f "$SHELL_RC" ]] && grep -q "jarvis-aliases\.sh" "$SHELL_RC" 2>/dev/null; then
    ok "jarvis aliases already sourced from $SHELL_RC"
elif [[ -f "$SHELL_RC" || "$SHELL" == *zsh ]]; then
    # Ajout en fin de fichier : les alias sourcés écrasent d'éventuels homonymes définis
    # plus haut, la migration est donc effective même sans nettoyage manuel.
    printf '\n# Jarvis launchd shortcuts\n%s\n' "$ALIASES_LINE" >> "$SHELL_RC"
    ok "Added jarvis aliases to $SHELL_RC (run 'source $SHELL_RC' or open a new terminal)"
    if grep -qE '^\s*alias jarvis-(start|stop|reload)=.*launchctl' "$SHELL_RC" 2>/dev/null; then
        warn "Anciens alias launchctl encore présents dans $SHELL_RC — désormais sans effet (surchargés), à supprimer quand tu veux"
    fi
else
    warn "Non-zsh shell — manually add to your rc file: $ALIASES_LINE"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────
bold "6/6 Done — what's left"

if [[ "$FRESH_USERS" == "1" && "$INTERACTIVE" == "1" ]]; then
cat <<EOF

  The essentials are configured. What is left is optional:

  1. .env holds a working default for everything else — open it only if you
     want to change one. Notably: OPENAI_API_KEY + LLM_LOCAL=no to run on a
     cloud API instead, or REASONING_MODEL_LOCAL for a separate reasoning tier.
  2. Add the other household members to $USERS_LIST — same shape, "code" is
     each one's API secret, "admin" stays true for you alone.
EOF
else
cat <<EOF

  1. Edit .env:
       - pick your local models (or keep the defaults) — see the
         "LLM providers — full local by default" section
       - fill in HF_TOKEN only if a chosen model is gated
       - optionally set OPENAI_API_KEY if you'd rather use a cloud API
         (then set LLM_LOCAL=no)
  2. Edit $USERS_LIST — one entry per user, "code" is their API secret.
EOF
fi

cat <<EOF
  3. If LLM_LOCAL=yes (default), download the models:
       source venv/bin/activate && python scripts/download_models.py
  4. Start Jarvis:
       jarvis-start          # via launchd (open a new terminal first, or: source $SHELL_RC)
       bash scripts/jarvis-entrypoint.sh   # or run it directly in the foreground
  5. Verify:
       curl http://localhost:8000/status

EOF
