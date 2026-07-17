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
    warn "Docker daemon still not up — start it manually before running ./start.sh"
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

USERS_LIST=jarvis-core/JarvisData/users_list.json
if [[ ! -f "$USERS_LIST" ]]; then
    cp DOCS/examples/users_list.example.json "$USERS_LIST"
    ok "Created $USERS_LIST from template — edit it before starting Jarvis"
else
    ok "$USERS_LIST already exists — left untouched"
fi

# ── 5. launchd service ────────────────────────────────────────────────────
bold "5/6 launchd service"

PLIST_DEST="$HOME/Library/LaunchAgents/com.jarvis.api.plist"
if [[ ! -f "$PLIST_DEST" ]]; then
    EXTRA_PATH=""
    [[ -d "/Applications/OrbStack.app" ]] && EXTRA_PATH="$HOME/.orbstack/bin:"
    mkdir -p "$HOME/Library/LaunchAgents"
    sed -e "s#__JARVIS_HOME__#$JARVIS_HOME#g" \
        -e "s#__EXTRA_PATH__#$EXTRA_PATH#g" \
        DOCS/examples/com.jarvis.api.plist.template > "$PLIST_DEST"
    ok "Installed $PLIST_DEST (not loaded yet — see next steps)"
else
    ok "$PLIST_DEST already exists — left untouched"
fi

ALIASES_LINE="source $JARVIS_HOME/DOCS/examples/jarvis-aliases.sh"
SHELL_RC="$HOME/.zshrc"
if [[ -f "$SHELL_RC" ]] && grep -qE "jarvis-aliases\.sh|alias jarvis-start=" "$SHELL_RC" 2>/dev/null; then
    ok "jarvis-start/stop/reload aliases already in $SHELL_RC"
elif [[ -f "$SHELL_RC" || "$SHELL" == *zsh ]]; then
    printf '\n# Jarvis launchd shortcuts\n%s\n' "$ALIASES_LINE" >> "$SHELL_RC"
    ok "Added jarvis-start/stop/reload aliases to $SHELL_RC (run 'source $SHELL_RC' or open a new terminal)"
else
    warn "Non-zsh shell — manually add to your rc file: $ALIASES_LINE"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────
bold "6/6 Done — what's left"

cat <<EOF

  1. Edit .env:
       - pick your local models (or keep the defaults) — see the
         "LLM providers — full local by default" section
       - fill in HF_TOKEN only if a chosen model is gated
       - optionally set OPENAI_API_KEY if you'd rather use a cloud API
         (then set LLM_LOCAL=no)
  2. Edit $USERS_LIST — one entry per user, "code" is their API secret.
  3. If LLM_LOCAL=yes (default), download the models:
       source venv/bin/activate && python scripts/download_models.py
  4. Start Jarvis:
       jarvis-start          # via launchd (open a new terminal first, or: source $SHELL_RC)
       ./start.sh            # or run it directly in the foreground
  5. Verify:
       curl http://localhost:8000/status

EOF
