# Claude Code → Qwen local via Jarvis

Utiliser Claude Code sur n'importe quelle machine du réseau,
en faisant tourner l'inférence sur le Mac Mini (jarvis.local) plutôt que sur les serveurs Anthropic.

---

## Architecture

```
Claude Code (ta machine)
    │  format Anthropic API
    ▼
jarvis.local:8090  ←  anthropic-proxy.py
    │  format OpenAI  +  no_think / thinking_budget
    ▼
jarvis.local:8000  ←  Jarvis /v1/raw/chat/completions
    │  MLX direct, aucun pipeline Jarvis
    ▼
Qwen3.6-35B (Apple Silicon)
```

---

## Prérequis

- Claude Code installé sur la machine cliente (`npm install -g @anthropic-ai/claude-code`)
- Jarvis et le proxy tournent sur le Mac Mini (vérifier : `curl http://jarvis.local:8090/health`)

---

## Configuration (à faire une fois par machine)

### 1. Fichier de config Claude Code

Il y a deux emplacements possibles :

| Fichier | Portée |
|---|---|
| `~/.claude/settings.json` | Global — toutes les sessions sur cette machine |
| `.claude/settings.json` | Local — ce répertoire de projet uniquement |

**Option A — global** (machine dédiée à Jarvis) : crée/complète `~/.claude/settings.json` :

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://jarvis.local:8090",
    "ANTHROPIC_API_KEY":  "local"
  },
  "maxTokens": 8000
}
```

`maxTokens` limite les tokens de sortie par requête (cohérent avec `RAW_MAX_TOKENS=8000` côté proxy).

**Option B — par projet** (machine mixte cloud/local) : crée `.claude/settings.json` à la racine du projet, même contenu. Les aliases `cl`/`cc` restent utiles pour switcher sans toucher ce fichier.

> `effortLevel` dans le settings.json contrôle le niveau de thinking par défaut
> (`"low"`, `"medium"`, `"high"`, `"max"`). Peut se surcharger en session avec `/effort`.

### 2. Aliases shell — `~/.zshrc` (ou `~/.bashrc`)

```bash
# Claude Code — switch local (Qwen via Jarvis) / cloud (Anthropic)
claude-local() {
  ANTHROPIC_BASE_URL=http://jarvis.local:8090 ANTHROPIC_API_KEY=local claude "$@"
}
alias cl=claude-local   # Qwen local
alias cc=claude         # Anthropic cloud
```

Puis recharge :

```bash
source ~/.zshrc
```

---

## Utilisation

```bash
# Dans n'importe quel répertoire de projet :
cl        # lance Claude Code → Qwen local
cc        # lance Claude Code → Anthropic cloud (tokens facturés)
```

Les deux commandes acceptent les mêmes arguments que `claude` :

```bash
cl --model claude-opus-4-8   # inutile, le proxy ignore le nom de modèle
cl -p "résume ce fichier"    # mode one-shot non-interactif
```

---

## Contrôle du thinking (Qwen3.6)

Depuis Claude Code, utilise `/effort` pour doser le raisonnement :

| Commande Claude Code | Thinking Qwen | Budget |
|---|---|---|
| `/effort low`    | désactivé | — |
| `/effort medium` | actif | ~2 000 tok |
| `/effort high`   | actif | ~4 000 tok |
| `/effort max`    | actif | ~10 000 tok |

Le bloc `<think>` est strippé avant envoi — Claude Code ne le voit jamais.
Par défaut (sans `/effort`) : thinking désactivé, réponses rapides.

---

## Variables d'environnement du proxy (sur le Mac Mini)

À mettre dans `/opt/jarvis/.env` si besoin d'ajuster :

| Variable | Défaut | Description |
|---|---|---|
| `PROXY_MAX_CTX_CHARS` | `112000` (~28K tok) | Taille max du contexte entrant — messages anciens supprimés au-delà |
| `RAW_MAX_TOKENS` | `8000` | Tokens max en sortie — doit correspondre à `maxTokens` dans settings.json |
| `RAW_NO_THINK` | `true` | Thinking désactivé par défaut (contrôlé par `/effort`) |
| `PROXY_PORT` | `8090` | Port du proxy |
| `PROXY_JARVIS_URL` | `http://localhost:8000` | URL de Jarvis (connexion interne Mac Mini → Mac Mini, ne pas changer) |

Après modification, relancer le proxy :
```bash
launchctl stop  com.jarvis.anthropic-proxy
launchctl start com.jarvis.anthropic-proxy
```

---

## Diagnostic

```bash
# Proxy vivant ?
curl http://jarvis.local:8090/health

# Logs du proxy (sur le Mac Mini)
tail -f /opt/jarvis/logs/anthropic-proxy.log

# Test rapide depuis la machine cliente
curl -s http://jarvis.local:8090/v1/models | python3 -m json.tool
```

---

## Retour au cloud

```bash
cc          # alias standard, ANTHROPIC_BASE_URL non défini → Anthropic direct
```

Ou temporairement dans un shell :

```bash
unset ANTHROPIC_BASE_URL
claude
```
