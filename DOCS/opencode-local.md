# OpenCode sur le LLM local de Jarvis

Remplace l'ancien montage Claude Code / proxy Anthropic (supprimé le 08/08/2026), qui
émulait les appels d'outil au niveau du prompt et échouait dès qu'un appel était mal formé.
OpenCode parle OpenAI nativement et le function calling est désormais réel : les schémas
d'outils sont passés au template de chat, et le modèle répond dans le format sur lequel il
a été entraîné.

## Architecture

```
opencode  ──►  http://jarvis:8000/v1/raw/chat/completions  ──►  Qwen3.6-35B (MLX)
             (Tailscale, depuis n'importe quelle machine)
```

`/v1/raw` et non `/v1/chat/completions` : la route principale est l'assistant personnel,
elle ne garde que le dernier message, écrase le `system` du client, injecte profil, mémoire,
RAG, Gmail et Calendar, et **écrit dans la mémoire** à chaque appel. Un agent de code y
polluerait le convlog et Qdrant. `/v1/raw` ne fait rien de tout ça.

## Installation

```bash
brew install opencode          # tire node + ripgrep
mkdir -p ~/.config/opencode
cp /opt/jarvis/DOCS/opencode.json.example ~/.config/opencode/opencode.json
```

Sur une autre machine : même chose, seul le fichier de config est nécessaire — il pointe sur
le nom Tailscale `jarvis`, pas sur `localhost`, donc il est valable partout tel quel.
Prérequis : la machine doit être sur le tailnet, et le Mac Mini démarré.

## Réglages

| Champ | Valeur | Pourquoi |
|---|---|---|
| `baseURL` | `http://jarvis:8000/v1/raw` | le client compose `/chat/completions` derrière |
| `limit.output` | `8000` | = `RAW_MAX_TOKENS` (`routes/proxy.py`) |
| `limit.context` | `32768` | prudent : le modèle annonce 262 144, mais le cache KV correspondant ne tient pas en RAM sur le Mac Mini. À monter progressivement en surveillant la latence. |

Le thinking est désactivé par défaut sur cette route (`RAW_NO_THINK=true`) et les blocs
`<think>` sont strippés : OpenCode ne verra jamais de raisonnement dans ses réponses.

## Fonctionnement du function calling

Le template (`models/templates/qwen36_ninja.jinja`) rend les outils en tête du bloc système
et impose ce format de réponse :

```
<tool_call>
<function=read_file>
<parameter=path>
src/main.py
</parameter>
</function>
</tool_call>
```

`jarvis-core/src/tool_calls.py` fait la traduction dans les deux sens :
- **sortie modèle → OpenAI** : `parse_tool_calls()`, avec typage des paramètres d'après le
  schéma JSON de l'outil (un `42` doit revenir en entier, pas en `"42"`, sinon la validation
  côté client échoue) ;
- **entrée OpenAI → template** : `normalise_messages_for_template()`, qui reconvertit
  `arguments` de chaîne JSON en dict — le template itère `arguments|items` et n'accepte pas
  une chaîne. Sans ça, le second tour produit un prompt corrompu.

En présence d'outils, la réponse est bufferisée avant émission : tant que `<tool_call>` n'est
pas clos, on ne peut pas savoir si le texte en cours est de la prose ou le début d'un appel.

## Sécurité — dette assumée

`/v1/raw` n'a **aucune authentification** et uvicorn écoute sur `0.0.0.0`. Acceptable tant que
la machine reste sur le réseau domestique et que l'accès distant passe par Tailscale. À
reprendre si elle se retrouve sur un réseau partagé.

## Diagnostic

```bash
./jarvis-status.sh          # section « Endpoint agents de code (OpenCode) »
```
