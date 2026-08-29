# OpenCode on Jarvis's local LLM

Replaces the earlier Claude Code / Anthropic-proxy setup, which emulated tool calls at the
prompt level and broke as soon as one call came out malformed. OpenCode speaks OpenAI
natively and function calling is now real: tool schemas are passed to the chat template, and
the model answers in the format it was trained on.

## Architecture

```
opencode  ──►  http://jarvis:8000/v1/raw/chat/completions  ──►  Qwen3.6-35B (MLX)
             (Tailscale, from any machine)
```

`/v1/raw` and not `/v1/chat/completions`: the main route *is* the personal assistant. It
keeps only the last message, overwrites the client's `system`, injects profile, memory, RAG,
Gmail and Calendar, and **writes to memory** on every call. A coding agent would pollute the
convlog and Qdrant. `/v1/raw` does none of that.

## Install

```bash
brew install opencode          # pulls node + ripgrep
mkdir -p ~/.config/opencode
cp /opt/jarvis/DOCS/opencode.json.example ~/.config/opencode/opencode.json
```

On another machine: same thing, the config file is all you need — it points at the Tailscale
name `jarvis`, not `localhost`, so it works anywhere as-is. Prerequisites: the machine is on
the tailnet, and the Mac Mini is up.

## Settings

| Field | Value | Why |
|---|---|---|
| `baseURL` | `http://jarvis:8000/v1/raw` | the client appends `/chat/completions` |
| `limit.output` | `8000` | matches `RAW_MAX_TOKENS` (`routes/proxy.py`) |
| `limit.context` | `32768` | conservative: the model advertises 262 144, but the matching KV cache does not fit in RAM on the Mac Mini. Raise gradually while watching latency. |

Thinking is off by default on this route (`RAW_NO_THINK=true`) and `<think>` blocks are
stripped: OpenCode never sees reasoning in its responses.

## How function calling works

The template (`models/templates/qwen36_ninja.jinja`) renders the tools at the top of the
system block and imposes this response format:

```
<tool_call>
<function=read_file>
<parameter=path>
src/main.py
</parameter>
</function>
</tool_call>
```

`jarvis-core/src/tool_calls.py` translates both ways:

- **model output → OpenAI**: `parse_tool_calls()`, typing parameters from the tool's JSON
  schema — a `42` must come back as an integer, not `"42"`, or client-side validation fails;
- **OpenAI input → template**: `normalise_messages_for_template()`, which converts
  `arguments` back from a JSON string to a dict. The template iterates `arguments|items` and
  will not accept a string; without this, the second turn produces a corrupted prompt.

When tools are present, the response is buffered before emission: until `<tool_call>` is
closed, there is no way to tell whether the text in flight is prose or the start of a call.

## Security — acknowledged debt

`/v1/raw` has **no authentication** and uvicorn listens on `0.0.0.0`. Acceptable as long as
the machine stays on a home network and remote access goes through Tailscale. Revisit if it
ever lands on a shared network. See [SECURITY.md](SECURITY.md).

## Diagnostics

```bash
./scripts/jarvis-status.sh          # section "Endpoint agents de code (OpenCode)"
```
