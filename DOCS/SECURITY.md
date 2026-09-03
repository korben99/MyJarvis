# Security

> Jarvis reads your mail, your calendar and your documents, and runs a model that can be
> given a shell. The threat model is not theoretical: read this page before exposing
> anything.

---

## Deployment model

Jarvis is built to run **on a trusted network**, on a machine you control. There is no TLS,
no accounts, no passwords and no built-in rate limiting. Everything below assumes that.

| Surface | Default bind | Authentication | What to know |
|---|---|---|---|
| Jarvis API (8000) | `0.0.0.0` | user code | Reverse proxy + TLS if you leave the LAN |
| `/v1/raw/chat/completions` | `0.0.0.0` | **none** | Never expose this route beyond the LAN |
| Open WebUI (3000) | `0.0.0.0` | Open WebUI account | `WEBUI_SECRET_KEY` is in cleartext in `docker-compose.yml` — change it |
| Qdrant (6333/6334) | `0.0.0.0` | **none** | The whole vector store is readable from the LAN |
| Redis (6379) | `0.0.0.0` | **none** | The whole memory is readable *and writable* from the LAN |

**These three services are published on every interface**, not just loopback:
`docker-compose.yml` maps `6379:6379`, `6333:6333` and `3000:8080` with no bind address. On
a shared home network, any device on the LAN can therefore read every user's full memory
with no credentials whatsoever.

If only the API needs to be reachable from the network — which is the nominal case, since
the iOS app and Open WebUI talk to the API and not to the stores — restrict both stores to
loopback:

```yaml
# docker-compose.yml
  redis:
    ports:
      - "127.0.0.1:6379:6379"
  qdrant:
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
```

For remote access, the recommended route is a VPN (Tailscale, WireGuard) rather than
publishing to the internet.

---

## Authentication

Authentication is **by user code** — an opaque string defined in
`jarvis-core/JarvisData/users_list.json`, sent in the request body (`user_code`) or as
`Authorization: Bearer <code>`.

Consequences to accept:

- **A code is a secret equivalent to a password.** Whoever knows it reads that user's entire
  memory, mail and calendar.
- There is **no expiry and no rotation**. A compromised code is changed by hand in
  `users_list.json`.
- There is **no rate limiting**: nothing stops brute-forcing a code. Choose them long and
  random — not a first name, not a date.
- Memory is **partitioned by user code**: Redis keys and Qdrant filters all carry the code.
  One user cannot read another's memory.

**Partitioning by key prefix is not authorization.** Namespacing data by user code only
protects it if every endpoint checks that the caller *is* the user it names in the path.
An endpoint that merely verifies "the token is a known user code" grants every valid token
access to everyone's data, because the path parameter chooses the namespace. Both checks are
required, and they are different:

```python
_auth(authorization) in USER_CODES     # the caller is someone — NOT an authorization check
_auth(authorization) == user_code      # the caller is THIS someone — the actual check
```

The `/portfolio/*` endpoints are the reference implementation
(`routes/portfolio.py::_garde`): caller identity, user exists, feature enabled for that
user. Any new per-user endpoint must apply the same three, and there is deliberately **no
admin exception** — an admin token gives no access to another user's financial data.

**A feature flag is an authorization boundary too.** `"trading": true` decides whether a
user has a portfolio at all. Checking it in the scheduler alone is not enough: the HTTP
endpoints and the briefing injection each reach the same data by their own path, and each
must check it. A flag consulted in one place out of three is not a flag.

---

## Secrets

No secret belongs in the repository. The following are ignored by `.gitignore` and must
stay that way:

| File | Contents |
|---|---|
| `.env` | API keys (OpenAI, Tavily, Hugging Face), Google credentials and refresh token |
| `scripts/client_secret.json` | Google OAuth client secret |
| `keys/` | APNs signing key (`.p8`) and any other private key |
| `jarvis-core/JarvisData/users_list.json` | user codes, names, email addresses |

`docker-compose.yml`, on the other hand, **is** versioned and carries `WEBUI_SECRET_KEY` in
cleartext. The shipped value is a default shared by every clone of the repository: change it
before first start, or move it into `.env` via `${WEBUI_SECRET_KEY}`.

The `keys/` directory is ignored **entirely**, and the `.p8`, `.pem`, `.key` and `.p12`
extensions are ignored everywhere: a key dropped there by mistake does not travel in a
commit.

Before publishing a fork, check that the **history itself** is clean — `.gitignore` only
acts on future commits, never on ones already written. `git log --all -- .env` and
`git log --all -- '*users_list.json'` are the two commands worth running.

---

## The agent sandbox

The agentic loop (`jarvis-core/src/agent/`) can execute shell commands. It runs under the
user's account with full rights: confinement is not a configuration option, it is the
condition of the feature existing.

Three independent layers, described in detail in `agent/shell.py`:

1. **seatbelt (`sandbox-exec`)** — the only real barrier: the kernel refuses, not us.
   Writes limited to the task workspace and `/tmp`; reads denied on `.env`, `keys/`,
   `~/.ssh` and the keychain; network cut.
2. **Pattern blacklist** — a guard rail against honest mistakes (`sudo`, `rm -rf /`,
   `curl … | sh`, machine shutdown). This is **not** a security boundary: a blacklist can be
   worked around.
3. **Budgets** — per-command timeout, per-task call quota, truncated output.

The network is cut inside the shell even though the agent has `web_search` and `fetch_url`:
those two go through Jarvis's own code, logged and bounded. A `curl` in a shell is not, and
it is the shortest exfiltration path there is.

The shell agent is **disabled by default** (`AGENT_SHELL_ENABLED`). Enable it only
deliberately. See **[AGENT.md](AGENT.md)** for the full picture.

---

## Prompt injection

Jarvis ingests content it does not control: web pages, search results, email bodies,
documents indexed into the RAG. All of it must be treated as **hostile by default** — text
can carry instructions aimed at the model.

The guard rails in place are partial, and worth knowing as such:

- External content is injected inside delimited blocks, never into the system message.
- The agent's writing tools are confined to the task workspace.
- The proto-self's autonomous actions go through a closed catalogue of typed actions, not
  free-form code.

None of that is complete protection against prompt injection. If you point Jarvis at a
publicly reachable mailbox, assume a third party can influence what the model tells you.

---

## Logs

`logs/prompts.log` contains full prompts, therefore **the contents of your memory, your mail
and your documents in cleartext**. The `logs/` directory is git-ignored. Treat it as personal
data: do not attach it to a bug report without reading it first.

---

## Reporting a vulnerability

This project is a self-hosted personal assistant, not a product with a security team. If you
find a flaw, open an issue — or, if it is sensitive, make contact privately through the
repository's GitHub profile before any public disclosure.
