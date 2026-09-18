# Operations

> Logs, diagnostics and everyday commands.

All Jarvis modules share a single logging configuration defined in `helpers.py`.

## Setup

`setup_logging()` is called once at startup (in the FastAPI `lifespan`). It configures the
root logger with two rotating file handlers:

| File | Level | Rotation | Purpose |
|---|---|---|---|
| `jarvis-api.log` | INFO+ | 5 MB × 3 backups | Operational — the normal production log |
| `jarvis-debug.log` | DEBUG+ | 10 MB × 2 backups | Verbose — detailed trace for debugging |

Both are written to `logs/` and readable directly on the host.

## Prompt logs

Seven logs record the prompt and the **raw** LLM response:

| File | Gate | Contents |
|---|---|---|
| `prompts.log` | `LLM_DEBUG_PROMPTS` | conversational traffic |
| `analyzer-prompts.log` | `LLM_DEBUG_PROMPTS` | conversation analysis (hourly) |
| `nightly-prompts.log` | `LLM_DEBUG_PROMPTS` | nightly review — **everything** it calls |
| `reflection-prompts.log` | `LLM_DEBUG_PROMPTS` | reflection cycle, self-challenges included |
| `opencode-prompts.log` | `RAW_DEBUG_PROMPTS` | `/v1/raw` only (coding agents) |
| `agent-prompts.log` | `AGENT_DEBUG_PROMPTS` | agentic loop — tasks **you** hand it |
| `autocode-prompts.log` | `AGENT_DEBUG_PROMPTS` | one nightly autocoding cycle, **end to end** |

The last two split by the task's `origin`, and the reason is practical: a cycle reads from
the choice of finding to the final report, and scattering its steps among the human tasks
meant jumping between files to follow a single night. The two framing calls (choose, report)
used to land in `prompts.log`, in the middle of chat traffic, while the loop they framed
wrote elsewhere — the two halves of one cycle in two unrelated files.

**Two routing mechanisms, for two different needs.** OpenCode and the agent pass an
**explicit** path (`debug_log_path`) and have their own switch: their prompts carry an entire
repository's context or ten tool schemas, and you want to follow them without turning
conversational logging back on.

The background jobs go through a **context variable** (`journal_de_cycle`,
`llm/local.py`) — the autocoding cycle included, for its two framing calls. They share `prompts.log`'s gate — it is the same traffic, just filed
elsewhere. A context variable was preferred to a parameter because one cycle calls the LLM
from several modules: the nightly review delegates curation to `memory/cleaning.py` and the
narrative to `memory/profile.py`. Threading a path through `helpers` and then the four entry
points of `llm/local.py` would have meant a dozen changed signatures and would still have
missed the delegated calls. The context variable follows the cycle wherever it goes,
including across `asyncio.to_thread`, which copies the context.

For the analyzer, the context is set on `analyze_exchange()` rather than on the scheduled
job: two paths lead there (the scheduler and `POST /memory/analyze`), and that function
carries the single LLM call.

Rotation is `PROMPT_LOG_MAX_MB` × `PROMPT_LOG_BACKUPS` (20 MB × 3 by default), applied per
file — `_prompt_writer()` creates one `RotatingFileHandler` per path, so a new log is covered
without configuration. Before rotation existed, these logs were written with a raw
`open(…, "a")` and `prompts.log` had reached 42 MB.

> **Analysis trap.** These logs show LLM output **before** pydantic validation. A field
> visible here is not a field that reached the store — check Redis/Qdrant, not the prompt log.

## Usage in modules

Every module gets its named logger through `helpers.get_logger`:

```python
from helpers import get_logger
logger = get_logger("jarvis-memory")
```

This replaces the per-module `import logging` + `logging.getLogger(...)` pattern. `config.py`
is the only exception: it is imported by `helpers.py` and cannot import from it without a
circular dependency.

## Silenced noisy loggers

`setup_logging()` sets level `WARNING` on `httpx`, `httpcore`, `primp`,
`sentence_transformers`, `apscheduler`, `urllib3` and `asyncio`.

`yfinance` is pinned to `CRITICAL`, not `WARNING`: it logs conditions the caller already
handles — a 404 on an unknown symbol, "possibly delisted" — at `ERROR` level, and `ERROR` is
precisely what is noisy. `trading/market.py` turns them into `None` and logs the failure
itself with the ticker, so nothing diagnostic is lost. This matters beyond noise: those lines
count towards `erreurs_log_24h`, which feeds vitals — one invalid symbol in a portfolio would
raise a `degradation_interne` incident and push α up.

---

## Test suite

```bash
# Fast loop — 157 unit tests, about two seconds, no network at all
./venv/bin/python -m pytest jarvis-core/tests/ -m "not integration"

# Same thing: the default run skips everything that needs a server or the web
./venv/bin/python -m pytest jarvis-core/tests/
```

| File | Scope |
|---|---|
| `conftest.py` | Bootstrap: real `config.py`, test `.env`, test users, network guard |
| `test_analyzer.py` | Prompt ↔ pydantic contract — the silently-dropped-field trap |
| `test_memory.py` | Retraction, convlog, project whitelist |
| `test_self.py` | Action catalogue coherence, introspection axes, nightly prompt |
| `test_agent.py` | Sandbox, blacklist, tool schemas, budget relationships |
| `test_autocode.py` | Pool parsing, mechanical guards, verdict computation, tool set per origin |
| `test_integration.py` | Live pipeline — **opt-in**, writes to real memory |
| `test_web_search.py` | Real web queries — **opt-in** |
| `test_lru_cache.py` | GPU benchmark — needs Jarvis stopped |

**The unit suite cannot reach any store.** An autouse fixture severs outbound sockets for
every test not marked `integration`, so a forgotten `get_redis()` fails loudly with the
address it tried, instead of quietly writing into a memory that is kept for 90 days.

Two families need an explicit opt-in, because the dev machine always has Jarvis running and
a bare `pytest` would otherwise exercise them:

```bash
# Writes to the REAL Redis and Qdrant, under user code TEST only
JARVIS_INTEGRATION=1 ./venv/bin/python -m pytest jarvis-core/tests/test_integration.py

# GPU benchmark — stop Jarvis first
jarvis-stop && ./venv/bin/python -m pytest jarvis-core/tests/ -m benchmark
```

Cleaning up after an integration run — see [REDIS.md](REDIS.md):

```bash
docker exec jarvis-redis redis-cli --scan --pattern "chat:TEST:*" \
  | xargs docker exec jarvis-redis redis-cli DEL
docker exec jarvis-redis redis-cli DEL episodic:TEST:conversations user:TEST:profile
```

> **Why the suite imports the real `config.py`.** It used to rebuild a fake `config` module
> by hand-listing its constants. Every constant added to the code without being copied there
> broke collection with an `ImportError` unrelated to the test at hand. The drift had reached
> 69 missing constants out of 125 and the suite no longer started at all. The environment is
> controlled instead — `JARVIS_ENV_FILE`, `USERS_LIST` and the data paths point at
> throwaway fixtures — so `config` cannot diverge from the code again.

---

## Health check

```bash
./scripts/jarvis-status.sh
```

Reports container state, the launchd job and an actual HTTP response from the API, LLM
providers, external APIs, RAG vector count, memory, proto-self and the `/v1/raw` endpoint.

### When HTTP answers but nothing generates

A live process at 0 % CPU that serves `/docs` while every LLM call hangs means the GPU lock is
held with no owner. Nothing is logged — the last line in `jarvis-api.log` is simply the last
generation before the freeze, which is what dates the incident.

```bash
sample <pid> 5 -f /tmp/jarvis-stack.txt      # macOS, no install needed
grep -c "_pthread_cond_wait" /tmp/jarvis-stack.txt
```

The signature is the main asyncio thread parked in `lock_PyThread_acquire_lock` for the whole
sample, with every worker thread asleep. Take the sample **before** restarting: a restart clears
the state and the evidence with it.

The known cause is fixed. `asyncio.to_thread(_infer_lock.acquire)` is not cancellable — a
blocking `threading.Lock.acquire()` cannot be interrupted — so a client disconnect left the
coroutine gone while its thread went on to take the lock nobody would release. Acquisition now
goes through `_acquire_infer_lock_chat()`, which shields it and releases an orphaned acquisition,
logging `[INFER-LOCK] acquisition orpheline relâchée`. That warning is the one to watch: it means
the situation occurred and was recovered.

---

## Upgrading dependencies

**Automated shortcut**: `./scripts/backup-jarvis.sh updates` runs Case A end to end — backup,
then `docker compose pull/up` plus `pip install -r requirements.txt --upgrade` **in place**
(never a `mv` of the venv), all inside a **maintenance window**, so the errors and outages
caused by the operation are tagged "maintenance" rather than logged as incidents. Two things
stay manual: editing `requirements.txt` beforehand, and bumping the Python interpreter
(Case B).

### Case A — security bumps (routine, in place)

```bash
cd /opt/jarvis

# 0. rollback anchor + backup (USB key mounted, for the receipt)
/opt/jarvis/venv/bin/python -m pip freeze > requirements.freeze.$(date +%Y%m%d).txt
./scripts/backup-jarvis.sh

# 1. edit requirements.txt (targeted bump), SAVE, verify:
git diff -- requirements.txt        # or: grep cryptography requirements.txt

# 2. apply INSIDE the existing venv (no new venv, no mv)
/opt/jarvis/venv/bin/python -m pip install -r requirements.txt --upgrade

# 3. validate (see below), then restart Jarvis
```

### Case B — full rebuild or Python bump (in place, without moving the new venv)

```bash
cd /opt/jarvis
/opt/jarvis/venv/bin/python -m pip freeze > requirements.freeze.$(date +%Y%m%d).txt
./scripts/backup-jarvis.sh

# (if Python) brew upgrade python@3.13   # stay on 3.13, not 3.14

# move the OLD one aside (rollback), create the new one DIRECTLY at the final path
mv /opt/jarvis/venv /opt/jarvis/venv-old
/opt/homebrew/bin/python3.13 -m venv /opt/jarvis/venv   # created in place, never moved
/opt/jarvis/venv/bin/python -m pip install -U pip
/opt/jarvis/venv/bin/python -m pip install -r requirements.txt

# rollback if needed:
# rm -rf /opt/jarvis/venv && mv /opt/jarvis/venv-old /opt/jarvis/venv
```

The point of this ordering: the **old** venv is the one that moves, so a rollback returns it
to its original path intact, and the new one is created directly at `/opt/jarvis/venv`. The
new venv is never moved.

### Validation (before any restart)

```bash
/opt/jarvis/venv/bin/pip --version                    # entry points OK
/opt/jarvis/venv/bin/python -c "import mlx, torch, transformers, cryptography; print('core OK')"
/opt/jarvis/venv/bin/python -c "import sys;sys.path.insert(0,'/opt/jarvis/jarvis-core/src');import cve;r=cve.scan();print('venv scanned:', 'venv' in r['par_source'])"
```

Then run the test suite, and load a model and generate once.
