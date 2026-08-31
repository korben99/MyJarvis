# Performance

> TTFT measurements, KV cache design, and the map of every LLM call.

## Implemented optimisations

| Optimisation | TTFT gain | Details |
|---|---|---|
| **Conditional no_think** | −4 s on plain chat | `chat_no_think=True` except for RAG/web/reasoning. `thinking_budget=0` via chat template (KV-safe). |
| **ThinkingBudgetProcessor** | −2 to −5 s | Hard logit cut on `</think>` at the exact budget (COMPACT/MEDIUM/DEEP) — prevents runaway reasoning. |
| **Trimmed system prompt** | −0.3 s | `SYSTEM_BASE_FR` reduced by ~400 chars / ~100 tokens. |
| **KV prefix caching** | −1 to −3 s from turn 2 | Per-session MLX KV cache (LRU ×8). Only new tokens are computed each turn. |
| **Vision resize to 1024 px** | ~3–5× on VLM inference | iPhone photo (12 MP) resized before `vlm_generate` — from 8–10 tiles to 2–4. `max_tokens` 1200 → 700. |
| **Qwen2.5-1.5B LoRA router** | −0.5 to −1.2 s vs Hermes-3B | Fine-tuned on 492 samples (val loss 0.047). Warmup with `ROUTER_SYSTEM` → LRU hit from the first call. Turn 2+: 95 % cache hit (1044/1093 tok). |
| **Stable profile in system prompt** | ~0.1 s / turn | `<profil_utilisateur>` (~80 tokens) injected per-user into the system prompt — never reprocessed after warmup. |

## KV cache design

```
Turn 1: [SYS + user profile ~310 tok] + [dynamic CTX + msg1 ~600 tok]
         ↑ all computed                 ↑ all computed
         └── cached ────────────────────┘

Turn 2: [SYS + profile ~310 tok] + [CTX1+msg1+reply1 ~900 tok] + [CTX2+msg2 ~600 tok]
         ↑ cache hit                ↑ cache hit                    ↑ only this computed

Turn N: skips (N−1) × ~900 tokens → only ~600 new tokens
```

`<profil_utilisateur>` (~80 tokens: family, size, job…) lives in the per-user system prompt
and is never reprocessed.

The system prompt is **token-identical** on every turn for a given user
(`SYSTEM_BASE_FR` + name + `<profil_utilisateur>`). Dynamic context — memories, opinions,
date — is prefixed onto the current user message instead. That prefix is stripped for
display in `/history`.

## Reference measurements — Mac Mini M4 Pro 48 GB (2026-05-23)

### Models compared

| Model | Metal weights | Total Metal at rest | Headroom / 48 GB |
|---|---|---|---|
| `spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit` | ~23.6 GB | n/a (before Metal logging) | — |
| `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-6bit` | ~26.3 GB | **~35.9 GB** | ~12 GB |
| `majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-5bit` | ~21.9 GB | **~31.7 GB** | ~16 GB |

Total Metal = primary model + Hermes 3.2B + VLM 8B + OS/Python. Measured through
`mx.metal.get_active_memory()`.

### Prefill speed (remaining_tokens / first_token_time, no_think)

| Model | Remaining tok | First token | Prefill tok/s |
|---|---|---|---|
| 5.4bit standard | ~1 560–1 960 | ~2.2–2.7 s | **~695–725** |
| RotorQuant 6bit | ~1 715–1 916 | ~2.5–2.7 s | **~699–713** |
| RotorQuant 5bit | ~2 184–2 265 | ~3.2–3.3 s | **~683–694** |

The 5bit prompts were ~400 tok longer (later in the session) — normalised prefill is
equivalent to the others. RotorQuant should show its advantage on very large contexts
(>4 000 remaining), where memory bandwidth becomes dominant.

### Decode speed (tok/s, no_think)

| Model | Decode tok/s |
|---|---|
| 5.4bit standard | ~55–60 |
| RotorQuant 6bit | ~62–82 |
| RotorQuant 5bit | **~72+** |

### End-to-end TTFT (iPhone → first visible token, no_think)

| Model | Typical TTFT | of which router | of which RotorQuant prefill |
|---|---|---|---|
| 5.4bit standard | ~3.5–4.7 s | ~1.2–2.6 s | ~2.2–2.7 s |
| RotorQuant 6bit | ~3.8–5.1 s | ~1.2–2.6 s | ~2.5–2.7 s |
| RotorQuant 5bit | ~4.5–5.4 s | ~1.2–2.6 s | ~3.2–3.3 s |

The old router (Hermes 3.2B, since replaced by Qwen2.5-1.5B LoRA) accounted for 30–50 % of
TTFT depending on cache hit. RotorQuant prefill of ≈2.5 s is incompressible for
~1700–1900 remaining tokens.

**Update 2026-05-30 — Qwen2.5-1.5B LoRA router:**

| Measurement | Hermes 3.2B | Qwen2.5-1.5B LoRA |
|---|---|---|
| gather1 (router + ctx) — turn 1 | ~1.4 s | ~1.4 s (LRU miss) |
| gather1 — turn 2+ | — | **~0.8 s** (LRU hit 1044/1093 tok) |
| Visible TTFT — turn 1 | ~5.3 s | ~5.3 s |
| Visible TTFT — turn 2+ | — | **~4.7 s** (−0.6 s) |

Warmup now uses `ROUTER_SYSTEM` instead of the generic prompt, so the LRU is seeded
correctly from startup.

**Update 2026-06-14 — context-aware router + jailbreak:**

- `ROUTER_SYSTEM` grows from ~1340 to ~1510 tok (added `<last_jarvis>` instruction + 2 examples).
- `ROUTER_USER` injects Jarvis's last reply truncated to 300 chars → contextual routing of
  elliptical messages.
- `SYSTEM_BASE_FR` grows from ~190 to ~224 tok: the "do not generate at all costs" rule is
  replaced by "always answer, extrapolate, never a flat refusal".

### Thinking mode (first visible token = after the think block)

| Model | Visible TTFT | Think generation | Think decode |
|---|---|---|---|
| 5.4bit standard | ~28–42 s | ~2 048–3 072 tok | ~55–57 tok/s |
| RotorQuant 6bit | ~59 s (1 measurement, budget 3 072) | ~3 072 tok | ~55 tok/s |

---

## LLM call map

Every LLM call in the codebase. Each row states whether thinking is on, the real budget, and
whether `ThinkingBudgetProcessor` engages.

### Legend

| Column | Description |
|---|---|
| **Think** | `think` = reasoning mode on (`no_think=False`) · `no_think` = direct mode |
| **Budget** | Allocated tokens (thinking and answer share it — a kill switch, not a hard cap) |
| **Processor** | `✅` = `ThinkingBudgetProcessor` active (hard `</think>` cut at the exact budget through logits) · `—` = inactive |
| **Rationale** | Why this mode for this task |

### Conversations (`routes/chat.py`)

| Context | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| Plain chat (memory/conversation intent) | PRIMARY | `no_think` | 1 500 | — | Fast reply, no reasoning needed |
| Web / RAG chat (synthesis) | PRIMARY | `think` | 8 000 | ✅ 2048 tok | Synthesising multiple sources — thinking improves coherence |
| Reasoning chat (`use_reasoning`) | PRIMARY | `think` | 10 000 | ✅ 2048 tok | Complex request explicitly routed to thinking |

> `ThinkingBudgetProcessor` engages as soon as `thinking_budget > 0` and
> `USE_THINKING_BUDGET_PROCESSOR=yes`. It forces `</think>` by manipulating logits (soft boost
> at 90 % of budget, hard cut at 100 %), keeping reasoning from eating into the answer budget.
> The value matters: too short a budget truncates the reasoning and degrades quality.

### Background — Analyzer (`analyzer.py`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `analyze_exchange` | PRIMARY | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Structured extraction (topics, mood, facts, projects). Pure classification — thinking generates ~1500 tok of verbose English without ever closing `</think>`, proven by test. no_think gets the same result in under 5 s. |

### Background — Memory (`memory.py`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `_normalize_profile_keys_batch` | ROUTER | `no_think` | 250 | — | Key normalisation — deterministic task, short answer |
| `_normalize_profile_key` | ROUTER | `no_think` | 150 | — | Same, single call |
| `_consolidate_user_memories` | PRIMARY | `no_think` | 400 | — | Fact dedup / merge — classification, no creativity |
| `curative_profile_cleanup` | PRIMARY | `no_think` | 600 | — | Curative profile cleanup — like prune_self_memory, thinking causes high variance (tested) |

### Background — Self-reflection (`self/`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `_call_global_reflection_llm` | REASONING | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Phase 1 — picks ONE global action per chain step |
| `_call_user_reflection_llm` | REASONING | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Phase 2 — same, per user |
| `generate_proactive_push` | REASONING | `no_think` | `MAX_TOKENS_COMPACT` (600) | — | Binary decision + one short sentence — thinking is superfluous |
| `_action_prune_self_memory` | REASONING | `think` | `MAX_TOKENS_THINK_COMPACT` (2 048) | ✅ `THINKING_BUDGET_COMPACT` (1 024) | Selecting entries to delete — short thinking for coherence without aggression |
| `_llm_review_before_action` | REASONING | `think` | `MAX_TOKENS_THINK_COMPACT` (2 048) | ✅ `THINKING_BUDGET_COMPACT` (1 024) | Self-challenge before an outbound action — thinking improves contextual judgement |

### Background — Nightly review (`self/nightly.py`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `_nightly_facts_user` (call 1) | REASONING | `no_think` | `MAX_TOKENS_NO_THINK` (1 500) | — | Durable facts about the user → autobio Qdrant, relationship, suggestions. Structured parsing, thinking adds no measurable value |
| `_nightly_self_user` (call 2) | REASONING | `no_think` | `MAX_TOKENS_NO_THINK` (1 500) | — | Jarvis's introspection (9 axes) + opinions. Revision, not accumulation — see *Self-Introspection* |
| `_nightly_cleaning_user` (call 3) | REASONING | `no_think` | `MAX_TOKENS_COMPACT` (600) | — | Autobio Qdrant curation (archive/delete) — pure classification |
| `update_profile_narrative` | PRIMARY | `no_think` | `PROFILE_NARRATIVE_TOKENS` (400) | — | ~300-token narrative portrait — fluent prose, thinking superfluous |
| `_action_refine_prompt` (+ retry) | REASONING | `think` | `MAX_TOKENS_REASONING` (10 000) | ✅ `THINKING_BUDGET_DEEP` (4 000) | **Creative**: rewriting a system prompt. Thinking essential. ~6 000 tok left for the rewritten prompt + rationale. |

### Routing & web search (`llm/router.py`, `web_search.py`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `llm_route` | ROUTER (Qwen2.5-1.5B LoRA) | `no_think` | 300 | — | Intent classification — short deterministic JSON |
| `_llm_judge_relevance` | ROUTER | `no_think` | `MAX_TOKENS_SHORT` (300) | — | Relevance score — binary, very short |
| `_generate_optimized_query` | ROUTER | `no_think` | `MAX_TOKENS_TINY` (80) | — | Query rewrite — simple task |
| `_refine_web_queries` | ROUTER | `no_think` | `MAX_TOKENS_TINY` (80) | — | Same, two refined queries |

### Trading (`trading/core.py`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `_ticker_llm_call_async` | PRIMARY | `no_think` | `MAX_TOKENS_TINY` (80) | — | Ticker symbol extraction — very short answer |
| `evaluate_alerts` | PRIMARY | `no_think` | `MAX_TOKENS_MEDIUM` (1 000) | — | Alert threshold evaluation — technical classification |
| `suggest_thresholds_llm` | PRIMARY | `think` | `MAX_TOKENS_THINK_MEDIUM` (5 048) | ✅ `THINKING_BUDGET_MEDIUM` (2 048) | Quantitative reasoning on price thresholds. ~3 000 tok left for multi-position JSON. |

### Briefing (`briefing.py`)

| Function | Model | Think | Budget | Processor | Rationale |
|---|---|---|---|---|---|
| `_assemble_with_llm` | PRIMARY | `no_think` | 3 000 | — | Assembling the daily briefing — structured formatting, no reasoning |

### think vs no_think decision rules

```
Classification / extraction / formatting task
  → no_think=True  (fast, deterministic, identical result)

Conversational or creative task (chat, refine_prompt, trading thresholds)
  → think=True, thinking_budget=THINKING_BUDGET_MEDIUM or DEEP → processor on (precise hard cut)

Judgement task with limited context (prune, action review)
  → think=True, thinking_budget=THINKING_BUDGET_COMPACT (1024) → brief thinking, coherent result

Never use thinking_budget=0 in production
  → Risk: runaway reasoning (~1900 tok) → eats the answer budget, unpredictable timeout
```

### Control variables (`.env`)

| Variable | Default | Role |
|---|---|---|
| `TOKEN_SPEED_TPS` | 50 | Estimated generation speed (tok/s) — calibrates timeouts |
| `TIMEOUT_MARGIN` | 1.3 | Multiplicative margin for `llm_timeout()` |
| `USE_THINKING_BUDGET_PROCESSOR` | yes | Enables `ThinkingBudgetProcessor` on calls with `thinking_budget > 0` |
| `THINKING_BUDGET_COMPACT` | 1 024 | Short thinking budget (prune, action review) |
| `THINKING_BUDGET_MEDIUM` | 2 048 | Medium thinking budget (chat synthesis, trading) |
| `THINKING_BUDGET_DEEP` | 4 000 | Long thinking budget (refine_prompt, reasoning) |
| `MAX_TOKENS_TINY` | 80 | Ticker, web optimiser — very short answer |
| `MAX_TOKENS_SHORT` | 300 | Router, calendar, web judge |
| `MAX_TOKENS_COMPACT` | 600 | Push, nightly cleaning, memory ops |
| `MAX_TOKENS_MEDIUM` | 1 000 | Analyzer, reflection, alerts |
| `MAX_TOKENS_NO_THINK` | 1 500 | Plain chat, nightly facts |
| `MAX_TOKENS_BRIEFING` | 3 000 | Daily briefing |
| `MAX_TOKENS_THINK_COMPACT` | `COMPACT + 1024` | prune / action review (thinking + answer) |
| `MAX_TOKENS_THINK_MEDIUM` | `MEDIUM + 3000` | Trading thresholds (thinking + answer) |
| `MAX_TOKENS_SYNTHESIS` | 8 000 | Web/RAG chat with thinking |
| `MAX_TOKENS_REASONING` | 10 000 | `use_reasoning` chat + refine_prompt |
| `MAX_TOKENS_HARD_CAP` | 16 000 | Absolute kill switch on all local calls |
| `HIST_CONV_TOKEN_BUDGET` | 800 | Token budget for raw history injected per turn |
| `SESSION_SUMMARY_TOKENS` | 600 | Token budget of the session summary (~2 400 chars) |
| `HIST_CONV_SUMMARIZE_THRESHOLD` | 1 500 | Chars of uncovered conversation that trigger session compression |
| `PROFILE_NARRATIVE_TOKENS` | 600 | Token budget of the narrative portrait from `update_profile_narrative` (nightly, 7-day TTL) |

---

## Speculative decoding (MLX) — not applicable

`mlx-lm` supports native speculative decoding through
`stream_generate(draft_model=..., num_draft_tokens=N)`, but the feature is **incompatible
with Qwen3.6**, a hybrid Transformer + GatedDeltaNet model.

GatedDeltaNet layers keep a recurrent state (`ArraysCache`) that cannot be rolled back when
a draft token is rejected — mlx-lm refuses it explicitly (`cache non-trimmable`). Tested and
confirmed: output degenerates into noise from the first rejections.

**Blocked on**: recurrent-state rollback support in mlx-lm, or a move to a pure-attention
primary model (dense Qwen3-30B, etc.).
