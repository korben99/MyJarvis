# The Tier-1 router — a fine-tuned model, not a prompt

The intent router that settles what each request costs is **a LoRA fine-tune of
Qwen2.5-1.5B-Instruct**, served merged and quantised as
`models/hub/Qwen2.5-1.5B-router-v3-6bit`.

That one sentence has a consequence the rest of this file exists for: **its prompt is part
of the trained artefact.** `ROUTER_SYSTEM` and `ROUTER_USER` are not instructions the model
reads at inference — they are the fixed prefix it was conditioned on. Editing either one
without retraining degrades the router instead of improving it.

> **Training artefacts are not in Git.** `RouterData/` is gitignored — datasets, adapters
> and the frozen per-version prompts live only on the machine. A clone has the serving code
> and this file, not the means to reproduce the model.

---

## It is French-only

`build_dataset_v2.py` imports `ROUTER_SYSTEM` and `ROUTER_USER` from `prompts`, which
resolves to the **French** set, and writes them into every line of `train.jsonl`. There has
never been an English training run.

So on `JARVIS_LANG=en` the router is served `prompts_en.ROUTER_SYSTEM` — a prompt this LoRA
has never seen. The router is, in practice, francophone: an English instance runs it out of
distribution on every request. Nothing in the code reports this, because nothing can see it.

Two ways out, neither taken: keep serving the French router prompt whatever the UI language
(the router's output is JSON, not prose, so its prompt need not be localised), or build a
second dataset and a second adapter. If you ever switch an instance to English, decide this
first.

## Why the prompt cannot be edited in isolation

`lora_config_v3.yaml` sets `mask_prompt: true`. The system prompt is ~1155 tokens, identical
on all 730 training lines, and masked out of the loss — the model never learns to reproduce
it, only to emit the ~10 output tokens *given* it. The prefix is therefore closer to a
calibration constant than to an instruction.

The cost of drift is measured, twice, on this very model:

| Change | Effect |
|---|---|
| One example added to `ROUTER_SYSTEM`, no retrain (v2, 18/08) | validation **88 % → 82 %** |
| v2 evaluated with v3's prompt instead of its own (31/08) | **−6 points**, same cause |

Both are recorded in `lora_config_v3.yaml` and `compare_v2_v3.py`. This is why
`router_system_v2.txt` exists: each version's prompt is kept as an artefact of that version,
so an older adapter can still be evaluated on the text it was trained with.

### Guard rails now in place

- `ROUTER_SYSTEM` and `ROUTER_USER` are **excluded from `REFINABLE_PROMPTS`**. The
  prompt-refinement loop could otherwise propose a rewrite, and an approval writes the
  override live, with no restart and no retrain — exactly the −6 point gesture.
  `REFLECTION_PROMPT` tells the model so, and routes a wrong-routing observation to
  `alert_admin` instead.
- `test_le_prompt_du_routeur_correspond_a_son_dataset` compares `prompts_fr.ROUTER_SYSTEM`
  byte for byte with the system message of the first line of `train.jsonl`. It skips when
  the dataset is absent (a fresh clone), and fails the moment the two diverge. It compares
  the **French** constant regardless of `JARVIS_LANG`, since the dataset is French.

---

## Version history

Each version changed **one axis**, deliberately: hyperparameters were held identical from
v1 through v3 so that a comparison means something. Every improvement came from the input.

| | Samples | What changed | Outcome |
|---|---|---|---|
| **v1** | 443 | labels merged from 5 heterogeneous routers | failed — dataset was 97-100 % `null` on every extraction field and 87 % `memory`; the model collapsed onto the majority class |
| **v2** | 698 | relabelled by the 35 B then arbitrated by hand; `memory` brought down 87 % → 44.8 %; every field ≥ 3.7 % | 88 % on its validation; no majority collapse, no length confound (84 % beyond 130 characters) |
| **v3** | 730 | topic coverage: +12 internal-state examples (CVE, vitals, backups), +12 on valuing an asset considered for purchase, +8 long messages to break the class/length correlation the rebalancing had introduced; `ROUTER_SYSTEM` gained one `{"intents":["web","portfolio"]}` example | in production as `-v3-6bit` |

v1's real weakness was never the tuning. v2's was not a collapse either — it was **topic
coverage**: out of 698 examples, zero concerned Jarvis's own technical state and zero
concerned evaluating an asset before buying it; `self` examples were only greetings and
`portfolio` only spoke of positions already held.

The v3 fixture score is not recorded anywhere in the repository. If you measured it, this
table is where it belongs.

Hyperparameters, constant across all three: `fine_tune_type: lora`, `num_layers: 8`,
`batch_size: 4`, `iters: 550`, `learning_rate: 2e-4`, `optimizer: adamw`,
`max_seq_length: 2048`, rank 8, scale 20.0, dropout 0.05, `seed: 0`. Training always runs on
the **bf16** base; the deployment quantisation is decided afterwards, by measurement — the
memory of that decision is that 8-bit went to production and bf16 bought nothing.

## Evaluating

**`fixtures_validation.jsonl` is the gate — 37 hand-written cases, verified absent from
train/valid.** The real corpus barely contains `calendar`, `weather`, `gmail` or `briefing`
(2.0 %, 1.2 %, 0.1 %, 0.4 % of 856 messages), so a validation split drawn from it never
exercises those families at all.

**Never compare two versions on `valid.jsonl`.** It is a 10 % draw from the training pool,
so it changes between versions and the comparison means nothing.

```bash
cd /opt/jarvis/RouterData

# One model, against the current prompt from `prompts`
../venv/bin/python check_fixtures.py /opt/jarvis/models/hub/Qwen2.5-1.5B-router-v3-6bit

# Two models, each judged with the prompt it was trained on
../venv/bin/python compare_v2_v3.py <model_v2> <model_v3> --prompt-v2 router_system_v2.txt
```

## Retraining

The order matters: the prompt goes into the dataset, so the dataset is rebuilt *after* any
prompt change and never before.

1. **Freeze the current prompt** as an artefact, the way `router_system_v2.txt` was kept —
   otherwise the outgoing adapter can no longer be evaluated fairly.
2. **Rebuild the dataset.** `build_dataset_v2.py` re-imports `ROUTER_SYSTEM` / `ROUTER_USER`
   from `prompts`, so the new text lands on every line. It writes `train.jsonl` and
   `valid.jsonl` (90/10).
3. **Train** with a new `lora_config_v4.yaml` — change one axis, and say in the header which
   one and why. Keep the hyperparameters unless the hyperparameters *are* the axis.
4. **Measure on the fixtures**, not on `valid.jsonl`.
5. **Merge and quantise**, then measure again: the deployment quantisation is a separate
   decision from the training.
6. **Point `ROUTER_MODEL`** at the new path and restart.
7. **Re-run the test suite.** `test_le_prompt_du_routeur_correspond_a_son_dataset` turns
   green again once the dataset carries the prompt being served.

## Files in `RouterData/`

| File | Role |
|---|---|
| `routing_samples.jsonl` | raw collector output — every routing decision logged in production |
| `relabel_with_teacher.py` · `routing_samples_teacher.jsonl` | relabelling by the 35 B |
| `curate_router_dataset.py` · `routing_samples_curated.jsonl` | curation and class rebalancing |
| `desaccords_a_arbitrer.jsonl` · `appliquer_arbitrage.py` · `routing_samples_arbitre.jsonl` | teacher/collector disagreements, arbitrated by hand then applied |
| `exemples_simules.py` · `exemples_simules.jsonl` | hand-written examples for the families the real corpus lacks |
| `build_dataset_v2.py` | assembles `train.jsonl` + `valid.jsonl`, embedding the prompts |
| `lora_config_v{2,3}.yaml` | training configs; the header states the one axis that changed |
| `router_system_v2.txt` | v2's frozen prompt, so v2 stays evaluable |
| `check_fixtures.py` · `compare_v2_v3.py` · `fixtures_validation.jsonl` | evaluation |
| `check_frontiere.py` | boundary cases between neighbouring intents |
| `router_lora_adapterv1.py` | `routing_to_label()` — the routing dict → training label mapping |
| `adapters/qwen25-1.5b-router-v{1,2,3}/` | the LoRA adapters themselves |

## Where it sits in the request path

Tier 0 (`embed_router.py`, cosine similarity against pre-embedded examples) answers first
and settles most requests with no LLM at all. This router is **Tier 1**: it only sees what
Tier 0 declined — a score below 0.74, or an ambiguity margin under 0.06. See
**[ARCHITECTURE.md](ARCHITECTURE.md)** for the four tiers and the router's output fields.

A router failure (timeout, parse error) sets every `use_*` flag to `False`: no context is
fetched and the primary model answers from the system prompt alone.
