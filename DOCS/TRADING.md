# Trading

Portfolio surveillance built on a Boursorama CSV export, live prices from yfinance, and a
year of daily history used to read moves in context.

The module answers three different questions, and keeps them apart on purpose:

| Question | Answered by | Cadence |
|---|---|---|
| What do I hold, and at what price? | `trading/core.py` | hourly |
| Is something happening *right now* that deserves a push? | `trading/core.py` → `evaluate_alerts` | hourly |
| Where is this heading, and what is coming up? | `trading/market.py` | daily, from the briefing |

---

## Package layout

```
jarvis-core/src/trading/
    __init__.py   re-exports the public surface — every historical `from trading import …` still works
    keys.py       Redis key schema. No dependencies; imported alone by self/actions.py
    core.py       CSV import, yfinance prices, portfolio read, alert evaluation
    market.py     daily history: trends, statistics, market context, upcoming dates
```

Dependency graph (acyclic): `core → keys`, `market → core`.

`core` and `market` are separate for a operational reason, not a cosmetic one. `core` lives in
the **hourly** loop and only ever handles a snapshot; `market` downloads a year of history and
runs **once a day**. Merging them would drop a slow network call into a frequent path.

---

## Data flow

1. You drop a Boursorama CSV export into **your own subfolder**,
   `TRADE_DATA_DIR/{USER_CODE}/` (`/opt/jarvis/RAGData/Trade/KORBEN99/` by default layout).
   **CSV only** — the importer globs `*.csv`, an `.xlsx` is ignored.

   The per-user subfolder is not cosmetic: a broker export carries no mark of its owner, so
   a shared directory makes "whose file is this?" unanswerable, and every user importing
   receives the most recently dropped CSV whoever produced it. The folder is the only
   ownership marker available. There is **no fallback to the root directory** — a user with
   no subfolder imports nothing, which is the right default: importing zero positions beats
   importing someone else's.
2. The scheduler calls `run_trade_check()`:
   - `import_csv_to_redis()` — parses the most recent CSV. **mtime-gated**: an unchanged file
     is skipped, so quantities and buying prices stay frozen at the date of your last export.
   - `fetch_live_prices()` — yfinance per ISIN → current price + intraday %, cached 55 min.
   - `update_prices_in_redis()` — writes the prices back into the position hashes.
   - `evaluate_alerts()` — the primary model decides whether an alert is worth sending.
3. Fired alerts are queued in Redis and injected into the next `/chat` response.

Only users flagged `"trading": true` participate — see `DOCS/CONFIGURATION.md`.

---

## Trends and statistics (`market.py`)

Everything below comes from **one** series per ticker, downloaded once a day and cached in
Redis for 20 h. `core.py` alone knows only a snapshot — `last_price` and `intraday_var_pct`,
overwritten hourly — so no trend was computable from it. A point is not a trajectory.

### Per position

| Field | Meaning |
|---|---|
| `perf_5j` / `perf_1m` / `perf_6m` | rolling performance over 5, 22 and 127 sessions |
| `vs_mm50` / `vs_mm200` | above or below the 50- and 200-session moving average |
| `tendance` | one word, derived from the two averages (see below) |
| `volatilite_quotidienne_pct` | standard deviation of daily returns — the unit a move is measured in |
| `volatilite_annuelle_pct` | the same, annualised (× √252) |
| `plus_haut_52s` / `plus_bas_52s` | 52-week range |
| `perte_max_52s_pct` | worst drawdown from a peak over the period |

**Trend is defined explicitly, not left to the model**: above both averages = `haussière`,
below both = `baissière`, disagreement = `transition`. The disagreement case is the one that
carries information — Valneva measured at +23 % over a month *inside* a −43 % six-month slide
is a rebound within a downtrend, not a reversal, and only the MM50/MM200 split says so.

**Why volatility matters.** It is the unit of measurement for a move, not decoration.
Measured on the live portfolio: +3 % on an MSCI World ETF (±0.69 %/day) is over four standard
deviations — an event; +3 % on 2CRSI (±6.5 %/day) is half a deviation — an ordinary Tuesday.

### Market context

Nine reference series, each chosen because it changes how the portfolio reads: CAC 40,
EuroStoxx 50, S&P 500, Nasdaq, VIX (is the market calm or nervous), EUR/USD (it weighs on
every dollar-denominated asset, World ETFs first), gold, Brent, US 10-year yield.

### Upcoming dates

`Ticker.calendar` supplies real ex-dividend and earnings dates, replacing the hand-typed
`dividend_date` field. **Future dates only** — this filter is not cosmetic: measured on
2026-08-31, Yahoo returned a 2023-12-05 ex-dividend date for 2CRSI and an already-passed
earnings date for Valneva. Without the filter the briefing would announce a three-year-old
event as upcoming. ETFs return an empty calendar; that is normal, not a failure.

---

## Alert thresholds

`evaluate_alerts()` asks the primary model to decide. It fires only on:

- a user-defined `threshold_high` / `threshold_low` being crossed;
- an **abnormal** intraday move — |J| above that line's own `seuil_anormal`;
- a total daily portfolio loss above 2 %;
- a dividend or earnings date less than 5 calendar days away.

`seuil_anormal` is **two standard deviations of that line's own daily variation**. The rule
used to be a flat 3 % for everything, which on the live portfolio was wrong for four lines
out of six:

| Line | Old threshold | Volatility-relative |
|---|---|---|
| iShares MSCI World | 3 % | **1.4 %** |
| Amundi MSCI EMU | 3 % | 1.9 % |
| Sanofi / Solvay | 3 % | 3.1 % |
| Valneva | 3 % | **8.6 %** |
| 2CRSI | 3 % | **13.0 %** |

The alert path reads the **cache only** (`telecharger=False`) — it runs hourly and must never
pay a dozen Yahoo downloads. On a cold cache, before the day's first briefing, no
`seuil_anormal` is emitted and the model falls back to the historical 3 %, which is stated
explicitly in the alert system prompt.

Per-position rate limit: the same position will not alert twice within 8 h.

---

## Briefing integration

`market.render_briefing_block(user_code)` produces the `<perspectives_marche>` section, next
to `<portefeuille>`. The two do not overlap: `<portefeuille>` lists positions, prices and P&L;
`<perspectives_marche>` adds only what a snapshot cannot say — trajectory, normal variation,
deadlines.

Cost: ~4.5 s on the first call of the day (15 downloads), ~0.01 s afterwards. The morning
briefing pays it and nothing else does.

Two prompt guardrails (`prompts_fr.py` / `prompts_en.py`, `RÈGLE MARCHÉ` / `MARKET RULE`):

- **Last known session, never "today".** Measured on 08-31, Yahoo still served the 08-27
  close. Presenting those figures as today's would be wrong three days out of seven.
- **Perspective, never instruction.** Jarvis explains how a move reads in its context. It
  gives no buy, sell or reallocation advice and predicts no future price.

Moving averages are spelled out in full in the rendered block. Abbreviated as `MM200`, the
model unfolded them wrongly — "below its 200-**year** moving average" appeared in a test
briefing.

---

## Redis keys

| Pattern | Type | TTL | Contents |
|---|---|---|---|
| `trade:{code}:index` | set | — | Portfolio ISIN index |
| `trade:{code}:pos:{isin}` | hash | — | One position (price, thresholds…) |
| `trade:{code}:last_import_ts` | string | — | mtime of the last imported CSV |
| `trade:{code}:pending_alerts` | string | 24 h | Queued alerts, JSON list |
| `trade:price_cache:{isin}` | string | 55 min | yfinance price cache |
| `jarvis:market:hist:{ticker}` | string | 20 h | One year of daily closes, JSON |
| `jarvis:market:dates:{ticker}` | string | 20 h | Upcoming ex-dividend / earnings dates |

Only the closes are cached, not the full OHLC: everything the module computes derives from
them, and a list of floats reads back from JSON without pandas — the cache stays usable by any
process.

```bash
# Force a fresh history download on the next call
docker exec jarvis-redis redis-cli --scan --pattern "jarvis:market:*" \
  | xargs docker exec jarvis-redis redis-cli DEL
```

---

## Position hash fields

**CSV-sourced** (refreshed on each import): `name`, `isin`, `quantity`, `buying_price`,
`last_price_csv`, `total_var_pct_csv`, `last_movement_date`, `imported_at`.

**Price feed** (hourly, yfinance): `last_price`, `intraday_var_pct`, `price_updated_at`.

**Jarvis-managed** (never overwritten by import or price feed): `yahoo_ticker`,
`dividend_eur`, `dividend_date`, `threshold_high`, `threshold_low`, `last_alert_at`,
`last_alert_reason`, `notes`.

Yahoo dates are read in preference to `dividend_date`, but never written over it — a manually
entered value is left alone and used as the fallback.

---

## Known limitations

- **No price validation.** A guard once compared the live price to the CSV price and
  invalidated the ticker beyond 30 % divergence. It was removed: the CSV reference is a frozen
  snapshot of unbounded age, so any real trend eventually crossed it. Valneva went from 2.17
  to 3.08 in twelve days — a genuine +40 % taken for a resolution error, and the guard could
  not recover, freezing the price for 12 days. A drifting value is visible by eye on the
  portfolio.
- **ISIN identity does not replace it either**: yfinance returns `-` for VLA.PA and 2CRSI.PA,
  and `FR001400UEY9` instead of `FR0000120578` for Sanofi.
- **A wrong-but-real ticker is not detectable.** Resolution verifies that a symbol *returns a
  series*, not that it designates the right security. A symbol that exists but points at
  another company produces plausible prices, P&L and alerts with nothing in the logs. Removing
  the generative fallback closed the main source of this (search results are at least already
  associated by Yahoo with the ISIN or name); the residual risk is a same-name listing on
  another exchange, which is why ISIN search runs before name search.
- yfinance logs an HTTP 404 at `ERROR` level when asked for ETF fundamentals (`WPEA.PA`,
  `CMU.PA`), and again for any symbol it does not know. Non-fatal — the calendar simply comes
  back empty and `market.py` handles it. The `yfinance` logger is therefore pinned to
  `CRITICAL` in `helpers/logging_setup.py`: these lines are a handled condition, and left at
  `ERROR` they count towards `erreurs_log_24h`, which feeds vitals — an invalid symbol in a
  portfolio would raise a `degradation_interne` incident and push α up.

---

## See also

- `DOCS/CONFIGURATION.md` — `TRADE_DATA_DIR`, per-user `"trading"` flag
- `DOCS/API.md` — `/portfolio/*` endpoints
- `DOCS/REDIS.md` — full key reference
