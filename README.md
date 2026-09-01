# ⚾ MLB Schedule & Probable Pitchers

A web app that shows today's MLB slate with, for every game:

- Both teams' logos, overall / home / away / L10 records, streak, run
  differential, vs-LHP/RHP splits, Pythagorean record, and team offense/pitching
- Both **probable pitchers** with standard stats, sabermetrics (FIP, xFIP, ERA-,
  FIP-, WAR), expected/Statcast metrics (xERA, xwOBA, K%, Whiff%, Barrel%, …),
  pitch arsenal, and recent form
- **Batter-vs-Pitcher** career matchups vs the opposing roster
- **Bullpen availability & fatigue** — each reliever's recent usage (from the
  last 6 days of box scores) graded 🟢🟡🟠🔴, with advanced stats
- A 🔮 **win-probability model** (expected runs → Pythagorean win %)
- An on-page **date picker** (Prev / date / Today / Next / ⟳ Refresh)

All data comes from free, key-free sources: **MLB StatsAPI** and
**Baseball Savant**. Values are rounded to 3 significant figures.

---

## Run locally

```bash
pip install -r requirements.txt
python3 app.py
# open http://localhost:8000
```

Set a port with `PORT=5000 python3 app.py`.

There's also a no-Flask fallback — `python3 mlb_schedule.py` runs the same site
on Python's stdlib server, and `python3 mlb_schedule.py --static 2025-09-01`
writes a single offline `.html` file.

## Production command

```bash
gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
```

**Keep it to one worker.** The in-memory page cache and the background
"prewarm" thread (which rebuilds *today* every 15 min so visitors get an instant
page) live inside the process; multiple workers would each rebuild and waste API
calls. Threads handle concurrent requests fine.

### Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `PORT` | 8000 | Port to bind |
| `MLB_CACHE_TTL` | 600 | Seconds a built date stays cached |
| `MLB_REFRESH_SECONDS` | 900 | How often the background thread rebuilds *today* |
| `ODDS_API_KEY` | *(unset)* | A free key from the-odds-api.com. When set, each game shows the live moneyline, the model's edge vs the market, and a ✓ VALUE flag. Unset = the odds row simply doesn't render. |

---

## Deploy

The repo includes a `Procfile`, `requirements.txt`, `runtime.txt`, and a
`render.yaml` blueprint, so it works on most Python hosts. A full build takes
~30–40s, so a host **without** an aggressive request timeout is ideal — the
background prewarm keeps *today* instant, but the `--timeout 120` matters for
first-time loads of other dates.

### Render (easiest — free tier)

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com): **New + → Blueprint**, pick the repo.
   Render reads `render.yaml` and configures everything (build, start,
   health check, env vars).
3. Deploy. You get a `https://<name>.onrender.com` URL.

*(Free Render services sleep when idle and cold-start on the next request; the
first hit after sleeping will be slow while it wakes and prewarms.)*

### Railway

1. Push to GitHub. On [railway.app](https://railway.app): **New Project →
   Deploy from GitHub repo**.
2. Railway detects the `Procfile`. Add env vars from the table above if you want
   to tune them. Generate a public domain under the service's **Settings →
   Networking**.

### Fly.io

```bash
fly launch --no-deploy      # generates fly.toml; keep 1 machine
fly deploy
```
Ensure the internal port matches `$PORT` (Fly sets it) and the process uses the
gunicorn command above.

### Any VPS / your own machine

```bash
pip install -r requirements.txt
gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:8000
```
Put nginx/Caddy in front for TLS if exposing publicly.

---

## How it stays fast & polite to the APIs

- ~70 upstream calls per build, parallelized with a thread pool.
- Bulk endpoints (all pitchers' season + sabermetric stats, Savant leaderboards,
  a league-wide box-score sweep for fatigue) instead of per-player fan-out.
- Per-date in-memory cache + background refresh of today.

## Prediction model (`model.py`)

Shared by the live site and the backtest so both run identical math. Expected
runs per team come from:

- **Handedness offense** — team wOBA vs the *opposing starter's hand* (vs RHP /
  vs LHP), not just runs/game.
- **Offensive quality** — wOBA (linear weights) + ISO + OPS, with a wRC+ estimate.
- **Starter** — a true-talent run rate blending ERA / FIP / xFIP / xERA.
- **Expected starter innings** — projected from IP/GS; a 6.5-IP arm shifts more
  weight onto the starter and less onto the bullpen than a 4.5-IP arm.
- **Bullpen = quality × availability** — fatigue-weighted true talent of the
  *available* arms (no saves/holds as a quality proxy).
- **Park factors** — Coors vs Oracle are not the same run environment.

- **Recency + regression** — starter, bullpen, and offense inputs blend
  season-to-date with the **last 30 days**, each regressed to the league mean by
  sample size (so small samples don't swing the projection).

Team run totals are then drawn from a **negative-binomial distribution**
(overdispersed, matching real MLB scoring) and the win probability is the
P(home runs > away runs) over the joint distribution, with a home-field bump.

### Live odds & value (optional, `odds.py`)

Set `ODDS_API_KEY` (free tier from the-odds-api.com) and each game card adds a
row: the current moneyline, the market's de-vigged implied probability, the
model's probability, the **edge**, and a **✓ VALUE** flag when the model's
number beats the market's. With no key the row is simply omitted. Backtesting
shows value-only + line-shopping is the only variant that reached roughly
break-even against closing lines — see `bet_backtest.py`. Not betting advice.

## Backtest (`backtest.py`)

Reconstructs each historical game's inputs **as they were the day before**
(season-to-date via StatsAPI `byDateRange` — no look-ahead), runs the same
`model.predict()`, and compares to the actual final score.

```bash
python3 backtest.py --start 2025-06-01 --end 2025-06-30 --out backtest.html
```

Reports win-probability **accuracy, Brier score, log loss**, a **calibration
table**, and run-total **MAE/RMSE/bias**, each against sensible baselines
(50/50, always-pick-home). Example (187 games, first half of June 2025): Brier
0.246 vs 0.250 baseline, near-monotonic calibration, totals MAE ~3.6 with ~0
bias — a small but real edge, which is about what an honest MLB model looks like.

*Point-in-time caveat:* handedness splits and Statcast xERA can't be
reconstructed historically from the free endpoints, and bullpen fatigue isn't
rebuilt, so the backtest evaluates a slightly reduced feature set (overall wOBA
+ ERA/FIP talent, quality-only bullpen). Those live-only refinements are neither
credited nor penalized — every evaluated signal is strictly pre-game.

## Notes

- **Not betting advice.** The win-probability model is an analytical estimate
  built from the displayed stats.
- Early in a season or in the offseason, some advanced metrics may be sparse.
