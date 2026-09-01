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

## Notes

- **Not betting advice.** The win-probability model is an analytical estimate
  built from the displayed stats.
- Early in a season or in the offseason, some advanced metrics may be sparse.
