#!/usr/bin/env python3
"""
backtest.py — evaluate the prediction model against historical games.

For every completed game in a date range this:
  1. Reconstructs each team's stats AS THEY WERE the day BEFORE the game
     (season-to-date through D-1, via StatsAPI byDateRange) — no look-ahead.
  2. Feeds those point-in-time features to the SAME model.predict() the live
     site uses (imported from model.py), producing a pre-game win probability
     and projected total.
  3. Compares to the actual final score.

Then it reports win-probability calibration/accuracy (Brier, log loss,
calibration table, vs. sensible baselines) and run-total error (MAE/RMSE/bias).

Honest limitations (documented on purpose):
  - Handedness splits and Statcast xERA/K%/BB% are NOT reconstructable
    point-in-time from the free endpoints, so the backtest uses as-of-date
    OVERALL wOBA and an ERA+computed-FIP talent blend. The live site adds
    handedness + Statcast peripherals on top.
  - Bullpen fatigue IS reconstructed here: recent pitch counts are harvested
    from the box scores of the six days before each game, so the model's
    availability x usage weighting is applied exactly as it is live (talent is
    the reduced FIP-based estimate noted above).
  This keeps every evaluated signal strictly pre-game and reproducible.

Usage:
    python3 backtest.py --start 2025-06-01 --end 2025-06-30
    python3 backtest.py --start 2025-07-01 --end 2025-07-31 --out backtest.html
    python3 backtest.py --start 2025-08-01 --end 2025-08-07 --limit 50
"""

import io
import ssl
import csv
import math
import json
import argparse
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request

import model

STATS = "https://statsapi.mlb.com/api/v1"
_CTX_V = ssl.create_default_context()
_CTX_U = ssl._create_unverified_context()


def get_json(url):
    req = Request(url, headers={"User-Agent": "mlb-backtest/1.0"})
    for ctx in (_CTX_V, _CTX_U):
        try:
            with urlopen(req, timeout=60, context=ctx) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue  # try the next context (verified may fail on some Macs)
    return None


def season_start(season):
    # A safe lower bound; StatsAPI clips to actual first game.
    return f"{season}-03-01"


# --- Per-date, point-in-time feature stores (cached) --------------------------

_asof_cache = {}


def asof_stats(cutoff):
    """Everything through `cutoff` (a date): pitcher map + team offense/defense."""
    if cutoff in _asof_cache:
        return _asof_cache[cutoff]
    season = cutoff.year
    start = season_start(season)
    end = cutoff.isoformat()

    pit = {}
    d = get_json(f"{STATS}/stats?stats=byDateRange&group=pitching&sportId=1"
                 f"&startDate={start}&endDate={end}&playerPool=all&limit=3000")
    if d and d.get("stats"):
        for s in d["stats"][0].get("splits", []):
            pid = s.get("player", {}).get("id")
            st = s.get("stat", {})
            if pid is None:
                continue
            pit[pid] = {
                "team": s.get("team", {}).get("id"),
                "era": st.get("era"),
                "fip": model.calc_fip(st),
                "ip": model.ip_to_float(st.get("inningsPitched")),
                "gs": model._num(st, "gamesStarted"),
                "gp": model._num(st, "gamesPitched"),
                "sv": model._num(st, "saves"),
                "hld": model._num(st, "holds"),
            }

    off = {}
    d = get_json(f"{STATS}/teams/stats?stats=byDateRange&group=hitting&sportId=1"
                 f"&startDate={start}&endDate={end}")
    lg_agg = {}
    if d and d.get("stats"):
        for s in d["stats"][0].get("splits", []):
            tid = s.get("team", {}).get("id")
            st = s.get("stat", {})
            g = model._num(st, "gamesPlayed") or 1
            off[tid] = {"woba": model.calc_woba(st),
                        "rspg": model._num(st, "runs") / g}
            for k in ("atBats", "hits", "doubles", "triples", "homeRuns",
                      "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies"):
                lg_agg[k] = lg_agg.get(k, 0) + model._num(st, k)
    lg_woba = model.calc_woba(lg_agg) or 0.315

    deff = {}
    d = get_json(f"{STATS}/teams/stats?stats=byDateRange&group=pitching&sportId=1"
                 f"&startDate={start}&endDate={end}")
    eras = []
    if d and d.get("stats"):
        for s in d["stats"][0].get("splits", []):
            tid = s.get("team", {}).get("id")
            st = s.get("stat", {})
            g = model._num(st, "gamesPlayed") or 1
            deff[tid] = {"rapg": model._num(st, "runs") / g}
            if st.get("era") is not None:
                eras.append(float(st["era"]))
    lg_era = sum(eras) / len(eras) if eras else 4.15
    lg_r = (sum(o["rspg"] for o in off.values()) / len(off)) if off else 4.4

    # Precompute each team's bullpen: talent + role (for expected usage).
    # Availability is applied per-game from reconstructed recent usage.
    pen_by_team = {}
    for pid, p in pit.items():
        tid = p["team"]
        if tid is None or p["ip"] < 10:
            continue
        if p["gp"] > 0 and p["gs"] / p["gp"] >= 0.5:
            continue  # starter
        talent = model.pitcher_true_talent(p["era"], p["fip"], None, None, lg_era)
        role = ("CL" if p["sv"] >= 8 else "SU" if p["hld"] >= 8
                else "SW" if p["gs"] >= 3 else "RP")
        pen_by_team.setdefault(tid, []).append(
            {"pid": pid, "talent": talent, "role": role})

    store = {"pit": pit, "off": off, "deff": deff, "pen": pen_by_team,
             "lg_woba": lg_woba, "lg_era": lg_era, "lg_r": lg_r}
    _asof_cache[cutoff] = store
    return store


# --- Point-in-time reliever fatigue (reconstructed from box scores) -----------

_box_cache = {}     # gamePk -> (date, {pid: pitches})
_sched_cache = {}   # (start, end) -> [(gamePk, date), ...]


def _final_games(start, end):
    key = (start, end)
    if key in _sched_cache:
        return _sched_cache[key]
    d = get_json(f"{STATS}/schedule?sportId=1&startDate={start}&endDate={end}")
    pk_dates = []
    if d:
        for dd in d.get("dates", []):
            for g in dd.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    pk_dates.append((g["gamePk"], dd["date"]))
    _sched_cache[key] = pk_dates
    return pk_dates


def _box_pitch_counts(pk_date):
    pk, dstr = pk_date
    if pk in _box_cache:
        return pk, _box_cache[pk]
    b = get_json(f"{STATS}/game/{pk}/boxscore")
    counts = {}
    if b:
        for side in ("home", "away"):
            tm = b.get("teams", {}).get(side, {})
            for pid in tm.get("pitchers", []):
                st = (tm.get("players", {}).get(f"ID{pid}", {})
                      .get("stats", {}).get("pitching", {}))
                np = st.get("numberOfPitches")
                if np:
                    counts[pid] = counts.get(pid, 0) + float(np)
    _box_cache[pk] = (dstr, counts)
    return pk, (dstr, counts)


_usage_cache = {}


def recent_usage_asof(game_day, lookback=6):
    """{pid: {date: pitches}} over the `lookback` days before game_day."""
    if game_day in _usage_cache:
        return _usage_cache[game_day]
    start = (game_day - timedelta(days=lookback)).isoformat()
    end = (game_day - timedelta(days=1)).isoformat()
    pk_dates = _final_games(start, end)
    usage = {}
    if pk_dates:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for pk, (dstr, counts) in ex.map(_box_pitch_counts, pk_dates):
                d = datetime.strptime(dstr, "%Y-%m-%d").date()
                for pid, np in counts.items():
                    usage.setdefault(pid, {})
                    usage[pid][d] = usage[pid].get(d, 0) + np
    _usage_cache[game_day] = usage
    return usage


def games_on(day_str):
    """Completed games on a date with final scores + probable pitchers."""
    d = get_json(f"{STATS}/schedule?sportId=1&date={day_str}"
                 f"&hydrate=probablePitcher,linescore")
    out = []
    if not d:
        return out
    for dd in d.get("dates", []):
        for g in dd.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            h = g["teams"]["home"]
            a = g["teams"]["away"]
            hs, as_ = h.get("score"), a.get("score")
            if hs is None or as_ is None:
                continue
            out.append({
                "home_id": h["team"]["id"], "away_id": a["team"]["id"],
                "home_sp": (h.get("probablePitcher") or {}).get("id"),
                "away_sp": (a.get("probablePitcher") or {}).get("id"),
                "home_score": hs, "away_score": as_,
            })
    return out


def build_side(team_id, sp_id, store, usage, game_day):
    p = store["pit"].get(sp_id) if sp_id else None
    lg_era = store["lg_era"]
    if p:
        sp_ra = model.pitcher_true_talent(p["era"], p["fip"], None, None, lg_era)
        proj_ip = model._project_ip(p["ip"], p["gs"], model.DEFAULT_CFG)
    else:
        sp_ra, proj_ip = lg_era + 0.25, 4.5

    # Bullpen: talent x (reconstructed availability) x (role-based usage).
    arms = []
    for a in store["pen"].get(team_id, []):
        avail = model.availability_from_usage(usage.get(a["pid"], {}), game_day)
        arms.append({"talent": a["talent"], "avail": avail,
                     "usage": model.usage_weight(a["role"])})
    pen_ra = model.bullpen_run_prevention(arms, lg_era)

    off = store["off"].get(team_id, {})
    deff = store["deff"].get(team_id, {})
    return {
        "off_woba": off.get("woba"), "lg_off_woba": store["lg_woba"],
        "sp_ra": sp_ra, "proj_ip": proj_ip, "pen_ra": pen_ra,
        "rspg": off.get("rspg"), "rapg": deff.get("rapg"),
    }


def predict_historical(game, store, usage, game_day, cfg=None):
    home = build_side(game["home_id"], game["home_sp"], store, usage, game_day)
    away = build_side(game["away_id"], game["away_sp"], store, usage, game_day)
    ctx = {"lg_r": store["lg_r"],
           "park_factor": model.park_factor(game["home_id"])}
    return model.predict(home, away, ctx, cfg)


# --- Metrics ------------------------------------------------------------------

def evaluate(rows):
    """rows: list of (p_home, home_win, exp_total, actual_total)."""
    n = len(rows)
    if not n:
        return None
    brier = sum((p - y) ** 2 for p, y, _, _ in rows) / n
    eps = 1e-9
    logloss = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                   for p, y, _, _ in rows) / n
    picks_right = sum(1 for p, y, _, _ in rows if (p >= 0.5) == (y == 1))
    acc = picks_right / n
    home_win_rate = sum(y for _, y, _, _ in rows) / n

    # Calibration deciles
    bins = [[] for _ in range(10)]
    for p, y, _, _ in rows:
        bins[min(9, int(p * 10))].append((p, y))
    calib = []
    for i, b in enumerate(bins):
        if b:
            calib.append((f"{i*10}-{i*10+10}%", len(b),
                          sum(p for p, _ in b) / len(b),
                          sum(y for _, y in b) / len(b)))

    # Totals
    tot_rows = [(e, a) for _, _, e, a in rows if e is not None]
    mae = rmse = bias = None
    if tot_rows:
        mae = sum(abs(e - a) for e, a in tot_rows) / len(tot_rows)
        rmse = math.sqrt(sum((e - a) ** 2 for e, a in tot_rows) / len(tot_rows))
        bias = sum(e - a for e, a in tot_rows) / len(tot_rows)

    return {
        "n": n, "acc": acc, "brier": brier, "logloss": logloss,
        "home_win_rate": home_win_rate,
        "brier_baseline_50": 0.25,
        "brier_baseline_homerate": home_win_rate * (1 - home_win_rate),
        "acc_baseline_home": max(home_win_rate, 1 - home_win_rate),
        "logloss_baseline_homerate": -(home_win_rate * math.log(home_win_rate + eps)
                                       + (1 - home_win_rate) * math.log(1 - home_win_rate + eps)),
        "calib": calib,
        "mae": mae, "rmse": rmse, "bias": bias,
    }


# --- Report -------------------------------------------------------------------

def print_report(m, start, end):
    print("\n" + "=" * 60)
    print(f" BACKTEST  {start} .. {end}   ({m['n']} games)")
    print("=" * 60)
    print(f" Win-prob accuracy      : {m['acc']*100:5.1f}%   "
          f"(baseline pick-home {m['acc_baseline_home']*100:.1f}%)")
    print(f" Brier score            : {m['brier']:.4f}   "
          f"(50/50 {m['brier_baseline_50']:.4f}, "
          f"home-rate {m['brier_baseline_homerate']:.4f})  lower=better")
    print(f" Log loss               : {m['logloss']:.4f}   "
          f"(home-rate {m['logloss_baseline_homerate']:.4f})  lower=better")
    print(f" Home win rate (actual) : {m['home_win_rate']*100:5.1f}%")
    if m["mae"] is not None:
        print(f" Total runs  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
              f"bias={m['bias']:+.2f} (model minus actual)")
    print("\n Calibration (predicted vs actual home-win rate):")
    print("   bucket        n     pred    actual")
    for label, cnt, pred, act in m["calib"]:
        print(f"   {label:<10} {cnt:4d}   {pred*100:5.1f}%   {act*100:5.1f}%")
    print("=" * 60 + "\n")


def write_html(m, start, end, path):
    rows = "".join(
        f"<tr><td>{lb}</td><td>{cnt}</td><td>{pr*100:.1f}%</td>"
        f"<td>{ac*100:.1f}%</td></tr>"
        for lb, cnt, pr, ac in m["calib"])
    totals = (f"MAE {m['mae']:.2f} · RMSE {m['rmse']:.2f} · bias {m['bias']:+.2f}"
              if m["mae"] is not None else "n/a")
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Backtest {start}..{end}</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;
color:#e6edf3;max-width:760px;margin:24px auto;padding:0 16px;line-height:1.5}}
h1{{font-size:22px}}table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #2a3241;padding:6px 10px;text-align:center}}
th{{color:#8b949e}} .k{{color:#8b949e}} .big{{font-size:19px;font-weight:700}}
.card{{background:#161b22;border:1px solid #2a3241;border-radius:10px;padding:14px;margin:10px 0}}</style>
</head><body>
<h1>⚾ Model backtest — {start} to {end}</h1>
<div class=card><span class=k>Games</span> <span class=big>{m['n']}</span></div>
<div class=card>
<span class=k>Win-prob accuracy</span> <span class=big>{m['acc']*100:.1f}%</span>
&nbsp; (pick-home baseline {m['acc_baseline_home']*100:.1f}%)<br>
<span class=k>Brier</span> <b>{m['brier']:.4f}</b> (50/50={m['brier_baseline_50']:.3f}, home-rate={m['brier_baseline_homerate']:.3f}) — lower better<br>
<span class=k>Log loss</span> <b>{m['logloss']:.4f}</b> (home-rate={m['logloss_baseline_homerate']:.4f}) — lower better<br>
<span class=k>Actual home win rate</span> {m['home_win_rate']*100:.1f}%<br>
<span class=k>Run total</span> {totals}
</div>
<div class=card><b>Calibration</b> — if calibrated, pred≈actual in each bucket
<table><tr><th>Predicted bucket</th><th>n</th><th>Model</th><th>Actual</th></tr>{rows}</table></div>
<p class=k>Point-in-time reconstruction via StatsAPI byDateRange + box-score
recent-usage. Bullpen availability×usage IS reconstructed; handedness splits and
Statcast peripherals are live-only. Analysis only, not betting advice.</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {path}")


def daterange(start, end):
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap number of games")
    ap.add_argument("--out", default="", help="write an HTML report here")
    args = ap.parse_args()

    rows = []
    dates = list(daterange(args.start, args.end))
    for i, d in enumerate(dates, 1):
        gs = games_on(d.isoformat())
        if not gs:
            continue
        cutoff = d - timedelta(days=1)   # stats as of the day BEFORE
        store = asof_stats(cutoff)
        usage = recent_usage_asof(d)     # reconstructed reliever fatigue
        for g in gs:
            pred = predict_historical(g, store, usage, d)
            home_win = 1 if g["home_score"] > g["away_score"] else 0
            actual_total = g["home_score"] + g["away_score"]
            rows.append((pred["p_home_raw"], home_win,
                         pred["exp_total"], actual_total))
        print(f"  [{i}/{len(dates)}] {d}  (+{len(gs)} games, total {len(rows)})",
              flush=True)
        if args.limit and len(rows) >= args.limit:
            rows = rows[:args.limit]
            break

    m = evaluate(rows)
    if not m:
        print("No completed games found in range.")
        return
    print_report(m, args.start, args.end)
    if args.out:
        write_html(m, args.start, args.end, args.out)


if __name__ == "__main__":
    main()
