#!/usr/bin/env python3
"""
nrfi.py — No-Run-First-Inning model, built on the same data feeds as the site.

Estimates P(no run scored in the 1st inning by either team) using, per side:
  1. Actual top-3 hitter quality (posted lineup, else the team's 3 highest-wOBA
     regulars) — a top-of-order boost over team-average offense.
  2. First-inning pitcher performance WITH SHRINKAGE — the starter's real 1st-
     inning run rate (StatsAPI sitCode i01) regressed toward their K/BB/HR-based
     talent and the league 1st-inning rate (small samples don't dominate).
  3. K/BB/HR pitcher profile — folded in via FIP, which the 1st-inning rate is
     shrunk toward.
  4. Data-derived first-inning baseline — league P(team scores in 1st) and runs
     per half-inning computed from recent linescores, not hardcoded.
  5. Backtesting/calibration — reconstructs pre-game NRFI probabilities for past
     games (no look-ahead) and scores them against actual 1st-inning results
     (Brier, calibration), fitting a global calibration multiplier.

Usage:
    python3 nrfi.py <awayId|abbr> <homeId|abbr> [date]   # analyze one game
    python3 nrfi.py --game <date>                        # first TOR/CLE-style lookup
    python3 nrfi.py --backtest <start> <end>             # calibrate/validate

Analysis only, not betting advice.
"""

import sys
import math
import json
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import model
import backtest as B          # reuse get_json + point-in-time helpers
import mlb_schedule as M

STATS = "https://statsapi.mlb.com/api/v1"
gj = B.get_json

# Calibration multiplier on scoring odds, fit by --backtest (1.0 = uncalibrated).
# Fit on Jun 1-7 2025 (188 half-innings): model over-predicted scoring, so <1.
CALIB = 0.90


# --------------------------------------------------------------------------
# League first-inning baseline (data-derived)
# --------------------------------------------------------------------------

def league_first_inning(start, end):
    """From linescores in [start,end]: P(a team scores in the 1st) and runs/half."""
    d = gj(f"{STATS}/schedule?sportId=1&startDate={start}&endDate={end}"
           f"&hydrate=linescore")
    halves = scored = 0
    runs = 0.0
    if d:
        for dd in d.get("dates", []):
            for g in dd.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                inns = g.get("linescore", {}).get("innings", [])
                if not inns:
                    continue
                for side in ("away", "home"):
                    r = inns[0].get(side, {}).get("runs")
                    if r is None:
                        continue
                    halves += 1
                    runs += r
                    if r > 0:
                        scored += 1
    p_score = scored / halves if halves else 0.28
    runs_per_half = runs / halves if halves else 0.52
    return {"p_score": p_score, "runs_per_half": runs_per_half, "n": halves}


# --------------------------------------------------------------------------
# First-inning pitcher run-prevention, shrunk toward K/BB/HR talent
# --------------------------------------------------------------------------

def fi_pitcher_rate(pid, season, lg_era, lg_fi_runrate, cutoff=None):
    """
    Expected runs allowed in the 1st inning by a starter, per inning.
    Blends the pitcher's actual 1st-inning rate (sitCode i01) with their
    FIP-based talent (K/BB/HR), shrunk by sample size and toward league.
    `cutoff` (date) bounds the sample for point-in-time backtests.
    """
    url = (f"{STATS}/people/{pid}?hydrate=stats(group=[pitching],"
           f"type=[statSplits],sitCodes=[i01],season={season})")
    d = gj(url)
    fi_ip = fi_runs = 0.0
    if d and d.get("people"):
        for grp in d["people"][0].get("stats", []):
            for s in grp.get("splits", []):
                st = s.get("stat", {})
                fi_ip += model.ip_to_float(st.get("inningsPitched"))
                fi_runs += model._num(st, "runs")
    # Pitcher's FIP-based talent as a per-inning run rate (encodes K/BB/HR).
    prof = M.fetch_pitcher(pid, season, {}, {})
    talent9 = (model.pitcher_true_talent(prof["era"], prof["fip"], prof["xfip"],
                                         prof["xera"], lg_era) if prof else lg_era)
    talent_per_inn = talent9 / 9.0
    # 1st innings tend to run slightly above a pitcher's overall rate (top of
    # order, no book on the hitters yet): nudge the talent anchor up to the
    # league 1st-inning level ratio.
    talent_fi = talent_per_inn * (lg_fi_runrate / (lg_era / 9.0))
    # Shrinkage: regress the observed 1st-inning rate toward talent_fi.
    K = 15.0  # innings of prior weight
    obs = fi_runs / fi_ip if fi_ip > 0 else talent_fi
    shrunk = (fi_runs + talent_fi * K) / (fi_ip + K)
    return {"rate": shrunk, "fi_ip": fi_ip, "fi_runs": fi_runs, "obs": obs,
            "talent9": talent9, "hand": (prof or {}).get("hand", "R"),
            "name": (prof or {}).get("name", str(pid))}


# --------------------------------------------------------------------------
# Top-3 hitter quality (top-of-order boost)
# --------------------------------------------------------------------------

def _woba_for(pid, season):
    d = gj(f"{STATS}/people/{pid}?hydrate=stats(group=[hitting],"
           f"type=[season],season={season})")
    if d and d.get("people"):
        for grp in d["people"][0].get("stats", []):
            for s in grp.get("splits", []):
                w = model.calc_woba(s.get("stat", {}))
                pa = model._num(s.get("stat", {}), "plateAppearances")
                if w:
                    return w, pa
    return None, 0


def top3_boost(team_id, season, lineup_ids, team_woba, lg_woba):
    """avg wOBA of the top-3 hitters / team-average wOBA (clamped)."""
    ids = lineup_ids
    if not ids:
        # Fallback: the team's 3 highest-wOBA regulars (>=150 PA).
        d = gj(f"{STATS}/teams/{team_id}/roster?rosterType=active")
        cand = []
        if d:
            pids = [p["person"]["id"] for p in d.get("roster", [])
                    if p.get("position", {}).get("abbreviation") != "P"]
            with ThreadPoolExecutor(max_workers=8) as ex:
                for pid, (w, pa) in zip(pids, ex.map(lambda x: _woba_for(x, season), pids)):
                    if w and pa >= 150:
                        cand.append((w, pid))
        cand.sort(reverse=True)
        ids = [pid for _, pid in cand[:3]]
    if not ids:
        return 1.0, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda x: _woba_for(x, season), ids))
    wobas = [w for w, _ in res if w]
    if not wobas:
        return 1.0, ids
    avg = sum(wobas) / len(wobas)
    base = team_woba or lg_woba
    return max(0.85, min(1.30, avg / base)), ids


# --------------------------------------------------------------------------
# The NRFI model
# --------------------------------------------------------------------------

def nrfi_for_game(home_id, away_id, home_sp, away_sp, season, lgfi,
                  lg_era, hand_splits, lg_woba, park, lineup=None):
    lg_p = lgfi["p_score"]
    lg_rr = lgfi["runs_per_half"]
    base_odds = lg_p / (1 - lg_p)

    home_fi = fi_pitcher_rate(home_sp, season, lg_era, lg_rr)
    away_fi = fi_pitcher_rate(away_sp, season, lg_era, lg_rr)

    def team_woba(tid, opp_hand):
        return (hand_splits.get(tid, {}).get(opp_hand, {}) or {}).get("woba")

    lu = lineup or {}
    home_boost, home_ids = top3_boost(home_id, season, lu.get("home", []),
                                      team_woba(home_id, away_fi["hand"]), lg_woba)
    away_boost, away_ids = top3_boost(away_id, season, lu.get("away", []),
                                      team_woba(away_id, home_fi["hand"]), lg_woba)

    def p_score(tid, opp_hand, opp_fi, boost):
        w = team_woba(tid, opp_hand)
        lw = lg_woba.get(opp_hand, 0.31)
        off = (w / lw) if (w and lw) else 1.0            # platoon offense
        off *= boost                                     # top-of-order quality
        pitch = opp_fi["rate"] / lg_rr                   # 1st-inning pitching
        odds = base_odds * off * pitch * (park / 100.0) * CALIB
        return odds / (1 + odds)

    ph = p_score(home_id, away_fi["hand"], away_fi, home_boost)  # home bats vs away SP
    pa = p_score(away_id, home_fi["hand"], home_fi, away_boost)
    nrfi = (1 - ph) * (1 - pa)
    return {"nrfi": nrfi, "yrfi": 1 - nrfi, "p_home": ph, "p_away": pa,
            "home_fi": home_fi, "away_fi": away_fi,
            "home_boost": home_boost, "away_boost": away_boost,
            "home_ids": home_ids, "away_ids": away_ids}


# --------------------------------------------------------------------------
# Single-game entry
# --------------------------------------------------------------------------

def analyze(away, home, day):
    season = int(day[:4])
    d = gj(f"{STATS}/schedule?sportId=1&date={day}"
           f"&hydrate=probablePitcher,lineups,team,venue")
    want = {str(home), str(away)}
    game = None
    for dd in (d or {}).get("dates", []):
        for g in dd.get("games", []):
            hid = str(g["teams"]["home"]["team"]["id"])
            aid = str(g["teams"]["away"]["team"]["id"])
            if {hid, aid} == want:
                game = g
                break
    if not game:
        print(f"No game matching {away} @ {home} on {day}.")
        return
    t = game["teams"]
    home_id, away_id = t["home"]["team"]["id"], t["away"]["team"]["id"]
    hn, an = t["home"]["team"]["name"], t["away"]["team"]["name"]
    home_sp = (t["home"].get("probablePitcher") or {}).get("id")
    away_sp = (t["away"].get("probablePitcher") or {}).get("id")
    if not (home_sp and away_sp):
        print("Probable pitchers not both posted yet.")
        return

    records = M.fetch_standings(season); ts = M.fetch_team_stats(season)
    lg = M.compute_league(records, ts)
    hand_splits, lg_woba = M.fetch_team_hand_splits(season, [home_id, away_id])
    gd = datetime.strptime(day, "%Y-%m-%d").date()
    lgfi = league_first_inning((gd - timedelta(days=21)).isoformat(),
                               (gd - timedelta(days=1)).isoformat())
    park = model.park_factor(home_id)

    lineup = {}
    lu = game.get("lineups", {})
    if lu.get("homePlayers"):
        lineup["home"] = [p["id"] for p in lu["homePlayers"][:3]]
    if lu.get("awayPlayers"):
        lineup["away"] = [p["id"] for p in lu["awayPlayers"][:3]]

    r = nrfi_for_game(home_id, away_id, home_sp, away_sp, season, lgfi,
                      lg["ERA"], hand_splits, lg_woba, park, lineup)

    print(f"\n=== NRFI: {an} @ {hn} ({day}) ===")
    print(f"League 1st-inning baseline: P(score) {lgfi['p_score']*100:.1f}% · "
          f"{lgfi['runs_per_half']:.2f} runs/half ({lgfi['n']} halves) · park {park}\n")
    for lbl, fi in [(an, r['away_fi']), (hn, r['home_fi'])]:
        print(f"  {fi['name']} ({fi['hand']}HP, {lbl}): 1st-inn obs {fi['obs']:.2f} "
              f"R/inn over {fi['fi_ip']:.0f} IP → shrunk {fi['rate']:.2f} "
              f"(talent {fi['talent9']:.2f} R/9)")
    lu_note = "posted lineup" if lineup else "top-3 by wOBA (no lineup yet)"
    print(f"\n  {hn} top-3 boost {r['home_boost']:.2f} · {an} top-3 boost "
          f"{r['away_boost']:.2f}  [{lu_note}]")
    print(f"  P({hn} score 1st) {r['p_home']*100:.1f}% · "
          f"P({an} score 1st) {r['p_away']*100:.1f}%\n")
    print(f"  ===> NRFI {r['nrfi']*100:.1f}%   YRFI {r['yrfi']*100:.1f}%")
    print(f"       Fair decimal — NRFI {1/r['nrfi']:.2f} | YRFI {1/r['yrfi']:.2f}")
    print("       Analysis only, not betting advice.\n")


# --------------------------------------------------------------------------
# Backtest / calibration
# --------------------------------------------------------------------------

def run_backtest(start, end):
    season = int(start[:4])
    records = M.fetch_standings(season); ts = M.fetch_team_stats(season)
    lg = M.compute_league(records, ts)
    rows = []  # (p_run_half, actual_run_0/1)
    dcur = datetime.strptime(start, "%Y-%m-%d").date()
    dend = datetime.strptime(end, "%Y-%m-%d").date()
    while dcur <= dend:
        day = dcur.isoformat()
        d = gj(f"{STATS}/schedule?sportId=1&date={day}"
               f"&hydrate=probablePitcher,linescore,team")
        lgfi = league_first_inning((dcur - timedelta(days=21)).isoformat(),
                                   (dcur - timedelta(days=1)).isoformat())
        games = [g for dd in (d or {}).get("dates", []) for g in dd.get("games", [])
                 if g.get("status", {}).get("abstractGameState") == "Final"]
        for g in games:
            t = g["teams"]
            hid, aid = t["home"]["team"]["id"], t["away"]["team"]["id"]
            hsp = (t["home"].get("probablePitcher") or {}).get("id")
            asp = (t["away"].get("probablePitcher") or {}).get("id")
            inns = g.get("linescore", {}).get("innings", [])
            if not (hsp and asp and inns):
                continue
            hand_splits, lg_woba = M.fetch_team_hand_splits(season, [hid, aid])
            park = model.park_factor(hid)
            r = nrfi_for_game(hid, aid, hsp, asp, season, lgfi, lg["ERA"],
                              hand_splits, lg_woba, park)
            ah = inns[0].get("home", {}).get("runs")
            aa = inns[0].get("away", {}).get("runs")
            if ah is not None:
                rows.append((r["p_home"], 1 if ah > 0 else 0))
            if aa is not None:
                rows.append((r["p_away"], 1 if aa > 0 else 0))
        print(f"  {day}: {len(games)} games (rows {len(rows)})", flush=True)
        dcur += timedelta(days=1)

    n = len(rows)
    if not n:
        print("No data."); return
    brier = sum((p - y) ** 2 for p, y in rows) / n
    base = sum(y for _, y in rows) / n
    # global calibration multiplier on odds to match observed rate
    pred = sum(p for p, _ in rows) / n
    print(f"\n=== NRFI BACKTEST {start}..{end} — {n} half-innings ===")
    print(f" Actual P(score in 1st): {base*100:.1f}%   Model mean: {pred*100:.1f}%")
    print(f" Brier: {brier:.4f}  (base-rate {base*(1-base):.4f})")
    print(" Calibration (predicted vs actual):")
    bins = [[] for _ in range(6)]
    for p, y in rows:
        bins[min(5, int(p * 10 // 1.6667))].append((p, y))
    for i, b in enumerate(bins):
        if b:
            print(f"   {sum(p for p,_ in b)/len(b)*100:5.1f}% pred → "
                  f"{sum(y for _,y in b)/len(b)*100:5.1f}% actual  (n={len(b)})")
    print(f"\n Suggested CALIB adjust: ×{base/pred:.3f} "
          f"(current CALIB={CALIB}); set CALIB≈{CALIB*base/pred:.3f}")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--backtest":
        run_backtest(a[1], a[2]); return
    if len(a) >= 2:
        day = a[2] if len(a) > 2 else datetime.now().strftime("%Y-%m-%d")
        analyze(a[0], a[1], day); return
    print(__doc__)


if __name__ == "__main__":
    main()
