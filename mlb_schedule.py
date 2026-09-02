#!/usr/bin/env python3
"""
mlb_schedule.py

Scrapes today's MLB schedule and builds a rich, readable HTML page. For every
game it shows both teams (overall / home / away records + season context) and a
deep probable-pitcher profile drawn from MULTIPLE free, key-free data sources:

  MLB StatsAPI (statsapi.mlb.com)
    - season standard pitching stats
    - sabermetrics: FIP, xFIP, FIP-, ERA-, WAR, RA9-WAR, RAR
    - expectedStatistics: xwOBA, xBA, xSLG
    - pitchArsenal: pitch mix % and average velocity per pitch
    - gameLog: recent-form (last 3 outings)
    - standings: streak, last-10, run diff, RS/RA, division rank, GB, vs L/R,
      Pythagorean (expected) record
    - team hitting & pitching season stats (bulk)

  Baseball Savant (baseballsavant.mlb.com)
    - expected_statistics leaderboard: xERA, xBA, xSLG, xwOBA
    - custom Statcast leaderboard: K%, BB%, whiff%, barrel%, hard-hit%,
      groundball%, average fastball velocity

Everything is numeric-rounded to 3 significant figures.

Usage:
    python3 mlb_schedule.py                # live server + browser; pick ANY date on the page
    python3 mlb_schedule.py 2026-09-01     # live server, opening on this date
    python3 mlb_schedule.py 8077           # live server on a specific port
    python3 mlb_schedule.py --static       # write a single offline .html for today
    python3 mlb_schedule.py --static 2026-09-01   # offline .html for one date
    python3 mlb_schedule.py --no-open      # don't auto-open the browser

No third-party dependencies required (uses urllib from the stdlib). If `certifi`
is installed it is used for proper TLS verification; otherwise the script falls
back gracefully (these are public, read-only endpoints).
"""

import os
import io
import ssl
import sys
import csv
import json
import html
import time
import webbrowser
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import model  # shared prediction model (also used by backtest.py)
import odds   # optional live odds / value layer (dormant without ODDS_API_KEY)

STATS = "https://statsapi.mlb.com/api/v1"
SAVANT = "https://baseballsavant.mlb.com"
SPORT_ID = 1  # MLB
UA = "mlb-schedule-script/2.0 (+https://statsapi.mlb.com)"


# ---------------------------------------------------------------------------
# HTTP helpers (with graceful macOS SSL fallback)
# ---------------------------------------------------------------------------

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_VERIFIED_CTX = _ssl_context()
_UNVERIFIED_CTX = ssl._create_unverified_context()


def _fetch(url, decode=True):
    req = Request(url, headers={"User-Agent": UA})
    for ctx in (_VERIFIED_CTX, _UNVERIFIED_CTX):
        try:
            with urlopen(req, timeout=45, context=ctx) as resp:
                raw = resp.read()
                return raw.decode("utf-8") if decode else raw
        except ssl.SSLError:
            continue
        except (URLError, HTTPError, ValueError) as e:
            print(f"  ! request failed: {url}\n    {e}", file=sys.stderr)
            return None
    print(f"  ! SSL failed (verified + fallback): {url}", file=sys.stderr)
    return None


def get_json(url):
    txt = _fetch(url)
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except ValueError as e:
        print(f"  ! bad JSON: {url}\n    {e}", file=sys.stderr)
        return None


def get_csv(url):
    """Return a list of dict rows from a CSV endpoint, or []."""
    txt = _fetch(url)
    if not txt:
        return []
    txt = txt.lstrip("﻿")
    return list(csv.DictReader(io.StringIO(txt)))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def sig3(v):
    """Round a numeric value to 3 significant figures; pass non-numbers through."""
    if v in (None, ""):
        return "—"
    s = str(v).strip()
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    if f == 0:
        return "0"
    out = f"{f:.3g}"
    if "." in out and "e" not in out and "E" not in out:
        out = out.rstrip("0").rstrip(".")
    return out


def esc(x):
    return html.escape(str(x))


def pct(v):
    """Format a value already expressed as a percent number (e.g. 28.5 -> 28.5%)."""
    v = sig3(v)
    return f"{v}%" if v != "—" else "—"


# ---------------------------------------------------------------------------
# StatsAPI: schedule
# ---------------------------------------------------------------------------

def fetch_schedule(day):
    hydrate = "probablePitcher(note),linescore,team,person,venue,weather"
    url = f"{STATS}/schedule?sportId={SPORT_ID}&date={day}&hydrate={hydrate}"
    data = get_json(url)
    if not data:
        return []
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games


# ---------------------------------------------------------------------------
# StatsAPI: standings (rich team records)
# ---------------------------------------------------------------------------

def fetch_standings(season):
    url = (f"{STATS}/standings?leagueId=103,104&season={season}"
           f"&standingsTypes=regularSeason&hydrate=team")
    data = get_json(url)
    records = {}
    if not data:
        return records
    for rec in data.get("records", []):
        for tr in rec.get("teamRecords", []):
            tid = tr.get("team", {}).get("id")
            if tid is None:
                continue
            splits = {s.get("type"): s for s in
                      tr.get("records", {}).get("splitRecords", [])}
            expected = {s.get("type"): s for s in
                        tr.get("records", {}).get("expectedRecords", [])}

            def rc(d):
                return f"{d.get('wins', 0)}-{d.get('losses', 0)}" if d else "—"

            records[tid] = {
                "overall": f"{tr.get('wins', 0)}-{tr.get('losses', 0)}",
                "pct": tr.get("winningPercentage", "—"),
                "home": rc(splits.get("home")),
                "away": rc(splits.get("away")),
                "l10": rc(splits.get("lastTen")),
                "streak": tr.get("streak", {}).get("streakCode", "—"),
                "run_diff": tr.get("runDifferential", "—"),
                "rs": tr.get("runsScored", "—"),
                "ra": tr.get("runsAllowed", "—"),
                "div_rank": tr.get("divisionRank", "—"),
                "gb": tr.get("gamesBack", "—"),
                "vs_left": rc(splits.get("left")),
                "vs_right": rc(splits.get("right")),
                "day": rc(splits.get("day")),
                "night": rc(splits.get("night")),
                "one_run": rc(splits.get("oneRun")),
                "expected": rc(expected.get("xWinLoss")
                               or expected.get("xWinLossSeason")),
            }
    return records


# ---------------------------------------------------------------------------
# StatsAPI: bulk team hitting & pitching
# ---------------------------------------------------------------------------

def fetch_team_stats(season):
    out = {}

    def load(group, keys):
        url = (f"{STATS}/teams/stats?season={season}&sportId={SPORT_ID}"
               f"&group={group}&stats=season")
        data = get_json(url)
        if not data or not data.get("stats"):
            return
        for split in data["stats"][0].get("splits", []):
            tid = split.get("team", {}).get("id")
            if tid is None:
                continue
            st = split.get("stat", {})
            out.setdefault(tid, {})
            for label, key in keys.items():
                out[tid][label] = st.get(key, "—")

    load("hitting", {
        "avg": "avg", "obp": "obp", "slg": "slg", "ops": "ops",
        "runs": "runs", "hr": "homeRuns", "sb": "stolenBases",
        "bb": "baseOnBalls", "so": "strikeOuts", "babip": "babip",
    })
    load("pitching", {
        "team_era": "era", "team_whip": "whip", "team_so": "strikeOuts",
        "team_hr": "homeRuns", "team_sv": "saves",
    })
    # Advanced hitting adds ISO (SLG-AVG) and BABIP context in one bulk call.
    url = (f"{STATS}/teams/stats?season={season}&sportId={SPORT_ID}"
           f"&group=hitting&stats=seasonAdvanced")
    data = get_json(url)
    if data and data.get("stats"):
        for split in data["stats"][0].get("splits", []):
            tid = split.get("team", {}).get("id")
            if tid is not None:
                out.setdefault(tid, {})
                out[tid]["iso"] = split.get("stat", {}).get("iso", "—")
    return out


def fetch_team_hand_splits(season, team_ids):
    """
    Per-team offense vs RHP / vs LHP as wOBA (+ OPS), plus league baselines.
    Returns (splits, league) where:
        splits[team_id] = {"R": {woba, ops}, "L": {woba, ops}}
        league = {"R": lg_woba_vs_R, "L": lg_woba_vs_L}
    'R'/'L' key the OPPOSING pitcher's throwing hand.
    """
    splits = {}
    agg = {"R": {}, "L": {}}  # league aggregate counting stats by opp hand

    def load(tid):
        url = (f"{STATS}/teams/{tid}/stats?stats=statSplits&sitCodes=vr,vl"
               f"&group=hitting&season={season}")
        data = get_json(url)
        res = {}
        if data:
            for s in data.get("stats", []):
                for sp in s.get("splits", []):
                    code = sp.get("split", {}).get("code")  # 'vr' or 'vl'
                    hand = "R" if code == "vr" else "L" if code == "vl" else None
                    if hand:
                        res[hand] = sp.get("stat", {})
        return tid, res

    with ThreadPoolExecutor(max_workers=10) as ex:
        for tid, res in ex.map(load, team_ids):
            entry = {}
            for hand in ("R", "L"):
                st = res.get(hand, {})
                entry[hand] = {
                    "woba": model.calc_woba(st),
                    "ops": st.get("ops"),
                    "pa": model._num(st, "plateAppearances"),
                }
                # accumulate league totals
                for k in ("atBats", "hits", "doubles", "triples", "homeRuns",
                          "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies"):
                    agg[hand][k] = agg[hand].get(k, 0) + model._num(st, k)
            splits[tid] = entry

    league = {"R": model.calc_woba(agg["R"]) or 0.315,
              "L": model.calc_woba(agg["L"]) or 0.315}
    return splits, league


# ---------------------------------------------------------------------------
# Recency (last-30-day) inputs, blended + regressed in the prediction model
# ---------------------------------------------------------------------------

def fetch_recent_pitching(day):
    """{pid: {era, fip, ip}} over the 30 days before `day` (starters + pen)."""
    gd = datetime.strptime(day, "%Y-%m-%d").date()
    sd = (gd - timedelta(days=30)).isoformat()
    ed = (gd - timedelta(days=1)).isoformat()
    out = {}
    d = get_json(f"{STATS}/stats?stats=byDateRange&group=pitching&sportId={SPORT_ID}"
                 f"&startDate={sd}&endDate={ed}&playerPool=all&limit=3000")
    if d and d.get("stats"):
        for s in d["stats"][0].get("splits", []):
            pid = s.get("player", {}).get("id")
            if pid is None:
                continue
            st = s.get("stat", {})
            out[pid] = {"era": st.get("era"), "fip": model.calc_fip(st),
                        "ip": model.ip_to_float(st.get("inningsPitched"))}
    return out


def fetch_offense_windows(day, season):
    """
    Per-team GENERAL offense multiplier (~wRC+/100, park-neutral): a blend of
    7/15/30-day rolling wOBA and season wOBA, each regressed to league by PA.
    Returns ({team_id: general_mult}, lg_woba).
    """
    gd = datetime.strptime(day, "%Y-%m-%d").date()
    ed = (gd - timedelta(days=1)).isoformat()

    def hit(sd):
        d = get_json(f"{STATS}/teams/stats?stats=byDateRange&group=hitting"
                     f"&sportId={SPORT_ID}&startDate={sd}&endDate={ed}")
        out = {}
        if d and d.get("stats"):
            for s in d["stats"][0].get("splits", []):
                tid = s.get("team", {}).get("id")
                st = s.get("stat", {})
                out[tid] = (model.calc_woba(st), model._num(st, "plateAppearances"))
        return out

    windows = {
        "d7": hit((gd - timedelta(days=7)).isoformat()),
        "d15": hit((gd - timedelta(days=15)).isoformat()),
        "d30": hit((gd - timedelta(days=30)).isoformat()),
        "season": hit(f"{season}-03-01"),
    }
    season_w = windows["season"]
    wobas = [w for w, _ in season_w.values() if w]
    lg_woba = sum(wobas) / len(wobas) if wobas else 0.315

    mult = {}
    for tid in season_w:
        blended = 0.0
        for key, weight in model.OFFENSE_WINDOW_WEIGHTS.items():
            wo, pa = windows[key].get(tid, (None, 0))
            blended += weight * model.regress(wo, pa, 170, lg_woba)
        mult[tid] = blended / lg_woba if lg_woba else 1.0
    return mult, lg_woba


def apply_split_fallback(splits, prev_splits):
    """
    Where a current-season handedness split has < MIN_SPLIT_PLATE_APPEARANCES,
    substitute the previous season's split for that team/hand.
    """
    for tid, hs in splits.items():
        for hand in ("R", "L"):
            cur = hs.get(hand) or {}
            if (cur.get("pa") or 0) < model.MIN_SPLIT_PLATE_APPEARANCES:
                pv = (prev_splits.get(tid) or {}).get(hand) or {}
                if pv.get("woba"):
                    cur = dict(cur)
                    cur["woba"] = pv["woba"]
                    cur["fallback"] = True
            hs[hand] = cur
    return splits


# ---------------------------------------------------------------------------
# Baseball Savant: bulk leaderboards keyed by player_id
# ---------------------------------------------------------------------------

def fetch_savant_expected(season):
    url = (f"{SAVANT}/leaderboard/expected_statistics?type=pitcher&year={season}"
           f"&position=&team=&min=1&csv=true")
    rows = get_csv(url)
    out = {}
    for r in rows:
        pid = (r.get("player_id") or "").strip()
        if pid:
            out[pid] = {
                "xera": r.get("xera", "—"),
                "xba": r.get("est_ba", "—"),
                "xslg": r.get("est_slg", "—"),
                "xwoba": r.get("est_woba", "—"),
            }
    return out


def fetch_savant_custom(season):
    sel = ("pa,k_percent,bb_percent,whiff_percent,hard_hit_percent,"
           "barrel_batted_rate,fastball_avg_speed,groundballs_percent")
    url = (f"{SAVANT}/leaderboard/custom?year={season}&type=pitcher&filter="
           f"&min=1&selections={sel}&sort=1&sortDir=desc&csv=true")
    rows = get_csv(url)
    out = {}
    for r in rows:
        pid = (r.get("player_id") or "").strip()
        if pid:
            out[pid] = {
                "k_pct": r.get("k_percent", "—"),
                "bb_pct": r.get("bb_percent", "—"),
                "whiff_pct": r.get("whiff_percent", "—"),
                "hardhit_pct": r.get("hard_hit_percent", "—"),
                "barrel_pct": r.get("barrel_batted_rate", "—"),
                "gb_pct": r.get("groundballs_percent", "—"),
                "fb_velo": r.get("fastball_avg_speed", "—"),
            }
    return out


# ---------------------------------------------------------------------------
# StatsAPI: per-pitcher deep profile
# ---------------------------------------------------------------------------

def _first_split_stat(group):
    sp = group.get("splits", [])
    return sp[0].get("stat", {}) if sp else {}


def fetch_pitcher(pid, season, savant_x, savant_c):
    if not pid:
        return None
    types = "season,sabermetrics,expectedStatistics,pitchArsenal,gameLog"
    url = (f"{STATS}/people/{pid}?hydrate=stats(group=[pitching],"
           f"type=[{types}],season={season})")
    data = get_json(url)
    if not data or not data.get("people"):
        return None
    person = data["people"][0]

    season_stat, saber, expected = {}, {}, {}
    arsenal_splits, gamelog_splits = [], []
    for grp in person.get("stats", []):
        name = grp.get("type", {}).get("displayName", "")
        if name == "season":
            season_stat = _first_split_stat(grp)
        elif name == "sabermetrics":
            saber = _first_split_stat(grp)
        elif name == "expectedStatistics":
            expected = _first_split_stat(grp)
        elif name == "pitchArsenal":
            arsenal_splits = grp.get("splits", [])
        elif name == "gameLog":
            gamelog_splits = grp.get("splits", [])

    def g(d, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""):
                return sig3(v)
        return "—"

    # Pitch arsenal: sort by usage, keep top 5
    arsenal = []
    for s in sorted(arsenal_splits,
                    key=lambda x: x.get("stat", {}).get("percentage", 0),
                    reverse=True)[:5]:
        st = s.get("stat", {})
        desc = st.get("type", {}).get("description") or st.get("type", {}).get("code", "?")
        arsenal.append({
            "name": desc,
            "usage": sig3((st.get("percentage") or 0) * 100),
            "velo": sig3(st.get("averageSpeed")),
        })

    # Recent form: last 3 outings from the game log
    recent = []
    for s in gamelog_splits[-3:][::-1]:
        st = s.get("stat", {})
        opp = s.get("opponent", {}).get("name", "")
        recent.append({
            "date": s.get("date", ""),
            "opp": opp,
            "ip": sig3(st.get("inningsPitched")),
            "er": sig3(st.get("earnedRuns")),
            "k": sig3(st.get("strikeOuts")),
            "bb": sig3(st.get("baseOnBalls")),
            "h": sig3(st.get("hits")),
        })

    sx = savant_x.get(str(pid), {})
    sc = savant_c.get(str(pid), {})

    return {
        "name": person.get("fullName", "TBD"),
        "hand": person.get("pitchHand", {}).get("code", ""),
        "number": person.get("primaryNumber", ""),
        # --- season standard ---
        "w": g(season_stat, "wins"), "l": g(season_stat, "losses"),
        "era": g(season_stat, "era"), "g": g(season_stat, "gamesPlayed"),
        "gs": g(season_stat, "gamesStarted"), "ip": g(season_stat, "inningsPitched"),
        "so": g(season_stat, "strikeOuts"), "bb": g(season_stat, "baseOnBalls"),
        "whip": g(season_stat, "whip"), "hr": g(season_stat, "homeRuns"),
        "k9": g(season_stat, "strikeoutsPer9Inn"), "bb9": g(season_stat, "walksPer9Inn"),
        "kbb": g(season_stat, "strikeoutWalkRatio"), "h9": g(season_stat, "hitsPer9Inn"),
        "avg": g(season_stat, "avg"), "bf": g(season_stat, "battersFaced"),
        "strike_pct": g(season_stat, "strikePercentage"),
        "go_ao": g(season_stat, "groundOutsToAirouts"),
        "cg": g(season_stat, "completeGames"), "sho": g(season_stat, "shutouts"),
        "sv": g(season_stat, "saves"), "hld": g(season_stat, "holds"),
        "hr9": g(season_stat, "homeRunsPer9"), "whip_": g(season_stat, "whip"),
        # --- sabermetrics ---
        "fip": g(saber, "fip"), "xfip": g(saber, "xfip"),
        "fip_minus": g(saber, "fipMinus"), "era_minus": g(saber, "eraMinus"),
        "war": g(saber, "war"), "ra9war": g(saber, "ra9War"), "rar": g(saber, "rar"),
        # --- expected (StatsAPI) ---
        "xwoba_api": g(expected, "woba"), "xba_api": g(expected, "avg"),
        "xslg_api": g(expected, "slg"),
        # --- expected (Savant) ---
        "xera": sig3(sx.get("xera", "—")), "xba": sig3(sx.get("xba", "—")),
        "xslg": sig3(sx.get("xslg", "—")), "xwoba": sig3(sx.get("xwoba", "—")),
        # --- Statcast (Savant custom) ---
        "k_pct": sc.get("k_pct", "—"), "bb_pct": sc.get("bb_pct", "—"),
        "whiff_pct": sc.get("whiff_pct", "—"), "hardhit_pct": sc.get("hardhit_pct", "—"),
        "barrel_pct": sc.get("barrel_pct", "—"), "gb_pct": sc.get("gb_pct", "—"),
        "fb_velo": sig3(sc.get("fb_velo", "—")),
        # --- nested ---
        "arsenal": arsenal, "recent": recent,
    }


# ---------------------------------------------------------------------------
# StatsAPI: active rosters (to keep BvP to current hitters)
# ---------------------------------------------------------------------------

def fetch_roster(team_id):
    """Return (all_person_ids:set, pitcher_ids:list) for a team's active roster."""
    url = f"{STATS}/teams/{team_id}/roster?rosterType=active"
    data = get_json(url)
    ids, pitchers = set(), []
    if data:
        for p in data.get("roster", []):
            pid = p.get("person", {}).get("id")
            if pid is None:
                continue
            ids.add(pid)
            if p.get("position", {}).get("abbreviation") == "P":
                pitchers.append(pid)
    return ids, pitchers


# ---------------------------------------------------------------------------
# StatsAPI: batter-vs-pitcher (whole opposing team in one call)
# ---------------------------------------------------------------------------

def fetch_bvp(pitcher_id, opp_team_id, roster_ids):
    """
    Career batter-vs-pitcher lines for a pitcher against an opposing team.
    Aggregates each batter's splits into a single career line and keeps only
    batters currently on the opponent's active roster. Returns a list sorted
    by at-bats (most history first).
    """
    if not pitcher_id or not opp_team_id:
        return []
    url = (f"{STATS}/people/{pitcher_id}?hydrate=stats(group=[pitching],"
           f"type=[vsPlayer],opposingTeamId={opp_team_id})")
    data = get_json(url)
    if not data or not data.get("people"):
        return []

    agg = {}  # batter_id -> accumulated counting stats
    for grp in data["people"][0].get("stats", []):
        if grp.get("type", {}).get("displayName") != "vsPlayer":
            continue
        for s in grp.get("splits", []):
            b = s.get("batter", {})
            bid = b.get("id")
            if bid is None or (roster_ids and bid not in roster_ids):
                continue
            st = s.get("stat", {})

            def num(key):
                try:
                    return float(st.get(key, 0) or 0)
                except (TypeError, ValueError):
                    return 0.0

            a = agg.setdefault(bid, {
                "name": b.get("fullName", "?"), "ab": 0.0, "h": 0.0, "hr": 0.0,
                "bb": 0.0, "so": 0.0, "pa": 0.0, "tb": 0.0, "hbp": 0.0, "sf": 0.0,
            })
            a["ab"] += num("atBats")
            a["h"] += num("hits")
            a["hr"] += num("homeRuns")
            a["bb"] += num("baseOnBalls")
            a["so"] += num("strikeOuts")
            a["pa"] += num("plateAppearances")
            a["tb"] += num("totalBases")
            a["hbp"] += num("hitByPitch")
            a["sf"] += num("sacFlies")

    out = []
    for a in agg.values():
        ab = a["ab"]
        if ab + a["bb"] + a["hbp"] < 1:
            continue
        avg = a["h"] / ab if ab else 0
        obp_den = ab + a["bb"] + a["hbp"] + a["sf"]
        obp = (a["h"] + a["bb"] + a["hbp"]) / obp_den if obp_den else 0
        slg = a["tb"] / ab if ab else 0
        out.append({
            "name": a["name"],
            "pa": sig3(a["pa"] or (ab + a["bb"] + a["hbp"] + a["sf"])),
            "ab": sig3(ab), "h": sig3(a["h"]), "hr": sig3(a["hr"]),
            "so": sig3(a["so"]), "bb": sig3(a["bb"]),
            "avg": sig3(round(avg, 3)), "ops": sig3(round(obp + slg, 3)),
            "_ab": ab,
        })
    out.sort(key=lambda x: x["_ab"], reverse=True)
    return out[:10]


# ---------------------------------------------------------------------------
# Bullpen: bulk season/saber stats, recent usage (fatigue), assembly
# ---------------------------------------------------------------------------

def fetch_bulk_pitching(season):
    """One-shot season + sabermetric pitching lines for EVERY pitcher."""
    out = {}
    d = get_json(f"{STATS}/stats?stats=season&group=pitching&season={season}"
                 f"&sportId={SPORT_ID}&playerPool=all&limit=3000")
    if d and d.get("stats"):
        for s in d["stats"][0].get("splits", []):
            pid = s.get("player", {}).get("id")
            if pid is not None:
                out[pid] = dict(s.get("stat", {}))
                out[pid]["_name"] = s.get("player", {}).get("fullName", "")
    d = get_json(f"{STATS}/stats?stats=sabermetrics&group=pitching&season={season}"
                 f"&sportId={SPORT_ID}&playerPool=all&limit=3000")
    if d and d.get("stats"):
        for s in d["stats"][0].get("splits", []):
            pid = s.get("player", {}).get("id")
            if pid is not None:
                out.setdefault(pid, {})
                out[pid]["fip"] = s.get("stat", {}).get("fip")
                out[pid]["xfip"] = s.get("stat", {}).get("xfip")
    return out


def fetch_recent_usage(day_str, lookback=6):
    """
    Return {pid: {date: pitches}} of pitches thrown in the `lookback` days
    before `day_str`, harvested from box scores of completed games.
    """
    game_day = datetime.strptime(day_str, "%Y-%m-%d").date()
    start = (game_day - timedelta(days=lookback)).isoformat()
    end = (game_day - timedelta(days=1)).isoformat()
    sched = get_json(f"{STATS}/schedule?sportId={SPORT_ID}"
                     f"&startDate={start}&endDate={end}")
    pk_dates = []
    if sched:
        for dd in sched.get("dates", []):
            for g in dd.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    pk_dates.append((g["gamePk"], dd["date"]))

    usage = {}

    def load(pk_date):
        pk, dstr = pk_date
        b = get_json(f"{STATS}/game/{pk}/boxscore")
        rows = []
        if not b:
            return rows
        for side in ("home", "away"):
            tm = b.get("teams", {}).get(side, {})
            for pid in tm.get("pitchers", []):
                pl = tm.get("players", {}).get(f"ID{pid}", {})
                st = pl.get("stats", {}).get("pitching", {})
                np = st.get("numberOfPitches")
                if np:
                    rows.append((pid, dstr, float(np)))
        return rows

    if pk_dates:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for rows in ex.map(load, pk_dates):
                for pid, dstr, np in rows:
                    d = datetime.strptime(dstr, "%Y-%m-%d").date()
                    usage.setdefault(pid, {})
                    usage[pid][d] = usage[pid].get(d, 0) + np
    return usage


def fatigue_for(byday, game_day):
    """Turn a {date: pitches} history into fatigue status + workload metrics.

    Exposes the raw availability inputs the model needs: pitches 1/2/3 days ago,
    appearances in the last 3 days, and days since last appearance.
    """
    if not byday:
        base = {"dot": "🟢", "label": "Fresh (no recent app)", "rest": "5+",
                "p1": 0, "p2": 0, "p3": 0, "app3": 0, "rest_n": None, "rank": 0}
        base["avail"] = model.availability_score(0, 0, 0, 0, None)
        return base
    last = max(byday)
    days_rest = (game_day - last).days
    d1, d2, d3 = (game_day - timedelta(days=n) for n in (1, 2, 3))
    p1 = byday.get(d1, 0)
    p2 = byday.get(d2, 0)
    p3d = byday.get(d3, 0)
    p3 = p1 + p2 + p3d
    app3 = sum(1 for d in (d1, d2, d3) if d in byday)
    b2b = (d1 in byday) and (d2 in byday)

    if days_rest <= 0:
        dot, label, rank = "🔴", "Threw today", 4
    elif app3 >= 3:
        dot, label, rank = "🔴", "3 apps in 3 days", 4
    elif b2b:
        dot, label, rank = "🟠", "Back-to-back", 3
    elif days_rest == 1 and p1 >= 35:
        dot, label, rank = "🟠", "35+ P yesterday", 3
    elif days_rest == 1:
        dot, label, rank = "🟡", "Threw yesterday", 2
    else:
        dot, label, rank = "🟢", f"{days_rest}d rest", 1

    return {"dot": dot, "label": label, "rest": str(days_rest),
            "p1": p1, "p2": p2, "p3": p3, "app3": app3, "rest_n": days_rest,
            "avail": model.availability_score(p1, p2, p3, app3, days_rest),
            "rank": rank}


def build_bullpen(team_id, exclude_pid, pitcher_ids, bulk, usage,
                  savant_c, savant_x, game_day):
    """Assemble a sorted list of available relievers with stats + fatigue.

    Each reliever carries the raw inputs for the bullpen model:
      talent  <- FIP, xFIP, xERA, K%, BB%
      avail   <- pitches 1/2/3 days ago, appearances 3d, days rest
      usage   <- role-based expected share of relief innings
    """
    pen = []
    for pid in pitcher_ids:
        if pid == exclude_pid:
            continue
        st = bulk.get(pid)
        if not st:
            continue  # no season data (unproven call-up) — nothing to show
        gp = float(st.get("gamesPitched", 0) or 0)
        gs = float(st.get("gamesStarted", 0) or 0)
        if gp > 0 and gs / gp >= 0.5:
            continue  # rotation starter, not a bullpen arm
        sv = float(st.get("saves", 0) or 0)
        hld = float(st.get("holds", 0) or 0)
        role = "CL" if sv >= 8 else "SU" if hld >= 8 else "SW" if gs >= 3 else "RP"
        fat = fatigue_for(usage.get(pid, {}), game_day)
        sc = savant_c.get(str(pid), {})
        sx = savant_x.get(str(pid), {})
        pen.append({
            "pid": pid,
            "name": st.get("_name") or "—", "role": role,
            "era": sig3(st.get("era")), "whip": sig3(st.get("whip")),
            "k9": sig3(st.get("strikeoutsPer9Inn")), "ip": sig3(st.get("inningsPitched")),
            "sv": int(sv), "hld": int(hld), "leverage": sv + hld,
            "fip": sig3(st.get("fip")), "xfip": sig3(st.get("xfip")),
            "xera": sig3(sx.get("xera")),
            "k_pct": sc.get("k_pct", "—"), "bb_pct": sc.get("bb_pct", "—"),
            "whiff_pct": sc.get("whiff_pct", "—"),
            "usage": model.usage_weight(role),
            "fat": fat,
        })
    # Most-available arms first, then higher expected usage.
    pen.sort(key=lambda x: (-x["fat"]["avail"], -x["usage"]))
    return pen


# ---------------------------------------------------------------------------
# Prediction model adapter — builds features and calls model.predict()
# ---------------------------------------------------------------------------
#
# The math lives in model.py (shared with backtest.py). Here we translate the
# page's data feeds into the feature dict the model expects:
#   - handedness-adjusted offense (team wOBA vs the opposing starter's hand)
#   - starter true talent (ERA/FIP/xFIP/xERA blend) + projected innings
#   - bullpen quality x availability (fatigue-weighted true talent)
#   - season run environment as a grounding fallback
# Park factor and the negative-binomial run distribution are applied in model.

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _season_run_env(rec):
    """(runs/game, runs-allowed/game) from a standings record, or (None, None)."""
    try:
        w, l = (int(x) for x in str(rec.get("overall", "")).split("-"))
        g = w + l
        if g:
            return (_f(rec.get("rs")) or 0) / g, (_f(rec.get("ra")) or 0) / g
    except (ValueError, AttributeError):
        pass
    return None, None


def build_features(rec, pitcher, sp_pid, pen, hand_split, opp_hand, lg,
                   team_id, recency):
    """Assemble one team's feature dict for model.predict().

    `recency` = {"pit": {pid:{era,fip,ip}} last 30d, "off": {tid: factor}}.
    Starter and reliever talent blend season with last-30-day form (regressed);
    offense wOBA is nudged by the team's recent-form multiplier.
    """
    lg_era = lg["ERA"]
    opp_hand = opp_hand if opp_hand in ("R", "L") else "R"
    rpit = (recency or {}).get("pit", {})
    K_IP = 40

    off = (hand_split or {}).get(opp_hand, {})
    base_woba = off.get("woba")
    lg_off_woba = (lg.get("woba") or {}).get(opp_hand)

    # Offense multiplier = 60% general (rolling 7/15/30 + season) + 40% hand.
    general_mult = (recency or {}).get("general", {}).get(team_id, 1.0)
    if base_woba and lg_off_woba:
        hand_mult = base_woba / lg_off_woba
        off_mult = (model.GENERAL_OFFENSE_WEIGHT * general_mult
                    + model.HAND_OFFENSE_WEIGHT * hand_mult)
    else:
        off_mult = general_mult

    rspg, rapg = _season_run_env(rec)

    if pitcher:
        season_t = model.pitcher_true_talent(pitcher["era"], pitcher["fip"],
                                             pitcher["xfip"], pitcher["xera"], lg_era)
        rp = rpit.get(sp_pid)
        recent_t = (model.pitcher_true_talent(rp["era"], rp["fip"], None, None, lg_era)
                    if rp else None)
        sp_ra = model.recency_blend(season_t, model.ip_to_float(pitcher["ip"]),
                                    recent_t, rp["ip"] if rp else 0, lg_era, K_IP)
        proj_ip = model._project_ip(pitcher["ip"], pitcher["gs"], model.DEFAULT_CFG)
    else:
        sp_ra, proj_ip = lg_era + 0.25, 4.5

    # Bullpen: season+L30 talent x availability x expected usage.
    arms = []
    for r in (pen or []):
        season_t = model.reliever_true_talent(r["fip"], r["xfip"], r["xera"],
                                              r["k_pct"], r["bb_pct"], lg_era)
        rp = rpit.get(r.get("pid"))
        recent_t = (model.pitcher_true_talent(rp["era"], rp["fip"], None, None, lg_era)
                    if rp else None)
        talent = model.recency_blend(season_t, model.ip_to_float(r["ip"]),
                                     recent_t, rp["ip"] if rp else 0, lg_era, K_IP)
        arms.append({"talent": talent, "avail": r["fat"]["avail"],
                     "usage": r.get("usage", 1.0)})
    pen_ra = model.bullpen_run_prevention(arms, lg_era)

    return {"off_mult": off_mult, "off_woba": base_woba,
            "lg_off_woba": lg_off_woba, "off_fallback": off.get("fallback", False),
            "sp_ra": sp_ra, "proj_ip": proj_ip, "pen_ra": pen_ra,
            "rspg": rspg, "rapg": rapg, "off_ops": off.get("ops")}


def compute_league(records, team_stats):
    """League runs/game and ERA baselines from all teams' season data."""
    rspg = []
    for r in records.values():
        rs_pg, _ = _season_run_env(r)
        if rs_pg:
            rspg.append(rs_pg)
    eras = [v for v in (_f(t.get("team_era")) for t in team_stats.values())
            if v is not None]
    lg_r = sum(rspg) / len(rspg) if rspg else 4.4
    lg_era = sum(eras) / len(eras) if eras else 4.15
    return {"R": lg_r, "ERA": lg_era}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def stat_grid(title, pairs, cls=""):
    cells = "".join(
        f'<div class="cell"><span class="k">{esc(k)}</span>'
        f'<span class="v">{esc(v)}</span></div>'
        for k, v in pairs
    )
    head = f'<div class="grid-title {cls}">{esc(title)}</div>' if title else ""
    return f'{head}<div class="grid">{cells}</div>'


def arsenal_html(arsenal):
    if not arsenal:
        return ""
    rows = "".join(
        f"<tr><td>{esc(a['name'])}</td><td>{esc(a['usage'])}%</td>"
        f"<td>{esc(a['velo'])} mph</td></tr>"
        for a in arsenal
    )
    return f"""
      <details class="extra">
        <summary>Pitch arsenal ({len(arsenal)})</summary>
        <table class="mini"><tr><th>Pitch</th><th>Usage</th><th>Velo</th></tr>{rows}</table>
      </details>"""


def recent_html(recent):
    if not recent:
        return ""
    rows = "".join(
        f"<tr><td>{esc(r['date'])}</td><td>{esc(r['opp'])}</td>"
        f"<td>{esc(r['ip'])}</td><td>{esc(r['h'])}</td><td>{esc(r['er'])}</td>"
        f"<td>{esc(r['k'])}</td><td>{esc(r['bb'])}</td></tr>"
        for r in recent
    )
    return f"""
      <details class="extra">
        <summary>Recent form (last {len(recent)})</summary>
        <table class="mini">
          <tr><th>Date</th><th>Opp</th><th>IP</th><th>H</th><th>ER</th><th>K</th><th>BB</th></tr>
          {rows}
        </table>
      </details>"""


def bvp_html(bvp, opp_name):
    if not bvp:
        return ""
    rows = "".join(
        f"<tr><td class='bn'>{esc(b['name'])}</td><td>{esc(b['pa'])}</td>"
        f"<td>{esc(b['ab'])}</td><td>{esc(b['h'])}</td><td>{esc(b['hr'])}</td>"
        f"<td>{esc(b['so'])}</td><td>{esc(b['bb'])}</td>"
        f"<td>{esc(b['avg'])}</td><td>{esc(b['ops'])}</td></tr>"
        for b in bvp
    )
    return f"""
      <details class="extra bvp">
        <summary>Batter vs Pitcher — vs {esc(opp_name)} ({len(bvp)})</summary>
        <table class="mini">
          <tr><th>Batter</th><th>PA</th><th>AB</th><th>H</th><th>HR</th><th>SO</th><th>BB</th><th>AVG</th><th>OPS</th></tr>
          {rows}
        </table>
      </details>"""


def pitcher_card(p, bvp=None, opp_name=""):
    if p is None:
        return '<div class="pitcher"><div class="p-name">Probable pitcher TBD</div></div>'
    hand = f" ({esc(p['hand'])}HP)" if p["hand"] else ""
    num = f"#{esc(p['number'])}" if p["number"] else ""
    return f"""
    <div class="pitcher">
      <div class="p-head">
        <span class="p-name">{esc(p['name'])}{hand}</span><span class="p-num">{num}</span>
      </div>
      <div class="p-line">{esc(p['w'])}-{esc(p['l'])} · {esc(p['era'])} ERA · {esc(p['ip'])} IP · {esc(p['so'])} K · {esc(p['whip'])} WHIP</div>

      {stat_grid("Standard", [
        ("ERA", p['era']), ("WHIP", p['whip']), ("K/9", p['k9']), ("BB/9", p['bb9']),
        ("K/BB", p['kbb']), ("H/9", p['h9']), ("HR/9", p['hr9']), ("AVG", p['avg']),
        ("GS", p['gs']), ("BF", p['bf']), ("Str%", p['strike_pct']), ("GO/AO", p['go_ao']),
      ])}

      {stat_grid("Advanced (sabermetric)", [
        ("FIP", p['fip']), ("xFIP", p['xfip']), ("ERA-", p['era_minus']),
        ("FIP-", p['fip_minus']), ("WAR", p['war']), ("RA9-WAR", p['ra9war']),
        ("RAR", p['rar']),
      ], cls="adv")}

      {stat_grid("Expected / Statcast", [
        ("xERA", p['xera']), ("xBA", p['xba']), ("xSLG", p['xslg']), ("xwOBA", p['xwoba']),
        ("K%", p['k_pct']), ("BB%", p['bb_pct']), ("Whiff%", p['whiff_pct']),
        ("Barrel%", p['barrel_pct']), ("HardHit%", p['hardhit_pct']),
        ("GB%", p['gb_pct']), ("FB velo", p['fb_velo']),
      ], cls="stc")}

      {arsenal_html(p['arsenal'])}
      {recent_html(p['recent'])}
      {bvp_html(bvp or [], opp_name)}
    </div>"""


def team_block(label, name, rec, ts, team_id):
    rec = rec or {}
    ts = ts or {}
    def r(k, d="—"):
        return esc(rec.get(k, d))
    def t(k, d="—"):
        return esc(sig3(ts.get(k, d)))
    logo = (f'<img class="logo" src="https://www.mlbstatic.com/team-logos/{team_id}.svg" '
            f'alt="{esc(name)} logo" loading="lazy" '
            f'onerror="this.style.display=\'none\'">') if team_id else ""
    return f"""
      <div class="team">
        <div class="t-label">{label}</div>
        {logo}
        <div class="t-name">{esc(name)}</div>
        <div class="t-rec">
          <span><b>{r('overall')}</b> ({r('pct')})</span>
          <span>{r('home')} home · {r('away')} away</span>
          <span>L10 {r('l10')} · streak {r('streak')}</span>
          <span>Run diff {r('run_diff')} · GB {r('gb')}</span>
          <span>vs L {r('vs_left')} · vs R {r('vs_right')}</span>
          <span>Pythag {r('expected')}</span>
        </div>
        <div class="t-team">
          <span>OPS {t('ops')}</span><span>Runs {t('runs')}</span><span>HR {t('hr')}</span>
          <span>Team ERA {t('team_era')}</span><span>WHIP {t('team_whip')}</span>
        </div>
      </div>"""


def bullpen_html(pen, team_name):
    if not pen:
        return ('<div class="pen"><div class="col-label">Bullpen</div>'
                '<div class="pen-empty">No bullpen data.</div></div>')
    counts = {"🟢": 0, "🟡": 0, "🟠": 0, "🔴": 0}
    for r in pen:
        counts[r["fat"]["dot"]] = counts.get(r["fat"]["dot"], 0) + 1
    summary = (f'{counts["🟢"]}🟢 rested · {counts["🟡"]}🟡 threw yest · '
               f'{counts["🟠"]}🟠 limited · {counts["🔴"]}🔴 down')
    rows = "".join(
        f"<tr><td class='bn'>{r['fat']['dot']} {esc(r['name'])}"
        f"<span class='role'>{esc(r['role'])}</span></td>"
        f"<td class='st'>{esc(r['fat']['label'])}</td>"
        f"<td>{round(r['fat']['avail']*100)}%</td>"
        f"<td>{esc(r['fat']['p3'])}</td><td>{esc(r['fat']['app3'])}</td>"
        f"<td>{esc(r['era'])}</td><td>{esc(r['fip'])}</td><td>{esc(r['xfip'])}</td>"
        f"<td>{esc(r['xera'])}</td><td>{esc(r['k_pct'])}</td><td>{esc(r['bb_pct'])}</td></tr>"
        for r in pen
    )
    return f"""
    <details class="pen" open>
      <summary><span class="col-label">Bullpen — availability &amp; fatigue ({len(pen)})</span></summary>
      <div class="pen-sum">{summary}</div>
      <div class="pen-scroll">
      <table class="mini">
        <tr><th>Reliever</th><th>Status</th><th>Avail</th><th>P·3d</th><th>App·3d</th>
            <th>ERA</th><th>FIP</th><th>xFIP</th><th>xERA</th><th>K%</th><th>BB%</th></tr>
        {rows}
      </table>
      </div>
    </details>"""


def prediction_html(pred, drv, away_name, home_name):
    if not pred:
        return ""
    fav = home_name if pred["fav_home"] else away_name
    fav_pct = max(pred["p_home"], pred["p_away"])
    pf = pred["park_factor"]
    pf_txt = f"{pf} ({'hitter' if pf > 102 else 'pitcher' if pf < 98 else 'neutral'})"
    o = pred.get("odds")
    odds_html = ""
    if o:
        side = home_name if o["pick_home"] else away_name
        dec = o.get("pick_dec", "—")
        badge = (f'<span class="val-yes">✓ VALUE +{o["edge_pct"]}%</span>'
                 if o["value"] else
                 f'<span class="val-no">no edge ({o["edge_pct"]}%)</span>')
        odds_html = f"""
      <div class="pred-odds">
        Market (decimal): {esc(away_name)} {esc(o['away_dec'])} / {esc(o['home_dec'])} {esc(home_name)}
        &nbsp;·&nbsp; model {o['model_home_pct']}% vs market {o['market_home_pct']}% (home)
        &nbsp;·&nbsp; pick {esc(side)} @ {esc(dec)} {badge}
        <br><b>Need ≥ {esc(o['min_dec'])}</b> (min decimal odds for value) — shop books above this.
        <span class="pred-note"> · {o['books']} books · analysis only, not betting advice</span>
      </div>"""
    return f"""
    <div class="predict">
      <div class="pred-head">🔮 Model projection
        <span class="pred-conf">{esc(pred['conf'])}: {esc(fav)} {fav_pct}%</span></div>
      <div class="pred-bar">
        <div class="pb-away" style="width:{pred['p_away']}%">{esc(away_name)} {pred['p_away']}%</div>
        <div class="pb-home" style="width:{pred['p_home']}%">{pred['p_home']}% {esc(home_name)}</div>
      </div>
      <div class="pred-nums">Projected runs: {esc(away_name)} <b>{pred['e_away']}</b> &ndash; <b>{pred['e_home']}</b> {esc(home_name)}
        &nbsp;·&nbsp; total <b>{pred['exp_total']}</b> &nbsp;·&nbsp; park {esc(pf_txt)}</div>
      <div class="pred-drivers">
        Offense wRC+ (60% form / 40% vs hand): {esc(away_name)} <b>{esc(drv['away_wrc'])}</b>{esc(drv['away_fb'])} vs <b>{esc(drv['home_wrc'])}</b>{esc(drv['home_fb'])} {esc(home_name)}
        &nbsp;·&nbsp; wOBA vs {esc(drv['away_hand'])}HP/{esc(drv['home_hand'])}HP {esc(drv['away_woba'])}/{esc(drv['home_woba'])}
        <br>Starter R/9: {esc(drv['away_sp'])} vs {esc(drv['home_sp'])} (~{esc(drv['away_ip'])}/{esc(drv['home_ip'])} IP)
        &nbsp;·&nbsp; Pen R/9: {esc(drv['away_pen'])} vs {esc(drv['home_pen'])}
        <span class="fb-note">{" ⚑ = split fell back to prior season (<150 PA)" if (drv['home_fb'] or drv['away_fb']) else ""}</span>
      </div>
      {odds_html}
      <div class="pred-note">Negative-binomial run model — recency-weighted handedness offense, starter+bullpen (quality×availability), park factor. Analysis only, not betting advice.</div>
    </div>"""


def game_card(game, records, team_stats, pitchers, bvp_map, bullpens, league,
              hand_splits, recency, odds_map):
    teams = game.get("teams", {})
    home_t = teams.get("home", {}).get("team", {})
    away_t = teams.get("away", {}).get("team", {})
    home_id, away_id = home_t.get("id"), away_t.get("id")
    home_name, away_name = home_t.get("name", "TBD"), away_t.get("name", "TBD")

    def rec_for(side, tid):
        r = records.get(tid)
        if r:
            return r
        lr = teams.get(side, {}).get("leagueRecord", {})
        if lr:
            return {"overall": f"{lr.get('wins', 0)}-{lr.get('losses', 0)}"}
        return {}

    home_pp = teams.get("home", {}).get("probablePitcher", {}) or {}
    away_pp = teams.get("away", {}).get("probablePitcher", {}) or {}
    home_pid, away_pid = home_pp.get("id"), away_pp.get("id")
    home_pitcher = pitchers.get(home_pid)
    away_pitcher = pitchers.get(away_pid)

    try:
        dt = datetime.fromisoformat(game.get("gameDate", "").replace("Z", "+00:00"))
        gametime = dt.astimezone().strftime("%-I:%M %p %Z")
    except (ValueError, TypeError):
        gametime = ""

    venue = game.get("venue", {}).get("name", "")
    status = game.get("status", {}).get("detailedState", "")
    weather = game.get("weather", {})
    wx = ""
    if weather.get("temp"):
        wx = f" · {esc(weather.get('temp'))}°F {esc(weather.get('condition', ''))}"

    # Away pitcher faces the home lineup; home pitcher faces the away lineup.
    away_bvp = bvp_map.get(away_pid, [])
    home_bvp = bvp_map.get(home_pid, [])
    away_pen = bullpens.get(away_id, [])
    home_pen = bullpens.get(home_id, [])

    # Prediction: handedness offense vs opposing starter, starter+pen defense.
    home_hand = (home_pitcher or {}).get("hand") or "R"   # home SP hand
    away_hand = (away_pitcher or {}).get("hand") or "R"   # away SP hand
    # Home offense faces the AWAY starter's hand; away offense faces HOME's.
    home_feat = build_features(rec_for("home", home_id), home_pitcher, home_pid,
                               home_pen, hand_splits.get(home_id), away_hand,
                               league, home_id, recency)
    away_feat = build_features(rec_for("away", away_id), away_pitcher, away_pid,
                               away_pen, hand_splits.get(away_id), home_hand,
                               league, away_id, recency)
    ctx = {"lg_r": league["R"], "park_factor": model.park_factor(home_id)}
    pred = model.predict(home_feat, away_feat, ctx)
    pred["odds"] = odds.evaluate(pred["p_home_raw"],
                                 odds_map.get((home_id, away_id))) if odds_map else None

    def _woba_txt(feat):
        w = feat.get("off_woba")
        return f"{w:.3f}".lstrip("0") if isinstance(w, float) else "—"

    def _wrc(feat):
        m = feat.get("off_mult")
        return round(m * 100) if m else "—"

    drv = {
        "home_hand": esc(home_hand), "away_hand": esc(away_hand),
        "home_woba": _woba_txt(home_feat), "away_woba": _woba_txt(away_feat),
        "home_wrc": _wrc(home_feat), "away_wrc": _wrc(away_feat),
        "home_fb": " ⚑" if home_feat.get("off_fallback") else "",
        "away_fb": " ⚑" if away_feat.get("off_fallback") else "",
        "home_sp": round(home_feat["sp_ra"], 2), "away_sp": round(away_feat["sp_ra"], 2),
        "home_pen": round(home_feat["pen_ra"], 2), "away_pen": round(away_feat["pen_ra"], 2),
        "home_ip": round(home_feat["proj_ip"], 1), "away_ip": round(away_feat["proj_ip"], 1),
    }

    # Pick metadata for the "Top plays" ranking.
    gid = game.get("gamePk", id(game))
    o = pred.get("odds")
    meta = {
        "gid": gid, "away": away_name, "home": home_name,
        "pick": home_name if pred["fav_home"] else away_name,
        "pick_pct": max(pred["p_home"], pred["p_away"]),
        "conf": abs(pred["p_home_raw"] - 0.5),
        "gametime": gametime,
        "edge": (o["edge_pct"] if o else None),
        "has_value": bool(o and o["value"]),
        "market_dec": (o.get("pick_dec") if o else None),
        "min_dec": (o.get("min_dec") if o else None),
    }

    card = f"""
  <div class="game" id="game-{gid}">
    <div class="matchup">
      {team_block("AWAY", away_name, rec_for('away', away_id), team_stats.get(away_id), away_id)}
      <div class="at">@</div>
      {team_block("HOME", home_name, rec_for('home', home_id), team_stats.get(home_id), home_id)}
    </div>
    <div class="meta">{esc(gametime)} · {esc(venue)} · {esc(status)}{wx}</div>
    {prediction_html(pred, drv, away_name, home_name)}
    <div class="pitchers">
      <div class="pitcher-col"><div class="col-label">Away probable</div>{pitcher_card(away_pitcher, away_bvp, home_name)}</div>
      <div class="pitcher-col"><div class="col-label">Home probable</div>{pitcher_card(home_pitcher, home_bvp, away_name)}</div>
    </div>
    <div class="pens">
      <div class="pen-col"><div class="pen-team">{esc(away_name)} pen</div>{bullpen_html(away_pen, away_name)}</div>
      <div class="pen-col"><div class="pen-team">{esc(home_name)} pen</div>{bullpen_html(home_pen, home_name)}</div>
    </div>
  </div>"""
    return card, meta


PAGE_CSS = """
:root{--bg:#0d1117;--card:#161b22;--card2:#1c2330;--line:#2a3241;--text:#e6edf3;
--muted:#8b949e;--accent:#4c9aff;--accent2:#f2a900;--good:#3fb950;--stc:#db6d9d;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
line-height:1.4;padding:24px;}
h1{font-size:26px;margin:0 0 4px;}
.sub{color:var(--muted);margin-bottom:24px;}
.legend{color:var(--muted);font-size:12px;margin-bottom:20px;max-width:1100px;}
.games{display:grid;gap:22px;grid-template-columns:1fr;max-width:1100px;margin:0 auto;}
.game{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:18px;box-shadow:0 2px 12px rgba(0,0,0,.35);}
.matchup{display:grid;grid-template-columns:1fr auto 1fr;align-items:start;gap:12px;}
.at{color:var(--muted);font-weight:700;font-size:18px;padding-top:20px;}
.team{text-align:center;}
.t-label{font-size:11px;letter-spacing:.08em;color:var(--muted);}
.logo{width:52px;height:52px;object-fit:contain;margin:2px auto 0;display:block;}
.t-name{font-size:18px;font-weight:700;margin:2px 0 6px;}
.t-rec{font-size:11px;color:var(--muted);display:flex;flex-direction:column;gap:1px;}
.t-rec b{color:var(--text);}
.t-team{font-size:11px;color:var(--accent);display:flex;flex-wrap:wrap;gap:8px;
justify-content:center;margin-top:6px;}
.meta{text-align:center;color:var(--muted);font-size:12px;margin:12px 0 14px;
border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:8px 0;}
.pitchers{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.col-label{font-size:11px;letter-spacing:.06em;color:var(--accent);margin-bottom:6px;text-transform:uppercase;}
.pitcher{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:11px;}
.p-head{display:flex;justify-content:space-between;align-items:baseline;}
.p-name{font-weight:700;font-size:14px;}
.p-num{color:var(--muted);font-size:12px;}
.p-line{color:var(--accent2);font-size:12px;margin:5px 0 10px;}
.grid-title{font-size:10px;letter-spacing:.06em;color:var(--muted);
text-transform:uppercase;margin:9px 0 4px;}
.grid-title.adv{color:var(--good);}
.grid-title.stc{color:var(--stc);}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;}
.cell{background:rgba(255,255,255,.03);border-radius:6px;padding:4px 3px;text-align:center;}
.cell .k{display:block;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;}
.cell .v{display:block;font-size:13px;font-weight:600;}
details.extra{margin-top:8px;font-size:12px;}
details.extra summary{cursor:pointer;color:var(--accent);font-size:11px;}
table.mini{width:100%;border-collapse:collapse;font-size:11px;margin-top:6px;}
table.mini th{color:var(--muted);font-weight:600;padding:2px 3px;border-bottom:1px solid var(--line);}
table.mini td{padding:3px;text-align:center;}
table.mini td.bn{text-align:left;white-space:nowrap;}
details.bvp summary{color:var(--stc);}
.pens{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;
border-top:1px solid var(--line);padding-top:12px;}
.pen-team{font-size:12px;font-weight:700;color:var(--text);margin-bottom:4px;}
details.pen summary{cursor:pointer;list-style:none;}
details.pen summary::-webkit-details-marker{display:none;}
.pen-sum{font-size:10px;color:var(--muted);margin:4px 0 6px;}
.pen-scroll{overflow-x:auto;}
.pen-empty{font-size:11px;color:var(--muted);}
.predict{background:linear-gradient(180deg,rgba(76,154,255,.07),rgba(76,154,255,.02));
border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-bottom:14px;}
.pred-head{font-size:13px;font-weight:700;margin-bottom:8px;display:flex;
justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;}
.pred-conf{font-size:11px;font-weight:600;color:var(--accent2);
background:rgba(242,169,0,.12);border-radius:6px;padding:2px 8px;}
.pred-bar{display:flex;height:26px;border-radius:6px;overflow:hidden;font-size:11px;font-weight:700;}
.pb-away{background:var(--accent);color:#04203f;display:flex;align-items:center;
padding:0 8px;white-space:nowrap;min-width:0;overflow:hidden;}
.pb-home{background:var(--accent2);color:#3a2600;display:flex;align-items:center;
justify-content:flex-end;padding:0 8px;white-space:nowrap;min-width:0;overflow:hidden;}
.pred-nums{font-size:12px;margin-top:8px;}
.pred-drivers{font-size:11px;color:var(--muted);margin-top:4px;}
.pred-note{font-size:9px;color:var(--muted);margin-top:6px;font-style:italic;}
.pred-odds{font-size:11px;margin-top:7px;padding-top:7px;border-top:1px dashed var(--line);}
.val-yes{color:var(--good);font-weight:700;background:rgba(63,185,80,.12);border-radius:5px;padding:1px 6px;}
.val-no{color:var(--muted);}
.fb-note{color:var(--accent2);font-size:10px;}
.top{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:14px 16px;margin:0 auto 22px;max-width:1100px;}
.top h2{font-size:14px;margin:0 0 4px;letter-spacing:.02em;}
.top .tsub{color:var(--muted);font-size:11px;margin-bottom:10px;}
.top-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.play{background:var(--card2);border:1px solid var(--line);border-radius:9px;
padding:10px 12px;text-decoration:none;color:var(--text);display:block;}
.play:hover{border-color:var(--accent);}
.play .rank{font-size:10px;color:var(--muted);}
.play .pick{font-size:15px;font-weight:700;margin:2px 0;}
.play .why{font-size:11px;color:var(--muted);}
.play .tag{display:inline-block;font-size:10px;font-weight:700;border-radius:5px;
padding:1px 7px;margin-top:6px;}
.tag.val{background:rgba(63,185,80,.15);color:var(--good);}
.tag.conf{background:rgba(76,154,255,.13);color:var(--accent);}
@media(max-width:640px){.top-grid{grid-template-columns:1fr;}}
.ranklist{max-width:1100px;margin:0 auto 22px;background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:8px 16px 14px;}
.ranklist summary{cursor:pointer;font-size:13px;font-weight:700;padding:6px 0;}
.rank-toggle{display:flex;gap:8px;margin:6px 0 10px;}
.rank-toggle button{background:var(--card2);color:var(--muted);border:1px solid var(--line);
border-radius:7px;padding:5px 12px;font-size:12px;font-weight:600;cursor:pointer;}
.rank-toggle button.active{color:var(--bg);background:var(--accent);border-color:var(--accent);}
.rank-scroll{overflow-x:auto;}
table.ranktable{width:100%;border-collapse:collapse;font-size:12px;}
table.ranktable th{color:var(--muted);font-weight:600;text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);}
table.ranktable td{padding:5px 8px;border-bottom:1px solid rgba(255,255,255,.03);}
table.ranktable tr{cursor:pointer;}
table.ranktable tr:hover td{background:rgba(76,154,255,.06);}
.ranktable .r-num{color:var(--muted);width:38px;}
.ranktable .r-pick{font-weight:700;white-space:nowrap;}
.ranktable .r-match{color:var(--muted);white-space:nowrap;}
.ranktable .r-val{color:var(--good);font-weight:700;}
.role{display:inline-block;font-size:8px;font-weight:700;color:var(--bg);
background:var(--accent);border-radius:4px;padding:1px 4px;margin-left:5px;vertical-align:middle;}
td.st{color:var(--muted);white-space:nowrap;font-size:10px;}
.empty{color:var(--muted);text-align:center;padding:40px;}
footer{color:var(--muted);font-size:12px;margin-top:28px;text-align:center;}
.datebar{display:flex;gap:8px;align-items:center;margin:0 0 18px;flex-wrap:wrap;}
.datebar input,.datebar button{background:var(--card2);color:var(--text);
border:1px solid var(--line);border-radius:8px;padding:7px 12px;font-size:13px;}
.datebar button{cursor:pointer;font-weight:600;}
.datebar button:hover{border-color:var(--accent);color:var(--accent);}
.datebar input[type=date]{color-scheme:dark;}
.datebar .hint{color:var(--muted);font-size:11px;margin-left:6px;}
"""


def top_plays_html(metas, have_odds):
    """Rank the day's games and render a Top-3 banner.

    With odds: rank by market edge (value). Without: rank by model confidence.
    """
    if not metas:
        return ""
    heading = "⭐ Top 3 plays"
    if have_odds:
        # Value picks only: model edge over the market, biggest first.
        value = sorted([m for m in metas if m["edge"] is not None and m["has_value"]],
                       key=lambda m: m["edge"], reverse=True)
        ranked = value
        basis = "biggest edge vs the market (value picks only)"
        heading = "⭐ Top 3 value picks"
        if not value:
            return ('<div class="top" style="max-width:1100px"><h2>⭐ Top value picks</h2>'
                    '<div class="tsub">No positive-edge value picks on the board today '
                    '— the market and model agree. Analysis only, not betting advice.</div></div>')
    else:
        ranked = sorted(metas, key=lambda m: m["conf"], reverse=True)
        basis = "highest model confidence (no odds key set)"
    top = ranked[:3]
    if not top:
        return ""
    cards = ""
    for i, m in enumerate(top, 1):
        if have_odds and m["edge"] is not None:
            dec = m.get("market_dec") or "—"
            mind = m.get("min_dec") or "—"
            tag = (f'<span class="tag val">+{m["edge"]}% edge</span>'
                   if m["has_value"] else
                   f'<span class="tag conf">{m["edge"]}% vs mkt</span>')
            why = f"model {m['pick_pct']}% · now {dec} · need ≥{mind}"
        else:
            tag = f'<span class="tag conf">{m["pick_pct"]}% win prob</span>'
            why = f"{esc(m['away'])} @ {esc(m['home'])} · {esc(m['gametime'])}"
        cards += f"""
      <a class="play" href="#game-{m['gid']}">
        <div class="rank">#{i} pick</div>
        <div class="pick">{esc(m['pick'])}</div>
        <div class="why">{why}</div>
        {tag}
      </a>"""
    return f"""
  <div class="top">
    <h2>{esc(heading)}</h2>
    <div class="tsub">Ranked by {esc(basis)}. Analysis only, not betting advice.</div>
    <div class="top-grid">{cards}</div>
  </div>"""


def ranked_table_html(metas, have_odds):
    """Full sortable list of every game — toggle between confidence and edge."""
    if not metas:
        return ""
    rows = ""
    for m in metas:
        edge = m["edge"]
        edge_attr = edge if edge is not None else -999
        dec = m.get("market_dec")
        dec_txt = f"{dec}" if dec is not None else "—"
        edge_cell = (f"{edge:+.1f}%" if edge is not None else "—")
        mind = m.get("min_dec")
        min_txt = f"{mind}" if mind is not None else "—"
        odds_cells = (f'<td>{esc(dec_txt)}</td><td>{esc(min_txt)}</td>'
                      f'<td class="{"r-val" if m["has_value"] else ""}">{edge_cell}</td>'
                      if have_odds else "")
        rows += (f'<tr data-conf="{m["conf"]:.4f}" data-edge="{edge_attr}" '
                 f'onclick="location.hash=\'#game-{m["gid"]}\'">'
                 f'<td class="r-num"></td><td class="r-pick">{esc(m["pick"])}</td>'
                 f'<td class="r-match">{esc(m["away"])} @ {esc(m["home"])}</td>'
                 f'<td>{m["pick_pct"]}%</td>{odds_cells}</tr>')
    edge_hdr = "<th>Dec</th><th>Need≥</th><th>Edge</th>" if have_odds else ""
    edge_btn = ('<button id="btn-edge" onclick="sortPlays(\'edge\')">By edge</button>'
                if have_odds else "")
    default = "edge" if have_odds else "conf"
    return f"""
  <details class="ranklist" open>
    <summary>Full ranked list — all {len(metas)} games</summary>
    <div class="rank-toggle">
      <button id="btn-conf" onclick="sortPlays('conf')">By confidence</button>
      {edge_btn}
    </div>
    <div class="rank-scroll"><table class="ranktable">
      <thead><tr><th>#</th><th>Pick</th><th>Matchup</th><th>Win%</th>{edge_hdr}</tr></thead>
      <tbody id="rank-body">{rows}</tbody></table></div>
    <script>
      function sortPlays(key){{
        var tb=document.getElementById('rank-body');
        var rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));
        rows.sort(function(a,b){{return parseFloat(b.dataset[key])-parseFloat(a.dataset[key]);}});
        rows.forEach(function(r,i){{r.querySelector('.r-num').textContent='#'+(i+1);tb.appendChild(r);}});
        var bs=document.querySelectorAll('.rank-toggle button');
        for(var i=0;i<bs.length;i++)bs[i].classList.remove('active');
        var el=document.getElementById('btn-'+key); if(el) el.classList.add('active');
      }}
      sortPlays('{default}');
    </script>
  </details>"""


def build_html(games, records, team_stats, day, pitchers, bvp_map, bullpens,
               league, hand_splits, recency, odds_map):
    if not games:
        body = '<div class="empty">No MLB games scheduled for this date.</div>'
    else:
        cards, metas = [], []
        for g in games:
            card, meta = game_card(g, records, team_stats, pitchers, bvp_map,
                                   bullpens, league, hand_splits, recency, odds_map)
            cards.append(card)
            metas.append(meta)
        have_odds = bool(odds_map)
        top = top_plays_html(metas, have_odds)
        ranked = ranked_table_html(metas, have_odds)
        body = top + ranked + '<div class="games">\n' + "\n".join(cards) + "\n</div>"

    pretty = datetime.strptime(day, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
    legend = (
        "Sources: MLB StatsAPI (season, sabermetrics, expected stats, pitch arsenal, "
        "game logs, standings, team stats) + Baseball Savant (xERA & Statcast "
        "leaderboards). Advanced keys — FIP/xFIP: fielding-independent ERA; "
        "ERA-/FIP-: 100 = league avg, lower is better; WAR: wins above replacement; "
        "xERA/xwOBA: contact-quality expected stats; Whiff%/Barrel%/HardHit%: Statcast "
        "contact metrics. Bullpen fatigue is derived from box scores of the last 6 days — "
        "🟢 rested · 🟡 threw yesterday · 🟠 back-to-back / heavy · 🔴 3 apps in 3 days or threw today; "
        "P·3d = pitches over the last 3 days, App·3d = appearances. "
        "🔮 Model projection blends each team's run environment, starter (ERA/FIP/xFIP/xERA), "
        "and fatigue-weighted bullpen into expected runs, then a Pythagorean win probability "
        "with home-field — an analytical estimate for context only, not betting advice. "
        "All values rounded to 3 significant figures."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Schedule — {esc(pretty)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
  <h1>⚾ MLB Schedule &amp; Probable Pitchers</h1>
  <div class="sub">{esc(pretty)} · {len(games)} game(s) · multi-source advanced metrics</div>
  <div class="datebar">
    <button onclick="shift(-1)">◀ Prev</button>
    <input type="date" id="dpick" value="{esc(day)}" onchange="goDate(this.value)">
    <button onclick="goDate(todayStr())">Today</button>
    <button onclick="shift(1)">Next ▶</button>
    <button onclick="refresh()" title="Bypass cache and pull fresh data">⟳ Refresh</button>
    <span class="hint" id="mode"></span>
  </div>
  <script>
    const CUR = "{esc(day)}";
    function refresh(){{
      if(location.protocol === 'file:') location.reload();
      else location.href = '?date=' + CUR + '&refresh=1';
    }}
    function todayStr(){{
      const t = new Date(), o = t.getTimezoneOffset();
      return new Date(t.getTime() - o*60000).toISOString().slice(0,10);
    }}
    function goDate(d){{
      if(!d) return;
      if(location.protocol === 'file:') location.href = 'mlb_schedule_' + d + '.html';
      else location.href = '?date=' + d;
    }}
    function shift(n){{
      const p = CUR.split('-').map(Number);
      const dt = new Date(Date.UTC(p[0], p[1]-1, p[2]));
      dt.setUTCDate(dt.getUTCDate() + n);
      goDate(dt.toISOString().slice(0,10));
    }}
    document.getElementById('mode').textContent = (location.protocol === 'file:')
      ? 'Static export — this saved file is one date only. Run `python3 mlb_schedule.py` (no flags) for a live picker.'
      : 'Live — pick any date above; recently viewed dates load instantly from cache. ⟳ Refresh pulls fresh data.';
  </script>
  <div class="legend">{esc(legend)}</div>
  {body}
  <footer>Generated {esc(datetime.now().strftime('%Y-%m-%d %H:%M'))} · statsapi.mlb.com + baseballsavant.mlb.com</footer>
</body>
</html>"""


# Per-date page cache: {day: (built_at_epoch, html)}. Repeat loads and flipping
# back to a recently viewed date are then instant. Tune with MLB_CACHE_TTL (secs).
_PAGE_CACHE = {}
CACHE_TTL = int(os.environ.get("MLB_CACHE_TTL", "600"))


def generate_page(day, force=False):
    """Fetch everything for `day` (YYYY-MM-DD) and return the full HTML string.

    Results are cached in memory for CACHE_TTL seconds; pass force=True to
    bypass the cache and rebuild from live data.
    """
    if not force:
        hit = _PAGE_CACHE.get(day)
        if hit and (time.time() - hit[0]) < CACHE_TTL:
            age = int(time.time() - hit[0])
            print(f"[cache] {day} served from cache ({age}s old, "
                  f"TTL {CACHE_TTL}s).")
            return hit[1]

    season = datetime.strptime(day, "%Y-%m-%d").year

    print(f"[1/6] Schedule for {day} ...")
    games = fetch_schedule(day)
    print(f"      {len(games)} game(s).")

    print("[2/6] Standings (rich team records) ...")
    records = fetch_standings(season)

    print("[3/6] Team hitting & pitching (bulk) ...")
    team_stats = fetch_team_stats(season)

    print("[4/6] Baseball Savant leaderboards (xERA + Statcast) ...")
    savant_x = fetch_savant_expected(season)
    savant_c = fetch_savant_custom(season)
    print(f"      Savant: {len(savant_x)} xstat rows, {len(savant_c)} statcast rows.")

    # Gather every probable pitcher and the team they face this day.
    team_ids = set()
    pitcher_jobs = {}   # pid -> opposing_team_id
    for g in games:
        t = g.get("teams", {})
        hid = t.get("home", {}).get("team", {}).get("id")
        aid = t.get("away", {}).get("team", {}).get("id")
        team_ids.update(x for x in (hid, aid) if x)
        hpp = (t.get("home", {}).get("probablePitcher") or {}).get("id")
        app = (t.get("away", {}).get("probablePitcher") or {}).get("id")
        if hpp:
            pitcher_jobs[hpp] = aid   # home pitcher faces away team
        if app:
            pitcher_jobs[app] = hid   # away pitcher faces home team

    # Each team's own probable starter (to exclude from its bullpen list).
    team_starter = {}
    for g in games:
        t = g.get("teams", {})
        for side in ("home", "away"):
            tid = t.get(side, {}).get("team", {}).get("id")
            spid = (t.get(side, {}).get("probablePitcher") or {}).get("id")
            if tid and spid:
                team_starter[tid] = spid

    print(f"[5/7] Active rosters ({len(team_ids)} teams, threaded) ...")
    roster_ids, roster_pitchers = {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for tid, res in zip(team_ids, ex.map(fetch_roster, team_ids)):
            roster_ids[tid], roster_pitchers[tid] = res

    print("[6/7] Bulk pitcher stats + recent-usage box scores (fatigue) ...")
    bulk = fetch_bulk_pitching(season)
    usage = fetch_recent_usage(day)
    game_day = datetime.strptime(day, "%Y-%m-%d").date()
    bullpens = {}
    for tid in team_ids:
        bullpens[tid] = build_bullpen(
            tid, team_starter.get(tid), roster_pitchers.get(tid, []),
            bulk, usage, savant_c, savant_x, game_day)
    print(f"      {sum(len(v) for v in bullpens.values())} relievers across "
          f"{len(bullpens)} teams; usage from {len(usage)} pitchers.")

    print(f"[7/7] Pitcher profiles + BvP ({len(pitcher_jobs)} pitchers, threaded) ...")
    pitchers, bvp_map = {}, {}

    def load_profile(pid):
        return pid, fetch_pitcher(pid, season, savant_x, savant_c)

    def load_bvp(item):
        pid, opp = item
        return pid, fetch_bvp(pid, opp, roster_ids.get(opp, set()))

    with ThreadPoolExecutor(max_workers=8) as ex:
        for pid, prof in ex.map(load_profile, list(pitcher_jobs)):
            pitchers[pid] = prof
        for pid, bvp in ex.map(load_bvp, list(pitcher_jobs.items())):
            bvp_map[pid] = bvp

    print("Handedness splits (vs RHP/LHP) + prev-season fallback (<150 PA) ...")
    hand_splits, lg_woba = fetch_team_hand_splits(season, team_ids)
    prev_splits, _ = fetch_team_hand_splits(season - 1, team_ids)
    hand_splits = apply_split_fallback(hand_splits, prev_splits)

    print("Rolling offense (7/15/30-day + season) + recency + odds ...")
    general_mult, _ = fetch_offense_windows(day, season)
    recency = {"pit": fetch_recent_pitching(day), "general": general_mult}
    # The live odds API only has UPCOMING games, so only spend a credit for
    # today's slate; other dates never call it (conserves the free quota).
    odds_map = odds.fetch_odds(day) if day == date.today().isoformat() else {}
    if odds_map:
        print(f"      live odds for {len(odds_map)} games.")

    print("Building page ...")
    league = compute_league(records, team_stats)
    league["woba"] = lg_woba
    page = build_html(games, records, team_stats, day, pitchers, bvp_map,
                      bullpens, league, hand_splits, recency, odds_map)
    _PAGE_CACHE[day] = (time.time(), page)
    return page


def valid_day(s):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def run_server(port, open_browser=True, start_day=None):
    """Serve the page locally so any date can be loaded live via the picker."""
    import http.server
    import socket
    from urllib.parse import urlparse, parse_qs

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            qs = parse_qs(parsed.query)
            day = (qs.get("date", [None])[0]) or date.today().isoformat()
            if not valid_day(day):
                day = date.today().isoformat()
            force = qs.get("refresh", ["0"])[0] in ("1", "true", "yes")
            try:
                page = generate_page(day, force=force)
            except Exception as e:  # noqa: BLE001 - keep the server alive
                page = f"<h1>Error building {esc(day)}</h1><pre>{esc(e)}</pre>"
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass  # quiet

    # Bind, trying a few ports in case the requested one is busy.
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = http.server.HTTPServer(("localhost", p), Handler)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        print(f"Could not bind any port in {port}-{port+19}.", file=sys.stderr)
        sys.exit(1)

    q = f"?date={start_day}" if start_day else ""
    url = f"http://localhost:{port}/{q}"
    print(f"\n  ✅ Open this in your browser:  {url}")
    print("     (the date picker at the top loads any date live)")
    print("     Press Ctrl+C here to stop.\n")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    argv = sys.argv[1:]
    open_browser = "--no-open" not in argv
    static = "--static" in argv
    rest = [a for a in argv if not a.startswith("--")]

    day = next((a for a in rest if valid_day(a)), None)
    if rest and any(not a.isdigit() and not valid_day(a) for a in rest):
        print("Date must be YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    # Default: run the live server so the on-page date picker works for ANY date.
    if not static:
        port = next((int(a) for a in rest if a.isdigit()), 8000)
        run_server(port, open_browser, start_day=day)
        return

    # --static: write a single self-contained file for one date (offline share).
    d = day or date.today().isoformat()
    page = generate_page(d)
    out = f"mlb_schedule_{d}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {out}")
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
