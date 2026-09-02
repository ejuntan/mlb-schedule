#!/usr/bin/env python3
"""
odds.py — optional live moneyline odds + value layer.

Dormant unless the environment variable ODDS_API_KEY is set (a free key from
https://the-odds-api.com). When present, fetch_odds() returns current MLB
moneylines keyed by (home_team_id, away_team_id), and evaluate() compares the
model's win probability to the market's implied probability to flag value.

No key => fetch_odds() returns {} and the site renders exactly as before.
"""

import os
import ssl
import json
import time
from urllib.request import urlopen, Request

ODDS_HOST = "https://api.the-odds-api.com/v4"
STATS = "https://statsapi.mlb.com/api/v1"
_CTX = ssl._create_unverified_context()


def _get(url):
    try:
        with urlopen(Request(url, headers={"User-Agent": "mlb-odds/1.0"}),
                     timeout=30, context=_CTX) as r:
            return json.loads(r.read().decode())
    except Exception:
        try:
            with urlopen(Request(url, headers={"User-Agent": "mlb-odds/1.0"}),
                         timeout=30, context=ssl.create_default_context()) as r:
                return json.loads(r.read().decode())
        except Exception:
            return None


def american_to_decimal(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return 1 + (o / 100.0 if o > 0 else 100.0 / abs(o))


_name_to_id = None


def _team_name_map():
    """{lowercased full team name: id} from StatsAPI, built once."""
    global _name_to_id
    if _name_to_id is not None:
        return _name_to_id
    _name_to_id = {}
    d = _get(f"{STATS}/teams?sportId=1")
    if d:
        for t in d.get("teams", []):
            for key in (t.get("name"), t.get("teamName"), t.get("clubName"),
                        t.get("shortName")):
                if key:
                    _name_to_id[key.lower()] = t["id"]
    return _name_to_id


def _to_id(name):
    if not name:
        return None
    m = _team_name_map()
    n = name.lower()
    if n in m:
        return m[n]
    # loose fallback: last word (e.g. "Yankees") matches clubName
    for k, v in m.items():
        if k in n or n in k:
            return v
    return None


# Cache the (single) upcoming-odds response so repeated page builds — e.g. the
# 15-minute prewarm — don't each spend an API credit. Tune with ODDS_CACHE_TTL.
_odds_cache = {"ts": 0, "data": {}}


def fetch_odds(day=None, regions="us", best_line=True):
    """
    {(home_id, away_id): {"home_dec", "away_dec", "home_ml", "away_ml", "books"}}
    for upcoming MLB games. Empty dict if ODDS_API_KEY is unset or the call fails.
    Cached for ODDS_CACHE_TTL seconds to conserve the API quota.
    """
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    ttl = int(os.environ.get("ODDS_CACHE_TTL", "1800"))  # 30 min
    if _odds_cache["data"] and (time.time() - _odds_cache["ts"]) < ttl:
        return _odds_cache["data"]
    url = (f"{ODDS_HOST}/sports/baseball_mlb/odds?apiKey={key}"
           f"&regions={regions}&markets=h2h&oddsFormat=american")
    data = _get(url)
    if not isinstance(data, list):
        return {}
    out = {}
    for g in data:
        hid = _to_id(g.get("home_team"))
        aid = _to_id(g.get("away_team"))
        if not (hid and aid):
            continue
        h_decs, a_decs, nbooks = [], [], 0
        for bk in g.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                nbooks += 1
                for oc in mk.get("outcomes", []):
                    dec = american_to_decimal(oc.get("price"))
                    if dec is None:
                        continue
                    if _to_id(oc.get("name")) == hid:
                        h_decs.append(dec)
                    elif _to_id(oc.get("name")) == aid:
                        a_decs.append(dec)
        if not (h_decs and a_decs):
            continue
        pick = max if best_line else (lambda xs: sum(xs) / len(xs))
        hd, ad = pick(h_decs), pick(a_decs)
        out[(hid, aid)] = {
            "home_dec": hd, "away_dec": ad,
            "home_ml": _dec_to_american(hd), "away_ml": _dec_to_american(ad),
            "books": nbooks,
        }
    _odds_cache["ts"] = time.time()
    _odds_cache["data"] = out
    return out


def _dec_to_american(dec):
    if not dec or dec <= 1:
        return None
    if dec >= 2:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))




def evaluate(p_home, o):
    """
    Compare model prob to market. Returns a dict for display, or None if no odds.
        p_home : model P(home win), 0..1
        o      : one fetch_odds() value, or None
    """
    if not o:
        return None
    hd, ad = o.get("home_dec"), o.get("away_dec")
    if not (hd and ad):
        return None
    # De-vig the two-way implied probs to a fair market probability.
    imp_h, imp_a = 1 / hd, 1 / ad
    s = imp_h + imp_a
    fair_h = imp_h / s if s else 0.5

    pick_home = p_home >= 0.5
    p_pick = p_home if pick_home else 1 - p_home
    dec_pick = hd if pick_home else ad
    implied_pick = (1 / dec_pick)
    edge = p_pick - implied_pick
    return {
        "home_ml": o.get("home_ml"), "away_ml": o.get("away_ml"),
        "home_dec": round(hd, 2), "away_dec": round(ad, 2),
        "pick_dec": round(dec_pick, 2),
        "books": o.get("books"),
        "market_home_pct": round(fair_h * 100),
        "model_home_pct": round(p_home * 100),
        "pick_home": pick_home,
        "edge_pct": round(edge * 100, 1),
        "value": edge > 0,
    }
