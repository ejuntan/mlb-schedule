#!/usr/bin/env python3
"""
bet_backtest.py — REAL-odds P/L backtest with strategy search.

Joins a local moneyline dataset (opening + closing lines per book, plus final
scores) to the prediction model (point-in-time, via backtest.py), then searches
for the best HONEST betting strategy:

  - opening vs closing line
  - consensus price vs best-of-books (line shopping)
  - a minimum-edge (value) threshold

Model predictions are cached to disk (predictions_cache.json) so re-tuning the
betting logic doesn't re-run the slow reconstruction. The best strategy is
chosen on a TRAIN split and reported OUT-OF-SAMPLE on the TEST split.

Usage:
    python3 bet_backtest.py --odds ~/Downloads/mlb_odds_dataset.json \\
        --start 2025-06-01 --end 2025-08-16 --split 2025-07-21 --out bet_report.html

Analysis only, not betting advice.
"""

import os
import json
import argparse
from datetime import datetime, timedelta

import model
import backtest

ALIAS = {"OAK": "ATH", "WAS": "WSH", "CHW": "CWS", "ARI": "AZ",
         "SFG": "SF", "SDP": "SD", "TBR": "TB", "KCR": "KC"}
CACHE_FILE = "predictions_cache.json"


def american_to_decimal(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return 1 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def team_abbr_map():
    d = backtest.get_json("https://statsapi.mlb.com/api/v1/teams?sportId=1")
    m = {}
    if d:
        for t in d.get("teams", []):
            if t.get("abbreviation"):
                m[t["abbreviation"].upper()] = t["id"]
    return m


def to_id(shortname, abbr):
    s = (shortname or "").upper()
    return abbr.get(s) or abbr.get(ALIAS.get(s, ""), None)


def _side_prices(books, which):
    """which='openingLine' or 'currentLine' -> (home_cons, away_cons, home_best, away_best)."""
    dh, da, nh, na, bh, ba = 0.0, 0.0, 0, 0, 0.0, 0.0
    for b in books:
        ln = b.get(which) or {}
        xh = american_to_decimal(ln.get("homeOdds"))
        xa = american_to_decimal(ln.get("awayOdds"))
        if xh:
            dh += xh; nh += 1; bh = max(bh, xh)
        if xa:
            da += xa; na += 1; ba = max(ba, xa)
    if not (nh and na):
        return None
    return {"home_cons": dh / nh, "away_cons": da / na,
            "home_best": bh, "away_best": ba}


def parse_day(games, abbr):
    out = {}
    for g in games:
        gv = g.get("gameView", {})
        if "Final" not in (gv.get("gameStatusText") or ""):
            continue
        hid = to_id(gv.get("homeTeam", {}).get("shortName"), abbr)
        aid = to_id(gv.get("awayTeam", {}).get("shortName"), abbr)
        hs, as_ = gv.get("homeTeamScore"), gv.get("awayTeamScore")
        if None in (hid, aid, hs, as_):
            continue
        books = g.get("odds", {}).get("moneyline", [])
        close = _side_prices(books, "currentLine")
        open_ = _side_prices(books, "openingLine") or close
        if close:
            out[(hid, aid, hs, as_)] = {"close": close, "open": open_}
    return out


def daterange(a, b):
    s = datetime.strptime(a, "%Y-%m-%d").date()
    e = datetime.strptime(b, "%Y-%m-%d").date()
    while s <= e:
        yield s
        s += timedelta(days=1)


# --- Phase A: predictions (cached) -------------------------------------------

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE))
        except Exception:
            return {}
    return {}


def gkey(ds, g):
    return f"{ds}|{g['home_id']}|{g['away_id']}|{g['home_score']}|{g['away_score']}"


def get_predictions(start, end):
    cache = load_cache()
    changed = False
    preds = {}
    for d in daterange(start, end):
        ds = d.isoformat()
        model_games = backtest.games_on(ds)
        if not model_games:
            continue
        need = [g for g in model_games if gkey(ds, g) not in cache]
        if need:
            store = backtest.asof_stats(d - timedelta(days=1))
            usage = backtest.recent_usage_asof(d)
            for g in need:
                pr = backtest.predict_historical(g, store, usage, d)
                cache[gkey(ds, g)] = [pr["p_home_raw"], pr["exp_total"]]
            changed = True
            print(f"  {ds}: +{len(need)} predictions (cache {len(cache)})", flush=True)
        for g in model_games:
            k = gkey(ds, g)
            if k in cache:
                preds[k] = {"date": ds, "game": g,
                            "p_home": cache[k][0], "exp_total": cache[k][1]}
    if changed:
        json.dump(cache, open(CACHE_FILE, "w"))
    return preds


# --- Phase B: betting evaluation ---------------------------------------------

def build_bets(preds, odds, abbr):
    day_odds_cache = {}
    bets = []
    for k, pr in preds.items():
        ds = pr["date"]
        if ds not in day_odds_cache:
            day_odds_cache[ds] = parse_day(odds.get(ds, []), abbr)
        g = pr["game"]
        okey = (g["home_id"], g["away_id"], g["home_score"], g["away_score"])
        o = day_odds_cache[ds].get(okey)
        if not o:
            continue
        p_home = pr["p_home"]
        pick_home = p_home >= 0.5
        p_pick = p_home if pick_home else 1 - p_home
        home_win = g["home_score"] > g["away_score"]
        side = "home" if pick_home else "away"
        prices = {}
        for line in ("open", "close"):
            for price in ("cons", "best"):
                prices[(line, price)] = o[line][f"{side}_{price}"]
        bets.append({
            "date": ds, "p_pick": p_pick, "pick_home": pick_home,
            "won": pick_home == home_win, "prices": prices,
            # for baselines
            "home_win": home_win,
            "close_home": o["close"]["home_cons"], "close_away": o["close"]["away_cons"],
        })
    return bets


def eval_strategy(bets, line, price, edge):
    n = wins = 0
    units = 0.0
    for b in bets:
        dec = b["prices"][(line, price)]
        if dec is None:
            continue
        implied = 1.0 / dec
        if b["p_pick"] - implied < edge:
            continue
        n += 1
        if b["won"]:
            wins += 1
            units += dec - 1
        else:
            units -= 1
    roi = units / n if n else 0.0
    return {"n": n, "wins": wins, "units": units, "roi": roi}


def search_best(bets, min_bets=60):
    grid = []
    for line in ("open", "close"):
        for price in ("cons", "best"):
            for edge in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
                r = eval_strategy(bets, line, price, edge)
                if r["n"] >= min_bets:
                    grid.append(((line, price, edge), r))
    grid.sort(key=lambda x: x[1]["roi"], reverse=True)
    return grid


def summarize(r, label):
    if not r["n"]:
        return f"  {label:<34} (no bets)"
    return (f"  {label:<34} bets {r['n']:4d}  "
            f"W {r['wins']}/{r['n']} ({r['wins']/r['n']*100:4.1f}%)  "
            f"units {r['units']:+7.2f}  ROI {r['roi']*100:+5.1f}%")


def report(bets, split, out):
    train = [b for b in bets if b["date"] < split]
    test = [b for b in bets if b["date"] >= split]

    naive = eval_strategy(bets, "close", "cons", -9.0)   # bet EVERY pick
    value = eval_strategy(bets, "close", "best", 0.0)     # value + line shop

    print("\n" + "=" * 78)
    print(f" REAL-ODDS STRATEGY SEARCH   ({len(bets)} bets;"
          f" train {len(train)} < {split} <= test {len(test)})")
    print("=" * 78)
    print(summarize(naive, "Bet EVERY pick, close consensus"))
    print(summarize(value, "Value bets (edge>0) + line-shop"))

    grid = search_best(train)
    print("\n Top strategies by TRAIN ROI (line / price / min-edge):")
    for (line, price, edge), r in grid[:6]:
        print(f"   {line:>5}/{price:<4} edge>={edge:.2f} :"
              f" {r['n']:4d} bets  ROI {r['roi']*100:+5.1f}%")

    if grid:
        best_cfg = grid[0][0]
        line, price, edge = best_cfg
        tr = eval_strategy(train, line, price, edge)
        te = eval_strategy(test, line, price, edge)
        print(f"\n Best TRAIN config: {line}/{price} edge>={edge:.2f}")
        print(summarize(tr, "  -> on TRAIN"))
        print(summarize(te, "  -> on TEST (out-of-sample)"))
        full = eval_strategy(bets, line, price, edge)
        print(summarize(full, "  -> on FULL range"))
    print("=" * 78)
    print(" Real closing/opening moneylines. Best config is picked on train and"
          " shown out-of-sample on test to avoid curve-fitting. Not betting advice.\n")

    if out:
        write_html(bets, naive, grid, split, out)


def write_html(bets, naive, grid, split, out):
    train = [b for b in bets if b["date"] < split]
    test = [b for b in bets if b["date"] >= split]
    rows = ""
    for (line, price, edge), r in grid[:8]:
        te = eval_strategy(test, line, price, edge)
        rows += (f"<tr><td>{line}/{price} · edge≥{edge:.2f}</td>"
                 f"<td>{r['n']}</td><td>{r['roi']*100:+.1f}%</td>"
                 f"<td>{te['n']}</td><td>{te['roi']*100:+.1f}%</td>"
                 f"<td>{te['units']:+.1f}</td></tr>")
    best = grid[0][0] if grid else None
    hero = ""
    if best:
        te = eval_strategy(test, *best)
        cls = "pos" if te["units"] >= 0 else "neg"
        hero = (f"<div class=big class={cls}>{te['units']:+.1f} units</div>"
                f"<div class=k>best train config ({best[0]}/{best[1]}, edge≥{best[2]:.2f}) "
                f"on out-of-sample test — {te['n']} bets, ROI "
                f"<b class={cls}>{te['roi']*100:+.1f}%</b></div>")
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Real-odds strategy search</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;
color:#e6edf3;max-width:780px;margin:24px auto;padding:0 16px;line-height:1.5}}
h1{{font-size:22px}}table{{border-collapse:collapse;width:100%;margin:10px 0}}
td,th{{border:1px solid #2a3241;padding:6px 9px;text-align:center;font-size:13px}}
th{{color:#8b949e}}.k{{color:#8b949e}}.big{{font-size:30px;font-weight:700;margin:6px 0}}
.pos{{color:#3fb950}}.neg{{color:#f85149}}
.card{{background:#161b22;border:1px solid #2a3241;border-radius:10px;padding:16px;margin:10px 0}}</style>
</head><body>
<h1>⚾ Real-odds strategy search</h1>
<div class=card><b>Bet every pick</b> at closing consensus:
<b class="{'neg' if naive['units']<0 else 'pos'}">{naive['units']:+.1f}u</b>
(ROI {naive['roi']*100:+.1f}%, {naive['n']} bets) — the flat-betting result we're trying to beat.</div>
<div class=card>{hero}</div>
<div class=card><b>Strategies</b> — chosen on train, verified on test
<table><tr><th>Strategy</th><th>train n</th><th>train ROI</th>
<th>test n</th><th>test ROI</th><th>test units</th></tr>{rows}</table></div>
<p class=k>Point-in-time predictions (recency-weighted, regressed; no look-ahead)
vs real opening/closing moneylines. Best strategy selected on the train split and
reported out-of-sample. Analysis only, not betting advice.</p>
</body></html>"""
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--split", default="2025-07-21")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("Computing/loading predictions ...", flush=True)
    preds = get_predictions(args.start, args.end)
    print(f"  {len(preds)} games predicted.", flush=True)

    print("Loading odds ...", flush=True)
    odds = json.load(open(args.odds))
    abbr = team_abbr_map()
    bets = build_bets(preds, odds, abbr)
    print(f"  {len(bets)} games matched to odds.", flush=True)
    if not bets:
        print("No matches.")
        return
    report(bets, args.split, args.out)


if __name__ == "__main__":
    main()
