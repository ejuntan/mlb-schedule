#!/usr/bin/env python3
"""
totals_backtest.py — test the run-total model against the real over/under market.

The moneyline market is razor-efficient; totals get less sharp money, so this is
where a marginal model is likeliest to find edge. This asks three things, in order
of importance:

  (1) PREDICTIVE: is the model's expected total closer to the actual total than the
      closing line is?  (model MAE < market MAE = genuine information the line lacks)
  (2) DIRECTIONAL: when the model disagrees with the line by >= T runs, does that
      side (over/under) actually hit more than the ~52.4% needed to beat the vig?
  (3) CLV: betting the model's side at the OPENING line, does the line then move our
      way by close?  (fast-converging edge signal, same logic as the moneyline CLV)

Uses the cached point-in-time predictions (exp_total) from bet_backtest, so it is
strictly pre-game and reproducible. Pure stdlib. Not betting advice.

    python3 totals_backtest.py --odds ~/Downloads/mlb_odds_dataset.json \
        --start 2025-03-27 --end 2025-08-16 --split 2025-07-01 --out totals_report.txt
"""

import json
import argparse

import backtest
import bet_backtest as B


def american_to_decimal(o):
    return B.american_to_decimal(o)


def parse_totals(games, abbr):
    """Per game key (hid,aid,hs,as_) -> consensus opening/closing total + O/U odds."""
    out = {}
    for g in games:
        gv = g.get("gameView", {})
        if "Final" not in (gv.get("gameStatusText") or ""):
            continue
        hid = B.to_id(gv.get("homeTeam", {}).get("shortName"), abbr)
        aid = B.to_id(gv.get("awayTeam", {}).get("shortName"), abbr)
        hs, as_ = gv.get("homeTeamScore"), gv.get("awayTeamScore")
        if None in (hid, aid, hs, as_):
            continue
        books = g.get("odds", {}).get("totals", [])
        rec = {}
        for which, tag in (("openingLine", "open"), ("currentLine", "close")):
            tot_sum = ov_sum = un_sum = 0.0
            nt = nov = nun = 0
            for b in books:
                ln = b.get(which) or {}
                t = ln.get("total")
                if t is not None:
                    tot_sum += t; nt += 1
                ov = american_to_decimal(ln.get("overOdds"))
                un = american_to_decimal(ln.get("underOdds"))
                if ov:
                    ov_sum += ov; nov += 1
                if un:
                    un_sum += un; nun += 1
            if nt:
                rec[tag] = {"line": tot_sum / nt,
                            "over": ov_sum / nov if nov else None,
                            "under": un_sum / nun if nun else None}
        if "close" in rec:
            rec.setdefault("open", rec["close"])
            out[(hid, aid, hs, as_)] = rec
    return out


def build(preds, odds, abbr):
    day_tot = {}
    rows = []
    for k, pr in preds.items():
        ds = pr["date"]
        if ds not in day_tot:
            day_tot[ds] = parse_totals(odds.get(ds, []), abbr)
        g = pr["game"]
        key = (g["home_id"], g["away_id"], g["home_score"], g["away_score"])
        t = day_tot[ds].get(key)
        if not t:
            continue
        actual = g["home_score"] + g["away_score"]
        rows.append({"date": ds, "exp": pr["exp_total"], "actual": actual,
                     "open": t["open"], "close": t["close"]})
    return rows


def mae(rows, pred_key):
    xs = [abs(r[pred_key] - r["actual"]) for r in rows]
    return sum(xs) / len(xs) if xs else 0.0


def bias(rows, pred_key):
    xs = [r[pred_key] - r["actual"] for r in rows]
    return sum(xs) / len(xs) if xs else 0.0


def directional(rows, thresh, price_field="close"):
    """Model bets over/under vs the closing line when |exp-line|>=thresh runs.
    Settle vs actual (push on exact). Bet 1u at that line's consensus price."""
    n = wins = push = 0
    units = 0.0
    for r in rows:
        line = r[price_field]["line"]
        if abs(r["exp"] - line) < thresh:
            continue
        over = r["exp"] > line
        dec = r[price_field]["over" if over else "under"]
        if not dec:
            continue
        n += 1
        if r["actual"] == line:
            push += 1
            continue
        hit = (r["actual"] > line) if over else (r["actual"] < line)
        if hit:
            wins += 1
            units += dec - 1
        else:
            units -= 1
    dec_n = n - push
    return {"n": n, "push": push, "wins": wins,
            "wr": 100 * wins / dec_n if dec_n else 0.0,
            "units": units, "roi": 100 * units / n if n else 0.0}


def clv_totals(rows, thresh):
    """Bet model's side at the OPENING line; did the line move our way by close?"""
    n = beat = worse = same = 0
    move = 0.0
    for r in rows:
        oline = r["open"]["line"]
        if abs(r["exp"] - oline) < thresh:
            continue
        over = r["exp"] > oline
        cline = r["close"]["line"]
        n += 1
        d = cline - oline
        # over wants the number to FALL; under wants it to RISE.
        favor = (-d) if over else d
        move += favor
        if favor > 1e-9:
            beat += 1
        elif favor < -1e-9:
            worse += 1
        else:
            same += 1
    return {"n": n, "beat": beat, "worse": worse, "same": same,
            "beat_pct": 100 * beat / n if n else 0.0,
            "avg_move": move / n if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--split", default="2025-07-01")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("Loading predictions + odds ...", flush=True)
    preds = B.get_predictions(args.start, args.end)
    odds = json.load(open(args.odds))
    rows = build(preds, odds, B.team_abbr_map())

    L = []
    def out(s=""):
        print(s, flush=True); L.append(s)

    out("=" * 80)
    out(f" TOTALS (OVER/UNDER) BACKTEST   {len(rows)} games with a market total")
    out("=" * 80)

    json.dump(rows, open("totals_rows.json", "w"))   # cache for instant re-analysis

    out("\n(1) PREDICTIVE — is the model's total closer to reality than the line?")
    out(f"    model  MAE {mae(rows,'exp'):.3f}   bias {bias(rows,'exp'):+.3f} runs")
    close_rows = [dict(r, cl=r["close"]["line"]) for r in rows]
    out(f"    market MAE {mae(close_rows,'cl'):.3f}   (closing line vs actual)")
    out(f"    open   MAE {mae([dict(r, ol=r['open']['line']) for r in rows],'ol'):.3f}")
    # De-bias check: scale the model total (calibration fix) and re-measure.
    tr = [r for r in rows if r["date"] < args.split]
    te = [r for r in rows if r["date"] >= args.split]
    for sc in (1.0, 1.033):
        sr = [dict(r, s=r["exp"] * sc) for r in rows]
        srt = [dict(r, s=r["exp"] * sc) for r in tr]
        sre = [dict(r, s=r["exp"] * sc) for r in te]
        out(f"    scale x{sc:.3f}: MAE {mae(sr,'s'):.3f} bias {bias(sr,'s'):+.3f}"
            f"  | train MAE {mae(srt,'s'):.3f}  test MAE {mae(sre,'s'):.3f}")
    out("    -> model MAE < market MAE means the model knows something the line doesn't.")

    out("\n(2) DIRECTIONAL — bet model's side vs CLOSING line, by disagreement size:")
    out("    (need ~52.4% to beat -110 vig)")
    for th in (0.0, 0.5, 1.0, 1.5, 2.0):
        d = directional(rows, th)
        out(f"    |edge|>={th:.1f}r : {d['n']:4d} bets ({d['push']} push)  "
            f"W {d['wins']} ({d['wr']:4.1f}%)  units {d['units']:+7.2f}  "
            f"ROI {d['roi']:+5.1f}%")

    out("\n(3) CLV — bet model's side at OPEN, did the line move our way by close?")
    for th in (0.0, 0.5, 1.0):
        c = clv_totals(rows, th)
        out(f"    |edge|>={th:.1f}r : {c['n']:4d} bets  moved-our-way "
            f"{c['beat']}/{c['n']} ({c['beat_pct']:4.1f}%)  "
            f"avg line move {c['avg_move']:+.3f}r (+=our way)")

    # train/test on the headline threshold
    split = args.split
    tr = [r for r in rows if r["date"] < split]
    te = [r for r in rows if r["date"] >= split]
    out(f"\n    Out-of-sample check (split {split}, |edge|>=1.0r):")
    for label, sub in (("train", tr), ("test ", te)):
        d = directional(sub, 1.0); c = clv_totals(sub, 1.0)
        out(f"      {label}: {d['n']:4d} bets  W {d['wr']:4.1f}%  ROI {d['roi']:+5.1f}%"
            f"   | CLV beat {c['beat_pct']:4.1f}% ({c['n']} bets)")

    out("\n" + "=" * 80)
    out(" Model MAE vs market MAE is the honest headline; directional ROI on small")
    out(" samples is noisy, CLV converges faster. Not betting advice.")
    out("=" * 80)

    if args.out:
        open(args.out, "w").write("\n".join(L) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
